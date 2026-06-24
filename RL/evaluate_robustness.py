"""
evaluate_robustness.py
======================
学習済みモデルを用いたロバスト性評価スクリプト

使用方法:
  python evaluate_robustness.py \
    --baseline default \
    --model-dir /path/to/TH_PPO_models/default/seed_0 \
    --output-dir results/robustness_eval

特徴:
  - 複数ベースライン対応 (default, gainonly, torque, nofeedback)
  - 全堅牢性条件をテスト (ノイズ, 質量, 外乱など)
  - CSV, JSON, Markdown, グラフ (matplotlib) で結果出力
  - CoT⁻¹, 跳躍高さ, 姿勢誤差, 周期など複合指標を集計
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

# OpenMP競合回避
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# パス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
for p in [root_dir,
          os.path.join(root_dir, 'GymEnv'),
          os.path.join(root_dir, 'NN'),
          os.path.join(root_dir, 'CPG')]:
    if p not in sys.path:
        sys.path.append(p)


# =============================================================================
# ユーティリティ関数
# =============================================================================
def safe_mean_std(arr, min_length=2):
    """
    NaN/Inf 値をフィルタリングして安全に平均と標準偏差を計算
    
    Args:
        arr: 入力配列
        min_length: 最小要素数（満たさない場合は (0.0, 0.0) を返す）
    
    Returns:
        (mean_val, std_val): 平均と標準偏差
    """
    arr = np.asarray(arr, dtype=float)
    # NaN と Inf をフィルタリング
    valid_mask = np.isfinite(arr)
    valid_arr = arr[valid_mask]
    
    if len(valid_arr) >= min_length:
        mean_val = float(np.mean(valid_arr))
        std_val = float(np.std(valid_arr))
        return mean_val, std_val
    else:
        return 0.0, 0.0


def safe_cv(values, min_length=2):
    """
    NaN/Inf 値をフィルタリングして変動係数を安全に計算
    変動係数 = 標準偏差 / 平均
    
    Args:
        values: 入力配列
        min_length: 最小要素数
    
    Returns:
        cv: 変動係数（計算できない場合は 0.0）
    """
    mean_val, std_val = safe_mean_std(values, min_length=min_length)
    if mean_val > 1e-6:
        return float(std_val / mean_val)
    else:
        return 0.0


# =============================================================================
# 環境ファクトリ
# =============================================================================
def make_env(baseline: str,
             render_mode=None,
             noise_std: float = 0.0,
             mass_scale: float = 1.0,
             ext_force: float = 0.0,
             max_episode_steps: int = 4000,
             seed: int = 0):
    """
    指定条件の環境を生成して返す
    """
    kwargs = dict(render_mode=render_mode,
                  noise_std=noise_std,
                  mass_scale=mass_scale,
                  ext_force=ext_force)

    if baseline == "default":
        from GymEnv.Tegotae_Hopper_PPO_V3 import Tegotae_Hopper_PPO_v2_Env as EnvCls
    elif baseline == "gainonly":
        from GymEnv.Tegotae_Hopper_PPO_V3_gainonly import Tegotae_Hopper_PPO_v2_Env as EnvCls
    elif baseline == "nofeedback":
        from GymEnv.Tegotae_hopper_PPO_v3_nofeedback import Tegotae_Hopper_PPO_v2_Env as EnvCls
    elif baseline == "torque":
        from GymEnv.DirectTorque_Hopper import DirectTorque_Hopper as EnvCls
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    env = EnvCls(**kwargs)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env


def make_vec_env(baseline, noise_std, mass_scale, ext_force, seed,
                 max_episode_steps=4000):
    """
    ベクトル化環境を作成（単一環境）
    """
    def _init():
        return make_env(baseline, noise_std=noise_std,
                        mass_scale=mass_scale, ext_force=ext_force,
                        max_episode_steps=max_episode_steps, seed=seed)
    env = DummyVecEnv([_init])
    env = VecNormalize(env, norm_obs=True, norm_reward=False,
                       clip_obs=10., clip_reward=10.)
    return env


# =============================================================================
# ロバスト性評価
# =============================================================================
def evaluate_robustness(model, baseline: str, stats_path: str, n_trials: int = 5):
    """
    学習済みモデルをさまざまな条件下でテストする
    
    Returns:
        results: {condition_name: {mean, std, ...}, ...}
    """
    conditions = {
        "nominal":            dict(noise_std=0.0,  mass_scale=1.0,  ext_force=0.0),
        "noise_low":          dict(noise_std=0.01, mass_scale=1.0,  ext_force=0.0),
        "noise_high":         dict(noise_std=0.05, mass_scale=1.0,  ext_force=0.0),
        "mass_light":         dict(noise_std=0.0,  mass_scale=0.8,  ext_force=0.0),
        "mass_heavy":         dict(noise_std=0.0,  mass_scale=1.2,  ext_force=0.0),
        "ext_force_moderate": dict(noise_std=0.0,  mass_scale=1.0,  ext_force=10.0),
        "ext_force_large":    dict(noise_std=0.0,  mass_scale=1.0,  ext_force=30.0),
        "combined":           dict(noise_std=0.02, mass_scale=1.1,  ext_force=10.0),
    }

    results = {}

    for cond_name, cond_kwargs in conditions.items():
        print(f"  テスト中: {cond_name:25s} ...", end=" ", flush=True)
        
        rewards = []
        cot_inv_list_all = []
        max_height_list_all = []
        pose_error_list_all = []
        interval_list_all = []
        jump_reward_list_all = []
        cv_h_list_all = []
        term_list_all = []
        peak_grf_list_all = []
        cv_t_list_all = []

        for trial in range(n_trials):
            ev = make_vec_env(baseline, seed=trial+9000, **cond_kwargs)
            ev = VecNormalize.load(stats_path, ev)
            ev.training = False
            ev.norm_reward = False

            res = ev.reset()
            obs = res[0] if isinstance(res, tuple) else res
            done = False
            total = 0.0
            dt = ev.get_attr('dt')[0]

            # サイクル計測用変数
            cot_inv_ep = []
            max_height_ep = []
            pose_error_ep = []
            jump_reward_ep = []
            landing_step = -1
            step_count = 0
            interval_ep = []
            ep_terminated = False
            ep_peak_grf = 0.0
            prev_grf = 0.0
            cycle_z_max = -np.inf
            cycle_z_max_list = []
            cycle_energy = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step_res = ev.step(action)
                if len(step_res) == 5:
                    obs, r, terminated, truncated, infos = step_res
                else:
                    obs, r, terminated, infos = step_res
                    truncated = np.array([False])

                term_val = terminated[0] if isinstance(terminated, np.ndarray) else terminated
                trunc_val = truncated[0] if isinstance(truncated, np.ndarray) else truncated
                done = bool(term_val or trunc_val)

                info_dict = infos[0] if isinstance(infos, (list, tuple)) else infos

                # 報酬積算
                total += float(info_dict.get('reward', r[0] if isinstance(r, np.ndarray) else r))

                # 高さ・エネルギー・GRF取得
                z = float(info_dict.get('com_z', info_dict.get('height', 0.0)))
                power = float(info_dict.get('power', 0.0))
                grf_now = float(info_dict.get('peak_grf', info_dict.get('grf', 0.0)))
                pose_error = float(info_dict.get('pose_error', 0.0))

                max_height_ep.append(z)
                pose_error_ep.append(pose_error)
                cycle_energy += power * dt

                # 最大高さ更新
                if z > cycle_z_max and z > 0.1:
                    cycle_z_max = z

                # ピークGRF更新
                if grf_now > ep_peak_grf:
                    ep_peak_grf = grf_now

                # 着地判定
                just_landed = info_dict.get('just_landed', False)
                if (grf_now > 10.0 and prev_grf <= 10.0 and cycle_z_max > 0.01) or just_landed:
                    if landing_step >= 0:
                        interval_ep.append((step_count - landing_step) * dt)
                    landing_step = step_count
                    cycle_z_max_list.append(cycle_z_max)

                    # サイクル指標を計算
                    if cycle_energy > 1e-6:
                        cot_inv = 75.5 * 9.8 * cycle_z_max / cycle_energy
                        cot_inv_ep.append(cot_inv)
                    
                    if info_dict.get('efficiency_score') is not None:
                        jump_reward_ep.append(float(info_dict['efficiency_score']))

                    cycle_energy = 0.0
                    cycle_z_max = -np.inf

                # 早期終了検出
                if term_val and not trunc_val:
                    ep_terminated = True

                prev_grf = grf_now
                step_count += 1

            rewards.append(total)

            if cot_inv_ep:
                cot_inv_list_all.append(float(np.mean(cot_inv_ep)))
            if max_height_ep:
                max_height_list_all.append(float(np.max(max_height_ep)))
            if pose_error_ep:
                pose_error_list_all.append(float(np.mean(pose_error_ep)))
            if interval_ep:
                interval_list_all.append(float(np.mean(interval_ep)))
            if jump_reward_ep:
                jump_reward_list_all.append(float(np.mean(jump_reward_ep)))

            if len(cycle_z_max_list) >= 2:
                cv_h = safe_cv(cycle_z_max_list, min_length=2)
                cv_h_list_all.append(cv_h)

            term_list_all.append(1.0 if ep_terminated else 0.0)
            peak_grf_list_all.append(ep_peak_grf)

            if len(interval_ep) >= 2:
                cv_t = safe_cv(interval_ep, min_length=2)
                cv_t_list_all.append(cv_t)

            ev.close()

        # 条件別集計（NaN/Inf 対策）
        res_dict = {}
        
        # 報酬（基本統計）
        mean_r, std_r = safe_mean_std(rewards, min_length=1)
        res_dict['mean'] = mean_r
        res_dict['std'] = std_r
        rewards_valid = np.asarray(rewards)
        rewards_valid = rewards_valid[np.isfinite(rewards_valid)]
        res_dict['min'] = float(np.min(rewards_valid)) if len(rewards_valid) > 0 else 0.0
        res_dict['max'] = float(np.max(rewards_valid)) if len(rewards_valid) > 0 else 0.0
        
        # CoT逆数
        mean_cot, std_cot = safe_mean_std(cot_inv_list_all, min_length=1)
        res_dict['mean_cot_inv'] = mean_cot
        res_dict['std_cot_inv'] = std_cot
        
        # 最大高さ
        mean_h, std_h = safe_mean_std(max_height_list_all, min_length=1)
        res_dict['mean_max_height'] = mean_h
        res_dict['std_max_height'] = std_h
        
        # 姿勢誤差
        mean_pe, std_pe = safe_mean_std(pose_error_list_all, min_length=1)
        res_dict['mean_pose_error'] = mean_pe
        res_dict['std_pose_error'] = std_pe
        
        # 着地間隔
        mean_int, std_int = safe_mean_std(interval_list_all, min_length=1)
        res_dict['mean_interval'] = mean_int
        res_dict['std_interval'] = std_int
        
        # ジャンプ報酬
        mean_jr, std_jr = safe_mean_std(jump_reward_list_all, min_length=1)
        res_dict['mean_jump_reward'] = mean_jr
        res_dict['std_jump_reward'] = std_jr
        
        # CV_h
        mean_cv_h, std_cv_h = safe_mean_std(cv_h_list_all, min_length=1)
        res_dict['cv_h'] = mean_cv_h
        res_dict['std_cv_h'] = std_cv_h
        
        # 早期終了率
        mean_term, std_term = safe_mean_std(term_list_all, min_length=1)
        res_dict['term_rate'] = mean_term
        res_dict['std_term_rate'] = std_term
        
        # Peak GRF
        mean_grf, std_grf = safe_mean_std(peak_grf_list_all, min_length=1)
        res_dict['mean_peak_grf'] = mean_grf
        res_dict['std_peak_grf'] = std_grf
        
        # CV_T
        mean_cv_t, std_cv_t = safe_mean_std(cv_t_list_all, min_length=1)
        res_dict['cv_t'] = mean_cv_t
        res_dict['std_cv_t'] = std_cv_t

        results[cond_name] = res_dict
        print(f"✓ (reward={res_dict['mean']:.1f}±{res_dict['std']:.1f})")

    return results


# =============================================================================
# 結果出力
# =============================================================================
def save_results(results: dict, baseline: str, output_dir: str):
    """
    評価結果を複数形式で保存
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- CSV ----
    rows = []
    for cond_name, metrics in results.items():
        row = {'Condition': cond_name}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"robustness_{baseline}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved: {csv_path}")

    # ---- JSON ----
    json_path = os.path.join(output_dir, f"robustness_{baseline}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON saved: {json_path}")

    # ---- Markdown ----
    md_path = os.path.join(output_dir, f"robustness_{baseline}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Robustness Evaluation - {baseline}\n\n")
        f.write(f"Evaluated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # サマリーテーブル
        f.write("## Summary (Reward & Primary Metrics)\n\n")
        summary_cols = ['Condition', 'mean', 'std', 'min', 'max', 'mean_cot_inv', 'mean_max_height']
        df_summary = df[summary_cols] if all(col in df.columns for col in summary_cols) else df
        f.write(df_summary.to_markdown(index=False))
        f.write("\n\n")
        
        # 詳細テーブル
        f.write("## Detailed Metrics\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        # 指標説明
        f.write("## Metrics Description\n\n")
        f.write("| Metric | Description |\n|---|---|\n")
        f.write("| mean | Average accumulated reward per episode |\n")
        f.write("| std | Standard deviation of reward |\n")
        f.write("| mean_cot_inv | Average Cost of Transport (inverse) |\n")
        f.write("| mean_max_height | Average maximum hop height [m] |\n")
        f.write("| mean_pose_error | Average pose tracking error |\n")
        f.write("| mean_interval | Average landing interval [s] |\n")
        f.write("| cv_h | Coefficient of variation for hop height |\n")
        f.write("| cv_t | Coefficient of variation for hop period |\n")
        f.write("| term_rate | Early termination rate (0=none, 1=always) |\n")
        f.write("| mean_peak_grf | Average peak ground reaction force [N] |\n")
    print(f"✓ Markdown saved: {md_path}")

    return df


def plot_results(all_results: dict, output_dir: str):
    """
    複数ベースラインの結果をグラフで比較
    """
    os.makedirs(output_dir, exist_ok=True)

    # 条件名を統一（キーの一覧を取得）
    all_conditions = set()
    for baseline_results in all_results.values():
        all_conditions.update(baseline_results.keys())
    conditions = sorted(list(all_conditions))

    baselines = list(all_results.keys())
    metrics_to_plot = [
        ('mean', 'Accumulated Reward'),
        ('mean_cot_inv', 'CoT⁻¹ (Efficiency)'),
        ('mean_max_height', 'Maximum Hop Height [m]'),
        ('cv_h', 'Height Variation (CV_h)'),
        ('cv_t', 'Period Variation (CV_T)'),
        ('mean_peak_grf', 'Peak GRF [N]'),
        ('term_rate', 'Early Termination Rate'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx]
        x = np.arange(len(conditions))
        width = 0.2
        
        for i, baseline in enumerate(baselines):
            values = []
            errors = []
            for cond in conditions:
                if cond in all_results[baseline]:
                    v = all_results[baseline][cond].get(metric, 0.0)
                    e = all_results[baseline][cond].get(f'std_{metric}', 0.0)
                else:
                    v, e = 0.0, 0.0
                values.append(v)
                errors.append(e)
            
            ax.bar(x + i*width, values, width, label=baseline, yerr=errors, capsize=3)
        
        ax.set_xlabel('Condition')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x + width * (len(baselines)-1) / 2)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    # 余りのサブプロットを隠す
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "robustness_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {plot_path}")
    plt.close()


# =============================================================================
# メイン
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="学習済みモデルのロバスト性評価"
    )
    parser.add_argument("--baseline", "-b", nargs="*", default=["default"],
                        choices=["default", "gainonly", "torque", "nofeedback"],
                        help="評価対象のベースライン")
    parser.add_argument("--model-dir", "-m", required=True,
                        help="学習済みモデルのディレクトリ (例: TH_PPO_models/default/seed_0)")
    parser.add_argument("--output-dir", "-o", default="results/robustness_eval",
                        help="出力ディレクトリ")
    parser.add_argument("--trials", "-t", type=int, default=5,
                        help="各条件でのテスト試行回数")
    parser.add_argument("--compare", action="store_true",
                        help="複数モデルを比較（model-dir に複数ベースラインのパスをカンマ区切りで指定）")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("ロバスト性評価スクリプト")
    print("=" * 70)

    all_results = {}

    # 単一ベースラインの場合
    if not args.compare:
        baseline = args.baseline[0] if isinstance(args.baseline, list) else args.baseline
        model_dir = args.model_dir
        stats_path = os.path.join(model_dir, "vec_normalize.pkl")
        model_path = os.path.join(model_dir, "final_model.zip")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"Stats not found: {stats_path}")

        print(f"\nベースライン: {baseline}")
        print(f"モデル: {model_dir}")
        print(f"試行回数: {args.trials}\n")

        model = PPO.load(model_path)
        print("ロバスト性テスト実行中...\n")
        results = evaluate_robustness(model, baseline, stats_path, n_trials=args.trials)

        save_results(results, baseline, output_dir)
        all_results[baseline] = results

    # 複数ベースラインの比較
    else:
        baseline_dirs = {
            "default": "/mnt/ssd1/MKitao/vertical_hopper_3/RL/results/TH_v7_tegotae/models/default",
            "gainonly": "/mnt/ssd1/MKitao/vertical_hopper_3/RL/results/TH_v7_tegotae/models/gainonly",
            "torque": "/mnt/ssd1/MKitao/vertical_hopper_3/RL/results/TH_v7_torque/models/torque",
            "nofeedback": "/mnt/ssd1/MKitao/vertical_hopper_3/RL/results/TH_v7_tegotae/models/nofeedback",
        }

        for baseline in args.baseline:
            if baseline not in baseline_dirs:
                print(f"⚠ {baseline} is not in predefined paths. Skipping...")
                continue

            model_dir = baseline_dirs[baseline]
            seed_dir = os.path.join(model_dir, "seed_0")  # 最初のシードを使用

            if not os.path.isdir(seed_dir):
                print(f"⚠ Model directory not found: {seed_dir}. Skipping...")
                continue

            stats_path = os.path.join(seed_dir, "vec_normalize.pkl")
            model_path = os.path.join(seed_dir, "final_model.zip")

            if not os.path.isfile(model_path) or not os.path.isfile(stats_path):
                print(f"⚠ Model or stats not found for {baseline}. Skipping...")
                continue

            print(f"\n{'='*70}")
            print(f"ベースライン: {baseline}")
            print(f"{'='*70}")

            model = PPO.load(model_path)
            print(f"ロバスト性テスト実行中... ({args.trials} trials/condition)\n")
            results = evaluate_robustness(model, baseline, stats_path, n_trials=args.trials)

            save_results(results, baseline, output_dir)
            all_results[baseline] = results

        # グラフで比較
        if len(all_results) > 1:
            print(f"\n{'='*70}")
            print("複数ベースラインの比較グラフを生成中...")
            print(f"{'='*70}\n")
            plot_results(all_results, output_dir)

    print(f"\n{'='*70}")
    print("完了！")
    print(f"結果は以下に保存されました: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
