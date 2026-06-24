"""
ablation_study.py
=================
アブレーション解析スクリプト

各コンポーネント (GainNet, ReactionNet, ActionNet) を
それぞれ無効化した場合の性能を比較する。

使い方:
    python ablation_study.py --timesteps 100000 --seeds 0 1 2

生成物:
    ablation_results.csv  -- plot_results.py で可視化に使う
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.dirname(current_dir)
for p in [root_dir,
          os.path.join(root_dir, 'GymEnv'),
          os.path.join(root_dir, 'NN'),
          os.path.join(root_dir, 'CPG')]:
    sys.path.append(p)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
N_ENVS = 16  # 並列環境数（CPUコア数に合わせて調整）
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit


# アブレーション条件: キー = 条件名, 値 = 環境へのフラグ
# ※ 各環境クラスが disable_gain / disable_reaction / disable_action
#   等のフラグをサポートしている前提。サポートしていない場合は
#   フラグを渡しても無視されるため、手動でラッパーを追加する。
ABLATION_CONDITIONS = {
    "full_model":        dict(),                          # フルモデル (比較基準)
    "no_gain_net":       dict(disable_gain=True),         # GainNet 無効
    "no_reaction_net":   dict(disable_reaction=True),     # ReactionNet 無効
    "no_action_net":     dict(disable_action=True),       # ActionNet 無効 (CPGのみ)
    "no_cpg_modulation": dict(disable_cpg_mod=True),      # CPG位相変調なし
}


def make_ablation_env(baseline_cls, extra_kwargs, seed, max_episode_steps=4000,
                      use_subproc=False):
    def _init():
        try:
            env = baseline_cls(**extra_kwargs)
        except TypeError:
            # フラグ非対応の環境クラスは無視して通常起動
            env = baseline_cls()
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        return env
    if use_subproc and N_ENVS > 1:
        env = SubprocVecEnv([_init for _ in range(N_ENVS)])
    else:
        env = DummyVecEnv([_init])
    env = VecNormalize(env, norm_obs=True, norm_reward=False,
                       clip_obs=10., clip_reward=10.)
    return env


def run_condition(condition_name, extra_kwargs, baseline_cls, seeds, timesteps,
                  out_dir, eval_episodes=10, log_dir=None):
    seed_means = []
    for seed in seeds:
        np.random.seed(seed)
        set_random_seed(seed)

        env = make_ablation_env(baseline_cls, extra_kwargs, seed, use_subproc=True)  # 学習は並列

        try:
            from GymEnv.Tegotae_Policy import Tegotae_Policy
            policy = Tegotae_Policy
        except ImportError:
            policy = "MlpPolicy"

        # TensorBoardログ出力先
        tb_log = log_dir if log_dir else os.path.join(out_dir, "tb_logs")

        model = PPO(policy, env, seed=seed, verbose=0,
                    learning_rate=3e-4, n_steps=4096, batch_size=512,
                    n_epochs=5, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, device="auto",
                    tensorboard_log=tb_log,
                    policy_kwargs=dict(squash_output=False))
        
        eval_env_cb = make_ablation_env(baseline_cls, extra_kwargs, seed + 9999, use_subproc=False)  # 評価はDummy
        eval_env_cb.training = False
        eval_env_cb.norm_reward = False
        eval_cb = EvalCallback(
            eval_env_cb,
            n_eval_episodes=eval_episodes,
            eval_freq=40960,  # 10ロールアウトごと
            deterministic=True,
            render=False,
            verbose=0,
        )

        model.learn(total_timesteps=timesteps,
                    callback=eval_cb,
                    tb_log_name=f"{condition_name}_s{seed}",
                    )

        seed_out = os.path.join(out_dir, condition_name, f"seed_{seed}")
        os.makedirs(seed_out, exist_ok=True)
        model.save(os.path.join(seed_out, "model"))
        stats_path = os.path.join(seed_out, "vecnorm.pkl")
        env.save(stats_path)
        env.close()

        # 評価
        eval_env = make_ablation_env(baseline_cls, extra_kwargs, seed + 5000)
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        rewards = []
        for _ in range(eval_episodes):
            res = eval_env.reset()
            obs = res[0] if isinstance(res, tuple) else res
            done = False
            total = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step = eval_env.step(action)
                obs, r = step[0], step[1]
                terminated = step[2]
                truncated  = step[3] if len(step) == 5 else False
                total += float(np.squeeze(r))
                done = bool(np.squeeze(terminated) or np.squeeze(truncated))
            rewards.append(total)
        eval_env.close()
        

        mean_r = float(np.mean(rewards))
        seed_means.append(mean_r)
        print(f"  [{condition_name}] seed={seed}  mean={mean_r:.1f}")

    return dict(
        Condition=condition_name,
        Mean_reward=round(np.mean(seed_means), 2),
        Std_reward= round(np.std(seed_means),  2),
        N=len(seeds),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", "-t", type=int, default=100_000)
    parser.add_argument("--seeds",     "-s", type=int, nargs="*", default=[0,1,2])
    parser.add_argument("--output-dir","-o", default="ablation_results")
    parser.add_argument("--log-dir",   "-l", default=None,
                        help="TensorBoardログ出力先")
    parser.add_argument("--num-workers", "-w", type=int, default=None,
                        help="並列処理数（デフォルト：CPUコア数）")
    args = parser.parse_args()

    try:
        from GymEnv.Tegotae_Hopper_PPO_V3 import Tegotae_Hopper_PPO_v2_Env as EnvCls
    except ImportError:
        print("環境クラスが見つかりません。GymEnv ディレクトリを確認してください。")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 並列処理数の決定
    num_workers = args.num_workers or max(1, mp.cpu_count() // 2)  # 半数のコアを使用
    print(f"並列処理数: {num_workers}")
    
    tasks = []
    for cond_name, extra_kwargs in ABLATION_CONDITIONS.items():
        tasks.append((cond_name, extra_kwargs, EnvCls, args.seeds, args.timesteps, 
                     args.output_dir, 10, args.log_dir))
    
    rows = []
    print(f"\n=== {len(ABLATION_CONDITIONS)}個の条件を並列実行 ===\n")
    
    # 複数プロセスで条件を並列実行
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_cond = {
            executor.submit(run_condition, *task): task[0] 
            for task in tasks
        }
        
        for future in as_completed(future_to_cond):
            cond_name = future_to_cond[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"✓ {cond_name}: mean={row['Mean_reward']:.1f} ± {row['Std_reward']:.1f}")
            except Exception as e:
                print(f"✗ {cond_name} でエラー: {e}")
    
    # 結果をソート（元の条件の順序を保持）
    rows.sort(key=lambda r: list(ABLATION_CONDITIONS.keys()).index(r['Condition']))
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nアブレーション結果 → {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    # マルチプロセッシングのコンテキストを設定
    mp.set_start_method('spawn', force=True)
    main()