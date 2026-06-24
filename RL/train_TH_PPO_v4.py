"""
train_TH_PPO_v4.py
==================
改善版トレーニングスクリプト (Robomech/卒論 対応)

主な特徴:
  - ニューラルネット参照軌道（ref_traj_nn.pt）を使用
  - 複数乱数シードで実行し、平均±標準偏差を自動集計
  - ベースライン比較 (default / gainonly / torque)
  - 堅牢性テスト (ノイズ, 外乱, 質量スケール)
  - ハイパーパラメータをJSONで記録 + CSV/Markdownで結果表出力
  - TensorBoardコールバック強化 (位相速度, CPG状態)

参照軌道について:
  - 各環境では trajectories_nn/ref_traj_nn.pt を自動的に読み込み
  - NN参照軌道は複数周波数で学習された軌道を補間対応
  - 周波数は環境のCPG周波数に合わせて自動更新される
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import mujoco

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import TimeLimit

# OpenMP競合回避
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
N_ENVS = 16  # 並列環境数 (CPUコア数に応じて調整 - 高速化のため増加)

# パス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
for p in [root_dir,
          os.path.join(root_dir, 'GymEnv'),
          os.path.join(root_dir, 'NN'),
          os.path.join(root_dir, 'CPG')]:
    sys.path.append(p)

# =============================================================================
# カスタムTensorBoardコールバック (強化版)
# =============================================================================
class TegotaeCallback(BaseCallback):
    """
    infoディクショナリからカスタム値をTensorBoardに記録する。
    跳躍ごとの蓄積報酬, CPG位相速度, ゲイン平均なども追跡。
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_rewards: list = []

    def _on_step(self) -> bool:
        infos = self.locals['infos'][0]

        def log_if(key: str, tag: str):
            if key in infos:
                val = infos[key]
                try:
                    arr = np.asarray(val, dtype=float)
                    self.logger.record(tag, float(arr.flat[0]))
                except (TypeError, ValueError):
                    return

        # 内部状態
        log_if('gain',              'Tegotae/Gain_mean')
        log_if('action',            'Tegotae/Action')
        log_if('reaction',          'Tegotae/Reaction')
        log_if('phi',               'CPG/Phase')
        log_if('phi_dot',           'CPG/PhaseVelocity')
        log_if('height',            'State/Height')
        log_if('forward_vel',       'State/ForwardVel')
        log_if('step_reward',       'Reward/Step')
        log_if('accumulated_reward','Reward/JumpAccumulated')

        return True


# =============================================================================
# 学習中レンダリングコールバック
# =============================================================================
class RenderCallback(BaseCallback):
    """
    学習中に一定ステップごとにrender付きで評価エピソードを実行する。
    SubprocVecEnvとは別にDummyVecEnvで単独環境を作成して表示する。
    """
    def __init__(self, baseline: str, render_freq: int = 20000,
                 n_episodes: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.baseline = baseline
        self.render_freq = render_freq
        self.n_episodes = n_episodes
        self._render_env = None

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            try:
                if self._render_env is not None:
                    self._render_env.close()
                    self._render_env = None
                self._render_env = make_env(self.baseline, render_mode="human")
                for _ in range(self.n_episodes):
                    res = self._render_env.reset()
                    obs = res[0] if isinstance(res, tuple) else res
                    done = False
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        step_res = self._render_env.step(action)
                        if len(step_res) == 5:
                            obs, _, terminated, truncated, _ = step_res
                        else:
                            obs, _, terminated, _ = step_res
                            truncated = False
                        self._render_env.render()
                        done = bool(terminated or truncated)
                self._render_env.close()
                self._render_env = None
                print(f"[RenderCallback] {self.n_calls}ステップ時点のrender完了")
            except Exception as e:
                print(f"[RenderCallback] render失敗: {e}")
                import traceback; traceback.print_exc()
        return True

    def _on_training_end(self) -> None:
        if self._render_env is not None:
            self._render_env.close()
            self._render_env = None


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
    指定条件の環境を生成して返す (DummyVecEnv + VecNormalize なし)。
    
    参照軌道について:
      - 各環境は trajectories_nn/ref_traj_nn.pt を自動的に読み込む
      - NN参照軌道は複数周波数で学習された軌道を補間対応
      - reset() 時に周波数に応じてNN軌道が自動更新される
    
    ラッパーは呼び出し側で追加する。
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
        # 直接トルク制御ベースライン
        from GymEnv.DirectTorque_Hopper import DirectTorque_Hopper as EnvCls
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    env = EnvCls(**kwargs)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env


def make_vec_env(baseline, noise_std, mass_scale, ext_force, seed,
                 max_episode_steps=4000, use_subproc=True):
    """
    ベクトル化環境を作成し、正規化ラッパーを追加。デフォルトで SubprocVecEnv を使用し、高速化を実現。
    
    参照軌道について:
      - 各子環境で ref_traj_nn.pt が自動的に読み込まれる
      - CPG周波数に応じてNN軌道が自動生成される
    """
    def _init():
        return make_env(baseline, noise_std=noise_std,
                        mass_scale=mass_scale, ext_force=ext_force,
                        max_episode_steps=max_episode_steps, seed=seed)
    if use_subproc and N_ENVS > 1:
        env = SubprocVecEnv([_init for _ in range(N_ENVS)])
    else:
        env = DummyVecEnv([_init])
    env = VecNormalize(env, norm_obs=True, norm_reward=False,
                       clip_obs=10., clip_reward=10.)
    return env


# =============================================================================
# 1回の学習 + 評価
# =============================================================================
def run_one_seed(baseline, seed, args, models_dir, log_dir):
    """1シードで学習し、評価報酬を返す。
    
    参照軌道について:
      - 学習環境はref_traj_nn.ptを自動的に読み込む
      - 各エピソードでCPG周波数に応じて軌道が更新される
      - 評価時もNN参照軌道が使用される
    """
    np.random.seed(seed)
    import torch as _torch
    _torch.manual_seed(seed)
    set_random_seed(seed)

    # 学習環境
    env = make_vec_env(baseline,
                       noise_std=args.noise,
                       mass_scale=args.mass_scale,
                       ext_force=args.ext_force,
                       seed=seed)
    
    eval_env_cb = make_vec_env(baseline,
                           noise_std=args.noise,
                           mass_scale=args.mass_scale,
                           ext_force=args.ext_force,
                           seed=seed + 500,
                           use_subproc=False)  # EvalCallbackはDummyVecEnv

    eval_cb = EvalCallback(
        eval_env_cb,
        n_eval_episodes=10,          # 評価エピソード数
        eval_freq=4096,              # 何ステップごとに評価するか（高速化に合わせて短縮）
        deterministic=True,          # deterministic=True で評価
        render=False,
        verbose=0,
    )

    # カスタムポリシーを試みる (なければMlpPolicyにフォールバック)
    if baseline == "torque":
        # トルク直接制御は標準MLPポリシーを使用 (action_dim=2)
        policy = "MlpPolicy"
    elif baseline == "gainonly":
        from GymEnv.Tegotae_Policy_gainonly import Tegotae_gainonly_Policy
        policy = Tegotae_gainonly_Policy
    else:
        from GymEnv.Tegotae_Policy import Tegotae_Policy
        policy = Tegotae_Policy
        

    model = PPO(
        policy,
        env,
        seed=seed,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        tensorboard_log=log_dir,
        device="cpu",  # MlpPolicy推奨：CPUで高速処理
        policy_kwargs={},
    )
    # models_dir is expected to already point to a per-seed folder
    # (caller constructs os.path.join(bl_models_dir, f"seed_{seed}")),
    # so we use it directly rather than nesting again.
    seed_dir = models_dir
    os.makedirs(seed_dir, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=2048,  # n_steps と同期（高速化に合わせて削減）
        save_path=seed_dir,
        name_prefix="ppo_ckpt",
    )
    log_cb = TegotaeCallback()

    callbacks = [checkpoint_cb, log_cb, eval_cb]
    if getattr(args, 'render_freq', 0) > 0:
        render_cb = RenderCallback(
            baseline=baseline,
            render_freq=args.render_freq,
            n_episodes=1,
        )
        callbacks.append(render_cb)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=f"{args.experiment}_{baseline}_s{seed}",
        )
    except KeyboardInterrupt:
        print(f"\n[seed={seed}] 学習中断。保存します...")

    # モデル・正規化パラメータ保存
    model.save(os.path.join(seed_dir, "final_model"))
    stats_path = os.path.join(seed_dir, "vec_normalize.pkl")
    env.save(stats_path)
    env.close()

    # ---- 評価 ----
    eval_env = make_vec_env(baseline,
                            noise_std=args.noise,
                            mass_scale=args.mass_scale,
                            ext_force=args.ext_force,
                            seed=seed + 1000,
                            use_subproc=False)  # 評価はDummyVecEnv
    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    rewards = []
    cot_inv_ep    = []   # エピソードごとのCoT^{-1}平均
    delta_omega_interval_ep = []  # エピソードごとのΔΩ_interval平均（周期誤差項）
    delta_omega_height_ep = []    # エピソードごとのΔΩ_height平均（高さ誤差項）
    interval_ep   = []   # エピソードごとの跳躍間隔平均 [s]
    # 新指標
    cv_h_ep       = []   # エピソードごとの跳躍高さ変動係数 CV_h
    term_ep       = []   # エピソードごとの早期終了フラグ (1=終了, 0=完走)
    peak_grf_ep   = []   # エピソードごとのPeak GRF最大値
    cv_t_ep       = []   # エピソードごとの跳躍周期変動係数 CV_T

    for _ in range(args.eval_episodes):
        res = eval_env.reset()
        obs = res[0] if isinstance(res, tuple) else res
        done = False
        total = 0.0
        # サイクル計測用
        cycle_energy       = 0.0
        cycle_z_max        = -np.inf
        cycle_phi_dot_peak = 0.0
        prev_grf           = 0.0
        cot_inv_list       = []
        delta_omega_interval_list = []  # 周期誤差項のリスト
        delta_omega_height_list   = []  # 高さ誤差項のリスト
        interval_list      = []   # 跳躍間隔（着地→次の着地）
        cycle_z_max_list   = []   # CV_h用：サイクルごとの最大高さ
        ep_terminated      = False
        ep_peak_grf        = 0.0
        dt = eval_env.get_attr('dt')[0]
        has_cpg = (baseline != "torque")
        landing_step       = -1
        step_count         = 0

        raw_env = eval_env.venv.envs[0].env.env
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_res = eval_env.step(action)
            
            # 戻り値の要素数による分岐
            if len(step_res) == 5:
                obs, r, terminated, truncated, info_raw = step_res
            else:
                obs, r, terminated, info_raw = step_res
                truncated = False
                
            total += float(np.squeeze(r))
            
            # 配列で返ってくるケースを考慮して安全に bool 判定
            term_val = terminated[0] if isinstance(terminated, np.ndarray) else terminated
            trunc_val = truncated[0] if isinstance(truncated, np.ndarray) else truncated
            done = bool(term_val or trunc_val)
            
            # ▼ 追加: info_raw がリストでも辞書でも対応できるように吸収
            info_dict = info_raw[0] if isinstance(info_raw, (list, tuple)) else info_raw

            # ▼ info[0] を info_dict に変更
            grf   = float(info_dict.get('grf',    0.0))
            power = float(info_dict.get('power',  0.0))
            z     = float(info_dict.get('height', 0.0))
            
            cycle_energy += power * dt
            if z > cycle_z_max and z > 0.1:
                cycle_z_max = z
                
            # Peak GRF（エピソード全体の最大値）
            if grf > ep_peak_grf:
                ep_peak_grf = grf
                
            # 早期終了検出
            if term_val:
                ep_terminated = True

            if has_cpg:
                # ▼ ここも info_dict に変更
                phi_dot = float(info_dict.get('phase_velocity', 0.0))
                phi_dot = np.clip(phi_dot, -4*np.pi, 4*np.pi)
                if abs(phi_dot) > abs(cycle_phi_dot_peak):
                    cycle_phi_dot_peak = phi_dot

            # --- 着地検出 → サイクル評価 ---
            if (grf > 10.0) and (prev_grf <= 10.0) and cycle_z_max > 0.01:
                if landing_step >= 0:
                    period = (step_count - landing_step) * dt
                    interval_list.append(period)
                landing_step = step_count

                # CV_h用：サイクルの最大高さを記録
                cycle_z_max_list.append(cycle_z_max)

                if has_cpg:
                    cot_inv = 75.5 * 9.8 * cycle_z_max / cycle_energy
                else:
                    cot_inv = 75.5 * 9.8 * cycle_z_max / (cycle_energy * 1000.0)
                cot_inv_list.append(cot_inv)
                omega = 2 * np.pi / 1.35625
                # 周期誤差項と高さ誤差項を独立に計算
                if interval_list and np.mean(interval_list) > 1e-6:
                    mean_interval = np.mean(interval_list)
                    dw_interval = -(omega - 2*np.pi / mean_interval)**2
                else:
                    dw_interval = -(omega)**2
                dw_height = -(1.0 - (cycle_z_max/0.45))**2
                delta_omega_interval_list.append(dw_interval)
                delta_omega_height_list.append(dw_height)
                # サイクルリセット
                cycle_energy = 0.0
                cycle_z_max  = -np.inf
                cycle_phi_dot_peak = 0.0

            prev_grf = grf
            step_count += 1
        rewards.append(total)
        if cot_inv_list:
            cot_inv_ep.append(float(np.mean(cot_inv_list)))
        if delta_omega_interval_list:
            delta_omega_interval_ep.append(float(np.mean(delta_omega_interval_list)))
        if delta_omega_height_list:
            delta_omega_height_ep.append(float(np.mean(delta_omega_height_list)))
        if interval_list:
            interval_ep.append(float(np.mean(interval_list)))
        # 新指標をエピソード単位で集計
        if len(cycle_z_max_list) >= 2:
            arr = np.array(cycle_z_max_list)
            cv_h_ep.append(float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 1e-6 else 0.0)
        term_ep.append(1.0 if ep_terminated else 0.0)
        peak_grf_ep.append(ep_peak_grf)
        if len(interval_list) >= 2:
            arr_t = np.array(interval_list)
            cv_t_ep.append(float(np.std(arr_t) / np.mean(arr_t)) if np.mean(arr_t) > 1e-6 else 0.0)
    eval_env.close()

    mean_r        = float(np.mean(rewards))
    std_r         = float(np.std(rewards))
    max_r         = float(np.max(rewards))
    mean_cot      = float(np.mean(cot_inv_ep))           if cot_inv_ep           else 0.0
    mean_dw_interval = float(np.mean(delta_omega_interval_ep)) if delta_omega_interval_ep else 0.0
    mean_dw_height   = float(np.mean(delta_omega_height_ep))   if delta_omega_height_ep   else 0.0
    mean_interval = float(np.mean(interval_ep))        if interval_ep          else 0.0
    mean_cv_h     = float(np.mean(cv_h_ep))        if cv_h_ep       else 0.0
    term_rate     = float(np.mean(term_ep))        if term_ep       else 0.0
    mean_peak_grf = float(np.mean(peak_grf_ep))    if peak_grf_ep   else 0.0
    mean_cv_t     = float(np.mean(cv_t_ep))        if cv_t_ep       else 0.0

    print(f"  [seed={seed}] mean={mean_r:.1f}  std={std_r:.1f}"
          f"  CoT⁻¹={mean_cot:.4f}  ΔΩ_interval={mean_dw_interval:.4f}  ΔΩ_height={mean_dw_height:.4f}"
          f"  interval={mean_interval:.3f}s  CV_T={mean_cv_t:.3f}"
          f"  CV_h={mean_cv_h:.3f}  PeakGRF={mean_peak_grf:.0f}N  term={term_rate:.2f}")

    return dict(seed=seed, mean=mean_r, std=std_r, max=max_r,
                rewards=rewards,
                cot_inv=mean_cot,
                delta_omega_interval=mean_dw_interval,
                delta_omega_height=mean_dw_height,
                jump_interval=mean_interval,
                cv_h=mean_cv_h,
                term_rate=term_rate,
                peak_grf=mean_peak_grf,
                cv_t=mean_cv_t)


# =============================================================================
# 堅牢性テスト
# =============================================================================
def robustness_test(baseline, trained_model_dir, stats_path, base_args, n_trials=5):
    """
    学習済みモデルを異なる条件でテストし、成功率と平均報酬を返す。
    torqueベースラインの場合は evaluate_robustness() を使って
    CoT⁻¹・最大到達高さ・姿勢誤差・跳躍間隔を追加集計する。
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
    # verify model file exists (zip extension is added by SB3)
    expected = os.path.join(trained_model_dir, "final_model.zip")
    if not os.path.isfile(expected):
        raise FileNotFoundError(
            f"trained model not found at {expected}. "
            "Have you run training for this seed?"
        )
    model = PPO.load(os.path.join(trained_model_dir, "final_model"))

    for cond_name, cond_kwargs in conditions.items():
        rewards = []
        # baseline共通の評価指標リスト（evaluate_robustness() で統一）
        cot_inv_list_all     = []
        max_height_list_all  = []
        pose_error_list_all  = []
        interval_list_all    = []
        jump_reward_list_all = []
        # 新指標
        cv_h_list_all        = []   # 跳躍高さ変動係数 CV_h
        term_list_all        = []   # 早期終了フラグ (1=終了, 0=完走)
        peak_grf_list_all    = []   # エピソードのPeak GRF最大値
        cv_t_list_all        = []   # 跳躍周期変動係数 CV_T

        for trial in range(n_trials):
            ev = make_vec_env(baseline, seed=trial+9000, use_subproc=False, **cond_kwargs)
            ev = VecNormalize.load(stats_path, ev)
            ev.training = False
            ev.norm_reward = False

            res = ev.reset()
            obs = res[0] if isinstance(res, tuple) else res
            done = False
            total = 0.0
            dt = ev.get_attr('dt')[0]

            # ---- baseline共通: evaluate_robustness() を使った統一評価 ----
            # VecNormalize → DummyVecEnv → TimeLimit → Monitor → 実環境
            raw_env = ev.venv.envs[0].env.env  # TimeLimit.env
            if hasattr(raw_env, '_reset_robustness_accumulators'):
                raw_env._reset_robustness_accumulators()

            cot_inv_ep     = []
            max_height_ep  = []
            pose_error_ep  = []
            jump_reward_ep = []
            landing_step   = -1
            step_count     = 0
            interval_ep    = []
            ep_terminated  = False   # 早期終了フラグ
            ep_peak_grf    = 0.0     # エピソード全体のGRF最大値

            # ▼ 追加: CV_h, Interval用のマニュアル計算変数
            prev_grf = 0.0
            cycle_z_max = -np.inf
            cycle_z_max_list = []
            
            # デバッグ: 着地検出状況
            landing_count = 0
            cycle_z_max_list = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step_res = ev.step(action)
                if len(step_res) == 5:
                    obs, r, terminated, truncated, infos = step_res
                else:
                    obs, r, terminated, infos = step_res
                    truncated = np.array([False])
                
                # NumPy配列でも安全にbool変換
                term_val = terminated[0] if isinstance(terminated, np.ndarray) else terminated
                trunc_val = truncated[0] if isinstance(truncated, np.ndarray) else truncated
                done = bool(term_val or trunc_val)

                # info から robustness データを取得
                info_dict = infos[0] if isinstance(infos, (list, tuple)) else infos
                
                # ▼ 修正: rob_infoの取得
                if 'pose_error' in info_dict or 'com_z' in info_dict:
                    rob_info = info_dict
                elif hasattr(raw_env, 'evaluate_robustness'):
                    rob_info = raw_env.evaluate_robustness()
                else:
                    rob_info = info_dict

                # 報酬を積算
                total += float(info_dict.get('reward', r[0] if isinstance(r, np.ndarray) else r))

                # ▼ 修正: すべて `info_dict` ではなく `rob_info` から取得する！
                z = float(rob_info.get('com_z', info_dict.get('height', 0.0)))
                max_height_ep.append(z)
                
                pose_error_ep.append(float(rob_info.get('pose_error', 0.0)))

                # ▼ 修正: 着地判定を run_one_seed と統一し、自前でサイクルをカウントする
                grf_now = float(rob_info.get('peak_grf', rob_info.get('grf', info_dict.get('grf', 0.0))))
                if grf_now > ep_peak_grf:
                    ep_peak_grf = grf_now

                # 跳躍サイクルの最大高さ更新
                if z > cycle_z_max and z > 0.1:
                    cycle_z_max = z

                # 着地判定
                just_landed = rob_info.get('just_landed', False)
                if (grf_now > 10.0 and prev_grf <= 10.0 and cycle_z_max > 0.01) or just_landed:
                    if landing_step >= 0:
                        interval_ep.append((step_count - landing_step) * dt)
                    landing_step = step_count
                    
                    cycle_z_max_list.append(cycle_z_max)
                    
                    # ▼ 着地時に指標を記録（着地時のみ意味のある値）
                    if rob_info.get('efficiency_score') is not None:
                        cot_inv_ep.append(float(rob_info['efficiency_score']))
                    if rob_info.get('jump_reward') is not None:
                        jump_reward_ep.append(float(rob_info['jump_reward']))
                    
                    landing_count += 1  # デバッグ
                    cycle_z_max = -np.inf # リセット

                # ▼ 修正: 早期終了検出 (truncated(時間切れ)ではなく terminated(転倒など) で判定)
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

            # ▼ 修正: CV_h は自前で貯めた cycle_z_max_list を使って計算
            if len(cycle_z_max_list) >= 2:
                arr = np.array(cycle_z_max_list)
                cv_h = float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 1e-6 else 0.0
                cv_h_list_all.append(cv_h)

            # 早期終了率（trial単位: 1=終了, 0=完走）
            term_list_all.append(1.0 if ep_terminated else 0.0)

            # Peak GRF
            peak_grf_list_all.append(ep_peak_grf)

            # CV_T: 跳躍周期の変動係数
            if len(interval_ep) >= 2:
                arr_t = np.array(interval_ep)
                cv_t = float(np.std(arr_t) / np.mean(arr_t)) if np.mean(arr_t) > 1e-6 else 0.0
                cv_t_list_all.append(cv_t)

            ev.close()

        # 指標記録状況をデバッグ出力
        n_trials = n_trials
        debug_msg = (f"    Trial metrics recorded (landings detected={landing_count}): "
                     f"cot_inv={len(cot_inv_list_all)}/{n_trials}, "
                     f"height={len(max_height_list_all)}/{n_trials}, "
                     f"pose_error={len(pose_error_list_all)}/{n_trials}, "
                     f"interval={len(interval_list_all)}/{n_trials}, "
                     f"jump_reward={len(jump_reward_list_all)}/{n_trials}, "
                     f"cv_h={len(cv_h_list_all)}/{n_trials}, "
                     f"term_rate={len(term_list_all)}/{n_trials}, "
                     f"peak_grf={len(peak_grf_list_all)}/{n_trials}, "
                     f"cv_t={len(cv_t_list_all)}/{n_trials}")
        print(debug_msg)

        res_dict = dict(
            mean=np.mean(rewards),
            std=np.std(rewards),
            min=np.min(rewards),
            max=np.max(rewards),
        )
        # baseline 共通の既存指標（evaluate_robustness() で統一）
        res_dict['mean_cot_inv']     = float(np.mean(cot_inv_list_all))     if cot_inv_list_all     else 0.0
        res_dict['std_cot_inv']      = float(np.std(cot_inv_list_all))      if cot_inv_list_all     else 0.0
        res_dict['mean_max_height']  = float(np.mean(max_height_list_all))  if max_height_list_all  else 0.0
        res_dict['std_max_height']   = float(np.std(max_height_list_all))   if max_height_list_all  else 0.0
        res_dict['mean_pose_error']  = float(np.mean(pose_error_list_all))  if pose_error_list_all  else 0.0
        res_dict['std_pose_error']   = float(np.std(pose_error_list_all))   if pose_error_list_all  else 0.0
        res_dict['mean_interval']    = float(np.mean(interval_list_all))    if interval_list_all    else 0.0
        res_dict['std_interval']     = float(np.std(interval_list_all))     if interval_list_all    else 0.0
        res_dict['mean_jump_reward'] = float(np.mean(jump_reward_list_all)) if jump_reward_list_all else 0.0
        res_dict['std_jump_reward']  = float(np.std(jump_reward_list_all))  if jump_reward_list_all else 0.0
        # 新指標
        res_dict['cv_h']          = float(np.mean(cv_h_list_all))       if cv_h_list_all       else 0.0
        res_dict['std_cv_h']      = float(np.std(cv_h_list_all))        if cv_h_list_all       else 0.0
        res_dict['term_rate']     = float(np.mean(term_list_all))        if term_list_all       else 0.0
        res_dict['std_term_rate'] = float(np.std(term_list_all))        if term_list_all       else 0.0
        res_dict['mean_peak_grf'] = float(np.mean(peak_grf_list_all))   if peak_grf_list_all   else 0.0
        res_dict['std_peak_grf']  = float(np.std(peak_grf_list_all))    if peak_grf_list_all   else 0.0
        res_dict['cv_t']          = float(np.mean(cv_t_list_all))        if cv_t_list_all       else 0.0
        res_dict['std_cv_t']      = float(np.std(cv_t_list_all))        if cv_t_list_all       else 0.0

        results[cond_name] = res_dict

        msg = (f"  Robustness [{cond_name:25s}]  "
               f"mean={res_dict['mean']:.1f}±{res_dict['std']:.1f}  "
               f"CoT⁻¹={res_dict['mean_cot_inv']:.4f}±{res_dict['std_cot_inv']:.4f}  "
               f"h_max={res_dict['mean_max_height']:.3f}±{res_dict['std_max_height']:.3f}m  "
               f"CV_h={res_dict['cv_h']:.3f}±{res_dict['std_cv_h']:.3f}  "
               f"interval={res_dict['mean_interval']:.3f}±{res_dict['std_interval']:.3f}s  "
               f"CV_T={res_dict['cv_t']:.3f}±{res_dict['std_cv_t']:.3f}  "
               f"PeakGRF={res_dict['mean_peak_grf']:.0f}±{res_dict['std_peak_grf']:.0f}N  "
               f"term={res_dict['term_rate']:.2f}±{res_dict['std_term_rate']:.2f}  "
               f"pose_err={res_dict['mean_pose_error']:.4f}±{res_dict['std_pose_error']:.4f}  "
               f"jump_rew={res_dict['mean_jump_reward']:.4f}±{res_dict['std_jump_reward']:.4f}")
        print(msg)

    return results


# =============================================================================
# 結果を Markdown テーブル + CSV に保存
# =============================================================================
def save_summary(all_results: dict, output_dir: str, hyperparams: dict):
    """
    all_results: {condition_label: [per-seed dict, ...], ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for label, seed_results in all_results.items():
        means = [r['mean'] for r in seed_results]
        
        # 各指標の平均値と標準偏差を計算
        row = {
            'Condition': label,
            'N_seeds':   len(seed_results),
            'Mean_reward':      round(np.mean(means), 2),
            'Std_reward':       round(np.std(means),  2),
            'Max_reward':       round(np.max(means),  2),
            'Min_reward':       round(np.min(means),  2),
        }
        
        # CoT_inv (存在する場合)
        cot_invs = [r.get('cot_inv', 0.0) for r in seed_results]
        if any(cot_invs):
            row['Mean_cot_inv'] = round(np.mean(cot_invs), 4)
            row['Std_cot_inv']  = round(np.std(cot_invs), 4)
        
        # Delta_Omega - 周期誤差項 (存在する場合)
        delta_omegas_interval = [r.get('delta_omega_interval', 0.0) for r in seed_results]
        if any(delta_omegas_interval):
            row['Mean_delta_omega_interval'] = round(np.mean(delta_omegas_interval), 4)
            row['Std_delta_omega_interval']  = round(np.std(delta_omegas_interval), 4)
        
        # Delta_Omega - 高さ誤差項 (存在する場合)
        delta_omegas_height = [r.get('delta_omega_height', 0.0) for r in seed_results]
        if any(delta_omegas_height):
            row['Mean_delta_omega_height'] = round(np.mean(delta_omegas_height), 4)
            row['Std_delta_omega_height']  = round(np.std(delta_omegas_height), 4)
        
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved → {csv_path}")

    # Markdown表
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# 実験サマリー\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## ハイパーパラメータ\n\n")
        f.write("| パラメータ | 値 |\n|---|---|\n")
        for k, v in hyperparams.items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## 条件別 評価報酬 (平均±標準偏差)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*N_seeds: 乱数シード数。報酬は10エピソード平均。各指標について平均値と標準偏差を記載。*\n")

    print(f"Summary Markdown saved → {md_path}")
    return df


# =============================================================================
# メイン
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Tegotae Hopper PPO v3 - 改善版トレーニングスクリプト"
    )
    # 実験設定
    parser.add_argument("--experiment", "-e", default="TH_PPO_v3_improved")
    parser.add_argument("--timesteps",  "-t", type=int, default=200_000)
    parser.add_argument("--seeds", "-s", type=int, nargs="*", default=[0,1,2,3,4],
                        help="乱数シードリスト (例: 0 1 2 3 4)")
    parser.add_argument("--baselines", "-b", nargs="*",
                        default=["default"],
                        choices=["default","gainonly","torque","nofeedback"],
                        help="比較するベースラインを複数指定可")
    # 堅牢性
    parser.add_argument("--noise",      type=float, default=0.0)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--ext-force",  type=float, default=0.0)
    parser.add_argument("--robustness", action="store_true",
                        help="学習後に堅牢性テストを実行")
    # 評価
    parser.add_argument("--eval-episodes", type=int, default=10)
    # PPOハイパーパラメータ
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--n-steps",     type=int,   default=2048)
    parser.add_argument("--batch-size",  type=int,   default=128)
    parser.add_argument("--n-epochs",    type=int,   default=10)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--gae-lambda",  type=float, default=0.95)
    parser.add_argument("--clip-range",  type=float, default=0.2)
    parser.add_argument("--vf-coef",     type=float, default=0.75)
    parser.add_argument("--render-freq", type=int,   default=0,
                        help="何ステップごとにrenderするか (0=無効, 例: 20000)")
    args = parser.parse_args()

    # ハイパーパラメータ記録
    hyperparams = {
        'learning_rate': args.lr,
        'n_steps':       args.n_steps,
        'batch_size':    args.batch_size,
        'n_epochs':      args.n_epochs,
        'gamma':         args.gamma,
        'gae_lambda':    args.gae_lambda,
        'clip_range':    args.clip_range,
        'vf_coef':       args.vf_coef,
        'total_timesteps': args.timesteps,
        'seeds':         args.seeds,
        'baselines':     args.baselines,
        'noise_std':     args.noise,
        'mass_scale':    args.mass_scale,
        'ext_force':     args.ext_force,
    }

    out_root   = os.path.join(current_dir, "results", args.experiment)
    log_dir    = os.path.join(current_dir, "TH_logs_3", args.experiment)
    models_dir = os.path.join(out_root, "models")
    os.makedirs(out_root,   exist_ok=True)
    os.makedirs(log_dir,    exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(out_root, 'hyperparams.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2, ensure_ascii=False)
    """
    env = make_env("gainonly")
    env.reset()
    mujoco.mj_resetData(env.env.env.model, env.env.env.data)
    mujoco.mj_forward(env.env.env.model, env.env.env.data)
    print("default qpos[1] =", env.env.env.data.qpos[1])
    """
    print("=" * 60)
    print(f"実験名: {args.experiment}")
    print(f"ベースライン: {args.baselines}")
    print(f"シード: {args.seeds}  (計 {len(args.seeds)} 回/条件)")
    print(f"Timesteps: {args.timesteps:,}")
    print("\n【参照軌道】")
    print("  - ソース: trajectories_nn/ref_traj_nn.pt (ニューラルネット)")
    print("  - 各環境で自動読み込み・周波数に応じた自動更新")
    print("=" * 60)

    all_results = {}  # {baseline_label: [seed_dict, ...]}

    for baseline in args.baselines:
        print(f"\n{'='*40}")
        print(f"▶ ベースライン: {baseline}")
        print(f"{'='*40}")
        bl_models_dir = os.path.join(models_dir, baseline)
        os.makedirs(bl_models_dir, exist_ok=True)
        seed_results = []

        for seed in args.seeds:
            print(f"\n--- seed={seed} ---")
            result = run_one_seed(
                baseline=baseline,
                seed=seed,
                args=args,
                models_dir=os.path.join(bl_models_dir, f"seed_{seed}"),
                log_dir=log_dir,
            )
            seed_results.append(result)

        # シード集計
        means = [r['mean'] for r in seed_results]
        print(f"\n[{baseline}] 全シード集計 (n={len(args.seeds)}):")
        print(f"  平均報酬 = {np.mean(means):.1f} ± {np.std(means):.1f}")
        print(f"  最大報酬 = {np.max(means):.1f}")
        all_results[baseline] = seed_results

        # 堅牢性テスト (最初のシードのモデルを使用)
        if args.robustness:
            best_seed = args.seeds[int(np.argmax(means))]
            best_model_dir  = os.path.join(bl_models_dir, f"seed_{best_seed}")
            best_stats_path = os.path.join(best_model_dir, "vec_normalize.pkl")
            print(f"\n[{baseline}] 堅牢性テスト (best seed={best_seed})...")
            rob = robustness_test(baseline, best_model_dir, best_stats_path,
                                  base_args=args, n_trials=5)
            rob_path = os.path.join(out_root, f"robustness_{baseline}.json")
            with open(rob_path, 'w') as f:
                json.dump(rob, f, indent=2)
            print(f"  → {rob_path} に保存")

    # サマリー出力
    df = save_summary(all_results, out_root, hyperparams)

    # 複数ベースライン比較表示
    if len(args.baselines) > 1:
        print("\n" + "="*60)
        print("ベースライン比較サマリー")
        print("="*60)
        print(df.to_string(index=False))

    print("\n全実験完了。")
    print(f"結果ディレクトリ: {out_root}")

if __name__ == "__main__":
    main()