import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

# --- 1. OpenMPエラー回避 (最優先) ---
# PyTorchとNumPyの競合によるクラッシュを防ぎます
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 2. パス設定 ---
# このファイルの場所を取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# プロジェクトルート (vertical_hopper)
root_dir = os.path.dirname(current_dir)

# 必要なフォルダにパスを通す
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'GymEnv'))
sys.path.append(os.path.join(root_dir, 'NN'))
sys.path.append(os.path.join(root_dir, 'CPG'))

# 環境クラスのインポート
# ※ファイル名が "Tegotae_Hopper_PPO_V3.py" で、クラス名が "Tegotae_Hopper_PPO_v2_Env" であると仮定
try:
    from GymEnv.Tegotae_Hopper_PPO_V3 import Tegotae_Hopper_PPO_v2_Env
except ImportError:
    try:
        # フォルダ構成が異なる場合のフォールバック
        from GymEnv.Tegotae_Hopper_PPO_V3_gainonly import Tegotae_Hopper_PPO_v2_Env
    except ImportError:
        print("エラー: 環境ファイル 'Tegotae_Hopper_PPO_V3.py' が見つかりません。")
        print("GymEnvフォルダの中にファイルがあるか確認してください。")
        sys.exit(1)

# カスタムPolicy (Tegotae_Policy.py)
try:
    from GymEnv.Tegotae_Policy import Tegotae_Policy
except ImportError:
    try:
        from GymEnv.Tegotae_Policy import Tegotae_Policy
    except ImportError:
        print("エラー: ポリシーファイル 'Tegotae_Policy.py' が見つかりません。")
        print("GymEnvフォルダの中にファイルがあるか確認してください。")
        sys.exit(1)

# --- 3. カスタムCallback (ログ記録用) ---
class TensorboardCallback(BaseCallback):
    """
    info辞書からカスタム値をTensorBoardに記録するコールバック
    跳躍ごとの報酬(accumulated_reward)などもここで記録します。
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # ベクトル環境対応: 最初の環境のinfoを取得
        infos = self.locals['infos'][0]
        
        # --- 内部状態の記録 ---
        # ゲイン、アクション、リアクション
        if 'gain' in infos:
            self.logger.record("Custom/Gain", infos['gain'])
        if 'action' in infos:
            self.logger.record("Custom/Action", infos['action'])
        if 'reaction' in infos:
            self.logger.record("Custom/Reaction", infos['reaction'])
            
        # --- 報酬関連の記録 ---
        # ステップごとの瞬時報酬
        if 'step_reward' in infos:
            self.logger.record("Custom/StepReward", infos['step_reward'])
            
        # 跳躍ごとの蓄積報酬 (着地時に値が入る)
        if 'accumulated_reward' in infos:
            # 着地した瞬間に値が跳ね上がるグラフになります
            self.logger.record("Custom/JumpReward", infos['accumulated_reward'])

        # 高さ
        if 'height' in infos:
            self.logger.record("Custom/Height", infos['height'])
            
        return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO on Tegotae Hopper environment with optional multiple seeds and baselines.")
    parser.add_argument("--experiment", "-e", default="PPO_JumpReward_v12_1", help="experiment name for logging and saving")
    parser.add_argument("--timesteps", "-t", type=int, default=50000, help="total timesteps per run")
    parser.add_argument("--seeds", "-s", type=int, nargs="*", default=[0], help="list of random seeds to run (space separated)")
    parser.add_argument("--baseline", "-b", choices=["default","gainonly","torque"], default="default", help="which environment/baseline to use")
    parser.add_argument("--noise", type=float, default=0.0, help="observation noise standard deviation for robustness tests")
    parser.add_argument("--mass-scale", type=float, default=1.0, help="scale factor applied to mass/friction for robustness tests")
    parser.add_argument("--ext-force", type=float, default=0.0, help="external force amplitude for occasional pulses")
    parser.add_argument("--output-csv", "-o", default=None, help="path to write summary CSV (appended)")
    args = parser.parse_args()

    EXPERIMENT_NAME = args.experiment
    TOTAL_TIMESTEPS = args.timesteps
    seeds = args.seeds
    baseline = args.baseline
    csv_path = args.output_csv

    # ログとモデルの保存先
    log_dir = os.path.join(current_dir, "tensorboard_logs")
    models_dir = os.path.join(current_dir, "models", EXPERIMENT_NAME)
    os.makedirs(models_dir, exist_ok=True)

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Log Dir: {log_dir}")
    print(f"Model Dir: {models_dir}")
    print(f"Seeds: {seeds}")
    print(f"Baseline env: {baseline}")
    # ハイパーパラメータの記録
    hyperparams = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'noise': args.noise,
        'mass_scale': args.mass_scale,
        'ext_force': args.ext_force,
        'total_timesteps': TOTAL_TIMESTEPS,
        'baseline': baseline
    }
    import json
    with open(os.path.join(models_dir, 'hyperparams.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2)

    # CSVヘッダを書き込む必要がある場合
    if csv_path is not None and not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("seed,final_reward,avg_eval_reward,steps,notes\n")

    # ループ処理: 各シードで訓練
    all_results = []
    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        # seed の再現性設定
        np.random.seed(seed)
        import torch as _torch
        _torch.manual_seed(seed)
        set_random_seed(seed)

        # --- 環境の構築 ---
        if baseline == "default":
            env_cls = Tegotae_Hopper_PPO_v2_Env
        elif baseline == "gainonly":
            from GymEnv.Tegotae_Hopper_PPO_V3_gainonly import Tegotae_Hopper_PPO_v2_Env as GainEnv
            env_cls = GainEnv
        else:  # torque baseline
            from GymEnv.DirectTorque_Hopper import DirectTorque_Hopper_Env as TorqueEnv
            env_cls = TorqueEnv

        env = env_cls(render_mode="human", noise_std=args.noise, mass_scale=args.mass_scale, ext_force=args.ext_force)
        env = TimeLimit(env, max_episode_steps=4000)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

        # --- モデルの構築 ---
        model = PPO(
            Tegotae_Policy,
            env,
            seed=seed,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=log_dir,
            device="auto",
            policy_kwargs=dict(squash_output=False)
        )

        # --- Callbackの設定 ---
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(models_dir, f"seed_{seed}"),
            name_prefix="ppo_jump_reward"
        )
        log_callback = TensorboardCallback()

        # 学習開始
        print("Start learning...")
        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=[checkpoint_callback, log_callback],
                tb_log_name=f"{EXPERIMENT_NAME}_s{seed}"
            )
        except KeyboardInterrupt:
            print("\n学習が手動で中断されました。保存を試みます...")

        # 結果保存
        final_model_path = os.path.join(models_dir, f"final_model_s{seed}")
        model.save(final_model_path)
        stats_path = os.path.join(models_dir, f"vec_normalize_s{seed}.pkl")
        env.save(stats_path)
        print(f"Model saved to {final_model_path}")
        print(f"Normalization stats saved to {stats_path}")

        # 簡易評価: 10エピソード分の平均報酬を計測
        eval_env = env_cls(render_mode=None, noise_std=args.noise, mass_scale=args.mass_scale, ext_force=args.ext_force)
        eval_env = TimeLimit(eval_env, max_episode_steps=4000)
        eval_env = Monitor(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_env = VecNormalize.load(stats_path, eval_env)
        evaluate_rewards = []
        # seedはVecNormalizeに渡せないため事前に設定
        try:
            eval_env.seed(seed)
        except Exception:
            pass  # 一部ラッパーでは無効だが問題なし
        for _ in range(10):
            res = eval_env.reset()
            # Reset may return obs or (obs, info) depending on gym version
            if isinstance(res, tuple):
                obs = res[0]
            else:
                obs = res
            done = False
            total = 0.0
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                step_res = eval_env.step(action)
                # step() may return 4 or 5 elements depending on gym version
                if len(step_res) == 5:
                    obs, reward, terminated, truncated, info = step_res
                else:
                    obs, reward, terminated, info = step_res
                    truncated = False
                total += reward
                done = terminated or truncated
            evaluate_rewards.append(total)
        avg_eval = np.mean(evaluate_rewards)
        final_reward = np.max(evaluate_rewards)
        print(f"Evaluation average reward: {avg_eval}")

        # CSV追記
        if csv_path is not None:
            with open(csv_path, "a") as f:
                f.write(f"{seed},{final_reward},{avg_eval},{TOTAL_TIMESTEPS},\n")

        all_results.append({'seed': seed, 'avg_eval': avg_eval, 'max_eval': final_reward})

        # クリーンアップ
        env.close()
        eval_env.close()

    # 全シードでの統計を表示
    if len(all_results) > 1:
        avgs = [r['avg_eval'] for r in all_results]
        maxs = [r['max_eval'] for r in all_results]
        print(f"\nSummary over seeds ({len(seeds)} runs):")
        print(f"  avg_eval mean±std = {np.mean(avgs):.2f} ± {np.std(avgs):.2f}")
        print(f"  max_eval mean±std = {np.mean(maxs):.2f} ± {np.std(maxs):.2f}")

    print("\nAll runs completed.")

    # --- Callbackの設定 ---
    # 定期的にモデルを保存 (50,000ステップごと)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=models_dir,
        name_prefix="ppo_jump_reward"
    )
    
    # カスタムログ記録
    log_callback = TensorboardCallback()

    # --- 学習開始 ---
    print("Start learning...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, log_callback],
            tb_log_name=EXPERIMENT_NAME
        )
    except KeyboardInterrupt:
        print("\n学習が手動で中断されました。現在のモデルを保存します...")

    # --- 最終保存 ---
    # 1. モデル本体の保存
    final_model_path = os.path.join(models_dir, "final_model")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # 2. 正規化パラメータの保存 (推論時に必須！)
    stats_path = os.path.join(models_dir, "vec_normalize.pkl")
    env.save(stats_path)
    print(f"Normalization stats saved to {stats_path}")
    
    env.close()
    print("Done.")

if __name__ == "__main__":
    main()