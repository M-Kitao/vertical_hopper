import optuna
import numpy as np
import torch
import os
import sys
# 現在のファイル (RL/train_TH_PPO.py) のディレクトリパスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 1つ上の階層 (vertical_hopper/) を取得
root_dir = os.path.dirname(current_dir)
# 検索パスに追加
sys.path.append(root_dir)

from GymEnv.Tegotae_Hopper_PPO import Tegotae_Hopper_Env_PPO
import mujoco


def optimize_ppo_hyperparameters(trial):
    # --- 1. 最適化するパラメータの探索範囲定義 ---
    
    # Gain: フィードバックの強さ
    gain_val = trial.suggest_float('gain', 0.1, 5.0) 
    
    # Reaction Weight: センサー値にかける係数
    reaction_weight = trial.suggest_float('reaction_weight', 0.001, 0.2)
    
    # Actuator KP: バネ定数 (ここも重要！)
    kp_val = trial.suggest_float('kp', 100.0, 2000.0) 

    # --- 2. 環境のセットアップ ---
    # ここではVecEnv(ベクトル環境)は不要です。単純なEnvを使います。
    env = Tegotae_Hopper_Env_PPO(render_mode=None)
    obs, info = env.reset()

    # XMLのアクチュエータKP値を書き換える
    hip_act_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_joint")
    knee_act_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_joint")
    env.model.actuator_gainprm[hip_act_id, 0] = kp_val
    env.model.actuator_gainprm[knee_act_id, 0] = kp_val

    # --- 3. 固定アクションの作成 ---
    # Env内部で「val = action * 係数」となっているため、逆算してActionを作る
    # Tegotae_Hopper_PPO.py の stepメソッド内: 
    # react_val = action[0] * 0.05 * ...
    # gain_val  = action[1] * 1.0
    
    act0 = reaction_weight / 0.05
    act1 = gain_val / 1.0
    
    # Action空間 [-1, 1] に収まるようにクリップ (はみ出るようなら範囲設定を見直す)
    act0 = np.clip(act0, -1.0, 1.0)
    act1 = np.clip(act1, -1.0, 1.0)
    
    fixed_action = np.array([act0, act1], dtype=np.float32)

    # --- 4. シミュレーション実行 ---
    max_height = 0.0
    
    # 500ステップ (約5秒間) 実行して評価
    for _ in range(500):
        # ★重要: 毎ステップ、同じ「最適化したい固定パラメータ」を入力し続ける
        obs, reward, terminated, truncated, info = env.step(fixed_action)
        
        # 高さ(rootz)を取得して最大値を記録
        # (obsの構成: [z, z_dot, phi, sin, cos, ...])
        # obs[0] が高さ
        current_z = obs[0]
        if current_z > max_height:
            max_height = current_z
        
        # 転倒したらそこで終了 (ペナルティを与える)
        if terminated:
            # 早く転ぶほどスコアが低くなるように、到達高さからペナルティを引くなど
            return max_height * 0.5 

    env.close()
    
    # 最大到達高さを評価値として返す（これを最大化するようにOptunaが頑張る）
    return max_height

# 最適化の実行
if __name__ == "__main__":
    # maximize: 高さを最大化したい
    study = optuna.create_study(direction="maximize")
    print("最適化を開始します...")
    
    # 100回試行
    study.optimize(optimize_ppo_hyperparameters, n_trials=100) 

    print("------------------------------------------------")
    print("【発見された最強のパラメータ】")
    print(study.best_params)
    print("到達高さ:", study.best_value)
    print("------------------------------------------------")