import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import sys
# 現在のファイル (RL/train_TH_PPO.py) のディレクトリパスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 1つ上の階層 (vertical_hopper/) を取得
root_dir = os.path.dirname(current_dir)
# 検索パスに追加
sys.path.append(root_dir)

from GymEnv.Tegotae_Hopper_PPO import Tegotae_Hopper_Env_PPO

def train():
    # ログ保存場所
    log_dir = "./TH_PPO_logs/"
    model_dir = "./TH_PPO_models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. 環境の作成
    # ベクトル化環境にすることで学習効率アップ
    env = DummyVecEnv([lambda: Tegotae_Hopper_Env_PPO(render_mode=None)])

    # 2. モデルの定義 (MlpPolicy: 多層パーセプトロン)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        tensorboard_log=log_dir
    )

    # 3. チェックポイント保存の設定
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path='./TH_PPO_models/',
        name_prefix='tegotae_hopper'
    )

    # 4. 学習開始 (timestepsは適宜増やしてください。100万回くらいが目安)
    print("学習を開始します...")
    model.learn(total_timesteps=100000, callback=checkpoint_callback, tb_log_name='PPO_Tegotae_Hopper')
    
    # 5. モデルの保存
    save_path = os.path.join(model_dir, "PPO_Tegotae_Hopper")
    model.save(save_path)
    print("学習完了。モデルを保存しました。")

    env.close()

if __name__ == "__main__":
    train()