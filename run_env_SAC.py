import gymnasium as gym
from stable_baselines3 import SAC
import os
import vh_env

# --- 1. モデルと環境の読み込み ---
model_dir = "SAC_models/"
model_name = "SAC_vertical_hopper"
model_path = os.path.join(model_dir, f"{model_name}.zip")

# 学習済みのPPOモデルをロード
try:
    model = SAC.load(model_path)
except FileNotFoundError:
    print(f"エラー: モデルファイルが見つかりません: {model_path}")
    print("先に practice_vh_SAC.py を実行してモデルを学習・保存してください。")
    exit()

# シミュレーションを可視化するための環境を作成
env = gym.make("vh-v0", render_mode="human")
obs, info = env.reset()

print("--- シミュレーション開始 ---")
print("ウィンドウを閉じるか、Ctrl+Cで終了します。")

# --- 2. シミュレーションの実行ループ ---
try:
    while True:
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        env.render()
        if terminated:
            obs, info = env.reset()
        """
        # モデルが現在の観測(obs)から最適な行動(action)を予測
        action, _states = model.predict(obs, deterministic=True)
        
        # 予測した行動を環境内で実行し、次の状態や報酬などを取得
        obs, reward, terminated, truncated, info = env.step(action)
        
        # エピソードが終了（転倒など）したら、環境をリセット
        if terminated or truncated:
            print("エピソード終了。リセットします。")
            obs, info = env.reset()
        """
finally:
    # ループが終了したら環境を閉じる
    env.close()
    print("--- シミュレーション終了 ---")
"""
for _ in range(100000):
    a = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(a)
    env.render()
    if terminated:
        obs, info = env.reset()
env.close()
print("--- シミュレーション終了 ---")
"""