import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
import mujoco.viewer
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- パス設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 環境のインポート
from GymEnv.Tegotae_Hopper_PPO import Tegotae_Hopper_Env_PPO

def main():
    # 1. 環境の作成 (render_mode="human" で画面表示を有効化)
    env = Tegotae_Hopper_Env_PPO(render_mode="human")
    
    # 2. 学習済みモデルのパス
    # train_TH_PPO.py で保存したパスを指定
    model_path = os.path.join(root_dir, "TH_PPO_models", "PPO_Tegotae_Hopper")
    
    # モデルのロード
    if not os.path.exists(model_path + ".zip"):
        print(f"モデルが見つかりません: {model_path}")
        return

    model = PPO.load(model_path)
    print("モデルをロードしました。シミュレーションを開始します...")

    # 3. テストループ
    obs, info = env.reset()

    # グラフ用データの保存リスト
    height_history = []
    reward_history = []
    
    # 無限ループで動作確認 (ウィンドウを閉じるまで続く)
    try:
        #while True:
        for _ in range(10000):  # 10000ステップだけ実行
            # 決定論的(deterministic=True)に推論させる
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)

            # 高さの計算 (ログ出力と同じ計算式を使用)
            current_height = env.data.qpos[1] + 0.965
            
            # データをリストに追加
            height_history.append(current_height)
            reward_history.append(reward)
            
            # ログ表示 (位相 phi や 報酬などを確認)
            #print(f"Phi: {info['phi']:.2f}, Reward: {reward:.2f}, Action: {action}, height: {env.data.qpos[1] + 0.965:.2f}, ground_force: {abs(env.grf[2]):.2f}")
            #print(f"Mass: {env.model.body_mass.sum()}" )

            if terminated or truncated:
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("終了します")
    finally:
        env.close()

    # --- グラフの描画 ---
        if height_history: # データが存在する場合のみ描画
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # 重心高さ（Height）のグラフ
            ax1.plot(height_history, color='blue', label='Height')
            ax1.set_ylabel('Height [m]')
            ax1.set_title('Model Center of Mass Height')
            ax1.grid(True)
            ax1.legend()

            # 報酬（Reward）のグラフ
            ax2.plot(reward_history, color='orange', label='Reward')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Reward')
            ax2.set_title('Reward per Step')
            ax2.grid(True)
            ax2.legend()

            plt.tight_layout()
            plt.show()
        else:
            print("データが記録されませんでした。")

if __name__ == "__main__":
    main()