import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# --- 1. パス設定の修正 (重要) ---
# このファイル (.../RL/visualize_networks.py) の場所
current_dir = os.path.dirname(os.path.abspath(__file__))
# プロジェクトルート (.../vertical_hopper) を取得
root_dir = os.path.dirname(current_dir)

# パスに追加 (これで GymEnv, NN, CPG が見えるようになります)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 念のためサブフォルダも明示的に追加しておくと安全です
sys.path.append(os.path.join(root_dir, 'GymEnv'))
sys.path.append(os.path.join(root_dir, 'NN'))
sys.path.append(os.path.join(root_dir, 'CPG'))

# --- 2. モジュールのインポート ---
try:
    # GymEnvパッケージとしてインポートを試みる
    from GymEnv.Tegotae_Policy import Tegotae_Policy
except ImportError:
    try:
        # パスが通っていれば直接インポートできる場合もある
        from GymEnv.Tegotae_Policy import Tegotae_Policy
    except ImportError as e:
        print("Error: Tegotae_Policy could not be imported.")
        print(f"Current sys.path: {sys.path}")
        raise e

def visualize_tegotae_functions(model_path):
    print(f"Loading model from {model_path}...")
    
    # デバイスの自動判定 (CUDAが使えるなら使う)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # モデルをロード (custom_objectsでPolicyを紐付ける必要がある場合も考慮)
        model = PPO.load(model_path, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # model.policy.action_net は Tegotae_Actor クラスのインスタンスです
    actor = model.policy.action_net
    
    print("Extracting networks from Tegotae_Actor...")
    
    # dir(actor) の結果に基づき、小文字の属性名を直接指定します
    try:
        action_net = actor.action_net
        reaction_net = actor.reaction_net
        gain_net = actor.gain_net
        print("Networks found successfully: action_net, reaction_net, gain_net")
    except AttributeError as e:
        print(f"Fatal Error: {e}")
        return

    # 評価モード
    action_net.eval()
    reaction_net.eval()
    gain_net.eval()

    # --- グラフ描画 ---
    fig = plt.figure(figsize=(18, 5))
    
    # ---------------------------------------------------------
    # 3. Action A(phi) の描画
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(131)
    phis = np.linspace(0, 2 * np.pi, 100)
    phi_inputs = torch.tensor(np.stack([np.sin(phis), np.cos(phis)], axis=1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        actions = action_net(phi_inputs).cpu().numpy().flatten()
    
    # r"..." を使って SyntaxWarning を回避
    ax1.plot(phis, actions, label=r'Action A($\phi$)', color='blue', linewidth=2)
    ax1.set_title(r"Action A($\phi$) vs Phase")
    ax1.set_xlabel(r"Phase $\phi$ [rad]")
    ax1.set_ylabel("Action Output")
    ax1.grid(True)
    ax1.set_ylim([-1.1, 1.1])
    
    # ---------------------------------------------------------
    # 4. Reaction R(s) の描画
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(132)
    grf_values = np.linspace(0, 50, 100)
    sensor_dim = 5 
    sensor_inputs = np.zeros((100, sensor_dim))
    sensor_inputs[:, 0] = grf_values 
    sensor_tensor = torch.tensor(sensor_inputs, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        reactions = reaction_net(sensor_tensor).cpu().numpy().flatten()
        
    ax2.plot(grf_values, reactions, label='Reaction R(s)', color='red', linewidth=2)
    ax2.set_title("Reaction R(s) vs GRF")
    ax2.set_xlabel("Ground Reaction Force (GRF)")
    ax2.set_ylabel("Reaction Output")
    ax2.grid(True)
    
    # ---------------------------------------------------------
    # 5. Gain G(phi, s) の描画
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(133)
    phi_grid = np.linspace(0, 2 * np.pi, 50)
    grf_grid = np.linspace(0, 50, 50)
    PHI, GRF = np.meshgrid(phi_grid, grf_grid)
    
    grid_points = PHI.size
    flat_phis = PHI.flatten()
    flat_grfs = GRF.flatten()
    
    phi_in = torch.tensor(np.stack([np.sin(flat_phis), np.cos(flat_phis)], axis=1), dtype=torch.float32).to(device)
    sens_in_np = np.zeros((grid_points, sensor_dim))
    sens_in_np[:, 0] = flat_grfs
    sens_in = torch.tensor(sens_in_np, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        gains = gain_net(phi_in, sens_in).cpu().numpy().flatten()
    
    GAIN = gains.reshape(PHI.shape)
    
    c = ax3.pcolormesh(PHI, GRF, GAIN, cmap='viridis', shading='auto')
    ax3.set_title(r"Gain G($\phi$, GRF)")
    ax3.set_xlabel(r"Phase $\phi$ [rad]")
    ax3.set_ylabel("GRF")
    fig.colorbar(c, ax=ax3, label='Gain Value')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # モデルのパスを指定（存在確認を含む）
    # logsフォルダが一つ上の階層にあるなら ../logs/..., 同じ階層なら logs/... 
    # train.pyの設定に合わせて適宜修正してください
    MODEL_PATH = os.path.join(current_dir, "models", "final_model.zip") 
    
    # もし models フォルダではなく tensorboard_logs 等にある場合や、
    # 学習スクリプトで指定した保存先 (例: ./models/final_model.zip) に合わせてください
    # 見つからない場合は手動でフルパスを書いてもOKです
    if not os.path.exists(MODEL_PATH):
        # 試しに一つ上を探す
        MODEL_PATH = os.path.join(root_dir, "RL", "models", "final_model.zip")
    
    if os.path.exists(MODEL_PATH):
        visualize_tegotae_functions(MODEL_PATH)
    else:
        print(f"Model file not found at: {MODEL_PATH}")
        print("Please edit MODEL_PATH in the script to point to your .zip file.")