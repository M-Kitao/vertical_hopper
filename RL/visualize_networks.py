import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
# パス設定（学習時と同じ構成にする）
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
root_dir = os.path.dirname(current_dir)

# パスに追加 (これで GymEnv, NN, CPG が見えるようになります)
if root_dir not in sys.path:
    sys.path.append(root_dir)
sys.path.append(os.path.join(current_dir, 'CPG'))
sys.path.append(os.path.join(current_dir, 'NN'))
sys.path.append(os.path.join(current_dir, 'GymEnv'))

# 必要なクラスを読み込み（Pickleエラー回避のため）
from GymEnv.Tegotae_Policy_gainonly import Tegotae_gainonly_Policy 

def visualize_tegotae_functions(model_path):
    # 1. モデルのロード
    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    actor = model.policy.action_net
    
    print("Extracting networks from Tegotae_Actor...")
    
    try:
        # 【修正箇所】 dir(actor)の結果に基づき、小文字の属性名でアクセス
        action_net = actor.action_net
        reaction_net = actor.reaction_net
        gain_net = actor.gain_net
        print("Networks found successfully: action_net, reaction_net, gain_net")
    except AttributeError as e:
        print(f"Fatal Error: ネットワークが見つかりません。属性名を確認してください。\nDetails: {e}")
        return

    # ネットワークを評価モードに
    action_net.eval()
    reaction_net.eval()
    gain_net.eval()

    # センサー入力次元数を先に定義
    # GRF のみを使用
    sensor_dim = 1

    # --- グラフ描画設定 ---
    fig = plt.figure(figsize=(22, 7))

    # π単位の目盛り位置とラベルを定義
    # 0, 90度, 180度, 270度, 360度
    tick_pos = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    tick_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
     # ---------------------------------------------------------
    # 1. Gain G(phi, s) の描画 (ヒートマップ) - 左側
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(131)
    
    # Phi と GRF の2変数でメッシュを作成
    phi_grid = np.linspace(0, 2 * np.pi, 500)
    grf_grid = np.linspace(0, 50, 5000)
    PHI, GRF = np.meshgrid(phi_grid, grf_grid)
    
    # データの準備
    grid_points = PHI.size
    flat_phis = PHI.flatten()
    flat_grfs = GRF.flatten()
    
    # GainNetへの入力作成
    # Phi入力 (sin, cos)
    phi_in = torch.tensor(np.stack([np.sin(flat_phis), np.cos(flat_phis)], axis=1), dtype=torch.float32)
    
    # Sensor入力 (GRFのみ)
    sens_in_np = flat_grfs.reshape(-1, 1)
    sens_in = torch.tensor(sens_in_np, dtype=torch.float32)
    
    with torch.no_grad():
        # GainNetは phi と sensor の両方を取るが、ここではGRFのみを渡す
        gains = gain_net(phi_in, sens_in[:, 0:1]).numpy().flatten()
    
    GAIN = gains.reshape(PHI.shape)
    
    # ヒートマップ描画
    c = ax1.pcolormesh(PHI, GRF, GAIN, cmap='viridis', shading='auto')
    ax1.set_title(r"Gain $\sigma$($\phi$, F)")
    ax1.set_xlabel(r"Phase $\phi$ [rad]")
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(tick_labels)
    ax1.set_ylabel("GRF [N]")
    fig.colorbar(c, ax=ax1, label='Gain Value')

    # ---------------------------------------------------------
    # 2. Action A(phi) の描画 - 中央
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(132)
    
    # 位相 phi を 0 ~ 2pi でスイープ
    phis = np.linspace(0, 2 * np.pi, 1085)
    # sin, cos に変換して入力
    phi_inputs = torch.tensor(np.stack([np.sin(phis), np.cos(phis)], axis=1), dtype=torch.float32)
    
    with torch.no_grad():
        # ActionNetの出力
        actions = action_net(phi_inputs).numpy().flatten()
        #actions = -np.cos(phis)  # 固定アクション関数もプロット
    
    ax2.plot(phis, actions, label=r'$\frac{\partial C}{\partial \phi}$', color='blue', linewidth=2)
    ax2.set_title(r"Action $\frac{\partial C}{\partial \phi}(\phi)$")
    ax2.set_xlabel(r"Phase $\phi$ [rad]")
    ax2.set_ylabel("Action Output")
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels)
    ax2.grid(True)
    ax2.set_ylim([-1.1, 1.1])
    
    # ---------------------------------------------------------
    # 3. Reaction R(s) の描画 (GRFに対する反応) - 右側
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(133)

    # センサー入力次元数 (obsから推測)
    # [GRF, Hip, Knee, HipVel, KneeVel] と仮定
# GRFだけを 0 ~ 6000 N くらいまで変化させる
    grf_values = np.linspace(0, 6000, 60000)
    sensor_inputs = grf_values.reshape(-1, 1)
    
    sensor_tensor = torch.tensor(sensor_inputs, dtype=torch.float32)
    
    with torch.no_grad():
        # ReactionNet expects GRF-only input (shape (N,1))
        reactions = reaction_net(sensor_tensor[:, 0:1]).numpy().flatten()
        #reactions= np.arcsinh(grf_values)  # 固定リアクション関数もプロット

    ax3.plot(grf_values, reactions, label=r'Reaction S(F)', color='red', linewidth=2)
    ax3.set_title("Reaction S(F)")
    ax3.set_xlabel("GRF [N]")
    ax3.set_ylabel("Reaction Output")
    ax3.grid(True)
    
    # グラフ間に隙間を設定
    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.20, wspace=0.20, hspace=0.3)
    plt.show()

if __name__ == "__main__":
    # ここに学習済みモデルのパスを指定
    MODEL_PATH = "RL\\results\\TH_baseline_debug_2\\models\\gainonly\\seed_0\\final_model.zip"  
    
    if os.path.exists(MODEL_PATH):
        visualize_tegotae_functions(MODEL_PATH)
    else:
        print(f"Model file not found: {MODEL_PATH}")
        # 最新のモデルを探す処理などを入れても良い