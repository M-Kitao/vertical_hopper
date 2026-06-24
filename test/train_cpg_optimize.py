import pandas as pd
import numpy as np
import math
import cma
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt

# ==========================================
# ユーザー設定
# ==========================================
csv_file = "squat_flying_data_20251209_204834.csv"      # すでに得たデータセット
xml_path = "vertical_hopper.xml"  # モデルファイル
freq = 0.5                        # 周波数 (0.5Hz -> omega=pi)
N_ORDER = 5                       # フーリエ級数の次数 (5くらいが精度良い)

# ==========================================
# 1. データ前処理 & 最小二乗法 (LS)
# ==========================================
print("1. CSVデータを読み込み、CPG軌道を学習中...")

# CSV読み込み
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"エラー: {csv_file} が見つかりません。パスを確認してください。")
    exit()

# 安定している区間 (例: 5.0秒〜9.0秒) を抽出して学習に使う
# ※データが短い場合は範囲を調整してください
subset = df[(df["Time"] >= 5.0) & (df["Time"] < 9.0)].copy()

if len(subset) == 0:
    print("エラー: 指定した時間範囲のデータがありません。範囲を変更するかCSVを確認してください。")
    exit()

# --- 位相 phi の計算 ---
# phi = (2 * pi * f * t) % 2pi
subset["phase"] = (subset["Time"] * 2 * np.pi * freq) % (2 * np.pi)

# --- フーリエ特徴量の作成 ---
# X = [1, sin(phi), cos(phi), sin(2phi), cos(2phi), ...]
phase_data = subset["phase"].values
features = [np.ones(len(phase_data))] # バイアス項

for n in range(1, N_ORDER + 1):
    features.append(np.sin(n * phase_data))
    features.append(np.cos(n * phase_data))

X = np.column_stack(features) # 入力行列

# --- 教師データ Y ---
# 目標角度 (target_hip, target_knee) がCSVにある前提
# もし実測角(q_hip)を使いたい場合はここを書き換えますが、教師あり学習ならtargetが推奨です
try:
    Y = subset[["Smoothed_Target_Hip", "Smoothed_Target_Knee"]].values
except KeyError:
    # カラム名が違う場合のフォールバック (squats_flying.pyの出力名に合わせて調整)
    # 例: "Smoothed_Target_Hip" など
    print("カラム名 'target_hip' が見つかりません。CSVのヘッダーを確認してください。")
    # 仮に q_hip を使う場合:
    Y = subset[["Q_Hip", "Q_Knee"]].values

# --- 最小二乗法で重み W を計算 ---
# W = (X^T X)^-1 X^T Y
W_init, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

print(f"初期学習完了。重み行列 W の形状: {W_init.shape}")

# 学習結果の確認 (グラフ表示)
pred = X @ W_init
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(subset["phase"], Y[:,0], '.', label="Teacher Hip", alpha=0.3)
plt.plot(subset["phase"], pred[:,0], '.', label="Learned Hip", alpha=0.3)
plt.xlabel("Phase")
plt.title("Hip Joint Learning")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(subset["phase"], Y[:,1], '.', label="Teacher Knee", alpha=0.3)
plt.plot(subset["phase"], pred[:,1], '.', label="Learned Knee", alpha=0.3)
plt.xlabel("Phase")
plt.title("Knee Joint Learning")
plt.show() # 閉じるまで一時停止します

# ==========================================
# 2. CMA-ESによる最適化 (Optimization)
# ==========================================
print("\n2. CMA-ESによる最適化を開始します...")

# 初期パラメータ (最小二乗法の結果)
initial_params = W_init.flatten()
n_features = X.shape[1] # 特徴量の次元数

# 高速化のためモデルをプリロード
model_spec = mujoco.MjModel.from_xml_path(xml_path)

# --- CPG予測関数 ---
def get_target_from_phase(phi, W_flat):
    W = W_flat.reshape(n_features, 2)
    
    # 特徴量ベクトル作成 (スカラ計算版)
    feats = [1.0]
    for n in range(1, N_ORDER + 1):
        feats.append(math.sin(n * phi))
        feats.append(math.cos(n * phi))
    
    return np.array(feats) @ W # [hip, knee]

# --- 評価関数 ---
def evaluate(params):
    data = mujoco.MjData(model_spec)
    
    # 安定化設定
    hip_jid = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
    knee_jid = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
    model_spec.dof_damping[model_spec.jnt_dofadr[hip_jid]] = 10.0
    model_spec.dof_damping[model_spec.jnt_dofadr[knee_jid]] = 10.0
    
    hip_act_id = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_joint")
    knee_act_id = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_joint")
    root_id = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_JOINT, "rootz")
    
    phase = 0.0
    dt = model_spec.opt.timestep
    max_height = 0.0
    
    # 4秒間シミュレーション
    while data.time < 4.0:
        # 位相更新
        phase += 2 * math.pi * freq * dt
        phase %= (2 * math.pi)
        
        # CPG予測 (最適化中のパラメータを使用)
        tgt = get_target_from_phase(phase, params)
        
        # 制御指令
        data.ctrl[hip_act_id] = math.degrees(tgt[0])
        data.ctrl[knee_act_id] = math.degrees(tgt[1])
        
        mujoco.mj_step(model_spec, data)
        
        # 高さ記録
        z = data.qpos[model_spec.jnt_qposadr[root_id]]
        max_height = max(max_height, z)
        
    # ペナルティ処理
    if max_height < 0.5: return 10.0 # 転倒
    if max_height > 1.0: return 5.0 + (max_height - 1.0) * 10.0 # 飛びすぎ防止
    
    return -max_height # 高いほど良い

# CMA-ES実行
# 初期分散 sigma0 は 0.1 (初期値が良いので小さめからスタート)
es = cma.CMAEvolutionStrategy(initial_params, 0.1, {'popsize': 8, 'maxiter': 20})
es.optimize(evaluate)

best_params = es.result.xbest
best_score = -es.result.fbest

print(f"\n最適化完了！")
print(f"最大到達高さ: {best_score:.3f} m")

# ==========================================
# 3. 最適化結果の再生
# ==========================================
input("\nEnterキーを押すと、最適化されたCPGで再生します...")

with mujoco.viewer.launch_passive(model_spec, mujoco.MjData(model_spec)) as viewer:
    data = viewer.data
    # 安定化再設定
    hip_jid = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
    knee_jid = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
    model_spec.dof_damping[model_spec.jnt_dofadr[hip_jid]] = 10.0
    model_spec.dof_damping[model_spec.jnt_dofadr[knee_jid]] = 10.0
    
    phase = 0.0
    
    while viewer.is_running():
        step_start = time.time()
        
        # 位相更新
        phase += 2 * math.pi * freq * model_spec.opt.timestep
        phase %= (2 * math.pi)
        
        # 最適化されたパラメータで予測
        tgt = get_target_from_phase(phase, best_params)
        
        data.ctrl[0] = math.degrees(tgt[0])
        data.ctrl[1] = math.degrees(tgt[1])
        
        mujoco.mj_step(model_spec, data)
        viewer.sync()
        
        # リアルタイム待機
        time_until_next = model_spec.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)