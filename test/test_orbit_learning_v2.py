import pandas as pd
import numpy as np
import math
import cma
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d  # 補間用

# ==========================================
# 設定
# ==========================================
csv_file = "csvdata20251215/squat_flying_data_20251215_162251.csv"      # データファイル名
xml_path = "vertical_hopper.xml"
T = 1.3585                        # データの周期 (秒)
freq = 1/T                        # 周波数 (Hz)
N_BASIS = 20                      # 基底関数の数 (多いほど滑らか)
H_PARAM = 20.0                    # ガウス山の鋭さ (帯域幅の逆数)

# ==========================================
# 1. 堅牢なデータ読み込み & 前処理
# ==========================================
print("1. データ読み込み中 (RBF)...")
try:
    df = pd.read_csv(csv_file) 
    
    # 列数による判定
    if len(df.columns) >= 5:
        # [time, ..., target_hip, target_knee] (5列パターン)
        time_col = df.iloc[:, 0].values
        y_hip = df.iloc[:, 6].values  # 4列目
        y_knee = df.iloc[:, 7].values # 5列目
    elif len(df.columns) >= 3:
        # [time, hip, knee] (3列パターン)
        time_col = df.iloc[:, 0].values
        y_hip = df.iloc[:, 1].values
        y_knee = df.iloc[:, 2].values
    else:
        raise ValueError("CSVの列数が足りません")

    df_clean = pd.DataFrame({'t': time_col, 'h': y_hip, 'k': y_knee}).dropna()
    
    # 学習用区間 (5.0s ~ 9.0s など安定している場所)
    # ※ここで切り出したデータを「1周期分」として扱います
    start_time = 5.0
    end_time = start_time + T # 正確に1周期分だけ切り出すのが理想
    
    mask = (df_clean['t'] >= start_time) & (df_clean['t'] < end_time)
    subset = df_clean[mask]
    
    if len(subset) == 0: raise ValueError("指定範囲のデータなし")

    # 時間を 0〜cycle_time に正規化
    t_ref = subset['t'].values - subset['t'].values[0]
    y_hip_ref = subset['h'].values
    y_knee_ref = subset['k'].values

    # ★補間関数の作成 (時刻 t を入れたら、正解の角度を返す関数)
    # fill_value="extrapolate" で範囲外エラーを防ぐ
    ref_func_hip = interp1d(t_ref, y_hip_ref, kind='linear', fill_value="extrapolate")
    ref_func_knee = interp1d(t_ref, y_knee_ref, kind='linear', fill_value="extrapolate")
    
    # 初期重みのためのデータ準備 (位相ベース)
    phase_ref = (t_ref * 2 * np.pi * freq) % (2 * np.pi)
    Y_target = np.column_stack([y_hip_ref, y_knee_ref])

    print(f"参照軌道作成完了: {len(subset)} サンプル使用")

except Exception as e:
    print(f"エラー: {e}")
    exit()

# ==========================================
# 2. Von Mises 基底関数の定義
# ==========================================
def get_rbf_features(phi_array, n_basis=N_BASIS, h=H_PARAM):
    if np.isscalar(phi_array): phi_array = np.array([phi_array])
    centers = np.linspace(0, 2*np.pi, n_basis, endpoint=False)
    features = [np.ones(len(phi_array))]
    for c in centers:
        features.append(np.exp(h * (np.cos(phi_array - c) - 1)))
    return np.column_stack(features)

print("初期学習(LS)完了。最適化へ移行します。")

# ==========================================
# 3. 最小二乗法 (初期化)
# ==========================================
X = get_rbf_features(phase_ref)
# W = (X^T X)^-1 X^T Y
W_init, _, _, _ = np.linalg.lstsq(X, Y_target, rcond=None)
initial_params = W_init.flatten()
n_feats = X.shape[1]
print(f"初期学習完了。重み数: {W_init.size}")

# ==========================================
# 4. CMA-ES 最適化
# ==========================================
print("CMA-ES 最適化を開始...")
model_spec = mujoco.MjModel.from_xml_path(xml_path)
initial_params = W_init.flatten()
n_feats = X.shape[1]

def evaluate(params):
    W = params.reshape(n_feats, 2)
    data = mujoco.MjData(model_spec)
    
    # 安定化設定
    hip_id = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
    knee_id = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
    hip_adr = model_spec.jnt_qposadr[hip_id]
    knee_adr = model_spec.jnt_qposadr[knee_id]
    model_spec.dof_damping[model_spec.jnt_dofadr[hip_id]] = 10.0
    model_spec.dof_damping[model_spec.jnt_dofadr[knee_id]] = 10.0
    
    phase = 0.0
    dt = model_spec.opt.timestep
    total_error = 0.0
    steps = 0

    sim_duration = T * 5  # 5周期分シミュレーション
    
    # 4秒シミュレーション
    while data.time < sim_duration:
        phase = (phase + 2*math.pi*freq*dt) % (2*math.pi)
        
        # RBF予測 (1サンプル分)
        feat_vec = get_rbf_features(phase).flatten()
        tgt = feat_vec @ W
        
        data.ctrl[0] = math.degrees(tgt[0])
        data.ctrl[1] = math.degrees(tgt[1])
        mujoco.mj_step(model_spec, data)

        # 誤差計算 
        # 現在の時刻に対応する「正解データ」を取得 (周期的にループさせる)
        ref_t = data.time % T
        
        target_h = ref_func_hip(ref_t)
        target_k = ref_func_knee(ref_t)
        
        # 実際の関節角度
        real_h = data.qpos[hip_adr]
        real_k = data.qpos[knee_adr]
        
        # 二乗誤差の累積
        error_h = (real_h - target_h) ** 2
        error_k = (real_k - target_k) ** 2
        
        # 最初の0.5秒は着地安定待ちとして無視しても良いが、今回は全期間評価
        if data.time > 0.5:
            total_error += (error_h + error_k)
            steps += 1
            
        # 転倒判定 (高さが低すぎたらペナルティ)
        if data.qpos[0] < 0.3: # rootz
            return 1000.0 # 大きなペナルティ

    if steps == 0: return 1000.0
    
    # 平均二乗誤差 (MSE) を返す
    return total_error / steps
    

es = cma.CMAEvolutionStrategy(initial_params, 0.05, {'popsize': 8, 'maxiter': 20})
es.optimize(evaluate)

best_params = es.result.xbest
best_score = es.result.fbest # これは最小化された誤差

print(f"最適化完了。最小誤差: {best_score:.6f}m")
best_W = best_params.reshape(n_feats, 2)

# ==========================================
# 5. 再生
# ==========================================
input("Enterで再生...")
data = mujoco.MjData(model_spec)
with mujoco.viewer.launch_passive(model_spec, data) as viewer:
    phase = 0.0
    hip_adr = model_spec.jnt_qposadr[mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")]

    # グラフ表示用
    hist_real = []
    hist_ref = []
    while viewer.is_running():
        step_start = time.time()
        phase = (phase + 2*math.pi*freq*model_spec.opt.timestep) % (2*math.pi)
        
        feat_vec = get_rbf_features(phase).flatten()
        tgt = feat_vec @ best_W
        
        data.ctrl[0] = math.degrees(tgt[0])
        data.ctrl[1] = math.degrees(tgt[1])
        mujoco.mj_step(model_spec, data)
        viewer.sync()

        time_until_next = model_spec.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)
        
        # グラフ用データ収集
        current_time = data.time % T
        hist_real.append(data.qpos[hip_adr])
        hist_ref.append(ref_func_hip(current_time))

# グラフ表示
plt.figure()
plt.plot(hist_real, label='Real Hip Angle')
plt.plot(hist_ref, label='Reference Hip Angle', linestyle='--')
plt.legend()
plt.title('Hip Angle Tracking')
plt.xlabel('Timestep')
plt.ylabel('Angle (rad)')
plt.grid()
plt.show()
