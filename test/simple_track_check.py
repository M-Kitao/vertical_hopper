import pandas as pd
import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ==========================================
# 設定
# ==========================================
csv_file = "CPG_orbit_bspline_cmaes_v2.csv" # 読み込むファイル名 (squat_flying_data_xxxx.csv に変更してください)
xml_path = "vertical_hopper.xml"

# ★重要: まずはここを True にして「足だけ動くか」確認してください
# Trueなら空中で胴体を固定します。Falseなら物理演算で自由落下させます。
PIN_ROOT = False

# 再生速度 (1.0 = 等倍, 0.2 = スロー)
PLAY_SPEED = 1.0  # 位相速度に合わせる

# ==========================================
# 1. データ読み込み (squats_flying.py形式に対応)
# ==========================================
print(f"[{csv_file}] を読み込んでいます...")

# ファイル検索 (最新のCSVを自動で探す補助ロジック)
import glob
import os
if not os.path.exists(csv_file):
    list_of_files = glob.glob('CPG_orbit_fourier_v2.csv')
    if list_of_files:
        csv_file = max(list_of_files, key=os.path.getctime)
        print(f"最新のファイルを発見: {csv_file}")
    else:
        print("エラー: CSVファイルが見つかりません。")
        exit()

try:
    df = pd.read_csv(csv_file)
    # squats_flying.py の出力形式に合わせる
    # ['Time', 'Foot_X', 'Foot_Z', 'Target_H', 'Q_Hip', 'Q_Knee', 'Smoothed_Target_Hip', 'Smoothed_Target_Knee', 'GRF_Z']
    t_vals = df['Phase'].values * (1.35625 / (2 * np.pi))  # 位相を時間に変換
    # 軌道データとして「目標値(Smoothed)」を使うか「実測値(Q_Hip)」を使うか
    # ここでは「こう動いてほしい」という正解データとして Smoothed を使います
    hip_vals = df['Hip'].values
    knee_vals = df['Knee'].values
    
    # 時間を 0 スタートに正規化
    t_vals = t_vals - t_vals[0]
    duration = 1.35625  # データ全体の長さ (秒)
    
    # 補間関数
    ref_hip = interp1d(t_vals, hip_vals, fill_value="extrapolate")
    ref_knee = interp1d(t_vals, knee_vals, fill_value="extrapolate")
    
    print(f"データ読み込み完了: {len(t_vals)}行, {duration:.2f}秒分")

except Exception as e:
    print(f"データ読み込みエラー: {e}")
    exit()

# ==========================================
# 2. シミュレーション実行
# ==========================================
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# ID取得
hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
# アクチュエータID (名前がjointと同じと仮定)
hip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_joint")
knee_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_joint")

# 初期姿勢をセット
data.qpos[model.jnt_qposadr[hip_id]] = hip_vals[0]
data.qpos[model.jnt_qposadr[knee_id]] = knee_vals[0]
# 胴体の高さ (空中固定用)
root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rootz")
if PIN_ROOT:
    data.qpos[model.jnt_qposadr[root_id]] = 1.0 # 少し浮かせる
else:
    data.qpos[model.jnt_qposadr[root_id]] = 0.0 # 地面に置く

mujoco.mj_forward(model, data)

print(">>> Enterキーを押すと再生開始 (グラフは終了後に表示) <<<")
input()

log_t = []
log_real_hip, log_ref_hip = [], []

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_sys_time = time.time()
    sim_t = 0.0
    
    for _ in range(8000):
    #while viewer.is_running():
        step_start = time.time()

        # CSVの時間をループ再生
        ref_t = sim_t % duration
        
        # 1. 目標値取得
        target_h = ref_hip(ref_t)
        target_k = ref_knee(ref_t)
        
        # 2. 指令 (squats_flying.py と同じ方法)
        data.ctrl[hip_act] = target_h
        data.ctrl[knee_act] = target_k
        
        # ★胴体を固定する場合 (デバッグ用)
        if PIN_ROOT:
            # 速度と加速度を殺して位置を固定
            data.qvel[model.jnt_dofadr[root_id]] = 0.0
            data.qpos[model.jnt_qposadr[root_id]] = 1.0

        # シミュレーション進行
        mujoco.mj_step(model, data)
        viewer.sync()
        
        sim_t += model.opt.timestep
        
        # ログ記録 (描画用)
        #if len(log_t) < 5000: # 重くならないよう制限
        log_t.append(sim_t)
        log_ref_hip.append(target_h)
        log_real_hip.append(data.qpos[model.jnt_qposadr[hip_id]])

        # 時間調整
        time_until_next = (model.opt.timestep / PLAY_SPEED) - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

# ==========================================
# 3. グラフ確認
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(log_t, log_ref_hip, 'r--', label="Reference (CSV)", linewidth=2)
plt.plot(log_t, log_real_hip, 'b-', label="Real (Sim)", alpha=0.7)
plt.title(f"Check Tracking (Pin Root: {PIN_ROOT})")
plt.xlabel("Phase [rad]")
plt.ylabel("Hip Angle [rad]")
plt.legend()
plt.grid()
plt.show()