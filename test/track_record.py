import pandas as pd
import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cv2  # 動画保存用

# ==========================================
# 設定
# ==========================================
csv_file = "CPG_orbit_fourier_order20_editted.csv"
xml_path = "vertical_hopper.xml"

# 動画保存の設定
RECORD_VIDEO = True          # 動画を保存するかどうか
VIDEO_FILENAME = "simulation_video.mp4"
VIDEO_FPS = 30               # 保存する動画のフレームレート
WIDTH, HEIGHT = 640, 480     # 動画サイズ

# グラフ保存の設定
GRAPH_FILENAME = "jump_height_log.png"

# ★重要: まずはここを True にして「足だけ動くか」確認
PIN_ROOT = False
PLAY_SPEED = 1.0

# ==========================================
# 1. データ読み込み
# ==========================================
print(f"[{csv_file}] を読み込んでいます...")
import glob
import os
if not os.path.exists(csv_file):
    list_of_files = glob.glob('CPG_orbit_fourier_v2.csv')
    if list_of_files:
        csv_file = max(list_of_files, key=os.path.getctime)
        print(f"最新のファイルを発見: {csv_file}")
    else:
        print("エラー: CSVファイルが見つかりません。")
        # テスト用にダミーデータを作る場合の処理（省略可）
        exit()

try:
    df = pd.read_csv(csv_file)
    t_vals = df['Phase'].values * (1.35625 / (2 * np.pi))
    hip_vals = df['Hip'].values
    knee_vals = df['Knee'].values
    
    t_vals = t_vals - t_vals[0]
    duration = 1.35625
    
    ref_hip = interp1d(t_vals, hip_vals, fill_value="extrapolate")
    ref_knee = interp1d(t_vals, knee_vals, fill_value="extrapolate")
    print(f"データ読み込み完了: {len(t_vals)}行")

except Exception as e:
    print(f"データ読み込みエラー: {e}")
    exit()

# ==========================================
# 2. シミュレーション準備
# ==========================================
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# ID取得
hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rootz") # 高さ取得用

hip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_joint")
knee_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_joint")

# 初期姿勢
data.qpos[model.jnt_qposadr[hip_id]] = hip_vals[0]
data.qpos[model.jnt_qposadr[knee_id]] = knee_vals[0]
data.qpos[model.jnt_qposadr[root_id]] = 1.0

mujoco.mj_forward(model, data)

# 動画用レンダラーとWriterの準備
renderer = None
video_writer = None
if RECORD_VIDEO:
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Macなら'avc1'などが良い場合もあり
    video_writer = cv2.VideoWriter(VIDEO_FILENAME, fourcc, VIDEO_FPS, (WIDTH, HEIGHT))
    print(f"動画記録を開始します: {VIDEO_FILENAME}")

print(">>> Enterキーを押すと再生開始 <<<")
input()

# ==========================================
# 3. ループ実行
# ==========================================
log_t = []
log_real_hip, log_ref_hip = [], []
log_height = [] # 高さ記録用

# 動画フレーム制御用
last_render_time = 0
render_interval = 1.0 / VIDEO_FPS


sim_t = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_sys_time = time.time()
    
    # 終了条件: viewerが開いていて、かつ一定時間(例:10秒)まで
    # while viewer.is_running() and sim_t < 10.0: 
    for _ in range(12000): # 既存コードに合わせるならこちら
        step_start = time.time()

        # CSVループ再生
        ref_t = sim_t % duration
        target_h = ref_hip(ref_t)
        target_k = ref_knee(ref_t)
        
        # 制御入力
        data.ctrl[hip_act] = target_h
        data.ctrl[knee_act] = target_k
        
        if PIN_ROOT:
            data.qvel[model.jnt_dofadr[root_id]] = 0.0
            data.qpos[model.jnt_qposadr[root_id]] = 1.0

        # ステップ進行
        mujoco.mj_step(model, data)
        viewer.sync()
        
        sim_t += model.opt.timestep
        
        # --- ログ記録 ---
        current_height = data.qpos[model.jnt_qposadr[root_id]]
        
        log_t.append(sim_t)
        log_ref_hip.append(target_h)
        log_real_hip.append(data.qpos[model.jnt_qposadr[hip_id]])
        log_height.append(current_height) # 高さを追加

        # --- 動画記録 ---
        if RECORD_VIDEO and (sim_t - last_render_time >= render_interval):
            renderer.update_scene(data, camera='track')
            frame = renderer.render()
            # MuJoCo(RGB) -> OpenCV(BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            last_render_time = sim_t

        # 時間調整
        time_until_next = (model.opt.timestep / PLAY_SPEED) - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

# 後処理
if video_writer:
    video_writer.release()
    print(f"動画を保存しました: {VIDEO_FILENAME}")

# ==========================================
# 4. グラフ保存・表示
# ==========================================
plt.figure(figsize=(12, 8))

# 上段: 股関節の追従確認
plt.subplot(2, 1, 1)
plt.plot(log_t, log_ref_hip, 'r--', label="Reference", linewidth=2)
plt.plot(log_t, log_real_hip, 'b-', label="Real Hip", alpha=0.7)
plt.title(f"Joint Tracking (Pin Root: {PIN_ROOT})")
plt.ylabel("Angle [rad]")
plt.legend()
plt.grid()

# 下段: 跳躍高さ
plt.subplot(2, 1, 2)
plt.plot(log_t, log_height, 'g-', label="Body Height (Root Z)", linewidth=2)
plt.title("Jump Height")
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(GRAPH_FILENAME) # 画像として保存
print(f"グラフを保存しました: {GRAPH_FILENAME}")
plt.show()