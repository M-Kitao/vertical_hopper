import time
import math
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# 1. モデルの読み込み
xml_path = "vertical_hopper.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
os.makedirs(f'csvdata{time.strftime("%Y%m%d")}', exist_ok=True) # ディレクトリ作成

# リンクの長さ設定 (XMLの定義値に基づく)
L1 = 0.5  # 大腿 (Thigh)
L2 = 0.4  # 下腿 (Shank) ※Sphereの中心までの概算

# ジョイントとアクチュエータのIDを取得
root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rootz")
hip_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_joint")
knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_joint")
qpos_root_adr = model.jnt_qposadr[root_id]
foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_site")
hip_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
knee_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
hip_addr = model.jnt_qposadr[hip_jid]
knee_addr = model.jnt_qposadr[knee_jid]
#model.dof_damping[model.jnt_dofadr[knee_jid]] = 15.0
#model.dof_damping[model.jnt_dofadr[hip_jid]] = 15.0
tau = 0.03                   # (s) 1次フィルタの時定数
prev_time = time.time()
prev_desired_h = None

# 目標角度を滑らかにするための現在の目標（初期値は現在角度）
smoothed_target_hip  = data.qpos[hip_addr]
smoothed_target_knee = data.qpos[knee_addr]
alpha = 0.15  # 補間係数（0->即応、0.1~0.2 程度が滑らか）
current_target_h = 0.6

# データを収集するためのリスト
time_data = []
smoothed_target_hip_list  = []
smoothed_target_knee_list = []
foot_x_data = []
foot_z_data = []
target_h_data = []
q_hip_data = []
q_knee_data = []
desired_h_list = []
h_dot_list = []
grf_list = []

# 地面反力の取得
try:
    #heel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'heel_grf')
    foot_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'foot_grf')
    
    #heel_sensor_adr = self.model.sensor_adr[heel_sensor_id]
    foot_sensor_adr = model.sensor_adr[foot_sensor_id]
except ValueError:
    print("エラー: センサー名がXML内に見つかりません。XMLを編集しましたか？")
    exit()

def inverse_kinematics(h):
    """
    足先を腰の真下(X=0)、距離hに保つための関節角度を計算する関数
    h: 腰から足先までの垂直距離
    """
    # 物理的に届かない距離の場合は制限する
    h = min(max(h, 0.32), L1 + L2 - 0.001)

    # 1. 膝の角度 (余弦定理)
    # h^2 = L1^2 + L2^2 - 2*L1*L2*cos(pi - theta_knee_internal)
    # cos_gamma = (L1^2 + L2^2 - h^2) / (2 * L1 * L2)
    # gamma = arccos(...)
    # knee_angle = gamma - pi (0度が真っ直ぐ、負の値で曲がる)
    
    val = (L1**2 + L2**2 - h**2) / (2 * L1 * L2)
    gamma = math.acos(val)
    q_knee = gamma - math.pi

    # 2. 股関節の角度 (余弦定理と幾何学)
    # 足先をX=0にするには、膝を曲げた分だけ太ももを前に出す必要がある
    # L2^2 = L1^2 + h^2 - 2*L1*h*cos(theta_hip)
    
    val2 = (L1**2 + h**2 - L2**2) / (2 * L1 * h)
    q_hip = math.acos(val2)

    return q_hip, q_knee

# ビューアを起動してシミュレーション
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()

    #初期位置
    data.qpos[qpos_root_adr] = 0.0  # 腰の高さ
    data.qvel[qpos_root_adr] = 0.0  # 速度ゼロ
    data.ctrl[hip_act_id] = math.acos((L1**2 + 0.85**2 - L2**2) / (2 * L1 * 0.85))  # 初期股関節角度
    data.ctrl[knee_act_id] = math.acos((L1**2 + L2**2 - 0.85**2) / (2 * L1 * L2)) - math.pi  # 初期膝関節角度

    phase_time = 0.0  # 位相時間

    is_grounded = False      # 現在の接地状態
    contact_threshold_ON = 400.0  # 接地とみなす閾値
    contact_threshold_OFF = -10.0  # 離陸とみなす閾値（低く設定してチャタリング防止）
    
    for _ in range(8000): #10秒間シミュレーション
    #while viewer.is_running():
        step_start = time.time()
        
        # --- 制御ロジック ---
        tnow = time.time()
        dt = max(1e-6, tnow - prev_time)

        grf = -data.sensordata[foot_sensor_adr + 2]  # Z成分で反力取得

        if is_grounded:
            if grf < contact_threshold_OFF:
                is_grounded = False
        else:
            if grf > contact_threshold_ON:
                is_grounded = True
        in_flight = not is_grounded

        # A. 軌道生成
        target_pos = [0, 0, 0.60 + 0.25 * math.cos(2 * math.pi * 3.0 * phase_time)] 

        # 実際のhを計測 (ワールド座標系での腰のZ - 足先のZ)
        Z_root_pos = data.qpos[qpos_root_adr]
        foot_z_pos = data.site_xpos[foot_site_id][2]
        actual_h = Z_root_pos - foot_z_pos
        desired_h = target_pos[2]

        # 足先の高さと上昇速度を取得
        foot_z = data.site_xpos[foot_site_id][2]

        
        if prev_desired_h is None:
            h_dot = 0.0
        else:
            h_dot = (desired_h - prev_desired_h) / dt
        prev_desired_h = actual_h
        prev_time = tnow        
        
        if  in_flight:
            desired_h = 0.85
        else:
            desired_h = target_pos[2]

        # 目標値が急に変わっても、徐々に追従させる (ローパスフィルタ)
        blend_rate = 0.8  # 0.05〜0.2くらいで調整（小さいほど滑らかだが遅れる）
        current_target_h = (1 - blend_rate) * current_target_h + blend_rate * desired_h
        
        # B. 逆運動学 (IK) で目標角度を計算
        #target_hip_angle, target_knee_angle = inverse_kinematics(current_target_h)
        
        target_hip_angle, target_knee_angle = inverse_kinematics(current_target_h)
        #目標角をスムーズに更新（急なジャンプ切替を和らげる）
        alpha = dt / (tau + dt)
        smoothed_target_hip  = (1 - alpha) * smoothed_target_hip  + alpha * target_hip_angle
        smoothed_target_knee = (1 - alpha) * smoothed_target_knee + alpha * target_knee_angle

        
        
        # C. モータへの指令         
        data.ctrl[hip_act_id] = smoothed_target_hip
        data.ctrl[knee_act_id] = smoothed_target_knee

        # --- 物理的な固定 ---
        
        # D. 腰の空中固定 (ルートジョイントの上書き)
        #data.qpos[qpos_root_adr] = 0.5  # 高さ固定
        #data.qvel[qpos_root_adr] = 0.0  # 速度ゼロ
        
        # --------------------

        # データ収集
        time_data.append(data.time)
        # foot_siteの位置はdata.site_xposに格納されている (X, Y, Z)
        foot_pos = data.site_xpos[foot_site_id]
        
        foot_x_data.append(foot_pos[0])
        foot_z_data.append(foot_pos[2])
        target_h_data.append(data.qpos[qpos_root_adr] - current_target_h)  
        desired_h_list.append(desired_h)

        q_hip_data.append(data.qpos[model.jnt_qposadr[hip_jid]])
        q_knee_data.append(data.qpos[model.jnt_qposadr[knee_jid]])

        smoothed_target_hip_list.append(smoothed_target_hip)
        smoothed_target_knee_list.append(smoothed_target_knee)

        grf_list.append(grf)


        # 物理ステップを進める
        mujoco.mj_step(model, data)
        viewer.sync()

        # リアルタイム制御のための待機
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

timestamp = time.strftime("%Y%m%d_%H%M%S")
# 2. データをCSVに保存
with open(f'csvdata{time.strftime("%Y%m%d")}/squat_flying_data_{timestamp}.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Time', 'Foot_X', 'Foot_Z', 'Target_H', 'Q_Hip', 'Q_Knee', 'Smoothed_Target_Hip', 'Smoothed_Target_Knee', 'GRF_Z'])
    for i in range(len(time_data)):
        csv_writer.writerow([time_data[i], foot_x_data[i], foot_z_data[i], target_h_data[i], q_hip_data[i], q_knee_data[i], smoothed_target_hip_list[i], smoothed_target_knee_list[i], grf_list[i]])

# 3. グラフのプロット
plt.figure(figsize=(12, 6))

# X軸の追従グラフ (X=0固定の確認)
plt.subplot(1, 2, 1)
plt.plot(time_data, foot_x_data, label='Foot X Position')
plt.axhline(0, color='r', linestyle='--', label='Target X = 0')
plt.title('Foot X Position Over Time (X-axis Constraint)')
plt.xlabel('Time (s)')
plt.ylabel('X Position (m)')
plt.legend()
plt.grid(True)

# Z軸の追従グラフ (高さ H(t) の追従確認)
plt.subplot(1, 2, 2)
# ワールド座標系での足先Z目標値は、腰の固定高さ(1.0) - 目標距離(target_h)
target_world_z = np.array(target_h_data) 

plt.plot(time_data, foot_z_data, label='Foot Z Position (Actual)')
plt.plot(time_data, target_world_z + 0.965, 'r--', label='Target Z Position')
plt.title('Foot Z Position Over Time (Squat Tracking)')
plt.xlabel('Time (s)')
plt.ylabel('Z Position (m)')
plt.legend()
plt.grid(True)

# 関節角度のプロット
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_data, q_hip_data, label='Hip Joint Angle')
plt.plot(time_data, smoothed_target_hip_list, 'g--', label='Target Hip Angle')
plt.title('Hip Joint Angle Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time_data, q_knee_data, label='Knee Joint Angle', color='orange')
plt.plot(time_data, smoothed_target_knee_list, 'r--', label='Target Knee Angle')
plt.title('Knee Joint Angle Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 4))
plt.plot(time_data, grf_list, label='Ground Reaction Force (Z)')
plt.title('Ground Reaction Force Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.grid(True)

#desired_hをプロット
plt.figure(figsize=(12, 4))
plt.plot(time_data, desired_h_list, label='Desired Height h(t)')
plt.title('Desired Height Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



print("シミュレーションが完了しました。グラフが出力されました。")
