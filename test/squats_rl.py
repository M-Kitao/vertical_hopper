import time
import math
import mujoco
import mujoco.viewer
import numpy as np

# 1. モデルの読み込みと初期設定
xml_path = "vertical_hopper.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# リンクの長さ設定 (XMLの定義値 L1=thigh=0.5m, L2=shank=0.4m に基づく)
L1 = 0.5  
L2 = 0.4  

# ジョイントとアクチュエータのIDを取得
hip_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_joint")
knee_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_joint")
root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rootz")

# rootzジョイントのqposアドレスを取得 (空中固定に使用)
qpos_root_adr = model.jnt_qposadr[root_id] 

# --- ダンピング強化 (揺れ対策) ---
# XMLを編集せずに、Pythonで関節の粘性摩擦を強化
knee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
knee_dof_adr = model.jnt_dofadr[knee_joint_id]
model.dof_damping[knee_dof_adr] = 10.0 # 10.0 (デフォルトの0から強化)

# 2. 逆運動学 (IK) 関数
def inverse_kinematics(h, L1, L2):
    """
    足先を垂直線上に、距離hに保つための関節角度を計算する。
    h: 腰から足先までの垂直距離 (Z方向の長さ)
    """
    # 物理的な可動域制限 (0.1m ~ 0.9m)
    h = min(max(h, 0.1), L1 + L2 - 0.001)

    # 膝の角度 q_knee (ラジアン)
    # 余弦定理: h^2 = L1^2 + L2^2 - 2*L1*L2*cos(gamma)
    # q_knee = gamma - pi (MuJoCoの定義に合わせる)
    try:
        val = (L1**2 + L2**2 - h**2) / (2 * L1 * L2)
        gamma = math.acos(np.clip(val, -1.0, 1.0)) # 誤差対策でクリップ
        q_knee = gamma - math.pi
    except ValueError:
        return 0, 0 

    # 股関節の角度 q_hip (ラジアン)
    # 余弦定理: L2^2 = L1^2 + h^2 - 2*L1*h*cos(q_hip)
    try:
        val2 = (L1**2 + h**2 - L2**2) / (2 * L1 * h)
        q_hip = math.acos(np.clip(val2, -1.0, 1.0))
    except ValueError:
        return 0, 0
    
    # MuJoCoの関節角度は正の値で前に倒れる(屈曲)ため、IKの結果を反転
    return -q_hip, q_knee # hip_jointはXMLで軸が反転している可能性を考慮し、符号を調整

# 3. シミュレーションループ
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    # CPG/MLシミュレーションのための初期設定
    # 周期 T=4秒 (0.25Hz), 中心高さ H_center=0.5m, 振幅 A=0.2m
    H_center = 0.55
    A = 0.2
    
    while viewer.is_running():
        step_start = time.time()
        
        # --- (A) ハイレベルポリシー (ML) のシミュレーション ---
        # MLポリシーが「目標とする脚の長さ (h)」を決定
        now = time.time() - start_time
        # サイン波により、0.35m (しゃがみ) から 0.75m (伸び) まで周期的に変化
        target_h = H_center + A * math.sin(2 * math.pi * 0.25 * now)
        
        # --- (B) ロウレベル制御 (IK) ---
        # 目標の長さ h を達成するための関節角度を計算 (X=0が保証される)
        target_hip_rad, target_knee_rad = inverse_kinematics(target_h, L1, L2)
        
        # --- (C) PD制御による指令 ---
        # MuJoCoのPositionアクチュエータはPD制御で目標角度に追従
        # XMLでangle="degree"が指定されているため、指令値は度に変換
        data.ctrl[hip_act_id] = math.degrees(target_hip_rad)
        data.ctrl[knee_act_id] = math.degrees(target_knee_rad)

        # --- (D) 物理的な固定 (Pinning) ---
        # 腰のZ位置を固定し、重力による落下を防ぐ
        data.qpos[qpos_root_adr] = 0.5  # 固定する高さ (足が床につかない適当な値)
        data.qvel[qpos_root_adr] = 0.0  # 速度ゼロ
        
        # --------------------

        mujoco.mj_step(model, data)
        viewer.sync()

        # リアルタイム制御のための待機 (タイムステップの調整)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)