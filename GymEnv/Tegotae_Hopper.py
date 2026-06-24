import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import torch
import os
import sys

# このファイルの場所 (GymEnv/) を取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 一つ上の階層 (vertical_hopper/ ルート) を取得
root_dir = os.path.dirname(current_dir)

# Pythonがモジュールを探すパスに、ルート、CPG、NNフォルダを追加
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'CPG'))
sys.path.append(os.path.join(root_dir, 'NN'))

from CPG.Kuramoto import Kuramoto_Oscillator
from NN.ActionNet import ActionNetMLP
from NN.ReactionNet import ReactionNetMLP
from NN.GainNet import GainNetMLP
from NN.CPG_to_joints import CPGToJointsMLP

class Tegotae_Hopper_Env(gym.Env):
    """
    Tegotae Feedback則に基づいて垂直Hopperを制御するGym環境
    step() 内でニューラルネットワークとKuramoto振動子の更新を行い、MuJoCoを駆動
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, model_path=None, render_mode=None, contact_model: str = "hunt_crossley"):
        super().__init__()

        # MuJoCoモデルの読み込み
        if model_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "vertical_hopper.xml")
        else:
            xml_path = model_path

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XMLファイルが{xml_path}に無いよー")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.contact_model = contact_model

        if self.contact_model == "hunt_crossley":
            self._apply_hunt_crossley_contact_model()

        self.dt = float(self.model.opt.timestep) #XMLのoption timestep=0.001250

        # Sensor: [ground_contact_force, hip_angle, knee_angle, hip_vel, knee_vel] -> dim=5
        self.sensor_dim = 5

        #networkの初期化
        self.action_out_dim = 1 #A(Φ)の出力次元
        self.reaction_out_dim = 1 #R(s)の出力次元
        self.gain_out_dim = 1 #G(Φ, s)の出力次元

        self.reaction_in_dim = 1 #センサー入力次元(一旦地面反力のみにする)
        self.gain_in_dim = 2 + self.reaction_in_dim #CPG位相2 + センサー入力次元

        #インスタンス化
        self.action_net = ActionNetMLP(output_dim=self.action_out_dim)
        self.reaction_net = ReactionNetMLP(input_dim=self.reaction_in_dim, output_dim=self.reaction_out_dim)
        self.gain_net = GainNetMLP(sensor_dim=self.reaction_in_dim, output_dim=self.gain_out_dim)

        #動作範囲を定義
        self.cpg_mapper = CPGToJointsMLP(
            hip_range=(-np.pi/2, np.pi/2),
            knee_range=(-3*np.pi/4, 0),
            vel_max=5.0
        )

        self.oscillator = Kuramoto_Oscillator(omega=np.pi*2.0, dt=self.dt) #1Hz

        #観測空間の定義
        obs_dim = self.reaction_in_dim + 1  #センサー入力 + CPG位相
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 行動空間: 
        # もし「重みの学習」ではなく「シミュレーション実行」が主なら、Emptyあるいはダミーでも可。
        # ここでは、外部からの摂動やパラメータ調整を受け付ける余地としてBoxを用意しますが、
        # 基本的に制御は内部のTegotaeループで完結します。
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        #rendererの初期化
        self.viewer = None

    def _apply_hunt_crossley_contact_model(self):
        """Apply Hunt-Crossley style contact parameters to the loaded MuJoCo model."""
        hunt_solimp = np.array([0.95, 0.99, 0.01, 0.5, 2.0], dtype=np.float64)
        hunt_solref = np.array([0.01, 10.0], dtype=np.float64)

        self.model.opt.o_solimp[:] = hunt_solimp
        self.model.opt.o_solref[:] = hunt_solref

        foot_geom_names = {
            "foot_sphere", "footsphere",
            "right_foot_geom", "left_foot_geom",
            "right_toe_geom", "left_toe_geom"
        }
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name in foot_geom_names:
                self.model.geom_solimp[geom_id] = hunt_solimp
                self.model.geom_solref[geom_id] = hunt_solref

    def _get_sensor_data(self):
        """
        MuJoCoの物理状態からニューラルネットに入力するためのセンサーベクトルを作成
        Sensor Vector Definition:
        0: Ground Reaction Force (approximate from contact)
        1: Hip Joint Angle
        2: Knee Joint Angle
        3: Hip Joint Velocity
        4: Knee Joint Velocity
        """
        # 地面反力の取得
        try:
            heel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'heel_grf')
            foot_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'foot_grf')
    
            heel_sensor_adr = self.model.sensor_adr[heel_sensor_id]
            foot_sensor_adr = self.model.sensor_adr[foot_sensor_id]
        except ValueError:
            print("エラー: センサー名がXML内に見つかりません。XMLを編集しましたか？")
            exit()

        # (viewer.sync() など)
        # センサーデータを取得 (各センサーは 3D ベクトル [Fx, Fy, Fz])
        heel_force = self.data.sensordata[heel_sensor_adr : heel_sensor_adr + 3]
        foot_force = self.data.sensordata[foot_sensor_adr : foot_sensor_adr + 3]

        # 足全体の合計地面反力
        self.grf = heel_force# + foot_force heel_forceのみで(多分)十分
        z_grf = self.grf[2]  # z方向成分

        # ジョイント角度と速度の取得
        hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
        knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")

        # jnt_qposadr[id] にそのジョイントのqpos配列上の開始インデックスが入っています
        hip_qpos_adr = self.model.jnt_qposadr[hip_id]
        knee_qpos_adr = self.model.jnt_qposadr[knee_id]
        
        hip_angle = self.data.qpos[hip_qpos_adr]
        knee_angle = self.data.qpos[knee_qpos_adr]
        
        # 速度は自由度(DOF)に対応するため jnt_dofadr を使います
        hip_dof_adr = self.model.jnt_dofadr[hip_id]
        knee_dof_adr = self.model.jnt_dofadr[knee_id]
        
        hip_vel = self.data.qvel[hip_dof_adr]
        knee_vel = self.data.qvel[knee_dof_adr]

        sensor_data = torch.tensor([z_grf, hip_angle, knee_angle, hip_vel, knee_vel], dtype=torch.float32)

        return sensor_data
    
    def _get_obs(self, sensor_data):
        phi = self.oscillator.phi
        z_pos = self.data.qpos[0]  # 垂直位置
        z_vel = self.data.qvel[0]  # 垂直速度

        obs = np.concatenate([
            [z_pos, z_vel],
            [phi, np.sin(phi), np.cos(phi)],
            sensor_data.numpy()
        ])
        return obs.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.oscillator.reset(phi0=0.0)

        # 初期姿勢
        self.data.qpos[0] = np.random.uniform(1.0, 2.0)  # 初期高さ
        mujoco.mj_forward(self.model, self.data) 

        sensor_data = self._get_sensor_data()
        return self._get_obs(sensor_data), {}
    
    def step(self, action):
        """
        1ステップの進行:
        1. センサー情報取得
        2. NNs (Action, Reaction, Gain) Forward Pass
        3. 振動子 (Kuramoto) の位相更新 (Tegotae Feedback)
        4. 位相 -> 目標関節角度の変換 (CPGToJoints)
        5. MuJoCoのアクチュエータ制御入力更新
        6. 物理シミュレーション進行
        """
        # 1. センサー情報取得
        sensor_tensor = self._get_sensor_data()[0:self.reaction_in_dim]  # 必要なセンサー情報のみ抽出
        phi_tensor = torch.tensor([self.oscillator.phi], dtype=torch.float32)

        # 2. NNs Forward Pass
        with torch.no_grad():
            act_val = self.action_net(phi_tensor.unsqueeze(0)).squeeze()  # A(Φ)
            react_val = self.reaction_net(sensor_tensor.unsqueeze(0)).squeeze()  # R(s)
            gain_val = self.gain_net(phi_tensor.unsqueeze(0), sensor_tensor.unsqueeze(0)).squeeze()  # G(Φ, s)

            joint_targets = self.cpg_mapper(phi_tensor.unsqueeze(0)).squeeze()  # CPG位相 -> 関節目標値

        # 3. 振動子の位相更新 (Tegotae Feedback) dphai_dt = ω + G(Φ, s) * A(Φ) * R(s)
        self.oscillator.step(gain=gain_val.item(), action=act_val.item(), reaction=react_val.item())

        # 4. MuJoCoのアクチュエータ制御入力更新
        # Actuator name="hip_joint", "knee_joint"
        # ctrl配列のインデックスはXMLの定義順 (hip, knee)
        target_hip_pos = joint_targets[0].item()
        target_knee_pos = joint_targets[1].item()

        # XMLのActuatorはPosition制御なので、ctrlに位置をセット
        self.data.ctrl[0] = target_hip_pos
        self.data.ctrl[1] = target_knee_pos

        # 5. 物理シミュレーション進行
        mujoco.mj_step(self.model, self.data)

        # 6. 観測値、報酬、終了判定の取得
        sensor_data_new = self._get_sensor_data()
        obs = self._get_obs(sensor_data_new)

        #報酬：輸送コストの逆数
        com_z = self.data.qpos[0]  # COMのz位置
        com_z_vel = self.data.qvel[0]  # z軸速度
        reward = 0.0
        apex_threshold = 0.01  # 頂点とみなすz速度の閾値
        if abs(com_z_vel) < apex_threshold:
            if not hasattr(self, 'prev_apex_z') or self.prev_apex_z is None:
                self.prev_apex_z = com_z
                height_gained = 0.0
            else:
                height_gained = max(0.0, com_z - self.prev_apex_z)  # 前の頂点からの高さの増加分
            if height_gained > 1e-6:
                energy_used = 0.0
                n_write = min(self.model.nu, 2)  # hipとkneeのアクチュエータ数
                qvel = self.data.qvel[:n_write]
                ctrl = self.data.ctrl[:n_write]
                energy_used = np.sum(np.abs(ctrl * qvel)) * self.dt  # 簡易的なエネルギー消費モデル

                cot = (energy_used / height_gained) if height_gained > 0 else np.inf # 輸送コスト(CoT)
                if cot > 0 and np.isfinite(cot):
                    reward += 1.0 / cot  # CoTの逆数を報酬に
                self.prev_apex_z = com_z  # 現在の頂点z位置を保存

        # 終了判定 (転倒など)
        terminated = bool(self.data.qpos[0] < 0.3 or self.data.qpos[0] > 3.0) # 地面より下(異常) or 高すぎ
        truncated = False

        info = {
            "phi": self.oscillator.phi,
            "gain": gain_val.item(),
            "reaction": react_val.item()
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
    
    #環境のレンダリング
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.viewer.is_running():
                self.viewer.sync()
        elif self.render_mode == "rgb_array":
            width, height = 640, 480
            rgb_array = mujoco.mj_render(self.model, self.data, width, height)
            return rgb_array

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None