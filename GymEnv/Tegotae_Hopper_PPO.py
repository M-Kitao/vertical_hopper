import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import torch
import os
import sys
import math
from gymnasium.envs.registration import register

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

class Tegotae_Hopper_Env_PPO(gym.Env):
    """
    Tegotae Feedback則に基づいて垂直Hopperを制御するGym環境
    step() 内でニューラルネットワークとKuramoto振動子の更新を行い、MuJoCoを駆動
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, model_path=None, render_mode=None):
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
        #self.reaction_net = ReactionNetMLP(input_dim=self.reaction_in_dim, output_dim=self.reaction_out_dim)
        #self.gain_net = GainNetMLP(sensor_dim=self.reaction_in_dim, output_dim=self.gain_out_dim)

        #動作範囲を定義
        self.cpg_mapper = CPGToJointsMLP(
            hip_range=(-np.pi/2, np.pi/2),
            knee_range=(-3*np.pi/4, 0),
            vel_max=5.0
        )

        self.oscillator = Kuramoto_Oscillator(omega=np.pi, dt=self.dt) 

        #観測空間の定義
        obs_dim = 2 + 3 + self.sensor_dim #センサー入力 + CPG位相
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 行動空間: 
        # もし「重みの学習」ではなく「シミュレーション実行」が主なら、Emptyあるいはダミーでも可。
        # ここでは、外部からの摂動やパラメータ調整を受け付ける余地としてBoxを用意しますが、
        # 基本的に制御は内部のTegotaeループで完結します。
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        #rendererの初期化
        self.viewer = None

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
            #heel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'heel_grf')
            foot_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'foot_grf')
    
            #heel_sensor_adr = self.model.sensor_adr[heel_sensor_id]
            foot_sensor_adr = self.model.sensor_adr[foot_sensor_id]
        except ValueError:
            print("エラー: センサー名がXML内に見つかりません。XMLを編集しましたか？")
            exit()

        #com_z_vel = self.data.qvel[1]  # z軸速度

        # (viewer.sync() など)
        # センサーデータを取得 (各センサーは 3D ベクトル [Fx, Fy, Fz])
        #heel_force = self.data.sensordata[heel_sensor_adr : heel_sensor_adr + 3]
        foot_force = self.data.sensordata[foot_sensor_adr : foot_sensor_adr + 3]

        # 足全体の合計地面反力
        self.grf = foot_force #heel_forceのみで(多分)十分
        z_grf = abs(self.grf[2])  # z方向成分(センサは負の値で認識するのでabs)

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
        z_pos = self.data.qpos[1]  # 垂直位置
        z_vel = self.data.qvel[1]  # 垂直速度

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
        self.data.qpos[1] = np.random.uniform(1.0, 1.5)  # 初期高さ
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
        sensor_tensor = self._get_sensor_data() # 必要なセンサー情報のみ抽出
        phi_tensor = torch.tensor([self.oscillator.phi], dtype=torch.float32)

        # 2. NNs Forward Pass
        with torch.no_grad():
            act_val = self.action_net(phi_tensor.unsqueeze(0)).squeeze() # A(Φ)
            act_val = act_val.item()
            react_val = float(action[0]) * 0.05 * sensor_tensor[0].item() # R(s)
            gain_val = float(action[1]) * 1.0 # G(Φ, s)

            # NNからの出力（学習初期はほぼゼロ）
            nn_output = self.cpg_mapper(phi_tensor.unsqueeze(0)).squeeze().numpy()
            
            # 強制的なベース動作（位相に同期したサイン波）を作成
            # 振幅 0.5 rad 程度で足を伸縮させる
            # ※ロボットの膝の曲がる方向によって + か - か調整が必要ですが、まずは + で
            base_motion = 0.5 * np.sin(self.oscillator.phi)
            
            # 最終的な指令値 = ベース動作 + NNの微調整
            joint_targets = base_motion + nn_output  # CPG位相 -> 関節目標値

        # 3. 振動子の位相更新 (Tegotae Feedback) dphai_dt = ω + G(Φ, s) * A(Φ) * R(s)
        self.oscillator.step(gain=gain_val, action=act_val, reaction=react_val)

        # 4. MuJoCoのアクチュエータ制御入力更新
        # Actuator name="hip_joint", "knee_joint"
        # ctrl配列のインデックスはXMLの定義順 (hip, knee)
        target_hip_pos = joint_targets[0].item()
        target_knee_pos = joint_targets[1].item()

        # XMLのActuatorはPosition制御なので、ctrlに位置をセット
        self.data.ctrl[0] = target_hip_pos
        self.data.ctrl[1] = target_knee_pos

        #空中での脱力 (Zero Torque in Air)
        
        # 1. アクチュエータのIDを取得 (initでやっておくのがベストですが、ここで書きます)
        hip_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_joint")
        knee_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_joint")

        # 2. 定義した強力なKP値 (XMLの値と合わせる)
        high_kp = 500.0 
        
        # 3. 状態に応じてゲインを書き換える
        # 接地判定の閾値 (例: 10N以下なら空中)
        if sensor_tensor[0].item() < 100.0:
            # 【空中】 バネ係数を0にして脱力させる
            self.model.actuator_gainprm[hip_act_id, 0] = 0.0
            self.model.actuator_gainprm[knee_act_id, 0] = 0.0
            
            # ※注意: 目標値(ctrl)は何であっても無視されますが、安全のため今の角度を入れておく手もあります
            # self.data.ctrl[0] = self.data.qpos[...] 
        else:
            # 【接地】 バネ係数を戻して力を出させる
            self.model.actuator_gainprm[hip_act_id, 0] = high_kp
            self.model.actuator_gainprm[knee_act_id, 0] = high_kp

        # 4. Actuator Control (通常通りctrlに値をセット)
        self.data.ctrl[0] = joint_targets[0].item()
        self.data.ctrl[1] = joint_targets[1].item()

        # 5. 物理シミュレーション進行
        mujoco.mj_step(self.model, self.data)

        # 6. 観測値、報酬、終了判定の取得
        sensor_data_new = self._get_sensor_data()
        obs = self._get_obs(sensor_data_new)

        #報酬：高さにガウス分布で重み付け
        com_z = self.data.qpos[1]  # COMのz位置
        desired_height = 0.7 # 目標高さ
        reward = 0.0
        height_reward = 0.0
        if sensor_data_new[0] < 100: #地面反力がない(跳んでいる)場合のみ報酬を与える(跳ばざる者食うべからず！)
            height_reward += math.exp(-((com_z - desired_height) ** 2))
        #輸送コスト(CoT)の逆数
        energy_used = 0.0
        n_write = min(self.model.nu, 2)  # 股関節と膝関節の2つ
        try:
            qvel =np.array(self.data.qvel, dtype=np.float32) #速度取得
        except Exception as e:
            qvel = np.zeros(self.model.nv, dtype=np.float32) # エラー時はゼロベクトルに設定
        qvel_jump = qvel[:n_write] if qvel.shape[0] >= n_write else np.zeros(n_write, dtype=np.float32) # ジャンプに関与する速度成分
        energy_step = np.sum(np.abs(self.data.ctrl[:n_write] * qvel_jump)) * self.dt  # 簡易的なエネルギー消費モデル
        energy_used += float(energy_step)
        height_sum = np.sum(com_z * self.dt) + 1e-6 #高さの総和(0割り防止で微小値加算)
        cot = energy_used / height_sum

        reward = height_reward + (1.0 / cot) * 0.01 #報酬の合成
        
        # 終了判定 (重心の位置が高いor低い、股関節か膝関節が地面に接触)
        terminated = bool(self.data.qpos[1] < -1.0 or self.data.qpos[1] > 1.5) # 地面より下(異常) or 高すぎ
        try:
            # サイトIDを取得
            hip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "s_hip")
            knee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "s_knee")
            
            # 現在のZ座標（高さ）を取得
            hip_z = self.data.site_xpos[hip_site_id][2]
            knee_z = self.data.site_xpos[knee_site_id][2]

            # 膝または腰が地面から非常に近い場合、終了とする
            if hip_z < 0.05 or knee_z < 0.05:
                terminated = True
                
        except ValueError:
            pass # XMLのエラー等でIDが取れない場合はスルー
        truncated = False

        info = {
            "phi": self.oscillator.phi,
            "gain": gain_val,
            "reaction": react_val
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

# 環境の登録
"""
register(
    id='TH-Hopper-v0',
    entry_point='GymEnv.Tegotae_Hopper_PPO:Tegotae_Hopper_Env_PPO',
)
"""