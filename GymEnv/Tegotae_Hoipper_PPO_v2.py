import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
import os
import sys
import math

# このファイルの場所 (GymEnv/) を取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 一つ上の階層 (vertical_hopper/ ルート) を取得
root_dir = os.path.dirname(current_dir)

# Pythonがモジュールを探すパスに、ルート、CPG、NNフォルダを追加
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'CPG'))
sys.path.append(os.path.join(root_dir, 'NN'))

from CPG.Kuramoto_v2 import KuramotoCPG
from NN.ActionNet import ActionNetMLP
from NN.ReactionNet import ReactionNetMLP
from NN.GainNet import GainNetMLP

class Tegotae_Hopper_PPO_v2_Env(gym.Env):

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # 1. パスの設定
        model_path = os.path.join(root_dir, 'vertical_hopper.xml')
        csv_path = os.path.join(root_dir, 'CPG_orbit_editted.csv')
        
        # 2. MuJoCoの初期化
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        
        # 3. CPG (Kuramoto_v2) の初期化
        # dtはMuJoCoのタイムステップに合わせるか、固定値 (0.01など)
        self.dt = self.model.opt.timestep
        self.cpg = KuramotoCPG(csv_path=csv_path, dt=self.dt)
        
        # 4. ReactionNet (環境の一部として使用)
        # ※学習済み重みがある場合はここで load_state_dict する
        self.sensor_dim = self.model.nsensordata
        self.reaction_net = ReactionNetMLP(input_dim=self.sensor_dim, output_dim=1)
        
        # 5. Action Space & Observation Space
        # Action: [Action_val, Gain_val] の2つ
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation: [sin(phi), cos(phi), sensor_data...]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + self.sensor_dim,), dtype=np.float32)
        
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

    def _get_obs(self):
        # 位相ベクトル [sin, cos]
        phi = self.cpg.phi
        phase_vec = np.array([np.sin(phi), np.cos(phi)], dtype=np.float32)
        
        # センサーデータ
        sensors = self.data.sensordata.copy().astype(np.float32)
        
        return np.concatenate([phase_vec, sensors])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # MuJoCoリセット
        mujoco.mj_resetData(self.model, self.data)
        
        # CPGリセット (Kuramoto_v2にresetメソッドがない場合はphiを直接0にする)
        self.cpg.phi = 0.0
        
        # 初期姿勢に少しランダム性を加えるならここでqposを変更
        # self.data.qpos[1] += np.random.uniform(-0.1, 0.1)
        
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
    
    def step(self, action):
        # PPOからの出力: action = [Network_Action, Network_Gain]
        # 値の範囲は tanh 等で -1~1 になっているはず
        net_action = action[0]
        net_reaction = action[1]
        net_gain = action[2] 
        
        # Gainは正の値であるべきなら変換 (例: softplus や exp, あるいは絶対値)
        # ここでは単純に正の値として扱うために絶対値やオフセットを加える例
        # real_gain = np.exp(net_gain)  # 例
        real_gain = net_gain * 5.0 # そのまま使う場合は負のフィードバックもあり得る
        
        # 1. 環境情報の取得 (Reactionの計算)
        sensors = self.data.sensordata
        sensor_tensor = torch.as_tensor(sensors, dtype=torch.float32)
        
        with torch.no_grad():
            # ReactionNetでセンサー値を「反力」に変換
            reaction_val = self.reaction_net(sensor_tensor).item()
        
        # 2. Kuramoto CPG の更新
        # Kuramoto_v2.update(gain, action, reaction) -> phase, hip_tgt, knee_tgt
        # 戻り値が Tensor の可能性があるので .item() で float に変換が必要かも確認
        phase, target_hip, target_knee = self.cpg.update(
            gain=real_gain, 
            action=net_action, 
            reaction=reaction_val
        )
        
        # Tensorが返ってきた場合の安全策
        if isinstance(target_hip, torch.Tensor): target_hip = target_hip.item()
        if isinstance(target_knee, torch.Tensor): target_knee = target_knee.item()

        # 3. MuJoCoの関節目標角度に反映
        hip_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
        knee_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
        self.data.qpos[ self.model.jnt_qposadr[hip_jid] ] = target_hip
        self.data.qpos[ self.model.jnt_qposadr[knee_jid] ] = target_knee

        # 4. MuJoCoシミュレーションのステップ実行
        mujoco.mj_step(self.model, self.data)

        sensor_data_new = self._get_sensor_data()
        obs = self._get_obs(sensor_data_new)

        #報酬：高さにガウス分布で重み付け
        com_z = self.data.qpos[1]  # COMのz位置
        desired_height = 0.45 # 目標高さ
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

        reward = height_reward + (1.0 / cot) * 0.1 #報酬の合成
        
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

        CPG_info = {
            'phase': phase,
            'target_hip': target_hip,
            'target_knee': target_knee,
            'net_action': net_action,
            'net_gain': net_gain,
            'real_gain': real_gain,
            'reaction_val': reaction_val
        }

        obs_info = {
            'com_z': self.data.qpos[1],
            'grf': self.grf,
            'energy_used': energy_used,
            'cot': cot
        }

        info = {'CPG': CPG_info, 'obs': obs_info}
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