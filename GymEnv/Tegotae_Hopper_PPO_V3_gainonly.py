import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import torch
import os
import sys
import math

# このファイルの場所 (GymEnv/) を取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 一つ上の階層 (vertical_hopper/ ルート) を取得
root_dir = os.path.dirname(current_dir)

# パス設定
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'CPG'))
sys.path.append(os.path.join(root_dir, 'NN'))

from CPG.Kuramoto_v2 import KuramotoCPG
from NN.ref_traj_loader import load_ref_traj_model, get_reference_trajectory_nn, create_phase_array
# ReactionNetはPolicy側で持つためEnvでのインポートは不要ですが、パス通しのために残しています

class Tegotae_Hopper_PPO_v2_Env(gym.Env):

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, noise_std: float = 0.0, mass_scale: float = 1.0, ext_force: float = 0.0,
                 disable_gain: bool = False, disable_reaction: bool = False, 
                 disable_action: bool = False, disable_cpg_mod: bool = False):
        super().__init__()

        # --- ロバストネステスト向けパラメータ ---
        # observation noise (gaussian std-dev)
        self.noise_std = noise_std
        # リセット時に質量・慣性・摩擦に掛けるスケール
        self.mass_scale = mass_scale
        # ステップごとにランダムに与える外力の振幅
        self.ext_force = ext_force
        
        # --- アブレーション用フラグ ---
        self.disable_gain = disable_gain          # GainNet を無効化
        self.disable_reaction = disable_reaction  # ReactionNet を無効化
        self.disable_action = disable_action      # ActionNet を無効化
        self.disable_cpg_mod = disable_cpg_mod    # CPG入力全て 0

        # 1. パスの設定
        model_path = os.path.join(root_dir, 'vertical_hopper.xml')
        ref_traj_path = os.path.join(root_dir, 'trajectories_nn', 'ref_traj_nn.pt')
        
        # 2. MuJoCoの初期化
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(ref_traj_path):
            raise FileNotFoundError(f"Reference trajectory model not found: {ref_traj_path}")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        
        # 3. 参照軌道ニューラルネットワークの読み込み
        self.ref_traj_model, self.ref_traj_meta = load_ref_traj_model(ref_traj_path, device='cpu')
        self.phase_array = create_phase_array(500)  # 評価用の位相配列
        self.current_freq_hz = 1.5  # デフォルト周波数
        self._update_reference_trajectory(self.current_freq_hz)
        
        # 4. CPG (Kuramoto_v2) の初期化
        self.dt = self.model.opt.timestep
        self.cpg = KuramotoCPG(ref_traj_model=self.ref_traj_model,
                               ref_traj_meta=self.ref_traj_meta,
                               omega=self.np_random.uniform(np.pi, 2 * np.pi),
                               dt=self.dt, 
                               device='cpu')
        
        # 4. Action Space
        # Policy (Tegotae_Actor) は [Action, Reaction, Gain] の3つを出力すると想定
        # shape=(3,) : 0:Action(Input to CPG), 1:Reaction(Input to CPG), 2:Gain(K)
        # SB3 requires finite bounds for continuous action spaces, so use [-1,1]
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        
        # 5. Observation Space
        # Sensor data (5 dims) + Phase (sin, cos) (2 dims) = 7 dims
        # _get_sensor_dataの実装に合わせて次元数を固定
        self.sensor_dim = 5 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + self.sensor_dim,), dtype=np.float32)
        
        self.viewer = None
        self.grf = np.zeros(3) # 初期化
        self.z_grf = 0.0

        # --- モデルパラメータスケーリング適用 ---
        if self.mass_scale != 1.0:
            # 質量・慣性・摩擦係数にscaleを掛ける
            for i in range(self.model.nbody):
                self.model.body_mass[i] *= self.mass_scale
            # 摩擦 (geom_friction) は (sliding, torsional, rolling)
            self.model.geom_friction *= self.mass_scale

    def _get_sensor_data(self):
        """
        MuJoCoの物理状態からニューラルネットに入力するためのセンサーベクトルを作成
        Returns:
            Tensor shape (5,): [z_grf, hip_angle, knee_angle, hip_vel, knee_vel]
        """
        # 地面反力の取得
        # --- 1. 地面反力の取得 (Joint_model_Envのロジックを移植) ---
        z_grf = 0.0
        
        # 【重要】XMLファイル内の足のパーツ名（geom名）に合わせて修正してください
        # 例: "foot_geom", "foot", "toe_geom" など
        # もし足がかかと・つま先に分かれているなら、そのリストを作ってください
        target_foot_geoms = ["foot_sphere", "footsphere", "hip_geom", "knee_geom", "thigh_geom", "shank_geom"]
        
        # geom名からIDを取得しておく
        foot_geom_ids = []
        for name in target_foot_geoms:
            try:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    foot_geom_ids.append(gid)
            except ValueError:
                pass # XMLにその名前がない場合はスキップ

        # 全接触点をループして反力を合算
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # 接触しているどちらかの物体が「足」か確認
            # (joint_model.pyではgeom2を見ていますが、念のため両方チェックします)
            if (contact.geom1 in foot_geom_ids) or (contact.geom2 in foot_geom_ids):
                
                # 接触力を計算するための配列確保
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                
                # c_array[0] が接触法線方向（地面なら垂直方向）の力
                z_grf += c_array[0]
                self.z_grf = z_grf  # デバッグ用に保存

        # ジョイント角度と速度の取得
        try:
            hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
            knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")

            hip_qpos_adr = self.model.jnt_qposadr[hip_id]
            knee_qpos_adr = self.model.jnt_qposadr[knee_id]
            
            hip_angle = self.data.qpos[hip_qpos_adr]
            knee_angle = self.data.qpos[knee_qpos_adr]
            
            hip_dof_adr = self.model.jnt_dofadr[hip_id]
            knee_dof_adr = self.model.jnt_dofadr[knee_id]
            
            hip_vel = self.data.qvel[hip_dof_adr]
            knee_vel = self.data.qvel[knee_dof_adr]
        except ValueError:
            # XML等の不整合
            hip_angle, knee_angle, hip_vel, knee_vel = 0, 0, 0, 0

        # 正規化等は必要に応じてここで行う（現在は生データ、NumPy で保持）
        sensor_data = np.array([z_grf, hip_angle, knee_angle, hip_vel, knee_vel], dtype=np.float32)
        # ノイズ付与
        if self.noise_std > 0.0:
            sensor_data += np.random.randn(5) * self.noise_std
        return sensor_data

    def _update_reference_trajectory(self, freq_hz):
        """参照軌道を周波数に応じて更新"""
        # 周波数を参照軌道モデルのサポート範囲内に制限
        f_min = self.ref_traj_meta['f_min']
        f_max = self.ref_traj_meta['f_max']
        freq_hz = np.clip(float(freq_hz), f_min, f_max)
        
        # ニューラルネットワークで参照軌道を生成
        hip_angles, knee_angles = get_reference_trajectory_nn(
            self.phase_array, freq_hz, self.ref_traj_model, self.ref_traj_meta, device='cpu'
        )
        
        # 内挿用の参照軌道を更新
        self._ref_phases = self.phase_array.astype(np.float64)
        self._ref_hip_angles = hip_angles.astype(np.float64)
        self._ref_knee_angles = knee_angles.astype(np.float64)

    def _get_obs(self, sensor_data_tensor=None):
        phi = self.cpg.phi
        if isinstance(phi, torch.Tensor):
            phi = float(phi.detach().flatten()[0])
        else:
            phi = float(phi)
        phase_vec = np.array([np.sin(phi), np.cos(phi)], dtype=np.float32)
        if sensor_data_tensor is None:
            sensor_data_tensor = self._get_sensor_data()
        if isinstance(sensor_data_tensor, torch.Tensor):
            sensor_data_tensor = sensor_data_tensor.detach().numpy().flatten()
        else:
            sensor_data_tensor = np.asarray(sensor_data_tensor, dtype=np.float32).flatten()
        obs = np.concatenate([phase_vec, sensor_data_tensor])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # MuJoCoリセット
        mujoco.mj_resetData(self.model, self.data)
        
        # CPGリセット
        self.cpg.reset(phi0=0.0)
        
        
        # 初期姿勢にランダム性を加える
        if seed is not None:
             np.random.seed(seed)
        #self.data.qpos[1] = 1.0 + np.random.uniform(-0.5, 0.5) # 高さの初期値を少しランダムに
        init_height = None
        if options is not None and 'init_height' in options:
            init_height = float(options['init_height'])
        elif getattr(self, 'init_height_override', None) is not None:
            init_height = float(self.init_height_override)

        if init_height is not None:
            # 固定値で初期化（ランダム無し）
            self.data.qpos[1] = init_height
        else:
            # これまで通りランダム
            self.data.qpos[1] = np.random.uniform(0.0, 2.0)
        mujoco.mj_forward(self.model, self.data)

        self.cpg.omega = self.np_random.uniform(np.pi, 2 * np.pi)  # 周波数をランダムに変更
        # 参照軌道を周波数に応じて更新
        self.current_freq_hz = self.cpg.omega / (2 * np.pi)
        self.cpg.update_frequency(self.cpg.omega)  # KuramotoCPGの周波数も更新
        self._update_reference_trajectory(self.current_freq_hz)
        # リセット時に質量スケールが指定されていれば反映
        if options is not None and 'mass_scale' in options:
            self.mass_scale = float(options['mass_scale'])
            for i in range(self.model.nbody):
                self.model.body_mass[i] *= self.mass_scale
            self.model.geom_friction *= self.mass_scale
        if options is not None and 'noise_std' in options:
            self.noise_std = float(options['noise_std'])
        if options is not None and 'ext_force' in options:
            self.ext_force = float(options['ext_force'])

        self.accumulated_jump_reward = 0.0
        # リワード周期(ジャンプサイクル)用変数初期化
        self.cycle_energy = 0.0
        self.cycle_max_height = self.data.qpos[1]
        # 最初は浮いているか接地しているか判定してセット
        sensor_data = self._get_sensor_data()
        self.prev_contact = (sensor_data[0] > 10.0)
        self.first_contact_done = False

        return self._get_obs(), {}
    
    def step(self, action):
        """
        args:
            action: np.array shape(3,) -> [Net_Action, Net_Reaction, Net_Gain]
        """
        # 1. Actionの解釈
        # PPOの出力は通常 -1 ~ 1 (tanh) なので、適切な範囲にスケーリングする
        net_action = -np.cos(self.cpg.phi)   # CPGへの入力 Action A
        net_reaction = self.z_grf/1000 # CPGへの入力 Reaction R (手応え)
        net_gain = action[0]     # フィードバックゲイン K
        
        # --- アブレーション処理 ---
        if self.disable_cpg_mod:
            # CPG制御を全て無効化
            net_action = 0.0
            net_reaction = 0.0
            net_gain = 1.0  # ゲインは最小限デフォルト
        else:
            # 個別無効化
            if self.disable_action:
                net_action = -np.cos(self.cpg.phi)  # CPGの位相に基づく単純なオシレーションに置き換え
            if self.disable_reaction:
                net_reaction = self.z_grf  # 環境側で計算した地面反力をそのまま渡す（学習させない）
            if self.disable_gain:
                net_gain = 1.0  # ゲイン = 1.0 (デフォルト)
        
        # ActionとReactionはそのまま使う（符号を含めて学習させる）
        # ただし、PolicyがReactionを出力しない設計の場合はここで環境側で計算する必要があるが、
        # 今回はPolicyに含まれている前提で進める。

        # 2. Kuramoto CPG の更新
        # step(gain, action, reaction) -> phase, hip_tgt, knee_tgt (Tensors)
        phase, target_hip, target_knee = self.cpg.step(
            gain=net_gain, 
            action=net_action, 
            reaction=net_reaction
        )
        
        # Tensor -> Float conversion
        def to_scalar(x):
            if isinstance(x, torch.Tensor): return float(x.detach().flatten()[0])
            return float(x)
        target_hip  = to_scalar(target_hip)
        target_knee = to_scalar(target_knee)
        phase       = to_scalar(phase)

        # 3. MuJoCoのアクチュエータ制御
        # qpos（位置）を直接書き換えると物理演算（反力）が正しく計算されないため、
        # ctrl（アクチュエータ指令値）を使用する。
        # XMLで position アクチュエータが設定されている前提。
        try:
             # アクチュエータIDの取得（名前はXMLに依存、ここでは順序0,1と仮定するか名前検索）
             # 安全のため名前で検索
             hip_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_motor")
             knee_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_motor")
             
             # もし名前が見つからない場合はインデックス0, 1を使用
             if hip_act_id == -1: hip_act_id = 0
             if knee_act_id == -1: knee_act_id = 1
             
             self.data.ctrl[hip_act_id] = target_hip
             self.data.ctrl[knee_act_id] = target_knee
        except Exception:
             # フォールバック：アクチュエータ未定義の場合はqpos書き換え（物理挙動は不正確になる）
             # ただし元のコードの挙動を再現するため残す
             hip_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
             knee_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
             self.data.qpos[self.model.jnt_qposadr[hip_jid]] = target_hip
             self.data.qpos[self.model.jnt_qposadr[knee_jid]] = target_knee

        # 外力パルス
        if self.ext_force > 0.0:
            # 確率0.01で外力を body0 (全身) に加える
            if np.random.rand() < 0.01:
                force = np.array([0.0, self.ext_force * (np.random.rand()*2-1), 0.0])
                # muJoco 2.4+ では mj_applyFT に torque, point, body, qfrc_target が必要
                torque = np.zeros((3,1), dtype=np.float64)
                point  = np.zeros((3,1), dtype=np.float64)
                body   = 0
                qfrc_target = np.zeros(self.model.nv, dtype=np.float64)
                mujoco.mj_applyFT(self.model, self.data,
                                  force.reshape(3,1),
                                  torque,
                                  point,
                                  body,
                                  qfrc_target)

        # 4. MuJoCoシミュレーションのステップ実行
        # 制御周波数とシミュレーション周波数の調整が必要な場合はここでループする (frame_skip)
        mujoco.mj_step(self.model, self.data)

        # 5. 観測と報酬の計算
        sensor_data_new = self._get_sensor_data()
        obs = self._get_obs(sensor_data_new)

        # 報酬計算
        com_z = self.data.qpos[1]  # COMのz位置 (vertical slider)
        # --- エネルギー消費とサイクル高さの記録 ---
        hip_id_j   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
        knee_id_j  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
        hip_dof_p  = self.model.jnt_dofadr[hip_id_j]
        knee_dof_p = self.model.jnt_dofadr[knee_id_j]
        power = (abs(self.data.actuator_force[hip_act_id]  * self.data.qvel[hip_dof_p]) +
                 abs(self.data.actuator_force[knee_act_id] * self.data.qvel[knee_dof_p]))
        self.cycle_energy += power * self.dt
        if com_z > self.cycle_max_height:                
            self.cycle_max_height = com_z

        desired_height = 0.45      # 目標高さ（論文で述べる）

        
        # 小さな生存ボーナスで安定ジャンプを奨励
        step_reward = 0.01 #* abs(self.data.qvel[1])  # 上昇速度に比例した報酬（高いほど良い）
        """
        is_airborne = (sensor_data_new[0] < 10.0)
        if is_airborne:
            step_reward += 0.1 * max(0.0, com_z - desired_height)  # 高いほど報酬
        else:
            step_reward -= 0.005  # 接地中はペナルティ（立ち続けを防ぐ）
        #base_height = 1.0  #直立よりちょい高
        #step_reward += 0.01 * max(0.0, com_z - base_height)
        """
        # エネルギーペナルティ
        step_reward -= 0.000005 * power
        # 接地時の大きな衝撃を避けるため GRF に対するペナルティ
        contact_penalty = max(0.0, sensor_data_new[0] - 2000.0) * 1e-6
        step_reward -= contact_penalty

        if self.first_contact_done:
            self.accumulated_jump_reward += step_reward

        current_contact = (sensor_data_new[0] > 10.0)
        just_landed = current_contact and not self.prev_contact
        if current_contact and not self.first_contact_done:
            self.first_contact_done = True  # 初回接地 → 以降は報酬を有効化
            self.accumulated_jump_reward = 0.0  # 落下中の蓄積をリセット
            self.cycle_energy = 0.0
            self.cycle_max_height = float(self.data.qpos[1])


        knee_touch = float(self.data.sensordata[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "knee_touch_sensor")])
        hip_touch  = float(self.data.sensordata[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "hip_touch_sensor")])
        # 終了判定
        terminated = False
        if com_z < -0.9:  # 低すぎる（転倒）
            terminated = True
        if com_z > 2.0:  # 高すぎる（爆発）
            terminated = True
        if knee_touch > 0.0 or hip_touch > 0.0:  # 膝または股関節が接地
            terminated = True
        truncated = False

        height_reward = None
        efficiency_score = None
        jump_reward = None

        reward_to_return = 0.0

        if terminated:
            # ペナルティを-10に緩和（元の-1000から改善）
            reward_to_return = -10.0
        elif just_landed:
            if self.cycle_max_height > 0.1:
                height_reward = math.exp(
                    -((self.cycle_max_height - desired_height) ** 2) * 10)
                if self.cycle_energy > 0.1:  # ゼロ除算ガード
                    efficiency_score = np.clip(
                        75.5 * 9.81 * self.cycle_max_height / self.cycle_energy,
                        0.0, 10.0)
                else:
                    efficiency_score = 0.0
                jump_reward = height_reward #+ 0.5 * efficiency_score
            else:
                jump_reward = 0.0
        
            # クリップした値を使う
            accumulated_clamped = np.clip(self.accumulated_jump_reward, -10.0, 10.0)
            reward_to_return = accumulated_clamped + jump_reward
        
            # 必ずリセット
            self.accumulated_jump_reward = 0.0  # ← コメントアウトを外す
            self.cycle_energy = 0.0
            self.cycle_max_height = com_z
        
        else:
            reward_to_return = step_reward
        
        # Info
        info = {
            'phase':            float(phase) if phase is not None else 0.0,
    'phase_velocity':   float(np.clip(self.cpg.phi_dot, -2*np.pi, 2*np.pi)),
    'target_hip':       float(target_hip),
    'target_knee':      float(target_knee),
    'grf':              float(sensor_data_new[0]),
    'gain':             float(net_gain.mean()) if hasattr(net_gain, 'mean') else float(net_gain),
    'action':           float(net_action.mean()) if hasattr(net_action, 'mean') else float(net_action),
    'reaction':         float(net_reaction.mean()) if hasattr(net_reaction, 'mean') else float(net_reaction),
    'height':           float(com_z),
    'power':            float(power),
    'cycle_energy':     float(getattr(self, 'cycle_energy', 0.0)),
    'cycle_max_height': float(getattr(self, 'cycle_max_height', com_z)),
    'step_reward':      float(step_reward),
    'accumulated_reward': self.accumulated_jump_reward.item() if hasattr(self.accumulated_jump_reward, 'item') else float(self.accumulated_jump_reward),
    'contact_penalty':  float(contact_penalty) if 'contact_penalty' in locals() else 0.0,
    'height_reward':    float(height_reward) if 'height_reward' in locals() and height_reward is not None else None,
    'efficiency_score': float(efficiency_score) if 'efficiency_score' in locals() and efficiency_score is not None else None,
    'jump_reward':      float(jump_reward) if 'jump_reward' in locals() and jump_reward is not None else None,
    # === robustness 評価用メトリクス ===
    'com_z':            float(com_z),
    'peak_grf':         float(self._cycle_grf_peak) if hasattr(self, '_cycle_grf_peak') else float(sensor_data_new[0]),
    'cycle_z_max_list': list(self._cycle_z_max_list) if hasattr(self, '_cycle_z_max_list') else [],
    'pose_error':       float(pose_error) if 'pose_error' in locals() else 0.0,
    'just_landed':      bool(just_landed) if 'just_landed' in locals() else False,
    'terminated':       bool(terminated),
        }

        # prev_contactを更新（重要：次のステップで正しくjust_landedを判定するため）
        self.prev_contact = current_contact
        #print(f"Step Reward: {reward_to_return:.4f}, Height: {com_z:.3f}, GRF: {sensor_data_new[0]:.1f}, Power: {power:.2f}")
        #if info['height_reward'] is not None and info['efficiency_score'] is not None and info['jump_reward'] is not None:
        #    print(f"  (Height Reward: {info['height_reward']:.4f}, Efficiency: {info['efficiency_score']:.4f}, Jump Reward: {info['jump_reward']:.4f})")

        return obs, reward_to_return, terminated, truncated, info
    
    # =========================================================================
    # robustness評価用メソッド（DirectTorque_Hopper と同一インターフェース）
    # =========================================================================

    def _reset_robustness_accumulators(self):
        """robustness評価用サイクル積算値を初期化（エピソード開始時に呼ぶ）。"""
        self.cycle_energy            = 0.0
        self.cycle_max_height        = float(self.data.qpos[1])
        self.accumulated_jump_reward = 0.0
        self.first_contact_done      = False
        self._cycle_z_max_list       = []    # CV_h 計算用：サイクルごとの最大高さ
        self._cycle_grf_peak         = 0.0   # 現サイクルのGRF最大値（Peak GRF用）
        sensor = self._get_sensor_data()
        self.prev_contact = (float(sensor[0]) > 10.0)

    def evaluate_robustness(self) -> dict:
        """robustnessテスト用の評価指標を返す（step() の直後に呼び出す）。

        DirectTorque_Hopper.evaluate_robustness() と同一のインターフェース・
        報酬スキームで指標を計算し、baseline間の比較を可能にする。

        Returns
        -------
        dict with keys:
            reward, step_reward, height_reward, efficiency_score, jump_reward,
            terminated, com_z, grf, power, is_airborne, just_landed,
            cycle_z_max, cycle_energy, height_error, pose_error, accumulated_reward
        """
        hip_act_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_motor")
        knee_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_motor")
        if hip_act_id  == -1: hip_act_id  = 0
        if knee_act_id == -1: knee_act_id = 1

        hip_id_j   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
        knee_id_j  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
        hip_dof_p  = self.model.jnt_dofadr[hip_id_j]
        knee_dof_p = self.model.jnt_dofadr[knee_id_j]

        # ---- 現在の状態 ----
        com_z      = float(self.data.qpos[1])
        hip_angle  = float(self.data.qpos[self.model.jnt_qposadr[hip_id_j]])
        knee_angle = float(self.data.qpos[self.model.jnt_qposadr[knee_id_j]])

        # 瞬時電力
        power = (abs(float(self.data.actuator_force[hip_act_id]))  * abs(float(self.data.qvel[hip_dof_p])) +
                 abs(float(self.data.actuator_force[knee_act_id])) * abs(float(self.data.qvel[knee_dof_p])))

        # GRF
        sensor = self._get_sensor_data()
        grf    = float(sensor[0])

        # ---- サイクル積算 ----
        self.cycle_energy += power * self.dt
        if com_z > self.cycle_max_height:
            self.cycle_max_height = com_z
        # GRF peak（着地衝撃の最大値をサイクル内で追跡）
        if not hasattr(self, '_cycle_grf_peak'):
            self._cycle_grf_peak = 0.0
        if grf > self._cycle_grf_peak:
            self._cycle_grf_peak = grf
        if not hasattr(self, '_cycle_z_max_list'):
            self._cycle_z_max_list = []

        # ---- step_reward（step()と同一ロジック） ----
        step_reward      = 0.01
        step_reward     -= 0.000005 * power
        contact_penalty  = max(0.0, grf - 2000.0) * 1e-6
        step_reward     -= contact_penalty

        # ---- 接触判定・着地検出 ----
        current_contact = grf > 10.0
        just_landed     = current_contact and not self.prev_contact

        # 初回接地でサイクル開始
        if current_contact and not self.first_contact_done:
            self.first_contact_done      = True
            self.accumulated_jump_reward = 0.0
            self.cycle_energy            = 0.0
            self.cycle_max_height        = com_z

        if self.first_contact_done:
            self.accumulated_jump_reward += step_reward

        # ---- 終了判定 ----
        knee_touch = float(self.data.sensordata[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "knee_touch_sensor")])
        hip_touch  = float(self.data.sensordata[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "hip_touch_sensor")])
        terminated = False
        if com_z < -0.9:                         terminated = True
        if com_z >  2.0:                         terminated = True
        if knee_touch > 0.0 or hip_touch > 0.0: terminated = True

        # ---- 報酬スキーム（step()と同一） ----
        desired_height   = 0.45
        height_reward    = None
        efficiency_score = None
        jump_reward      = None
        reward           = 0.0

        if terminated:
            reward = -10.0
        elif just_landed:
            if self.cycle_max_height > 0.1:
                height_reward = math.exp(
                    -((self.cycle_max_height - desired_height) ** 2) * 10)
                if self.cycle_energy > 0.1:
                    efficiency_score = float(np.clip(
                        75.5 * 9.81 * self.cycle_max_height / self.cycle_energy,
                        0.0, 10.0))
                else:
                    efficiency_score = 0.0
                jump_reward = height_reward  # + 0.5 * efficiency_score（step()に合わせる）
            else:
                height_reward    = 0.0
                efficiency_score = 0.0
                jump_reward      = 0.0

            accumulated_clamped = float(np.clip(self.accumulated_jump_reward, -10.0, 10.0))
            reward = accumulated_clamped + jump_reward

            # サイクルリセット
            self._cycle_z_max_list.append(float(self.cycle_max_height))  # CV_h用に記録
            self._cycle_grf_peak         = 0.0                           # 次サイクルへリセット
            self.accumulated_jump_reward = 0.0
            self.cycle_energy            = 0.0
            self.cycle_max_height        = com_z
        else:
            reward = step_reward

        # prev_contact更新（step()と共有するため末尾で）
        self.prev_contact = current_contact

        # ---- CPG位相に基づく姿勢誤差（__init__キャッシュ参照） ----
        phi = self.cpg.phi
        if isinstance(phi, torch.Tensor):
            phi = float(phi.detach().flatten()[0])
        else:
            phi = float(phi)
        ref_hip  = float(np.interp(phi % (2 * np.pi),
                                   self._ref_phases, self._ref_hip_angles))
        ref_knee = float(np.interp(phi % (2 * np.pi),
                                   self._ref_phases, self._ref_knee_angles))
        pose_error = (hip_angle - ref_hip) ** 2 + (knee_angle - ref_knee) ** 2

        return {
            'reward':             float(reward),
            'step_reward':        float(step_reward),
            'height_reward':      float(height_reward)    if height_reward    is not None else None,
            'efficiency_score':   float(efficiency_score) if efficiency_score is not None else None,
            'jump_reward':        float(jump_reward)      if jump_reward      is not None else None,
            'terminated':         terminated,
            'com_z':              com_z,
            'grf':                grf,
            'peak_grf':           float(self._cycle_grf_peak),   # 現サイクルのGRF最大値
            'cycle_z_max_list':   list(self._cycle_z_max_list),  # 全サイクルの最大高さ一覧
            'power':              power,
            'is_airborne':        not current_contact,
            'just_landed':        just_landed,
            'cycle_z_max':        float(self.cycle_max_height),
            'cycle_energy':       float(self.cycle_energy),
            'height_error':       abs(com_z - desired_height),
            'pose_error':         float(pose_error),
            'accumulated_reward': float(self.accumulated_jump_reward),
        }

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