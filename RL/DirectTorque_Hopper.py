import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import torch
import os, sys, math

# add paths for potential imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'NN'))

from NN.ref_traj_loader import load_ref_traj_model, get_reference_trajectory_nn, create_phase_array

class DirectTorque_Hopper_Env(gym.Env):
    """Simple 2-DOF vertical hopper where actions are joint torques.

    This environment is useful as a baseline for comparing
    against the CPG + Tegotae architecture. It exposes the
    same observation space (phase + sensors) but the policy
    outputs torques directly.
    """
    metadata = {'render_modes': ['human','rgb_array'], 'render_fps':60}

    def __init__(self, render_mode=None, noise_std: float = 0.0,
                 mass_scale: float = 1.0, ext_force: float = 0.0,
                 disable_gain: bool = False, disable_reaction: bool = False,
                 disable_action: bool = False, disable_cpg_mod: bool = False):
        super().__init__()
        model_path = os.path.join(root_dir, 'vertical_hopper_torque.xml')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.dt = self.model.opt.timestep
        self.steps_since_takeoff = 0
        self.max_grounded_steps = int(2.0 / self.dt)  # 2秒分のステップ数
        # robustness parameters
        self.noise_std = noise_std
        self.mass_scale = mass_scale
        self.ext_force = ext_force
        if self.mass_scale != 1.0:
            for i in range(self.model.nbody):
                self.model.body_mass[i] *= self.mass_scale
            self.model.geom_friction *= self.mass_scale
        
        # ablation flags (unused for DirectTorque, but maintained for compatibility)
        self.disable_gain = disable_gain
        self.disable_reaction = disable_reaction
        self.disable_action = disable_action
        self.disable_cpg_mod = disable_cpg_mod

        # Load reference trajectory
        import pandas as pd
        ref_traj_path = os.path.join(root_dir, 'trajectories_nn', 'ref_traj_nn.pt')
        
        if not os.path.exists(ref_traj_path):
            raise FileNotFoundError(f"Reference trajectory model not found: {ref_traj_path}")
        
        # ニューラルネットワークモデルを読み込み
        self.ref_traj_model, self.ref_traj_meta = load_ref_traj_model(ref_traj_path, device='cpu')
        self.phase_array = create_phase_array(500)  # 評価用の位相配列
        
        # デフォルト周波数で参照軌道を生成
        self.current_freq_hz = 1.5
        self._update_reference_trajectory(self.current_freq_hz)

        # Calculate phase_dot to match trajectory period
        len_trajectory = len(self.phase_array)
        T = len_trajectory * self.dt  # trajectory duration
        self.phase_dot = 2 * np.pi / T  # phase velocity
        self.desired_height = 0.45  # h*
        self.sigma_h = 0.5  # for r_hop Gaussian (wider)
        self.alpha = 1e-5  # for c_torque (very low penalty to allow large torques)
        self.beta = 0.5  # for c_contact (low penalty)

        # Reward function parameters
        self.xo = 0.45  # initial standing height
        self.xd = 0.45  # desired height
        self.w1 = 0.5   # weight for energy gain
        self.w2 = 2.0   # weight for height barrier
        self.w3 = 0.01  # weight for jerky action (reduced to allow more aggressive torque changes)
        self.w4 = 0.02  # weight for joint position penalty
        self.w5 = 0.005 # weight for joint velocity penalty
        
        # Joint limits
        self.ql_hip = -np.pi / 2   # lower limit hip
        self.qh_hip = np.pi / 2    # upper limit hip
        self.ql_knee = - 3 *np.pi / 4  # lower limit knee
        self.qh_knee = 0   # upper limit knee
        self.q_dot_h = 10.0    # max joint velocity

        # action: torque for hip and knee (larger range for strength)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)
        # observation: phase plus same 5 sensor dims
        self.sensor_dim = 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + self.sensor_dim,), dtype=np.float32)
        self.viewer = None
        self.grf = np.zeros(3)
        # simple phase variable not used for control but kept for comparability
        self.phase = 0.0 

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
        
        # 参照軌道を更新（evaluate_robustness() で使用）
        self.ref_phases = self.phase_array.astype(np.float64)
        self.ref_hip_angles = hip_angles.astype(np.float64)
        self.ref_knee_angles = knee_angles.astype(np.float64)

    def _get_sensor_data(self):
        # copy logic from other envs
        z_grf = 0.0
        target_foot_geoms = ["footsphere", "hip_geom", "knee_geom", "thigh_geom", "shank_geom"]
        foot_geom_ids = []
        for name in target_foot_geoms:
            try:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    foot_geom_ids.append(gid)
            except ValueError:
                pass
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 in foot_geom_ids) or (contact.geom2 in foot_geom_ids):
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                z_grf += c_array[0]
        try:
            hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
            knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
            hip_qpos_adr = self.model.jnt_qposadr[hip_id]
            knee_qpos_adr = self.model.jnt_qposadr[knee_id]
            hip_angle = self.data.qpos[hip_qpos_adr]
            knee_angle = self.data.qpos[knee_qpos_adr]
            hip_vel = self.data.qvel[self.model.jnt_dofadr[hip_id]]
            knee_vel = self.data.qvel[self.model.jnt_dofadr[knee_id]]
        except Exception:
            hip_angle=knee_angle=hip_vel=knee_vel=0.0
        sensor_data = torch.tensor([z_grf, hip_angle, knee_angle, hip_vel, knee_vel], dtype=torch.float32)
        if self.noise_std>0:
            sensor_data += torch.randn_like(sensor_data)*self.noise_std
        return sensor_data

    def _get_obs(self, sensor=None):
        phase_vec = np.array([np.sin(self.phase), np.cos(self.phase)], dtype=np.float32)
        if sensor is None:
            sensor = self._get_sensor_data()
        return np.concatenate([phase_vec, sensor.numpy()])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model,self.data)
        self.phase=0.0
        self.cycle_energy = 0.0
        self.accumulated_jump_reward = 0.0
        self.cycle_max_height = self.data.qpos[1]
        if options is not None and 'mass_scale' in options:
            self.mass_scale=float(options['mass_scale'])
            for i in range(self.model.nbody):
                self.model.body_mass[i]*=self.mass_scale
            self.model.geom_friction*=self.mass_scale
        if options is not None and 'noise_std' in options:
            self.noise_std=float(options['noise_std'])
        if options is not None and 'ext_force' in options:
            self.ext_force=float(options['ext_force'])
        self.data.qpos[1]=np.random.uniform(-0.5,0.5)
        # 追加: phase=0 の参照姿勢で初期化
        ref_hip0 = np.interp(0.0, self.ref_phases, self.ref_hip_angles)
        ref_knee0 = np.interp(0.0, self.ref_phases, self.ref_knee_angles)
        hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
        knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
        self.data.qpos[self.model.jnt_qposadr[hip_id]] = ref_hip0
        self.data.qpos[self.model.jnt_qposadr[knee_id]] = ref_knee0
        #print(f"reset後のqpos: {self.get_attr('data')[0].qpos}")
        mujoco.mj_forward(self.model,self.data)
        sensor_data = self._get_sensor_data()
        self.prev_contact = (sensor_data[0] > 10.0)
        self.first_contact_done = False
        self.prev_action = np.zeros(2)  # initialize previous action
        return self._get_obs(),{}

    def step(self, action):
        # Update phase
        self.phase += self.phase_dot * self.dt
        self.phase %= 2 * np.pi

        hip_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_motor")
        knee_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_motor")
        # apply torques directly
        torque = np.clip(action * 10, self.action_space.low * 10, self.action_space.high * 10)  # Scale up by 10x
        # ctrl arrays expects actuation; assuming torque actuators in XML
        self.data.ctrl[0]=torque[0]  # knee torque
        self.data.ctrl[1]=torque[1]  # hip torque
        # random external disturbance
        if self.ext_force>0 and np.random.rand()<0.01:
            ext_force_vec = np.array([0.0, self.ext_force*(np.random.rand()*2-1), 0.0])
            ext_torque = np.zeros((3,1), dtype=np.float64)
            point  = np.zeros((3,1), dtype=np.float64)
            body   = 0
            qfrc_target = np.zeros(self.model.nv, dtype=np.float64)
            mujoco.mj_applyFT(self.model,self.data,
                              ext_force_vec.reshape(3,1),
                              ext_torque,
                              point,
                              body,
                              qfrc_target)
        mujoco.mj_step(self.model,self.data)
        sensor_data_new = self._get_sensor_data()  # 1回だけ呼ぶ
        obs = self._get_obs(sensor_data_new)
        com_z = self.data.qpos[1]

        # Update cycle tracking
        hip_dof_p = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")]
        knee_dof_p = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")]
        power = (abs(self.data.actuator_force[hip_act_id]) * abs(self.data.qvel[hip_dof_p]) +
                 abs(self.data.actuator_force[knee_act_id]) * abs(self.data.qvel[knee_dof_p]))
        self.cycle_energy += power * self.dt
        if com_z > self.cycle_max_height:
            self.cycle_max_height = com_z

        # Get joint angles and velocities
        hip_angle = float(sensor_data_new[1])
        knee_angle = float(sensor_data_new[2])
        hip_vel = self.data.qvel[self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")]]
        knee_vel = self.data.qvel[self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")]]
        base_vel = self.data.qvel[0]  # base vertical velocity

        # Energy Gain (Ge)
        Ge = base_vel**2 + (com_z - self.xo)**2

        # Height Barrier Penalty (Ph)
        Ph = 1 - np.exp(-(com_z - self.xd)) if com_z >= self.xd else 0.0

        # Jerky Action Penalty (Pj)
        Pj = np.sum((action - self.prev_action)**2)

        # Joint Position Penalty (Pjp)
        Pjp = 0.0
        for q, ql, qh in [(hip_angle, self.ql_hip, self.qh_hip), (knee_angle, self.ql_knee, self.qh_knee)]:
            if q < ql or q > qh:
                Pjp += np.exp(-10 * (q - ql)) + np.exp(10 * (q - qh))
            else:
                Pjp += 0.0

        # Get reference pose from trajectory
        ref_hip = np.interp(self.phase, self.ref_phases, self.ref_hip_angles)
        ref_knee = np.interp(self.phase, self.ref_phases, self.ref_knee_angles)

        # Pose tracking reward (encourage following reference trajectory)
        pose_error = (hip_angle - ref_hip)**2 + (knee_angle - ref_knee)**2
        r_pose = np.exp(-10.0 * pose_error)  # Gaussian reward for pose matching

        # Joint Velocity Penalty (Pjv)
        Pjv = 0.0
        for q_dot in [hip_vel, knee_vel]:
            if abs(q_dot) > self.q_dot_h:
                Pjv += q_dot**2 - self.q_dot_h**2
            else:
                Pjv += 0.0

        # Total step reward (includes pose tracking)
        w_pose = 2.0  # weight for pose tracking
        step_reward = self.w1 * Ge - self.w2 * Ph - self.w3 * Pj - self.w4 * Pjp - self.w5 * Pjv + w_pose * r_pose

        # Update previous action
        self.prev_action = action.copy()

        # Accumulate step rewards during jump
        #if self.first_contact_done:
        #    self.accumulated_jump_reward += step_reward
        reward = step_reward

        knee_touch = float(self.data.sensordata[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "knee_touch_sensor")])
        hip_touch  = float(self.data.sensordata[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "hip_touch_sensor")])
        current_contact = (float(sensor_data_new[0]) > 10.0)
        just_landed = current_contact and not self.prev_contact

        if current_contact and not self.first_contact_done:
            self.first_contact_done = True  # 初回接地 → 以降は報酬を有効化
            self.accumulated_jump_reward = 0.0  # 落下中の蓄積をリセット
            self.cycle_energy = 0.0
            self.cycle_max_height = float(com_z)

        # Termination conditions
        terminated = False
        if com_z < -0.9:  # fallen
            terminated = True
        if com_z > 2.0:  # exploded
            terminated = True
        if knee_touch > 0.0 or hip_touch > 0.0:  # knee or hip touching ground
            terminated = True

        truncated = False

        height_reward = 0.0
        #reward = 0.0

        if terminated:
            reward += -10.0
        elif just_landed:
            if self.cycle_max_height > 0.1:
                height_reward += math.exp(-((self.cycle_max_height - self.desired_height)**2) / (self.sigma_h**2))
            else:
                height_reward = 0.0
            reward += height_reward
            # Reset for next cycle
            self.accumulated_jump_reward = 0.0
            self.cycle_energy = 0.0
            self.cycle_max_height = com_z
        else:
            reward = step_reward

        grf_val = float(sensor_data_new[0])
        info={'height':     float(com_z),
              'step_reward': float(step_reward),
              'Ge':         float(Ge),
              'Ph':         float(Ph),
              'Pj':         float(Pj),
              'Pjp':        float(Pjp),
              'Pjv':        float(Pjv),
              'r_pose':     float(r_pose),
              'pose_error': float(pose_error),
              'height_reward': float(height_reward) if height_reward is not None else None,
              'grf':        grf_val,
              'action':     float(np.asarray(action).flatten()[0]),
              'phase':      float(self.phase),
              'ref_hip':    float(ref_hip),
              'ref_knee':   float(ref_knee),
              }
        self.prev_contact = current_contact
        return obs, reward, terminated, truncated, info
    


    def render(self):
        if self.render_mode=="human":
            if self.viewer is None:
                self.viewer=mujoco.viewer.launch_passive(self.model,self.data)
            if self.viewer.is_running():
                self.viewer.sync()
        elif self.render_mode=="rgb_array":
            w,h=640,480
            return mujoco.mj_render(self.model,self.data,w,h)
    def close(self):
        if self.viewer is not None:
            self.viewer.close(); self.viewer=None