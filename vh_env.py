import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
from gymnasium.envs.registration import register

class vh_env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    """
    ＜作りたい環境＞
    ・エージェント(vertical_hopoper)がmujocoの3次元空間シミュレーション上で垂直方向にジャンプする環境
    ・観測対象ー＞ (垂直方向への移動速度)、地面からの反力、CPGの内部状態
    ・行動ー＞ CPG(kuramoto phase oscillator)の内部状態の変化
    ・報酬ー＞ 輸送コスト(CoT)の逆数(頂点に達した瞬間に計算)+CPGの同期度合いの向上に基づく報酬
    """
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        #MuJoCoモデルの読み込み
        xml_path = os.path.join(os.path.dirname(__file__), "vertical_hopper.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XMLファイルが{xml_path}に無いよー")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        #観測空間の定義[z軸速度、地面反力、CPG状態]
        obs_low  = np.array([-10.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([10.0, 1000.0, 2*np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        #行動空間の定義[CPG状態変化量]
        self.action_space = spaces.Box(low=0.0, high=2*np.pi, shape=(2,), dtype=np.float32)

        #CPG
        self.num_oscillators = 1
        self.phases = np.random.uniform(0, 2*np.pi, size=(self.num_oscillators,))  # 初期位相
        self.omega = np.array([2.0, 2.0])  # 固有振動数
        self.K = 1.0  # 結合強度

        self.energy_used = 0.0
        self.prev_apex_z = None # 前ステップの頂点のCOMのz位置
        self._last_apex_time = None

        self.prev_com_z = 0.0  # 前ステップのCOMのz位置

        self.dt = float(self.model.opt.timestep)

        #代替描写
        self.viewer = None
        self.renderer = None
        if self.render_mode == "rgb_array":
            try:
                self.renderer = mujoco.Renderer(self.model, width=640, height=480)
            except Exception as e:
                print("Rendererの初期化に失敗しました:", e)
                self.render_mode = None

        #デバッグ用情報表示
        #print("Action space shape:", self.action_space.shape)
        #print("model.nu(actuators):", self.model.nu)

    #観測の取得
    def _get_obs(self):
        #COM位置と速度
        mujoco.mj_comVel(self.model, self.data)
        try:
            com_z_vel = self.data.subtree_linvel[0][2]  # z軸速度
        except Exception as e:
            com_z_vel = float(self.data.qvel[0]) if self.data.qvel.shape[0] > 1 else 0.0 # エラー時はqvelから代替取得
        #地面反力
        try:
            ground_reaction_force = float(np.sum(self.data.cfrc_ext[:, 2]))  #z方向成分の合計
        except Exception as e:
            ground_reaction_force = 0.0 # エラー時は0に設定
        obs = np.array([com_z_vel, ground_reaction_force, self.phases[0]], dtype=np.float32)
        return obs
    
    def compute_jacobian(self):
        """足先のヤコビアン行列を計算"""
        # 足先（shank）のjacobian
        jacp = np.zeros((3, self.model.nv))  # 位置ヤコビアン
        jacr = np.zeros((3, self.model.nv))  # 回転ヤコビアン
    
        # shankの最後のボディのIDを取得
        end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "shank")
    
        # ヤコビアン計算
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, end_effector_id)
    
        return jacp  # 位置ヤコビアンのみ返す

    def step(self, action):
           
        # CPGの出力から目標力を計算
        max_force = 200.0  # 最大力[N]
        desired_force = np.array([0, 0, max_force * np.sin(self.phases[0])])  # z方向の力
    
        # ヤコビアン計算
        J = self.compute_jacobian()
    
        # トルク計算 (τ = J^T * F)
        joint_torques = np.dot(J.T, desired_force)
    
        # アクチュエータ制御
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        n_write = min(self.model.nu, 2)  # 股関節と膝関節の2つ
        ctrl[:n_write] = joint_torques[:n_write]
        self.data.ctrl[:] = ctrl
 
        #ジャンプ時トルク0
        if float(np.sum(self.data.cfrc_ext[:, 2])) < 1e-8:
            ctrl[:] = 0

        mujoco.mj_step(self.model, self.data)
        
        try:
            qvel =np.array(self.data.qvel, dtype=np.float32) #速度取得
        except Exception as e:
            qvel = np.zeros(self.model.nv, dtype=np.float32) # エラー時はゼロベクトルに設定

        qvel_jump = qvel[:n_write] if qvel.shape[0] >= n_write else np.zeros(n_write, dtype=np.float32) # ジャンプに関与する速度成分
        energy_step = np.sum(np.abs(ctrl[:n_write] * qvel_jump)) * self.dt  # 簡易的なエネルギー消費モデル
        self.energy_used += float(energy_step)

        obs = self._get_obs()

        #報酬計算
        com_z = np.mean(self.data.xipos[:, 2])  #COMのz位置
        reward = 0.0
        com_z_vel = float(obs[0])  # z軸速度

        apex_threshold = 0.01  # 頂点とみなすz速度の閾値
        if abs(com_z_vel) < apex_threshold:
            if self.prev_apex_z is None:
                self.prev_apex_z = com_z
                height_gained = 0.0
            else:
                height_gained = max(0.0, com_z - self.prev_apex_z)  # 前の頂点からの高さの増加分
            if height_gained > 1e-6:
                cot = (self.energy_used / height_gained) if height_gained > 0 else np.inf # 輸送コスト(CoT)
                if cot > 0 and np.isfinite(cot):
                    reward += 1.0 / cot  # CoTの逆数を報酬に
                self.energy_used = 0.0  # エネルギー消費のリセット
                self.prev_apex_z = com_z  # 現在の頂点z位置を保存

        #CPG同期度合いの報酬
        #sync = 1.0 - np.abs(np.sin(self.phases[0] - self.phases[1]))
        #reward += sync * 0.1  # 同期度合いに基づく報酬

        self.prev_com_z = com_z

        #エピソード終了条件
        terminated = False
        truncated = False # タイムアウトは無し

        if com_z is None or com_z < -0.5 or com_z > 3.0:  # COMのz位置が低すぎる場合
            terminated = True
            print("エピソード終了：転倒検出 com_z =", com_z)

        info = {}
        return obs, reward, terminated, truncated, info
    
    #環境のリセット
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        #初期姿勢
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        try:
            self.data.qpos[2] = 1.0  # 初期高さの設定
        except Exception as e:
            pass  # qposが存在しない場合はスキップ
        #CPG位相の初期化
        self.phases = np.random.uniform(0, 2*np.pi, size=(self.num_oscillators,))
        self.energy_used = 0.0
        self.prev_apex_z = None
        self._last_apex_time = None
        self.prev_com_z = np.mean(self.data.xipos[:, 2])        

        mujoco.mj_step(self.model, self.data)  # 初期ステップ

        obs = self._get_obs()
        print("初期CoM位置：", self.data.xipos[0])
        print("初期qpos：", self.data.qpos)
        return obs, {}

    
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
            
    #環境のクローズ
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
#環境の登録
register(
    id="vh-v0",
    entry_point="vh_env:vh_env",
    max_episode_steps=1000,
)