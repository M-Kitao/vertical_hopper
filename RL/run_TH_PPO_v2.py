import os, sys, time, shutil, tempfile, io, zipfile
import numpy as np
import mujoco
import cv2
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

RECORD_VIDEO  = True
VIDEO_FILENAME= "Gainonly_video.mp4"
VIDEO_FPS     = 60
WIDTH, HEIGHT = 640, 480
PLAY_SPEED    = 1.0
OMEGA         = 2 * np.pi / 1.35625
INIT_HEIGHT   = 0.0#np.random.uniform(-0.5, 0.5)  # ランダムな初期高さ

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.dirname(current_dir)
for p in [root_dir,
          os.path.join(root_dir, 'GymEnv'),
          os.path.join(root_dir, 'NN'),
          os.path.join(root_dir, 'CPG')]:
    if p not in sys.path:
        sys.path.append(p)


def main():
    # ===================== 設定 =====================
    BASELINE  = "gainonly"   # "default" / "gainonly" / "torque" / "nofeedback"
    EXP_NAME  = "TH_v5"
    SEED_NAME = "seed_0"
    RENDER_MODE = "human"
    # ================================================

    if BASELINE == "torque":
        from GymEnv.DirectTorque_Hopper import DirectTorque_Hopper as EnvCls
        USE_CUSTOM_POLICY = False
        has_cpg = False
    elif BASELINE == "gainonly":
        from GymEnv.Tegotae_Hopper_PPO_V3_gainonly import Tegotae_Hopper_PPO_v2_Env as EnvCls
        USE_CUSTOM_POLICY = True
        has_cpg = True
    elif BASELINE == "nofeedback":
        from GymEnv.Tegotae_hopper_PPO_v3_nofeedback import Tegotae_Hopper_PPO_v2_Env as EnvCls
        USE_CUSTOM_POLICY = True
        has_cpg = True
    else:
        from GymEnv.Tegotae_Hopper_PPO_V3 import Tegotae_Hopper_PPO_v2_Env as EnvCls
        USE_CUSTOM_POLICY = True
        has_cpg = True

    MODEL_DIR  = os.path.join(current_dir, "results", EXP_NAME, "models", BASELINE, SEED_NAME)
    MODEL_PATH = os.path.join(MODEL_DIR, "final_model.zip")
    STATS_PATH = os.path.join(MODEL_DIR, "vec_normalize.pkl")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: モデルが見つかりません\nPath: {MODEL_PATH}")
        return

    print(f"Loading: {MODEL_PATH}  (baseline={BASELINE})")

    raw_env = EnvCls(render_mode=RENDER_MODE)
    env = DummyVecEnv([lambda: raw_env])

    if os.path.exists(STATS_PATH):
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: vec_normalize.pkl が見つかりません。")

    import torch as _torch
    if USE_CUSTOM_POLICY:
        from GymEnv.Tegotae_Policy import Tegotae_Policy
        with zipfile.ZipFile(MODEL_PATH, 'r') as zf:
            with zf.open('policy.pth') as f:
                params = _torch.load(io.BytesIO(f.read()), map_location='cpu')
        for key in list(params.keys()):
            if 'action_net.log_std' in key:
                del params[key]
        tmp_dir = tempfile.mkdtemp()
        tmp_zip = os.path.join(tmp_dir, "model_fixed.zip")
        with zipfile.ZipFile(MODEL_PATH, 'r') as zi, \
             zipfile.ZipFile(tmp_zip, 'w', zipfile.ZIP_DEFLATED) as zo:
            for item in zi.infolist():
                if item.filename == 'policy.pth':
                    buf = io.BytesIO(); _torch.save(params, buf)
                    zo.writestr('policy.pth', buf.getvalue())
                elif item.filename == 'pytorch_optimizer.pth':
                    buf = io.BytesIO(); _torch.save({}, buf)
                    zo.writestr('pytorch_optimizer.pth', buf.getvalue())
                else:
                    zo.writestr(item, zi.read(item.filename))
        model = PPO.load(tmp_zip,
                         custom_objects={'Tegotae_Policy': Tegotae_Policy},
                         device="auto")
        shutil.rmtree(tmp_dir)
    else:
        model = PPO.load(MODEL_PATH, device="auto")

    print("\n--- Simulation Start (Press Ctrl+C to stop) ---")

    dt = env.envs[0].model.opt.timestep

    # --- 時系列ログ ---
    height_history, reward_history = [], []
    phi_history, phi_dot_history   = [], []
    grf_history, tegotae_history   = [], []
    hip_history, knee_history = [], []
    ref_hip_history, ref_knee_history = [], []
    hip_torque_history, knee_torque_history = [], []
    action_hip_history, action_knee_history = [], []
    action_integral = 0.0
    z_vel_history = []

    # --- サイクル評価指標 ---
    # CoT^{-1} = z_max / E
    # ΔΩ = -(ω - φ_dot_peak)^2 - (1 - z_max)^2
    cot_inv_history     = []
    delta_omega_history = []
    cycle_energy        = 0.0
    cycle_z_max         = -np.inf
    cycle_phi_dot_peak  = 0.0
    prev_grf            = 0.0
    omega               = OMEGA

    obs = env.reset()
    total_reward = 0.0
    sim_t = 0.0

    # Joint indices for angle logging
    mj_model = env.envs[0].model
    hip_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
    knee_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
    hip_qpos_adr = mj_model.jnt_qposadr[hip_joint_id]
    knee_qpos_adr = mj_model.jnt_qposadr[knee_joint_id]

    hip_act_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_joint")
    knee_act_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_joint")

    mj_data = env.envs[0].data  # Initialize mj_data outside the loop

    env.envs[0].data.qpos[1] = INIT_HEIGHT
    if has_cpg:
        env.envs[0].cpg.omega = OMEGA
    mujoco.mj_forward(env.envs[0].model, env.envs[0].data)

    renderer = video_writer = None
    if RECORD_VIDEO:
        renderer = mujoco.Renderer(env.envs[0].model, height=HEIGHT, width=WIDTH)
        video_writer = cv2.VideoWriter(VIDEO_FILENAME,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       VIDEO_FPS, (WIDTH, HEIGHT))
    last_render_time = 0.0

    try:
        for _ in range(10000):
            step_start = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            sim_t += dt

            # Log PPO action outputs
            if BASELINE == "torque":
                action_hip_history.append(float(action[0][1]))  # hip action
                action_knee_history.append(float(action[0][0]))  # knee action
            elif BASELINE == "gainonly":
                action_hip_history.append(float(action[0][0]))  # gain action
                action_knee_history.append(float('nan'))  # no knee action
            else:  # default, nofeedback
                action_hip_history.append(float(action[0][0]))  # gain
                action_knee_history.append(float(action[0][1]))  # action
                # action[0][2] would be reaction if available

            current_height = env.envs[0].data.qpos[1]
            grf = float(infos[0].get('grf', 0.0))

            hip_angle = float(env.envs[0].data.qpos[hip_qpos_adr])
            knee_angle = float(env.envs[0].data.qpos[knee_qpos_adr])
            hip_history.append(hip_angle)
            knee_history.append(knee_angle)

            hip_torque_history.append(float(env.envs[0].data.actuator_force[hip_act_id]))
            knee_torque_history.append(float(env.envs[0].data.actuator_force[knee_act_id]))

            phase_value = float(infos[0].get('phase', getattr(env.envs[0], 'phase', 0.0)))
            if hasattr(env.envs[0], 'ref_phases') and hasattr(env.envs[0], 'ref_hip_angles') and hasattr(env.envs[0], 'ref_knee_angles'):
                ref_hip = np.interp(phase_value, env.envs[0].ref_phases, env.envs[0].ref_hip_angles)
                ref_knee = np.interp(phase_value, env.envs[0].ref_phases, env.envs[0].ref_knee_angles)
                ref_hip_history.append(ref_hip)
                ref_knee_history.append(ref_knee)
            else:
                ref_hip_history.append(np.nan)
                ref_knee_history.append(np.nan)

            height_history.append(current_height)
            z_vel_history.append(float(env.envs[0].data.qvel[1]))  # 垂直速度
            reward_history.append(total_reward)
            grf_history.append(grf)

            if has_cpg:
                phi_dot = float(infos[0].get('phase_velocity', 0.0))
                action_integral += infos[0].get('action', 0.0) * dt
                tegotae_history.append(action_integral * infos[0].get('reaction', 0.0))
                phi_history.append(infos[0].get('phase', 0.0))
                phi_dot_history.append(phi_dot)
                omega = env.envs[0].cpg.omega
                if abs(phi_dot) > abs(cycle_phi_dot_peak):
                    cycle_phi_dot_peak = phi_dot

            # エネルギー積算
            ctrl = env.envs[0].data.ctrl
            qvel = env.envs[0].data.qvel
                        # 修正後：関節IDから正しいdofアドレスを取得

            mj_model = env.envs[0].model
            mj_data  = env.envs[0].data
            
            hip_id   = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
            knee_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
            hip_dof  = mj_model.jnt_dofadr[hip_id]
            knee_dof = mj_model.jnt_dofadr[knee_id]
            
            # ループ内でのエネルギー計算
            hip_vel  = mj_data.qvel[hip_dof]
            knee_vel = mj_data.qvel[knee_dof]
            power = abs(mj_data.actuator_force[hip_act_id]  * hip_vel) + \
                    abs(mj_data.actuator_force[knee_act_id] * knee_vel)
            cycle_energy += power * dt
            
            
            # 最大高さ更新
            if current_height > cycle_z_max:
                cycle_z_max = current_height

            # 着地検出 → 1サイクル評価
            just_landed = (grf > 10.0) and (prev_grf <= 10.0)
            if just_landed and cycle_z_max > 0.01:
                cot_inv = 75.5 * 9.81 * cycle_z_max / (cycle_energy + 1e-6)
                cot_inv_history.append(cot_inv)

                if has_cpg:
                    dw = -(omega - cycle_phi_dot_peak)**2 - (1.0 - cycle_z_max / 0.45)**2
                else:
                    dw = -(1.0 - cycle_z_max / 0.45)**2
                delta_omega_history.append(dw)

                print(f"  [cycle#{len(cot_inv_history):3d}]"
                      f"  z_max={cycle_z_max:.3f}m"
                      f"  E={cycle_energy:.3f}J"
                      f"  CoT⁻¹={cot_inv:.4f}"
                      f"  ΔΩ={dw:.4f}")

                # サイクルリセット
                cycle_energy = 0.0
                cycle_z_max  = -np.inf
                cycle_phi_dot_peak = 0.0

            prev_grf = grf

            if RECORD_VIDEO and (sim_t - last_render_time >= 1.0 / VIDEO_FPS):
                renderer.update_scene(env.envs[0].data, camera='track')
                video_writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
                last_render_time = sim_t

            wait = (dt / PLAY_SPEED) - (time.time() - step_start)
            if wait > 0:
                time.sleep(wait)

            env.envs[0].render()

            if dones[0]:
                print("Episode Finished. Resetting...")
                obs = env.reset()
                env.envs[0].data.qpos[1] = INIT_HEIGHT
                cycle_energy = 0.0
                cycle_z_max  = -np.inf
                cycle_phi_dot_peak = 0.0
                prev_grf = 0.0
                if has_cpg:
                    env.envs[0].cpg.omega = OMEGA
                mujoco.mj_forward(env.envs[0].model, env.envs[0].data)

    except KeyboardInterrupt:
        print("\nSimulation stopped.")

    env.close()
    if video_writer:
        video_writer.release()

    # --- サマリー出力 ---
    if cot_inv_history:
        print(f"\n=== 評価指標サマリー ({len(cot_inv_history)} サイクル) ===")
        print(f"  CoT⁻¹  : mean={np.mean(cot_inv_history):.4f}"
              f"  std={np.std(cot_inv_history):.4f}"
              f"  max={np.max(cot_inv_history):.4f}")
        if has_cpg:
            print(f"  ΔΩ     : mean={np.mean(delta_omega_history):.4f}"
                  f"  std={np.std(delta_omega_history):.4f}"
                  f"  max={np.max(delta_omega_history):.4f}")

    # --- グラフ描画 ---
    if height_history:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(height_history, color='blue', label='Height')
        ax1.set_ylabel('Height [m]')
        ax1.set_title(f'[{BASELINE}] Center of Mass Height')
        ax1.grid(True); ax1.legend()
        ax2.plot(reward_history, color='orange', label='Cumulative Reward')
        ax2.set_xlabel('Steps'); ax2.set_ylabel('Reward')
        ax2.set_title('Cumulative Reward')
        ax2.grid(True); ax2.legend()
        plt.tight_layout(); plt.show()

    if has_cpg and phi_history:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(phi_history, color='green', label='CPG Phase')
        ax1.set_ylabel('Phase [rad]'); ax1.set_title('CPG Phase')
        ax1.grid(True); ax1.legend()
        ax2.plot(phi_dot_history, color='red', label='Phase Velocity')
        ax2.set_xlabel('Steps'); ax2.set_ylabel('[rad/s]')
        ax2.set_title('CPG Phase Velocity')
        ax2.grid(True); ax2.legend()
        plt.tight_layout(); plt.show()

    # CoT^{-1} と ΔΩ のサイクルごとのグラフ
    if cot_inv_history:
        n_plots = 2 if (has_cpg and delta_omega_history) else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]

        axes[0].plot(cot_inv_history, color='teal', marker='o', markersize=4,
                     label='CoT⁻¹ = z_max / E')
        axes[0].axhline(np.mean(cot_inv_history), color='teal',
                        linestyle='--', alpha=0.6,
                        label=f'mean={np.mean(cot_inv_history):.4f}')
        axes[0].set_ylabel('CoT⁻¹')
        axes[0].set_title('Cost of Transport (inverse) per cycle')
        axes[0].grid(True); axes[0].legend()

        if has_cpg and delta_omega_history:
            axes[1].plot(delta_omega_history, color='darkred', marker='o', markersize=4,
                         label='ΔΩ = -(ω-φ̇_peak)² - (1-z_max/h)²')
            axes[1].axhline(np.mean(delta_omega_history), color='darkred',
                            linestyle='--', alpha=0.6,
                            label=f'mean={np.mean(delta_omega_history):.4f}')
            axes[1].set_ylabel('ΔΩ')
            axes[1].set_title('Phase Synchrony Index (ΔΩ) per cycle')
            axes[1].grid(True); axes[1].legend()

        axes[-1].set_xlabel('Cycle #')
        plt.tight_layout(); plt.show()

    if grf_history:
        plt.figure(figsize=(10, 4))
        plt.plot(grf_history, color='purple', label='GRF')
        plt.xlabel('Steps'); plt.ylabel('GRF [N]')
        plt.title('Ground Reaction Force'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.show()

    # Joint angle and reference trajectory plots
    if hip_history and knee_history:
        rad2deg = 180.0 / np.pi
        times = np.arange(len(hip_history)) * dt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(times, np.array(hip_history) * rad2deg, color='blue', label='Hip actual')
        if not np.all(np.isnan(ref_hip_history)):
            ax1.plot(times, np.array(ref_hip_history) * rad2deg, '--', color='cyan', label='Hip ref')
        ax1.set_ylabel('Hip angle [deg]')
        ax1.set_title('Hip Joint Angle vs Reference')
        ax1.grid(True); ax1.legend()

        ax2.plot(times, np.array(knee_history) * rad2deg, color='orange', label='Knee actual')
        if not np.all(np.isnan(ref_knee_history)):
            ax2.plot(times, np.array(ref_knee_history) * rad2deg, '--', color='red', label='Knee ref')
        ax2.set_ylabel('Knee angle [deg]')
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Knee Joint Angle vs Reference')
        ax2.grid(True); ax2.legend()

        plt.tight_layout(); plt.show()

    # Joint torque plots
    if hip_torque_history and knee_torque_history:
        times = np.arange(len(hip_torque_history)) * dt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(times, hip_torque_history, color='green', label='Hip torque')
        ax1.set_ylabel('Hip torque [Nm]')
        ax1.set_title('Hip Joint Torque')
        ax1.grid(True); ax1.legend()

        ax2.plot(times, knee_torque_history, color='purple', label='Knee torque')
        ax2.set_ylabel('Knee torque [Nm]')
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Knee Joint Torque')
        ax2.grid(True); ax2.legend()

        plt.tight_layout(); plt.show()

    # PPO action outputs plot
    if action_hip_history and action_knee_history:
        times = np.arange(len(action_hip_history)) * dt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(times, action_hip_history, color='blue', label='Hip action (PPO output)', linewidth=0.8)
        ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Hip action [-5 to 5]')
        ax1.set_title('PPO Policy Output: Hip Motor Commands')
        ax1.grid(True); ax1.legend()
        ax1.set_ylim([-5.5, 5.5])

        ax2.plot(times, action_knee_history, color='orange', label='Knee action (PPO output)', linewidth=0.8)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.set_ylabel('Knee action [-5 to 5]')
        ax2.set_xlabel('Time [s]')
        ax2.set_title('PPO Policy Output: Knee Motor Commands')
        ax2.grid(True); ax2.legend()
        ax2.set_ylim([-5.5, 5.5])

        plt.tight_layout(); plt.show()

        # Print statistics
        print(f"\n=== PPO Action Statistics ===")
        print(f"Hip action   : mean={np.mean(action_hip_history):.4f}, std={np.std(action_hip_history):.4f}, "
              f"min={np.min(action_hip_history):.4f}, max={np.max(action_hip_history):.4f}")
        print(f"Knee action  : mean={np.mean(action_knee_history):.4f}, std={np.std(action_knee_history):.4f}, "
              f"min={np.min(action_knee_history):.4f}, max={np.max(action_knee_history):.4f}")
        
        # ==========================================
        # ポアンカレ断面図 (Phase Portrait & Poincaré Section)
        # ==========================================
        print("\nGenerating Poincaré Section...")
        poincare_z = []
        poincare_z_vel = []
        
        # 「着地した瞬間（Touchdown）」を抽出する
        # 前のステップでGRFが閾値以下、現在のステップで閾値以上になった瞬間を「着地」と定義
        grf_threshold = 10.0  # 10Nを閾値とする（ノイズ対策）
        for i in range(1, len(grf_history)):
            if grf_history[i-1] < grf_threshold and grf_history[i] >= grf_threshold:
                poincare_z.append(height_history[i])
                poincare_z_vel.append(z_vel_history[i])

        plt.figure(figsize=(8, 6))
        
        # 1. 全体の連続的な軌道（相図 / Phase Portrait）を薄いグレーで描画
        plt.plot(height_history, z_vel_history, color='gray', alpha=0.4, linewidth=0.8, label='Continuous Trajectory')
        
        # 2. ポアンカレ断面（着地時の状態）を赤い点でプロット
        # 初期の過渡応答を無視したい場合は、最初の数点の色を変えるか除外するとより綺麗です
        plt.scatter(poincare_z[3:], poincare_z_vel[3:], color='red', s=40, zorder=5, label='Poincaré points (Touchdown)')
        if len(poincare_z) > 0:
            plt.scatter(poincare_z[0:3], poincare_z_vel[0:3], color='blue', s=20, alpha=0.5, zorder=4, label='Initial transients')

        plt.xlabel('Vertical Position $Z$ [m]', fontsize=12)
        plt.ylabel('Vertical Velocity $\dot{Z}$ [m/s]', fontsize=12)
        plt.title('Phase Portrait and Poincaré Section at Touchdown', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        # ==========================================

    print(f"\nFinal cumulative reward: {reward_history[-1] if reward_history else 0.0}")


if __name__ == "__main__":
    main()