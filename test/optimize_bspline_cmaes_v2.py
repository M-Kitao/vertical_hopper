"""
optimize_bspline_cmaes_v2.py

【v2 での主な変更点】
  1. 周期Bスプライン（Periodic B-spline）
       phi=0 と phi=2π で値・一階微分が自動的に一致する C¹ 連続な周期軌道を保証。
       制御点の末尾に先頭 degree 個を折り返して繰り返すことで周期性を実現。

  2. 跳躍時（飛翔期）の位相変化最小化
       ホッピング1周期のうち「飛翔期」（足が地面から離れている区間）では
       関節をなるべく動かさない（= dq/dphi ≈ 0）よう J_flight ペナルティを追加。
       飛翔期フラグは FLIGHT_PHASE_START / FLIGHT_PHASE_END [rad] で指定。

  3. エネルギー消費の最適化
       PDアクチュエータの仮想トルク τ = kp*(q_ref - q) + kv*(-dq/dt) を近似計算し、
       τ² の時間積分（= COT: Cost of Transport の代理変数）を J_energy として追加。
       vertical_hopper.xml: kp=5000, kv=50

対象モデル: vertical_hopper.xml
  hip_joint  : kp=5000, kv=50, ctrlrange=[-90, +90] deg
  knee_joint : kp=5000, kv=50, ctrlrange=[-135,  0] deg
  膝角度の符号: 屈曲=負方向、接地リスク=0 rad（伸展方向）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

try:
    import cma
except ImportError:
    raise ImportError("pip install cma を実行してください。")

from scipy.interpolate import make_interp_spline


# ============================================================
# 0. モデル定数 (vertical_hopper.xml)
# ============================================================

HIP_RANGE_MIN  = np.radians(-90)
HIP_RANGE_MAX  = np.radians( 90)
KNEE_RANGE_MIN = np.radians(-135)
KNEE_RANGE_MAX = np.radians(   0)   # 完全伸展 = 接地リスク

HIP_CTRL_MIN  = np.radians(-90)
HIP_CTRL_MAX  = np.radians( 90)
KNEE_CTRL_MIN = np.radians(-135)
KNEE_CTRL_MAX = np.radians(   0)

# PDアクチュエータゲイン (position actuator)
KP = 5000.0   # 比例ゲイン [N·m/rad]
KV =   50.0   # 微分ゲイン [N·m·s/rad]

KNEE_SAFETY_MARGIN    = np.radians(10)
KNEE_SAFE_RAD_DEFAULT = KNEE_RANGE_MAX - KNEE_SAFETY_MARGIN  # -0.1745 rad

MASS_TOTAL    = 75.0   # kg
GRF_THRESHOLD = 1.0   # [N] これ以下を飛翔期とみなす

# ============================================================
# 1. データ読み込み
# ============================================================

def load_latest_csv(prefix='csvdata20260401/squat_flying_data_'):
    files = glob.glob(f"{prefix}*.csv")
    if not files:
        raise FileNotFoundError("CSVファイルが見つかりません。")
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    return pd.read_csv(latest_file)


# ============================================================
# 2. 周期Bスプライン（C¹ 連続周期軌道）
# ============================================================

def make_periodic_bspline(phi, control_points, degree=3):
    """
    phi=0 と phi=2π で値・一階微分が一致する周期Bスプラインを構築する。

    実装方法：
      制御点列の末尾に先頭 degree 個を追加して「折り返し」を作り、
      対応するノット列と合わせて make_interp_spline に渡す。
      これにより端点での C¹ 連続性が自動的に保証される。

    Parameters
    ----------
    phi            : ndarray (N,)  評価点 [0, 2π]
    control_points : ndarray (n_ctrl,)  最適化変数
    degree         : int  スプライン次数（3推奨）

    Returns
    -------
    y : ndarray (N,)
    """
    n  = len(control_points)
    T  = 2 * np.pi

    # 等間隔ノット（制御点1個あたりの幅）
    dt = T / n

    # 制御点を周期的に拡張：末尾に先頭 degree 個を追加
    cp_ext  = np.concatenate([control_points,
                               control_points[:degree]])
    # 対応するノット時刻
    t_ext   = np.array([i * dt for i in range(n + degree)])

    # 評価範囲を [0, T) に収める（端点は t_ext の内側）
    phi_clipped = np.clip(phi, t_ext[0], t_ext[-1] - 1e-9)

    spl = make_interp_spline(t_ext, cp_ext, k=degree, bc_type=None)
    return spl(phi_clipped)


def make_periodic_bspline_derivative(phi, control_points, degree=3):
    """周期Bスプラインの一階微分 dq/dphi を返す"""
    n  = len(control_points)
    T  = 2 * np.pi
    dt = T / n

    cp_ext = np.concatenate([control_points, control_points[:degree]])
    t_ext  = np.array([i * dt for i in range(n + degree)])
    phi_clipped = np.clip(phi, t_ext[0], t_ext[-1] - 1e-9)

    spl = make_interp_spline(t_ext, cp_ext, k=degree, bc_type=None)
    return spl(phi_clipped, nu=1)   # nu=1 → 一階微分


def traj_and_deriv(theta, phi, joint='hip'):
    """軌道 q(phi) と dq/dphi を同時に返す"""
    n_ctrl = len(theta) // 2
    cp = theta[:n_ctrl] if joint == 'hip' else theta[n_ctrl:]
    q    = make_periodic_bspline(phi, cp)
    dqdt = make_periodic_bspline_derivative(phi, cp)
    return q, dqdt


# ============================================================
# 3. 目的関数
# ============================================================

def objective(theta, phi, dt,
              y_hip_ref, y_knee_ref,
              flight_mask,
              w_track=1.0, w_knee=10.0, w_smooth=1e-4,
              w_actuator=5.0, w_flight=5.0, w_energy=0.1,
              knee_safe_rad=KNEE_SAFE_RAD_DEFAULT):
    """
    J = J_track + J_knee + J_smooth + J_actuator + J_flight + J_energy

    ① J_track    : 参照軌道への追従（RMSE）
    ② J_knee     : 膝が伸展方向に入りすぎるペナルティ（2段階）
    ③ J_smooth   : 加速度二乗和（滑らかさ）
    ④ J_actuator : アクチュエータ制御範囲違反ペナルティ
    ⑤ J_flight   : 飛翔期の関節速度最小化（位相変化を抑える）
    ⑥ J_energy   : PDトルク二乗積分（エネルギー代理変数）

    Parameters
    ----------
    theta       : 最適化変数（Hip制御点 + Knee制御点 の連結）
    phi         : 位相配列 [0, 2π]
    dt          : 時間刻み [s]
    y_hip_ref   : Hip参照軌道
    y_knee_ref  : Knee参照軌道
    flight_mask : bool配列 True = 飛翔期のタイムステップ
    w_*         : 各項の重み
    knee_safe_rad : 膝安全限界 [rad]
    """
    yh, dyh = traj_and_deriv(theta, phi, 'hip')
    yk, dyk = traj_and_deriv(theta, phi, 'knee')

    dphi = phi[1] - phi[0]   # 位相刻み
    scale = (np.std(y_hip_ref) + np.std(y_knee_ref)) + 1e-8

    # ① 追従誤差（RMSE）
    J_track = w_track * (
        np.sqrt(np.mean((yh - y_hip_ref )**2))
      + np.sqrt(np.mean((yk - y_knee_ref)**2))
    )

    # ② 膝ペナルティ（2段階）
    soft_viol = np.maximum(0.0, yk - knee_safe_rad)
    hard_viol = np.maximum(0.0, yk - KNEE_RANGE_MAX)
    J_knee = w_knee * (np.mean(soft_viol**2) + 20.0 * np.mean(hard_viol**2))

    # ③ 滑らかさ（加速度二乗和を正規化）
    d2yh = np.diff(dyh) / dphi   # 二階微分（位相空間）
    d2yk = np.diff(dyk) / dphi
    J_smooth = w_smooth * (np.mean(d2yh**2) + np.mean(d2yk**2)) / scale**2

    # ④ アクチュエータ制御範囲違反
    hip_viol  = (np.maximum(0.0, yh - HIP_CTRL_MAX)
               + np.maximum(0.0, HIP_CTRL_MIN - yh))
    knee_viol = (np.maximum(0.0, yk - KNEE_CTRL_MAX)
               + np.maximum(0.0, KNEE_CTRL_MIN - yk))
    J_actuator = w_actuator * (np.mean(hip_viol**2) + np.mean(knee_viol**2))

    # ⑤ 飛翔期の関節速度最小化
    #    正規化の考え方：
    #      参照軌道の速度 dq_ref/dphi の RMS を基準にして
    #      「飛翔期の速度が参照速度の何倍か」を評価する。
    #      これで w_flight=1 のとき J_flight ≈ J_track と同オーダーになる。
    if flight_mask is not None and np.any(flight_mask):
        # 参照軌道の速度スケール（全区間の速度RMS）
        dyh_ref_rms = np.sqrt(np.mean(np.gradient(y_hip_ref,  dphi)**2)) + 1e-8
        dyk_ref_rms = np.sqrt(np.mean(np.gradient(y_knee_ref, dphi)**2)) + 1e-8
        J_flight = w_flight * (
            np.mean(dyh[flight_mask]**2) / dyh_ref_rms**2
          + np.mean(dyk[flight_mask]**2) / dyk_ref_rms**2
        )
    else:
        J_flight = 0.0

    # ⑥ エネルギー（PDトルク二乗積分）
    #    position アクチュエータの出力トルクを近似：
    #      τ ≈ kp * (q_ref - q_current) - kv * (dq/dt)
    #    ここでは q_ref = q_opt（最適化中の軌道自体を目標とする）とし、
    #    追従誤差由来のトルクではなく軌道変化由来のトルクを評価する。
    #    dq/dt = dq/dphi * dphi/dt = dq/dphi * (2π/T)
    omega = 2 * np.pi / (dt * len(phi))   # 角速度 [rad/s]
    dqh_dt = dyh * omega
    dqk_dt = dyk * omega
    #    静的なトルク推定（関節を保持するために必要な力の代理）
    #    ≈ kv * |dq/dt|（速度成分のみ、追従偏差は追従誤差項でカバー）
    tau_h = KV * np.abs(dqh_dt)
    tau_k = KV * np.abs(dqk_dt)
    J_energy = w_energy * (np.mean(tau_h**2) + np.mean(tau_k**2)) / (KV**2)

    return J_track + J_knee + J_smooth + J_actuator + J_flight + J_energy


# ============================================================
# 4. 初期制御点
# ============================================================

def make_initial_control_points(phi, y_hip_ref, y_knee_ref, n_ctrl):
    """参照軌道をダウンサンプリングして初期制御点を生成する"""
    idx     = np.linspace(0, len(phi) - 1, n_ctrl, dtype=int)
    cp_hip  = y_hip_ref[idx]
    cp_knee = y_knee_ref[idx]
    return np.concatenate([cp_hip, cp_knee])


# ============================================================
# 5. CMA-ES 最適化
# ============================================================

def optimize_with_cmaes(phi, dt,
                         y_hip_ref, y_knee_ref,
                         flight_mask,
                         n_ctrl=16,
                         sigma0=0.10,
                         maxiter=500,
                         w_track=1.0,
                         w_knee=10.0,
                         w_smooth=1e-4,
                         w_actuator=5.0,
                         w_flight=5.0,
                         w_energy=0.1,
                         knee_safe_rad=KNEE_SAFE_RAD_DEFAULT):

    theta0 = make_initial_control_points(phi, y_hip_ref, y_knee_ref, n_ctrl)

    lower = np.array([HIP_CTRL_MIN]  * n_ctrl + [KNEE_CTRL_MIN] * n_ctrl)
    upper = np.array([HIP_CTRL_MAX]  * n_ctrl + [KNEE_CTRL_MAX] * n_ctrl)

    eps = 1e-4
    theta0_clipped = np.clip(theta0, lower + eps, upper - eps)
    if not np.allclose(theta0, theta0_clipped):
        n_clip = int(np.sum(~np.isclose(theta0, theta0_clipped)))
        print(f"  ⚠️  初期制御点 {n_clip} 個をクリップしました")
    theta0 = theta0_clipped

    history = {k: [] for k in
               ['best_cost', 'J_track', 'J_knee', 'J_smooth',
                'J_actuator', 'J_flight', 'J_energy']}

    opts = cma.CMAOptions()
    opts['maxiter'] = maxiter
    opts['tolx']    = 1e-7
    opts['tolfun']  = 1e-7
    opts['verbose'] = -9
    opts['popsize'] = 4 + int(3 * np.log(len(theta0)))
    opts['bounds']  = [lower.tolist(), upper.tolist()]

    es = cma.CMAEvolutionStrategy(theta0, sigma0, opts)

    n_flight = int(np.sum(flight_mask)) if flight_mask is not None else 0
    print(f"\n=== CMA-ES 最適化開始 ===")
    print(f"  制御点数    : {n_ctrl}  (変数次元={len(theta0)})")
    print(f"  sigma0      : {sigma0}")
    print(f"  最大世代数  : {maxiter}  |  集団サイズ: {es.popsize}")
    print(f"  膝安全限界  : {np.degrees(knee_safe_rad):.1f} deg")
    print(f"  飛翔期サンプル数: {n_flight} / {len(phi)}")
    print(f"  重み: track={w_track} knee={w_knee} smooth={w_smooth} "
          f"act={w_actuator} flight={w_flight} energy={w_energy}")
    print("=" * 55)

    gen = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [
            objective(th, phi, dt, y_hip_ref, y_knee_ref, flight_mask,
                      w_track=w_track, w_knee=w_knee, w_smooth=w_smooth,
                      w_actuator=w_actuator, w_flight=w_flight,
                      w_energy=w_energy, knee_safe_rad=knee_safe_rad)
            for th in solutions
        ]
        es.tell(solutions, fitnesses)

        bt = es.result.xbest
        history['best_cost'].append(min(fitnesses))

        yh, dyh = traj_and_deriv(bt, phi, 'hip')
        yk, dyk = traj_and_deriv(bt, phi, 'knee')
        dphi  = phi[1] - phi[0]
        scale = (np.std(y_hip_ref) + np.std(y_knee_ref)) + 1e-8
        omega = 2 * np.pi / (dt * len(phi))

        history['J_track'].append(
            np.sqrt(np.mean((yh - y_hip_ref)**2))
          + np.sqrt(np.mean((yk - y_knee_ref)**2))
        )
        history['J_knee'].append(
            np.mean(np.maximum(0.0, yk - knee_safe_rad)**2)
        )
        d2yh = np.diff(dyh) / dphi
        d2yk = np.diff(dyk) / dphi
        history['J_smooth'].append(
            (np.mean(d2yh**2) + np.mean(d2yk**2)) / scale**2
        )
        hip_v  = np.maximum(0.0, yh - HIP_CTRL_MAX)  + np.maximum(0.0, HIP_CTRL_MIN - yh)
        knee_v = np.maximum(0.0, yk - KNEE_CTRL_MAX) + np.maximum(0.0, KNEE_CTRL_MIN - yk)
        history['J_actuator'].append(np.mean(hip_v**2) + np.mean(knee_v**2))

        if flight_mask is not None and np.any(flight_mask):
            dphi_log = phi[1] - phi[0]
            dyh_ref_rms = np.sqrt(np.mean(np.gradient(y_hip_ref,  dphi_log)**2)) + 1e-8
            dyk_ref_rms = np.sqrt(np.mean(np.gradient(y_knee_ref, dphi_log)**2)) + 1e-8
            history['J_flight'].append(
                np.mean(dyh[flight_mask]**2) / dyh_ref_rms**2
              + np.mean(dyk[flight_mask]**2) / dyk_ref_rms**2
            )
        else:
            history['J_flight'].append(0.0)

        tau_h = KV * np.abs(dyh * omega)
        tau_k = KV * np.abs(dyk * omega)
        history['J_energy'].append(
            (np.mean(tau_h**2) + np.mean(tau_k**2)) / KV**2
        )

        gen += 1
        if gen % 50 == 0:
            print(f"  Gen {gen:4d} | Cost={history['best_cost'][-1]:.5f} "
                  f"| track={history['J_track'][-1]:.4f} "
                  f"| knee={history['J_knee'][-1]:.4f} "
                  f"| flight={history['J_flight'][-1]:.4f} "
                  f"| energy={history['J_energy'][-1]:.4f} "
                  f"| σ={es.sigma:.4f}")

    print(f"\n最適化終了: {gen} 世代, 最終コスト = {history['best_cost'][-1]:.6f}")
    return es.result.xbest, history


# ============================================================
# 6. 可視化
# ============================================================

def plot_results(phi, dt,
                 y_hip_ref, y_knee_ref,
                 y_hip_opt, y_knee_opt,
                 dyh_opt, dyk_opt,
                 history, knee_safe_rad,
                 theta_opt, n_ctrl, flight_mask,
                 grf_z=None):          # ← GRF_Z 波形（オプション）

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        'Periodic BSpline Trajectory Optimization (CMA-ES)  |  vertical_hopper.xml\n'
        f'C¹-continuous, Flight-phase frozen, Energy-aware  '
        f'| Knee safe: {np.degrees(knee_safe_rad):.1f}°',
        fontsize=11, fontweight='bold'
    )

    omega      = 2 * np.pi / (dt * len(phi))
    y_knee_top = max(0.05, y_knee_ref.max(), y_knee_opt.max()) + 0.05

    # 飛翔期グレー帯を各軸に描画するヘルパー
    def shade_flight(ax):
        if flight_mask is not None and np.any(flight_mask):
            # 連続する飛翔区間ごとにシェード（区間が複数に分かれても対応）
            in_flight = False
            start_phi = None
            for i, (p, f) in enumerate(zip(phi, flight_mask)):
                if f and not in_flight:
                    start_phi = p
                    in_flight = True
                elif not f and in_flight:
                    ax.axvspan(start_phi, phi[i-1],
                               alpha=0.13, color='gray', zorder=0)
                    in_flight = False
            if in_flight:  # 末尾まで飛翔期が続く場合
                ax.axvspan(start_phi, phi[-1],
                           alpha=0.13, color='gray', zorder=0,
                           label='Flight phase')

    # ---- (1) 収束曲線 ----
    ax1 = fig.add_subplot(3, 3, 1)
    gens = np.arange(1, len(history['best_cost']) + 1)
    colors = {'best_cost':'k', 'J_track':'b', 'J_knee':'r',
              'J_smooth':'g', 'J_actuator':'m',
              'J_flight':'orange', 'J_energy':'cyan'}
    for key, col in colors.items():
        vals = np.array(history[key])
        if np.any(vals > 0):
            ax1.semilogy(gens, np.maximum(vals, 1e-12), color=col,
                         lw=1.5 if key != 'best_cost' else 2,
                         ls='-' if key == 'best_cost' else '--',
                         label=key)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Cost (log)')
    ax1.set_title('Convergence Curve')
    ax1.legend(fontsize=7)
    ax1.grid(True, which='both', alpha=0.4)

    # ---- (2) GRF_Z 波形 + 飛翔期フラグ ----
    ax2 = fig.add_subplot(3, 3, 2)
    if grf_z is not None:
        ax2.plot(phi, grf_z, 'steelblue', lw=1.5, label='GRF_Z [N]')
        ax2.axhline(GRF_THRESHOLD, color='orange', ls='--', lw=1.2,
                    label=f'Threshold ({GRF_THRESHOLD:.1f} N)')
        shade_flight(ax2)
        ax2.set_ylabel('GRF_Z [N]')
        ax2.set_title('Ground Reaction Force\n(Gray = detected flight phase)')
    else:
        ax2.text(0.5, 0.5, 'GRF_Z not available',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('GRF_Z (N/A)')
    ax2.set_xlabel('Phase φ [rad]')
    ax2.legend(fontsize=7); ax2.grid(True)

    # ---- (3) Hip 軌道 ----
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(phi, y_hip_ref, 'b-',  label='Reference', lw=2, alpha=0.6)
    ax3.plot(phi, y_hip_opt, 'r--', label='Optimized',  lw=1.8)
    shade_flight(ax3)
    ax3.axhline(HIP_CTRL_MAX, color='purple', ls=':', lw=1, alpha=0.7,
                label=f'Ctrl max ({np.degrees(HIP_CTRL_MAX):.0f}°)')
    ax3.axhline(HIP_CTRL_MIN, color='purple', ls=':', lw=1, alpha=0.7,
                label=f'Ctrl min ({np.degrees(HIP_CTRL_MIN):.0f}°)')
    ax3.set_xlabel('Phase φ [rad]')
    ax3.set_ylabel('Angle [rad]')
    ax3.set_title('Hip Trajectory')
    ax3.legend(fontsize=7); ax3.grid(True)

    # ---- (4) Knee 軌道 ----
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(phi, y_knee_ref, 'g-',  label='Reference', lw=2, alpha=0.6)
    ax4.plot(phi, y_knee_opt, 'm--', label='Optimized',  lw=1.8)
    shade_flight(ax4)
    ax4.axhline(knee_safe_rad, color='red', ls=':', lw=1.5,
                label=f'Safe limit ({np.degrees(knee_safe_rad):.1f}°)')
    ax4.axhline(KNEE_RANGE_MAX, color='darkred', ls='--', lw=1.2,
                label='Full ext. (0°)')
    ax4.fill_between(phi, knee_safe_rad, y_knee_top, color='red', alpha=0.08)
    ax4.set_xlabel('Phase φ [rad]')
    ax4.set_ylabel('Angle [rad]')
    ax4.set_title('Knee Trajectory\n(Red = danger zone)')
    ax4.legend(fontsize=7); ax4.grid(True)

    # ---- (5) 関節空間軌道 ----
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(y_hip_ref, y_knee_ref, 'k-',  label='Reference', lw=3, alpha=0.3)
    ax5.plot(y_hip_opt, y_knee_opt, 'r--', label='Optimized',  lw=1.8)
    ax5.plot(y_hip_opt[0],  y_knee_opt[0],  'go', ms=8, label='φ=0')
    ax5.plot(y_hip_opt[-1], y_knee_opt[-1], 'g^', ms=8, label='φ=2π')
    ax5.axhline(knee_safe_rad,  color='red',    ls=':', lw=1.5)
    ax5.axhline(KNEE_RANGE_MAX, color='darkred', ls='--', lw=1.2)
    x_min = min(y_hip_ref.min(), y_hip_opt.min()) - 0.05
    x_max = max(y_hip_ref.max(), y_hip_opt.max()) + 0.05
    ax5.fill_between([x_min, x_max], knee_safe_rad, y_knee_top,
                     color='red', alpha=0.07)
    ax5.set_xlabel('Hip Angle [rad]')
    ax5.set_ylabel('Knee Angle [rad]')
    ax5.set_title('Joint Space Orbit\n(●=φ=0, ▲=φ=2π → should overlap)')
    ax5.legend(fontsize=7); ax5.grid(True)

    # ---- (6) 関節速度 dq/dphi（飛翔期確認）----
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(phi, dyh_opt, 'r-',  label='dHip/dφ',  lw=1.5)
    ax6.plot(phi, dyk_opt, 'm--', label='dKnee/dφ', lw=1.5)
    ax6.axhline(0, color='k', lw=0.8)
    shade_flight(ax6)
    ax6.set_xlabel('Phase φ [rad]')
    ax6.set_ylabel('dq/dφ [rad/rad]')
    ax6.set_title('Joint Velocity in Phase\n(Gray zone = flight, should be ≈0)')
    ax6.legend(fontsize=7); ax6.grid(True)

    # ---- (7) 仮想トルク（エネルギー代理変数）----
    ax7 = fig.add_subplot(3, 3, 7)
    tau_h = KV * np.abs(dyh_opt * omega)
    tau_k = KV * np.abs(dyk_opt * omega)
    ax7.plot(phi, tau_h, 'r-',  label='τ_hip  [N·m]',  lw=1.5)
    ax7.plot(phi, tau_k, 'm--', label='τ_knee [N·m]', lw=1.5)
    shade_flight(ax7)
    ax7.set_xlabel('Phase φ [rad]')
    ax7.set_ylabel('|τ| = kv·|dq/dt| [N·m]')
    ax7.set_title('Virtual Torque (Energy Proxy)\n(kv·|dq/dt|)')
    ax7.legend(fontsize=7); ax7.grid(True)

    # ---- (8) C¹ 連続性の確認（端点拡大）----
    ax8 = fig.add_subplot(3, 3, 8)
    margin = int(len(phi) * 0.08)
    phi_head = phi[:margin]
    phi_tail = phi[-margin:] - 2*np.pi
    ax8.plot(phi_tail, y_hip_opt[-margin:],  'b--', lw=1.5, label='Hip tail (→0)')
    ax8.plot(phi_head, y_hip_opt[:margin],   'b-',  lw=1.5, label='Hip head (0→)')
    ax8.plot(phi_tail, y_knee_opt[-margin:], 'm--', lw=1.5, label='Knee tail')
    ax8.plot(phi_head, y_knee_opt[:margin],  'm-',  lw=1.5, label='Knee head')
    ax8.axvline(0, color='k', ls='--', lw=1.0, label='φ=0 boundary')
    ax8.set_xlabel('Phase φ (centered at 0)')
    ax8.set_ylabel('Angle [rad]')
    ax8.set_title('C¹ Continuity at φ=0\n(Lines should connect smoothly)')
    ax8.legend(fontsize=7); ax8.grid(True)

    # ---- (9) コスト内訳・統計テキスト ----
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    final = {k: history[k][-1] for k in history if history[k]}
    n_flight_samples = int(np.sum(flight_mask)) if flight_mask is not None else 0
    flight_ratio     = n_flight_samples / len(phi) * 100
    lines = [
        "=== Final Cost Breakdown ===",
        f"Total    : {final['best_cost']:.5f}",
        f"J_track  : {final['J_track']:.5f}",
        f"J_knee   : {final['J_knee']:.5f}",
        f"J_smooth : {final['J_smooth']:.5f}",
        f"J_act    : {final['J_actuator']:.5f}",
        f"J_flight : {final['J_flight']:.5f}",
        f"J_energy : {final['J_energy']:.5f}",
        "",
        "=== Flight Phase (from GRF_Z) ===",
        f"Samples : {n_flight_samples}/{len(phi)} ({flight_ratio:.1f}%)",
        "",
        "=== Trajectory Stats ===",
        f"Hip  : [{np.degrees(y_hip_opt.min()):.1f}, "
            f"{np.degrees(y_hip_opt.max()):.1f}]°",
        f"Knee : [{np.degrees(y_knee_opt.min()):.1f}, "
            f"{np.degrees(y_knee_opt.max()):.1f}]°",
        f"Knee max: {np.degrees(y_knee_opt.max()):.2f}° "
            + ("✅" if y_knee_opt.max() <= knee_safe_rad else "⚠️")
            + f" (safe<{np.degrees(knee_safe_rad):.1f}°)",
        "",
        "=== C¹ Continuity ===",
        f"Hip  Δ: {abs(y_hip_opt[0]-y_hip_opt[-1]):.2e} rad",
        f"Knee Δ: {abs(y_knee_opt[0]-y_knee_opt[-1]):.2e} rad",
    ]
    ax9.text(0.04, 0.97, "\n".join(lines),
             transform=ax9.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('CPG_orbit_bspline_cmaes_v2.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("図を保存しました: CPG_orbit_bspline_cmaes_v2.png")


# ============================================================
# 7. メイン
# ============================================================

def main():
    df = load_latest_csv()

    duration          = 1.325625  # 1周期の時間長 [s]（CSVデータから推定）
    start_time_offset = 5.0

    mask = ((df['Time'] >= start_time_offset) &
            (df['Time'] <  start_time_offset + duration))
    data_segment = df[mask].copy()

    if len(data_segment) == 0:
        print("指定された時間範囲のデータがありません。")
        return

    t_raw   = data_segment['Time'].values
    t_learn = t_raw - t_raw[0]
    dt      = t_learn[1] - t_learn[0]          # 時間刻み [s]
    phi     = 2 * np.pi * (t_learn / duration)  # 位相 [0, 2π]

    y_hip_ref  = data_segment['Smoothed_Target_Hip'].values.copy()
    y_knee_ref = data_segment['Smoothed_Target_Knee'].values.copy()

    # --- 周期境界処置（端点を平均値で揃える） ---
    # ※ 周期Bスプラインでは端点の明示的一致は不要だが、
    #   参照データの不連続を抑制するために実施
    y_hip_ref[0]  = y_hip_ref[-1]  = (y_hip_ref[0]  + y_hip_ref[-1])  / 2
    y_knee_ref[0] = y_knee_ref[-1] = (y_knee_ref[0] + y_knee_ref[-1]) / 2

    # ---- データ確認 ----
    print("\n=== データ範囲確認 ===")
    print(f"  Hip  : [{np.degrees(y_hip_ref.min()):.1f}, {np.degrees(y_hip_ref.max()):.1f}] deg")
    print(f"  Knee : [{np.degrees(y_knee_ref.min()):.1f}, {np.degrees(y_knee_ref.max()):.1f}] deg")

    # ============================================================
    # ★ 飛翔期フラグの自動検出（GRF_Z 列から）
    # ============================================================
    # CSVの 'GRF_Z' 列（鉛直床反力）が 0 の区間 = 飛翔期。
    # ノイズや数値誤差で完全に 0 にならない場合に備え、
    # モジュール定数 GRF_THRESHOLD 以下を「接地なし」と判定する。

    print("\n=== 飛翔期フラグ検出（GRF_Z） ===")

    if 'GRF_Z' in data_segment.columns:
        grf_z = data_segment['GRF_Z'].values

        # 生の飛翔期マスク（GRF_Z ≤ 閾値）
        flight_mask_raw = grf_z <= GRF_THRESHOLD

        # --- ノイズ除去：周期の 5% 未満の孤立区間を除去 ---
        min_duration_samples = max(3, int(len(phi) * 0.05))

        def remove_short_segments(mask, min_len):
            """min_len サンプル未満の True 区間を False に潰す"""
            result = mask.copy()
            i = 0
            while i < len(result):
                if result[i]:
                    j = i
                    while j < len(result) and result[j]:
                        j += 1
                    if (j - i) < min_len:
                        result[i:j] = False
                    i = j
                else:
                    i += 1
            return result

        flight_mask = remove_short_segments(flight_mask_raw, min_duration_samples)

        n_flight = int(np.sum(flight_mask))
        n_total  = len(phi)
        ratio    = n_flight / n_total * 100

        print(f"  GRF_Z 範囲  : min={grf_z.min():.3f}, max={grf_z.max():.3f} N")
        print(f"  閾値        : GRF_Z ≤ {GRF_THRESHOLD} N  "
              f"（変更は GRF_THRESHOLD 定数を編集）")
        print(f"  飛翔期      : {n_flight} / {n_total} サンプル ({ratio:.1f}%)")

        if n_flight > 0:
            flight_phi = phi[flight_mask]
            print(f"  飛翔期 φ範囲: [{flight_phi.min()/np.pi:.3f}π, "
                  f"{flight_phi.max()/np.pi:.3f}π]")
        else:
            print("  ⚠️  飛翔期サンプルが 0 です。GRF_THRESHOLD を大きくしてください。")
            flight_mask = None

    else:
        # GRF_Z 列が存在しない場合のフォールバック（手動指定）
        grf_z = None
        print("  ⚠️  'GRF_Z' 列が見つかりません。手動設定にフォールバックします。")
        print(f"  利用可能な列: {list(data_segment.columns)}")
        FLIGHT_PHI_START = np.pi
        FLIGHT_PHI_END   = 1.6 * np.pi
        flight_mask = (phi >= FLIGHT_PHI_START) & (phi <= FLIGHT_PHI_END)
        print(f"  フォールバック飛翔期: φ=[{FLIGHT_PHI_START/np.pi:.2f}π, "
              f"{FLIGHT_PHI_END/np.pi:.2f}π]  "
              f"({int(np.sum(flight_mask))} サンプル)")

    # ---- CMA-ES パラメータ ----
    KNEE_SAFE_RAD = KNEE_SAFE_RAD_DEFAULT   # -0.175 rad = -10 deg
    N_CTRL        = 16
    SIGMA0        = 0.10
    MAXITER       = 600

    # 重みの設計思想：
    #   J_track が常に支配的になるよう w_track を最大にする。
    #   他の項は「J_track の改善を邪魔しない」程度に設定する。
    #
    #   【診断結果を受けた修正】
    #   - w_track : 10.0 → 最重要項として強化
    #   - w_flight:  0.3 → 参照速度で正規化済みなので小さくてよい
    #                       （0.3 で飛翔期速度が参照の30%程度を目標）
    #   - w_energy:  0.1 → エネルギーは補助的に
    #   - w_knee  :  5.0 → 膝ペナルティはそのまま
    W_TRACK    = 10.0
    W_KNEE     =  5.0
    W_SMOOTH   =  1e-4
    W_ACTUATOR =  2.0
    W_FLIGHT   =  0.3
    W_ENERGY   =  0.1

    print(f"\n=== CMA-ES パラメータ ===")
    print(f"  N_CTRL={N_CTRL}, SIGMA0={SIGMA0}, MAXITER={MAXITER}")
    print(f"  w_track={W_TRACK}, w_knee={W_KNEE}, w_smooth={W_SMOOTH}")
    print(f"  w_actuator={W_ACTUATOR}, w_flight={W_FLIGHT}, w_energy={W_ENERGY}")

    theta_opt, history = optimize_with_cmaes(
        phi, dt, y_hip_ref, y_knee_ref, flight_mask,
        n_ctrl     = N_CTRL,
        sigma0     = SIGMA0,
        maxiter    = MAXITER,
        w_track    = W_TRACK,
        w_knee     = W_KNEE,
        w_smooth   = W_SMOOTH,
        w_actuator = W_ACTUATOR,
        w_flight   = W_FLIGHT,
        w_energy   = W_ENERGY,
        knee_safe_rad = KNEE_SAFE_RAD,
    )

    # ---- 最適化後の軌道・微分を生成 ----
    y_hip_opt,  dyh_opt = traj_and_deriv(theta_opt, phi, 'hip')
    y_knee_opt, dyk_opt = traj_and_deriv(theta_opt, phi, 'knee')

    # ---- 結果チェック ----
    print(f"\n=== 最適化結果チェック ===")
    print(f"  Hip  range: [{np.degrees(y_hip_opt.min()):.1f}, "
          f"{np.degrees(y_hip_opt.max()):.1f}] deg")
    print(f"  Knee range: [{np.degrees(y_knee_opt.min()):.1f}, "
          f"{np.degrees(y_knee_opt.max()):.1f}] deg")
    print(f"  C¹連続性 Hip  Δ: {abs(y_hip_opt[0]-y_hip_opt[-1]):.6f} rad")
    print(f"  C¹連続性 Knee Δ: {abs(y_knee_opt[0]-y_knee_opt[-1]):.6f} rad")
    k = KNEE_SAFE_RAD
    mk = y_knee_opt.max()
    print(f"  膝安全チェック  : max={np.degrees(mk):.2f}° "
          f"({'✅' if mk <= k else '⚠️'}  safe={np.degrees(k):.1f}°)")

    if flight_mask is not None and np.any(flight_mask):
        dq_flight_hip  = np.mean(np.abs(dyh_opt[flight_mask]))
        dq_flight_knee = np.mean(np.abs(dyk_opt[flight_mask]))
        print(f"  飛翔期平均速度  : |dHip/dφ|={dq_flight_hip:.4f}, "
              f"|dKnee/dφ|={dq_flight_knee:.4f} (小さいほどよい)")

    # ---- 可視化 ----
    plot_results(phi, dt,
                 y_hip_ref, y_knee_ref,
                 y_hip_opt, y_knee_opt,
                 dyh_opt, dyk_opt,
                 history, KNEE_SAFE_RAD,
                 theta_opt, N_CTRL, flight_mask,
                 grf_z=grf_z)

    # ---- CSV 保存 ----
    df_result = pd.DataFrame({
        'Phase':     phi,
        'Hip':       y_hip_opt,
        'Knee':      y_knee_opt,
        'dHip_dphi': dyh_opt,
        'dKnee_dphi':dyk_opt,
    })
    out_path = 'CPG_orbit_bspline_cmaes_v2.csv'
    df_result.to_csv(out_path, index=False)
    print(f"\n保存: {out_path}")
    print("列: Phase[rad], Hip[rad], Knee[rad], dHip_dphi, dKnee_dphi")


if __name__ == "__main__":
    main()