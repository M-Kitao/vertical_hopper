"""
optimize_orbit_bspline_cmaes.py

参照軌道をBスプラインで表現し、CMA-ESで最適化する。
目的：膝接地回避 + エネルギー最小化 + 元軌道への追従

対象モデル: vertical_hopper.xml
  ボディ構成:
    center (rootz: 鉛直スライド)
    ├── hat1     : 胴体  54.5 kg
    ├── thigh    : 大腿  17.0 kg  / hip_joint  [-90, +90] deg
    └── shank    : 下腿   3.5 kg  / knee_joint [-135,  0] deg
        └── footsphere: 接地球

  アクチュエータ（position制御）:
    hip_joint  : kp=5000, kv=50, ctrlrange=[-60, +60] deg = [-1.047, +1.047] rad
    knee_joint : kp=5000, kv=50, ctrlrange=[-60,   0] deg = [-1.047,  0.000] rad

  センサー:
    foot_grf          : 足底接触力 (touch)
    hip_touch_sensor  : 胴体接触 → 転倒検知
    knee_touch_sensor : 膝接触   → 膝接地検知 ★ペナルティに利用

  膝角度の符号:
    屈曲 = 負方向 (range: -135〜0 deg / -2.356〜0 rad)
    接地リスク = 0 rad に近い（伸展しすぎ）方向
    → KNEE_SAFE_RAD より大きい（0に近い）区間にペナルティ

元コード (test_orbit_learning_v3.py) との対応：
  - データ読み込み・前処理 : ほぼ同じ
  - 軌道表現             : フーリエ級数 → Bスプライン
  - パラメータ探索        : 最小二乗法 → CMA-ES
  - 出力CSV             : CPG_orbit_fourier.csv → CPG_orbit_bspline_cmaes.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# CMA-ES ライブラリ（pip install cma）
try:
    import cma
except ImportError:
    raise ImportError("cmaライブラリが必要です。'pip install cma' を実行してください。")

# Bスプライン（scipy）
from scipy.interpolate import make_interp_spline


# ============================================================
# 0. vertical_hopper.xml から読み取ったモデル定数
# ============================================================

# --- 関節可動域 [rad] ---
HIP_RANGE_MIN  = np.radians(-90)   # -1.5708 rad
HIP_RANGE_MAX  = np.radians( 90)   #  1.5708 rad

KNEE_RANGE_MIN = np.radians(-135)  # -2.3562 rad  (完全屈曲)
KNEE_RANGE_MAX = np.radians(   0)  #  0.0000 rad  (完全伸展 = 接地リスク)

# --- アクチュエータ制御範囲 [rad] ---
# joint range より狭い → 最適化後の制御点がこの範囲を超えないよう境界制約をかける
HIP_CTRL_MIN  = np.radians(-90)    # -1.0472 rad
HIP_CTRL_MAX  = np.radians( 90)    #  1.0472 rad
KNEE_CTRL_MIN = np.radians(-135)   # -2.3562 rad
KNEE_CTRL_MAX = np.radians(   0)   #  0.0000 rad

# --- 膝接地の安全限界 [rad] ---
# knee_joint range の上限(=0)から KNEE_SAFETY_MARGIN だけ離した値
# これより大きい（= 0に近い = 伸展しすぎ）区間にペナルティ
KNEE_SAFETY_MARGIN = np.radians(10)   # 10 deg のマージン
KNEE_SAFE_RAD_DEFAULT = KNEE_RANGE_MAX - KNEE_SAFETY_MARGIN  # = -0.1745 rad

# --- 質量 [kg] ---
MASS_HAT   = 54.5
MASS_THIGH = 17.0
MASS_SHANK =  3.5
MASS_TOTAL = MASS_HAT + MASS_THIGH + MASS_SHANK  # 75.0 kg


# ============================================================
# 1. データ読み込み（元コードと同じ）
# ============================================================

def load_latest_csv(prefix='csvdata20251215/squat_flying_data_'):
    """指定したプレフィックスを持つ最新のCSVファイルを読み込む"""
    files = glob.glob(f"{prefix}*.csv")
    if not files:
        raise FileNotFoundError(
            "CSVファイルが見つかりません。squats_flying.py を実行してください。"
        )
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading data from: {latest_file}")
    return pd.read_csv(latest_file)


# ============================================================
# 2. Bスプライン軌道の表現
# ============================================================

def make_bspline_trajectory(t, control_points, degree=3):
    """
    制御点からBスプライン軌道を生成する（周期的）。

    Parameters
    ----------
    t              : ndarray, shape (N,)  正規化時間 [0, T]
    control_points : ndarray, shape (n_ctrl,)  最適化変数
    degree         : int  スプライン次数（3 = 三次）

    Returns
    -------
    y : ndarray, shape (N,)  補間された軌道
    """
    n_ctrl = len(control_points)
    T = t[-1]

    # 制御点の時刻を等間隔に配置
    t_ctrl = np.linspace(0, T, n_ctrl)

    # 周期境界条件：先頭・末尾を一致させる
    # → 制御点の最後をコピーして端点を固定しない形で扱う
    cp = np.concatenate([control_points, [control_points[0]]])
    t_ctrl_ext = np.concatenate([t_ctrl, [T + (T / n_ctrl)]])

    # make_interp_spline で Bスプラインを構築
    spline = make_interp_spline(t_ctrl_ext, cp, k=degree,
                                 bc_type=None)
    return spline(t)


def control_points_to_trajectory(theta, t, joint='hip'):
    """
    最適化変数 theta を Hip / Knee の制御点に分割して軌道を返す。

    theta の構造：
        theta[:n_ctrl]         → Hip  制御点
        theta[n_ctrl:2*n_ctrl] → Knee 制御点
    """
    n_ctrl = len(theta) // 2
    if joint == 'hip':
        cp = theta[:n_ctrl]
    else:
        cp = theta[n_ctrl:]
    return make_bspline_trajectory(t, cp, degree=3)


# ============================================================
# 3. 目的関数（CMA-ESで最小化）
# ============================================================

def objective(theta, t, y_hip_ref, y_knee_ref,
               w_track=1.0, w_knee=10.0, w_smooth=1e-4, w_actuator=5.0,
               knee_safe_rad=KNEE_SAFE_RAD_DEFAULT):
    """
    目的関数 J(theta) = J_track + J_knee + J_smooth + J_actuator

    【vertical_hopper.xml に基づく符号の約束】
      knee_joint range: -135〜0 deg (-2.356〜0 rad)
      屈曲 = 負方向 / 伸展（接地リスク）= 0 rad 方向
      → knee > knee_safe_rad（デフォルト -0.175 rad）でペナルティ

    Parameters
    ----------
    theta         : 1D array  最適化変数（HipとKneeの制御点を連結）
    t             : 時間配列
    y_hip_ref     : Hip 参照軌道  [rad]
    y_knee_ref    : Knee 参照軌道 [rad]
    w_track       : 追従誤差の重み
    w_knee        : 膝接地ペナルティの重み
    w_smooth      : 滑らかさの重み
    w_actuator    : アクチュエータ制御範囲違反の重み
                    ctrlrange: hip [-1.047,+1.047] / knee [-1.047, 0] rad
    knee_safe_rad : 膝角度の安全限界 [rad]（これより大きい = 伸展危険域）

    Returns
    -------
    J : float  コスト（小さいほど良い）
    """
    y_hip_pred  = control_points_to_trajectory(theta, t, joint='hip')
    y_knee_pred = control_points_to_trajectory(theta, t, joint='knee')

    dt = t[1] - t[0]

    # ① 追従誤差（RMSE：スケール安定）
    J_track = w_track * (
        np.sqrt(np.mean((y_hip_pred  - y_hip_ref )**2))
      + np.sqrt(np.mean((y_knee_pred - y_knee_ref)**2))
    )

    # ② 膝接地ペナルティ（2段階）
    #    ソフト: knee_safe_rad を超えた量の二乗（= 伸展しすぎ）
    #    ハード: 0 rad（完全伸展）を超えた量に大ペナルティ
    soft_viol = np.maximum(0.0, y_knee_pred - knee_safe_rad)
    hard_viol = np.maximum(0.0, y_knee_pred - KNEE_RANGE_MAX)  # 0 rad 超過
    J_knee = w_knee * (np.mean(soft_viol**2) + 20.0 * np.mean(hard_viol**2))

    # ③ 滑らかさ（加速度二乗和を参照軌道stdで正規化）
    accel_hip  = np.diff(y_hip_pred,  n=2) / dt**2
    accel_knee = np.diff(y_knee_pred, n=2) / dt**2
    scale = (np.std(y_hip_ref) + np.std(y_knee_ref)) + 1e-8
    J_smooth = w_smooth * (np.mean(accel_hip**2) + np.mean(accel_knee**2)) / scale**2

    # ④ アクチュエータ制御範囲違反ペナルティ
    #    vertical_hopper.xml: hip ctrlrange=[-60,+60]deg / knee ctrlrange=[-60,0]deg
    hip_viol  = (np.maximum(0.0, y_hip_pred  - HIP_CTRL_MAX )
               + np.maximum(0.0, HIP_CTRL_MIN  - y_hip_pred ))
    knee_viol = (np.maximum(0.0, y_knee_pred - KNEE_CTRL_MAX)
               + np.maximum(0.0, KNEE_CTRL_MIN - y_knee_pred))
    J_actuator = w_actuator * (np.mean(hip_viol**2) + np.mean(knee_viol**2))

    return J_track + J_knee + J_smooth + J_actuator


# ============================================================
# 4. 初期制御点の作成（元のフーリエ学習結果から）
# ============================================================

def make_initial_control_points(t, y_hip_ref, y_knee_ref, n_ctrl=12):
    """
    参照軌道をダウンサンプリングして初期制御点を生成する。
    CMA-ESに良い初期解を与えることで収束を大幅に加速する。
    """
    idx = np.linspace(0, len(t) - 1, n_ctrl, dtype=int)
    cp_hip  = y_hip_ref[idx]
    cp_knee = y_knee_ref[idx]
    # Hip と Knee を連結して1つのベクトルにする
    return np.concatenate([cp_hip, cp_knee])


# ============================================================
# 5. CMA-ESによる最適化
# ============================================================

def optimize_with_cmaes(t, y_hip_ref, y_knee_ref,
                         n_ctrl=12,
                         sigma0=0.10,
                         maxiter=500,
                         w_track=1.0,
                         w_knee=10.0,
                         w_smooth=1e-4,
                         w_actuator=5.0,
                         knee_safe_rad=KNEE_SAFE_RAD_DEFAULT):
    """
    CMA-ESで制御点を最適化する。

    vertical_hopper.xml に合わせた設定：
      - 境界制約：アクチュエータ ctrlrange に基づく上下限
          hip  : [-1.047, +1.047] rad  (ctrlrange=[-60,+60] deg)
          knee : [-1.047,  0.000] rad  (ctrlrange=[-60,  0] deg)
      - sigma0=0.10：Hip振幅~1rad / Knee振幅~2rad のスケールに対応

    Parameters
    ----------
    n_ctrl        : 制御点の数（12が推奨、8〜16が実用的）
    sigma0        : CMA-ESの初期探索幅（0.10推奨）
    maxiter       : 最大世代数
    w_knee        : 膝ペナルティ重み
    w_actuator    : アクチュエータ制御範囲違反の重み
    knee_safe_rad : 膝安全限界 [rad]（デフォルト: -0.175 rad = -10 deg マージン）
    """
    theta0 = make_initial_control_points(t, y_hip_ref, y_knee_ref, n_ctrl)

    # 境界制約の配列を先に作る
    lower = np.array([HIP_CTRL_MIN]  * n_ctrl + [KNEE_CTRL_MIN] * n_ctrl)
    upper = np.array([HIP_CTRL_MAX]  * n_ctrl + [KNEE_CTRL_MAX] * n_ctrl)

    # 初期解が境界外だと CMA-ES の初期化時にエラーになるため必ずクリップ
    # （参照データが ctrlrange を超えている場合に発生）
    eps = 1e-4  # 境界端ぴったりを避けるための微小オフセット
    theta0_clipped = np.clip(theta0, lower + eps, upper - eps)
    if not np.allclose(theta0, theta0_clipped):
        n_clipped = int(np.sum(~np.isclose(theta0, theta0_clipped)))
        print(f"  ⚠️  初期制御点 {n_clipped} 個が ctrlrange 外 → クリップしました")
        print(f"     Hip  clip範囲: [{np.degrees(HIP_CTRL_MIN):.1f}, "
              f"{np.degrees(HIP_CTRL_MAX):.1f}] deg")
        print(f"     Knee clip範囲: [{np.degrees(KNEE_CTRL_MIN):.1f}, "
              f"{np.degrees(KNEE_CTRL_MAX):.1f}] deg")
    theta0 = theta0_clipped

    history = {'best_cost': [], 'J_track': [], 'J_knee': [],
               'J_smooth': [], 'J_actuator': []}

    opts = cma.CMAOptions()
    opts['maxiter'] = maxiter
    opts['tolx']    = 1e-6
    opts['tolfun']  = 1e-6
    opts['verbose'] = -9
    opts['popsize'] = 4 + int(3 * np.log(len(theta0)))

    # --- 境界制約（アクチュエータ ctrlrange に基づく） ---
    # theta = [hip_cp × n_ctrl, knee_cp × n_ctrl]
    opts['bounds'] = [lower.tolist(), upper.tolist()]

    es = cma.CMAEvolutionStrategy(theta0, sigma0, opts)

    print(f"\n=== CMA-ES 最適化開始 (vertical_hopper.xml) ===")
    print(f"  制御点数      : {n_ctrl}  (変数次元 = {len(theta0)})")
    print(f"  初期探索幅    : sigma0 = {sigma0}")
    print(f"  最大世代数    : {maxiter}")
    print(f"  集団サイズ    : lambda = {es.popsize}")
    print(f"  膝安全限界    : {knee_safe_rad:.4f} rad "
          f"= {np.degrees(knee_safe_rad):.1f} deg "
          f"(joint range上限から{np.degrees(-knee_safe_rad):.1f}deg手前)")
    print(f"  Hip  制御範囲 : [{np.degrees(HIP_CTRL_MIN):.0f}, "
          f"{np.degrees(HIP_CTRL_MAX):.0f}] deg")
    print(f"  Knee 制御範囲 : [{np.degrees(KNEE_CTRL_MIN):.0f}, "
          f"{np.degrees(KNEE_CTRL_MAX):.0f}] deg")
    print("=" * 50)

    gen = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [
            objective(th, t, y_hip_ref, y_knee_ref,
                      w_track=w_track, w_knee=w_knee,
                      w_smooth=w_smooth, w_actuator=w_actuator,
                      knee_safe_rad=knee_safe_rad)
            for th in solutions
        ]
        es.tell(solutions, fitnesses)

        # ログ記録（最良個体で各項を再計算）
        best_theta = es.result.xbest
        history['best_cost'].append(min(fitnesses))

        yh = control_points_to_trajectory(best_theta, t, 'hip')
        yk = control_points_to_trajectory(best_theta, t, 'knee')
        dt = t[1] - t[0]

        history['J_track'].append(
            np.sqrt(np.mean((yh - y_hip_ref)**2))
          + np.sqrt(np.mean((yk - y_knee_ref)**2))
        )
        history['J_knee'].append(
            np.mean(np.maximum(0.0, yk - knee_safe_rad)**2)
        )
        accel_h = np.diff(yh, n=2) / dt**2
        accel_k = np.diff(yk, n=2) / dt**2
        scale = (np.std(y_hip_ref) + np.std(y_knee_ref)) + 1e-8
        history['J_smooth'].append(
            (np.mean(accel_h**2) + np.mean(accel_k**2)) / scale**2
        )
        hip_viol  = (np.maximum(0.0, yh - HIP_CTRL_MAX)
                   + np.maximum(0.0, HIP_CTRL_MIN - yh))
        knee_viol = (np.maximum(0.0, yk - KNEE_CTRL_MAX)
                   + np.maximum(0.0, KNEE_CTRL_MIN - yk))
        history['J_actuator'].append(
            np.mean(hip_viol**2) + np.mean(knee_viol**2)
        )

        gen += 1
        if gen % 50 == 0:
            print(f"  Gen {gen:4d} | Cost={history['best_cost'][-1]:.5f} "
                  f"| J_track={history['J_track'][-1]:.5f} "
                  f"| J_knee={history['J_knee'][-1]:.5f} "
                  f"| sigma={es.sigma:.4f}")

    print(f"\n最適化終了: {gen} 世代, 最終コスト = {history['best_cost'][-1]:.6f}")
    return es.result.xbest, history


# ============================================================
# 6. 可視化
# ============================================================

def plot_results(t, phi,
                 y_hip_ref, y_knee_ref,
                 y_hip_opt, y_knee_opt,
                 history, knee_safe_rad,
                 theta_opt, n_ctrl):
    """最適化結果を6パネルで可視化する"""

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        'BSpline Trajectory Optimization via CMA-ES  |  vertical_hopper.xml\n'
        f'Knee safe limit: {np.degrees(knee_safe_rad):.1f} deg  '
        f'| Hip ctrl: [{np.degrees(HIP_CTRL_MIN):.0f},{np.degrees(HIP_CTRL_MAX):.0f}] deg  '
        f'| Knee ctrl: [{np.degrees(KNEE_CTRL_MIN):.0f},{np.degrees(KNEE_CTRL_MAX):.0f}] deg',
        fontsize=11, fontweight='bold'
    )

    t_ctrl_idx = np.linspace(0, len(t) - 1, n_ctrl, dtype=int)
    y_knee_top = max(0.05, y_knee_ref.max(), y_knee_opt.max()) + 0.05

    # ---- (1) 収束曲線 ----
    ax1 = fig.add_subplot(2, 3, 1)
    gens = np.arange(1, len(history['best_cost']) + 1)
    ax1.semilogy(gens, history['best_cost'],   'k-',  label='Total Cost',  lw=2)
    ax1.semilogy(gens, history['J_track'],     'b--', label='J_track',     lw=1.5)
    ax1.semilogy(gens, history['J_knee'],      'r--', label='J_knee',      lw=1.5)
    ax1.semilogy(gens, history['J_smooth'],    'g--', label='J_smooth',    lw=1.5)
    if 'J_actuator' in history and len(history['J_actuator']) > 0:
        ax1.semilogy(gens, history['J_actuator'], 'm--', label='J_actuator', lw=1.5)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Cost (log scale)')
    ax1.set_title('Convergence Curve')
    ax1.legend(fontsize=8)
    ax1.grid(True, which='both', alpha=0.4)

    # ---- (2) Hip 軌道比較 + アクチュエータ範囲 ----
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(phi, y_hip_ref, 'b-',  label='Reference',         lw=2, alpha=0.6)
    ax2.plot(phi, y_hip_opt, 'r--', label='Optimized (BSpline)', lw=1.8)
    ax2.plot(phi[t_ctrl_idx], theta_opt[:n_ctrl], 'rs',
             markersize=6, label='Control Points')
    # アクチュエータ制御範囲をハイライト
    ax2.axhline(y=HIP_CTRL_MAX, color='purple', linestyle=':', lw=1.2,
                label=f'Ctrl max ({np.degrees(HIP_CTRL_MAX):.0f}°)')
    ax2.axhline(y=HIP_CTRL_MIN, color='purple', linestyle=':', lw=1.2,
                label=f'Ctrl min ({np.degrees(HIP_CTRL_MIN):.0f}°)')
    ax2.fill_between(phi, HIP_CTRL_MIN, HIP_CTRL_MAX,
                     color='purple', alpha=0.05, label='Ctrl range')
    ax2.set_xlabel('Phase [rad]')
    ax2.set_ylabel('Angle [rad]')
    ax2.set_title('Hip Joint Trajectory\n(Purple band = actuator ctrl range)')
    ax2.legend(fontsize=7)
    ax2.grid(True)

    # ---- (3) Knee 軌道比較 + 危険域 + アクチュエータ範囲 ----
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(phi, y_knee_ref, 'g-',  label='Reference',         lw=2, alpha=0.6)
    ax3.plot(phi, y_knee_opt, 'm--', label='Optimized (BSpline)', lw=1.8)
    ax3.plot(phi[t_ctrl_idx], theta_opt[n_ctrl:], 'ms',
             markersize=6, label='Control Points')
    ax3.axhline(y=knee_safe_rad, color='red', linestyle=':', lw=1.5,
                label=f'Safe limit ({np.degrees(knee_safe_rad):.1f}°)')
    ax3.axhline(y=KNEE_RANGE_MAX, color='darkred', linestyle='--', lw=1.2,
                label=f'Full ext. ({np.degrees(KNEE_RANGE_MAX):.0f}°)')
    ax3.axhline(y=KNEE_CTRL_MIN, color='purple', linestyle=':', lw=1.2,
                label=f'Ctrl min ({np.degrees(KNEE_CTRL_MIN):.0f}°)')
    ax3.fill_between(phi, knee_safe_rad, y_knee_top,
                     color='red', alpha=0.10, label='Danger zone')
    ax3.fill_between(phi, KNEE_CTRL_MIN, KNEE_CTRL_MAX,
                     color='purple', alpha=0.05, label='Ctrl range')
    ax3.set_xlabel('Phase [rad]')
    ax3.set_ylabel('Angle [rad]')
    ax3.set_title('Knee Joint Trajectory\n(Red = danger, Purple = ctrl range)')
    ax3.legend(fontsize=7)
    ax3.grid(True)

    # ---- (4) 関節空間軌道（フェーズポートレート）----
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(y_hip_ref, y_knee_ref, 'k-',  label='Reference Orbit', lw=3, alpha=0.3)
    ax4.plot(y_hip_opt, y_knee_opt, 'r--', label='Optimized Orbit', lw=1.8)
    ax4.axhline(y=knee_safe_rad,  color='red',    linestyle=':',  lw=1.5,
                label=f'Safe limit ({np.degrees(knee_safe_rad):.1f}°)')
    ax4.axhline(y=KNEE_RANGE_MAX, color='darkred', linestyle='--', lw=1.2,
                label='Full extension')
    x_min = min(y_hip_ref.min(), y_hip_opt.min()) - 0.05
    x_max = max(y_hip_ref.max(), y_hip_opt.max()) + 0.05
    ax4.fill_between([x_min, x_max], knee_safe_rad, y_knee_top,
                     color='red', alpha=0.08)
    ax4.set_xlabel('Hip Angle [rad]')
    ax4.set_ylabel('Knee Angle [rad]')
    ax4.set_title('Joint Space Orbit (Phase Portrait)')
    ax4.legend(fontsize=8)
    ax4.grid(True)

    # ---- (5) 膝角度詳細（危険域ハイライト）----
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(phi, y_knee_ref, 'g-',  label='Reference', lw=2, alpha=0.6)
    ax5.plot(phi, y_knee_opt, 'm--', label='Optimized', lw=1.8)
    ax5.axhline(y=knee_safe_rad,  color='red',     linestyle=':',  lw=1.5,
                label=f'Safe limit ({np.degrees(knee_safe_rad):.1f}°)')
    ax5.axhline(y=KNEE_RANGE_MAX, color='darkred',  linestyle='--', lw=1.2,
                label='Full extension (0°)')
    ax5.fill_between(phi, knee_safe_rad, y_knee_top,
                     color='red', alpha=0.12, label='Danger zone')
    danger_ref = y_knee_ref > knee_safe_rad
    if np.any(danger_ref):
        ax5.fill_between(phi, y_knee_ref.min() - 0.05, y_knee_top,
                         where=danger_ref, color='orange', alpha=0.2,
                         label='Ref in danger zone')
    ax5.set_xlabel('Phase [rad]')
    ax5.set_ylabel('Knee Angle [rad]')
    ax5.set_title('Knee Angle Detail\n(Orange = reference violated margin)')
    ax5.legend(fontsize=8)
    ax5.grid(True)

    # ---- (6) 制御点 + アクチュエータ範囲境界 ----
    ax6 = fig.add_subplot(2, 3, 6)
    ctrl_x = np.linspace(0, 2 * np.pi, n_ctrl)
    ax6.stem(ctrl_x, theta_opt[:n_ctrl],
             linefmt='b-', markerfmt='bs', basefmt='b--',
             label='Hip ctrl pts')
    ax6.stem(ctrl_x, theta_opt[n_ctrl:],
             linefmt='g-', markerfmt='gs', basefmt='g--',
             label='Knee ctrl pts')
    # アクチュエータ制御範囲
    for val, lbl, col in [
        (HIP_CTRL_MAX,  f'Hip max ({np.degrees(HIP_CTRL_MAX):.0f}°)',  'blue'),
        (HIP_CTRL_MIN,  f'Hip min ({np.degrees(HIP_CTRL_MIN):.0f}°)',  'blue'),
        (KNEE_CTRL_MAX, f'Knee max ({np.degrees(KNEE_CTRL_MAX):.0f}°)', 'green'),
        (KNEE_CTRL_MIN, f'Knee min ({np.degrees(KNEE_CTRL_MIN):.0f}°)', 'green'),
        (knee_safe_rad, f'Knee safe ({np.degrees(knee_safe_rad):.1f}°)', 'red'),
    ]:
        ax6.axhline(y=val, color=col, linestyle=':', lw=1.0, alpha=0.7, label=lbl)
    ax6.set_xlabel('Phase [rad]')
    ax6.set_ylabel('Control Point Value [rad]')
    ax6.set_title('Optimized Control Points\n(Dotted lines = actuator limits)')
    ax6.legend(fontsize=7)
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig('CPG_orbit_bspline_cmaes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("図を保存しました: CPG_orbit_bspline_cmaes.png")


# ============================================================
# 7. メイン
# ============================================================

def main():
    # ---- データ読み込み（元コードと同じ） ----
    df = load_latest_csv()

    duration          = 1.35625
    start_time_offset = 5.0

    mask = ((df['Time'] >= start_time_offset) &
            (df['Time'] <  start_time_offset + duration))
    data_segment = df[mask].copy()

    if len(data_segment) == 0:
        print("指定された時間範囲のデータがありません。")
        return

    t_raw   = data_segment['Time'].values
    t_learn = t_raw - t_raw[0]
    phi     = 2 * np.pi * (t_learn / duration)

    y_hip_ref  = data_segment['Smoothed_Target_Hip'].values.copy()
    y_knee_ref = data_segment['Smoothed_Target_Knee'].values.copy()

    # 周期境界の簡易処置（元コードと同じ）
    y_hip_ref[0]  = y_hip_ref[-1]  = (y_hip_ref[0]  + y_hip_ref[-1])  / 2
    y_knee_ref[0] = y_knee_ref[-1] = (y_knee_ref[0] + y_knee_ref[-1]) / 2

    # ---- データ範囲確認 ----
    print("\n=== データ範囲確認 ===")
    print(f"  Hip  ref : min={y_hip_ref.min():.4f} rad ({np.degrees(y_hip_ref.min()):.1f}°), "
          f"max={y_hip_ref.max():.4f} rad ({np.degrees(y_hip_ref.max()):.1f}°)")
    print(f"  Knee ref : min={y_knee_ref.min():.4f} rad ({np.degrees(y_knee_ref.min()):.1f}°), "
          f"max={y_knee_ref.max():.4f} rad ({np.degrees(y_knee_ref.max()):.1f}°)")

    # モデルの関節可動域との整合性チェック
    print("\n=== 関節可動域チェック (vertical_hopper.xml) ===")
    for name, val, lo, hi in [
        ('Hip  min', y_hip_ref.min(),  HIP_RANGE_MIN,  HIP_RANGE_MAX),
        ('Hip  max', y_hip_ref.max(),  HIP_RANGE_MIN,  HIP_RANGE_MAX),
        ('Knee min', y_knee_ref.min(), KNEE_RANGE_MIN, KNEE_RANGE_MAX),
        ('Knee max', y_knee_ref.max(), KNEE_RANGE_MIN, KNEE_RANGE_MAX),
    ]:
        ok = lo <= val <= hi
        print(f"  {name}: {val:.4f} rad  {'✅' if ok else '⚠️ 範囲外'}"
              f"  (joint range [{lo:.3f}, {hi:.3f}])")

    # ---- CMA-ES パラメータ設定 ----
    # KNEE_SAFE_RAD はモデル定数から自動計算（= -0.175 rad = -10 deg マージン）
    # より厳しくしたい場合は KNEE_SAFETY_MARGIN を大きくする
    KNEE_SAFE_RAD = KNEE_SAFE_RAD_DEFAULT
    N_CTRL        = 16      # 制御点数
    SIGMA0        = 0.10    # Hip~1rad / Knee~2rad のスケールに合わせた探索幅
    MAXITER       = 500
    W_KNEE        = 10.0
    W_ACTUATOR    = 5.0

    print(f"\n=== CMA-ES パラメータ ===")
    print(f"  KNEE_SAFE_RAD : {KNEE_SAFE_RAD:.4f} rad = {np.degrees(KNEE_SAFE_RAD):.1f} deg")
    print(f"  N_CTRL        : {N_CTRL}")
    print(f"  SIGMA0        : {SIGMA0}")
    print(f"  W_KNEE        : {W_KNEE}  (膝ペナルティ重み)")
    print(f"  W_ACTUATOR    : {W_ACTUATOR}  (アクチュエータ制約重み)")

    theta_opt, history = optimize_with_cmaes(
        t_learn, y_hip_ref, y_knee_ref,
        n_ctrl        = N_CTRL,
        sigma0        = SIGMA0,
        maxiter       = MAXITER,
        w_track       = 1.0,
        w_knee        = W_KNEE,
        w_smooth      = 1e-4,
        w_actuator    = W_ACTUATOR,
        knee_safe_rad = KNEE_SAFE_RAD,
    )

    # ---- 最適化後の軌道を生成 ----
    y_hip_opt  = control_points_to_trajectory(theta_opt, t_learn, 'hip')
    y_knee_opt = control_points_to_trajectory(theta_opt, t_learn, 'knee')

    # ---- 結果チェック ----
    print(f"\n=== 最適化結果チェック ===")
    checks = [
        ('Hip  min',  y_hip_opt.min(),  HIP_CTRL_MIN,  HIP_CTRL_MAX),
        ('Hip  max',  y_hip_opt.max(),  HIP_CTRL_MIN,  HIP_CTRL_MAX),
        ('Knee min',  y_knee_opt.min(), KNEE_CTRL_MIN, KNEE_CTRL_MAX),
        ('Knee max',  y_knee_opt.max(), KNEE_CTRL_MIN, KNEE_CTRL_MAX),
    ]
    for name, val, lo, hi in checks:
        ok = lo <= val <= hi
        print(f"  {name}: {val:.4f} rad ({np.degrees(val):.1f}°)  "
              f"{'✅ ctrl range内' if ok else '⚠️ ctrl range外'}")

    max_knee_opt = y_knee_opt.max()
    if max_knee_opt <= KNEE_SAFE_RAD:
        print(f"  膝接地リスク : ✅ なし  (max={np.degrees(max_knee_opt):.1f}° < "
              f"safe={np.degrees(KNEE_SAFE_RAD):.1f}°)")
    elif max_knee_opt <= KNEE_RANGE_MAX:
        print(f"  膝接地リスク : ⚠️ マージン超過  (max={np.degrees(max_knee_opt):.1f}°)")
    else:
        print(f"  膝接地リスク : ❌ 完全伸展  → W_KNEE を大きくしてください")

    # ---- 可視化 ----
    plot_results(t_learn, phi,
                 y_hip_ref,  y_knee_ref,
                 y_hip_opt,  y_knee_opt,
                 history, KNEE_SAFE_RAD,
                 theta_opt, N_CTRL)

    # ---- CSV 保存（元コードと同じ形式） ----
    df_result = pd.DataFrame({
        'Phase': phi,
        'Hip':   y_hip_opt,
        'Knee':  y_knee_opt,
    })
    result_csv_path = 'CPG_orbit_bspline_cmaes.csv'
    df_result.to_csv(result_csv_path, index=False)
    print(f"\n最適化軌道を保存しました: {result_csv_path}")
    print("列構成: Phase [rad], Hip [rad], Knee [rad]")
    print(f"  ※ Hip  ctrlrange: [{np.degrees(HIP_CTRL_MIN):.0f}, {np.degrees(HIP_CTRL_MAX):.0f}] deg")
    print(f"  ※ Knee ctrlrange: [{np.degrees(KNEE_CTRL_MIN):.0f}, {np.degrees(KNEE_CTRL_MAX):.0f}] deg")


if __name__ == "__main__":
    main()