import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def load_latest_csv(prefix='csvdata20251215\squat_flying_data_'):
    """指定したプレフィックスを持つ最新のCSVファイルを探して読み込む"""
    files = glob.glob(f"{prefix}*.csv")
    if not files:
        raise FileNotFoundError("CSVファイルが見つかりません。squats_flying.pyを実行してください。")
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading data from: {latest_file}")
    return pd.read_csv(latest_file)

def make_design_matrix(t, omega, order=3):
    """
    最小二乗法のための計画行列（デザイン行列）を作成する
    基底関数: 1, cos(wt), sin(wt), cos(2wt), sin(2wt), ...
    """
    # 列ベクトルに変換
    t = t[:, np.newaxis]
    
    # バイアス項 (切片)
    X = np.ones_like(t)
    
    for k in range(1, order + 1):
        X = np.hstack([X, np.cos(k * omega * t), np.sin(k * omega * t)])
        
    return X

#最良の角振動数を探索する関数
def find_optimal_omega(t, y, initial_omega, search_range=5.0, steps=100):
    best_omega = initial_omega
    min_residual = float('inf')
    
    # initial_omega の前後を探る
    test_omegas = np.linspace(initial_omega - search_range, initial_omega + search_range, steps)
    
    for w in test_omegas:
        X = make_design_matrix(t, w, order=5) # 探索用は次数低めでOK
        _, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        
        # 残差の合計
        res_sum = residuals[0] if len(residuals) > 0 else 0
        
        if res_sum < min_residual:
            min_residual = res_sum
            best_omega = w
            
    return best_omega

def main():
    # 1. データの読み込み
    df = load_latest_csv()

    
    
    # 2. データの前処理と抽出
    # 安定していそうな時間帯（例: 3秒目以降）から1.4秒分を切り出す
    duration = 1.35625
    start_time_offset = 5.0

    # データ抽出フィルタ
    mask = (df['Time'] >= start_time_offset) & (df['Time'] < start_time_offset + duration)
    data_segment = df[mask].copy()
    
    if len(data_segment) == 0:
        print("指定された時間範囲のデータがありません。")
        return

    # 時間を0から始まるように正規化（学習のため）
    t_raw = data_segment['Time'].values
    t_learn = t_raw - t_raw[0] 
    phi = 2 * np.pi * (t_learn / duration)  # 正規化された位相 [0, 2pi]
    
    # 教師データ (Target Angles)
    y_hip = data_segment['Smoothed_Target_Hip'].values
    y_knee = data_segment['Smoothed_Target_Knee'].values

    # 3. 学習設定 (Least Squares)
    omega = 2 * np.pi / duration  # ユーザー指定の角振動数
    basis_order = 20 # フーリエ級数の次数（高いほど複雑な波形にフィット）
    optimal_omega = find_optimal_omega(t_learn, y_hip, initial_omega=2*np.pi)
    #print(f"最適化されたOmega: {optimal_omega}")

    # 計画行列 X の作成
    X = make_design_matrix(t_learn, omega, order=basis_order)

    # 最小二乗法の実行
    # w = (X^T X)^-1 X^T y
    # numpy.linalg.lstsq はこれを安定して解く関数です
    w_hip, residuals_hip, rank_hip, s_hip = np.linalg.lstsq(X, y_hip, rcond=None)
    w_knee, residuals_knee, rank_knee, s_knee = np.linalg.lstsq(X, y_knee, rcond=None)

    print("学習完了")
    print(f"Hip Weights: {w_hip[:3]}... (Total {len(w_hip)} weights)")
    print(f"Knee Weights: {w_knee[:3]}...")
    #phi=0, 2pi のときの出力値を確認
    y_hip_start = X[0, :] @ w_hip
    y_knee_start = X[0, :] @ w_knee
    print(f"At phi=0: Hip={y_hip_start:.4f}, Knee={y_knee_start:.4f}")
    y_hip_end = X[-1, :] @ w_hip
    y_knee_end = X[-1, :] @ w_knee
    print(f"At phi=2pi: Hip={y_hip_end:.4f}, Knee={y_knee_end:.4f}")

    # 4. 学習結果の再現（予測）
    y_hip_pred = X @ w_hip
    y_knee_pred = X @ w_knee

    # 5. 結果のプロット
    plt.figure(figsize=(14, 8))

    # --- 時間波形グラフ ---
    plt.subplot(2, 2, 1)
    plt.plot(phi, y_hip, 'b-', label='Teacher (Hip)', linewidth=2, alpha=0.6)
    plt.plot(phi, y_hip_pred, 'r--', label='Learned (Hip)')
    plt.title('Hip Joint Trajectory Learning')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(phi, y_knee, 'g-', label='Teacher (Knee)', linewidth=2, alpha=0.6)
    plt.plot(phi, y_knee_pred, 'm--', label='Learned (Knee)')
    plt.title('Knee Joint Trajectory Learning')
    plt.xlabel('Time (s) [normalized]')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.legend()

    # --- 軌道（Orbit）グラフ: ヒップ vs ニー ---
    plt.subplot(1, 2, 2)
    plt.plot(y_hip, y_knee, 'k-', label='Teacher Orbit', linewidth=3, alpha=0.3)
    plt.plot(y_hip_pred, y_knee_pred, 'r--', label='Learned Orbit', linewidth=1.5)
    plt.title(f'Joint Space Orbit (Phase Portrait)\nomega={omega:.2f}, duration={duration}s')
    plt.xlabel('Hip Angle (rad)')
    plt.ylabel('Knee Angle (rad)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    #[phi, y_hip_pred, y_knee_pred]として改めて保存
    df_result = pd.DataFrame({
        'Phase': phi,
        'Hip': y_hip_pred,
        'Knee': y_knee_pred
    })
    result_csv_path = 'CPG_orbit_fourier_order20.csv'
    df_result.to_csv(result_csv_path, index=False)

if __name__ == "__main__":
    main()
    