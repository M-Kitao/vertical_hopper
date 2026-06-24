import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSVファイルの読み込み
file_path = 'CPG_orbit_bspline_cmaes_v2.csv'
df = pd.read_csv(file_path)

phi = df['Phase'].values
y_hip_pred = df['Hip'].values
y_knee_pred = df['Knee'].values

plt.figure(figsize=(7, 8))

# π単位の目盛り位置とラベルを定義
# 0, 90度, 180度, 270度, 360度
tick_pos = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
tick_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']

# --- 時間波形グラフ ---
plt.subplot(2, 1, 1)
#plt.plot(phi, y_hip, 'b-', label='Teacher (Hip)', linewidth=2, alpha=0.6)
plt.plot(phi, y_hip_pred, 'r-', label='Hip')
plt.title('Learned Hip Joint Trajectory')
plt.xlabel('Phi (rad)')
# 変更点: 横軸の設定
plt.xlim(0, 2*np.pi)           # 左右の余白を削除（0～2πに固定）
plt.xticks(tick_pos, tick_labels) # 目盛りをπ単位に変更
plt.ylabel('Angle (rad)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
#plt.plot(phi, y_knee, 'g-', label='Teacher (Knee)', linewidth=2, alpha=0.6)
plt.plot(phi, y_knee_pred, 'm-', label='Knee')
plt.title('Learned Knee Joint Trajectory')
plt.xlabel('Phi (rad)')
# 変更点: 横軸の設定
plt.xlim(0, 2*np.pi)           # 左右の余白を削除（0～2πに固定）
plt.xticks(tick_pos, tick_labels) # 目盛りをπ単位に変更
plt.ylabel('Angle (rad)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()