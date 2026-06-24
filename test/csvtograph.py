import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
file_path = 'original_trajectory/omega_050.csv'
df = pd.read_csv(file_path)

# プロット用にデータを抽出
# CSVのヘッダー名に基づいて変数を割り当てます
time_data = df['Time']
q_hip_data = df['Q_Hip']
smoothed_target_hip_list = df['Smoothed_Target_Hip']
q_knee_data = df['Q_Knee']
smoothed_target_knee_list = df['Smoothed_Target_Knee']

# --- ご指定のプロットコード ---
plt.figure(figsize=(12, 6))

# 1. Hip Joint Angle
plt.subplot(2, 1, 1)
plt.plot(time_data, q_hip_data, label='Hip Joint Angle')
plt.plot(time_data, smoothed_target_hip_list, 'g--', label='Target Hip Angle')
plt.title('Hip Joint Angle Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

# 2. Knee Joint Angle
plt.subplot(2, 1, 2)
plt.plot(time_data, q_knee_data, label='Knee Joint Angle', color='orange')
plt.plot(time_data, smoothed_target_knee_list, 'r--', label='Target Knee Angle')
plt.title('Knee Joint Angle Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

# レイアウトの自動調整（グラフの重なりを防ぐため）
plt.tight_layout()

# グラフの表示
plt.show()