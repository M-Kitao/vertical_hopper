"""
test.orbit_learning・・・
squats_flying.pyで得たデータをCSVに保存
教師：各関節の目標角(smoothed_target_hip, smoothed_target_knee) 10秒間から1.4秒抽出(omega=pi)
入力：目標関節角(smoothed_target_hip, smoothed_target_knee)
最小二乗法で学習させる
CMA-ESでパラメータチューニング
CPG位相(0<phi<2pi)に対応した関節角を予測できるようにする((t, phi)=(0, 0)から(2, 2pi)まで線形に変化させる)
"""

import mujoco
import math
import time
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
import pickle

# データ読み込み
data_file = 'squat_flying_data_20251209_164704.csv'  # 例: 実際のファイル名に置き換えてください
time_data = []
foot_x_data = []
foot_z_data = []
target_h_data = []
q_hip_data = []
q_knee_data = []
smoothed_target_hip_list = []
smoothed_target_knee_list = []

with open(data_file, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # ヘッダーをスキップ
    for row in csv_reader:
        time_data.append(float(row[0]))
        foot_x_data.append(float(row[1]))
        foot_z_data.append(float(row[2]))
        target_h_data.append(float(row[3]))
        q_hip_data.append(float(row[4]))
        q_knee_data.append(float(row[5]))
        smoothed_target_hip_list.append(float(row[6]))
        smoothed_target_knee_list.append(float(row[7]))

# 入力データの整形
X = np.array(list(zip(q_hip_data, q_knee_data)))  # 形状: (サンプル数, 2)
y_hip = np.array(smoothed_target_hip_list)  # 形状: (サンプル数,)
y_knee = np.array(smoothed_target_knee_list)  # 形状: (サンプル数,)

# 線形回帰モデルの学習
model_hip = LinearRegression()
model_knee = LinearRegression()
model_hip.fit(X, y_hip)
model_knee.fit(X, y_knee)
print("Hip Joint Model Coefficients:", model_hip.coef_, "Intercept:", model_hip.intercept_)
print("Knee Joint Model Coefficients:", model_knee.coef_, "Intercept:", model_knee.intercept_)

# モデルの保存
with open('hip_joint_model.pkl', 'wb') as f:
    pickle.dump(model_hip, f)

with open('knee_joint_model.pkl', 'wb') as f:
    pickle.dump(model_knee, f)

# テスト: 学習したモデルで予測
def predict_target_angles(q_hip, q_knee):
    input_data = np.array([[q_hip, q_knee]])
    pred_hip = model_hip.predict(input_data)[0]
    pred_knee = model_knee.predict(input_data)[0]
    return pred_hip, pred_knee

# テスト例
test_q_hip = 0.1
test_q_knee = -0.2
predicted_hip, predicted_knee = predict_target_angles(test_q_hip, test_q_knee)
print(f"Predicted Target Angles for q_hip={test_q_hip}, q_knee={test_q_knee} => Hip: {predicted_hip}, Knee: {predicted_knee}")
