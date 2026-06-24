"""
test.csv_investigate の Docstring
CSVファイルからいろいろデータを取り出してみようというテストコード
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv_data(file_path):
    """
    指定されたCSVファイルからデータを読み込み、辞書形式で返す。
    各列のヘッダーをキー、列データを値とする。
    """
    data = {}
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                data[key].append(float(value))
    return data

#跳躍のピーク時間の間隔を計算する関数
def calculate_jump_intervals(foot_z_data, time_data, threshold=0.15):
    """
    足のZ位置データから跳躍のピーク時間の間隔を計算する。
    threshold: 跳躍とみなすZ位置の閾値
    """
    jump_times = []
    for i in range(1, len(foot_z_data)-1):
        if foot_z_data[i] > threshold and foot_z_data[i] > foot_z_data[i-1] and foot_z_data[i] > foot_z_data[i+1]:
            jump_times.append(time_data[i])
    jump_intervals = np.diff(jump_times)
    return jump_intervals

if __name__ == "__main__":
    csv_file_path = 'original_trajectory/omega_100.csv'  # 解析するCSVファイルのパスを指定
    data = read_csv_data(csv_file_path)
    time_data = data['Time']
    foot_x_data = data['Foot_X']
    foot_z_data = data['Foot_Z']
    target_h_data = data['Target_H']
    q_hip_data = data['Q_Hip']
    q_knee_data = data['Q_Knee']
    smoothed_target_hip_list = data['Smoothed_Target_Hip']
    smoothed_target_knee_list = data['Smoothed_Target_Knee']
    grf_list = data['GRF_Z']

    # 跳躍のピーク時間の間隔を計算
    jump_intervals = calculate_jump_intervals(foot_z_data, time_data)
    print("Jump Intervals (s):", jump_intervals)
    print("Average Jump Interval (s):", np.mean(jump_intervals[2:]) if len(jump_intervals) > 0 else "N/A")
    print("Number of Jumps Detected:", len(jump_intervals) + 1)
    