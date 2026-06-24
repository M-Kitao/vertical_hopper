"""各シードでの学習曲線をプロット"""
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# RL/tensorboard_logsフォルダ内のCSVファイルを探す
tb_logs_dir = r'c:\Users\masan\Documents\vertical_hopper_2\RL\tensorboard_logs'

# 方法1: tensorboard_logsの中から探す
csv_files = glob.glob(os.path.join(tb_logs_dir, "*/events_data.csv")) if os.path.exists(tb_logs_dir) else []

# 方法2: ルートディレクトリから探す
if not csv_files:
    root_dir = r'c:\Users\masan\Documents\vertical_hopper_2'
    csv_files = glob.glob(os.path.join(root_dir, "PPO*.csv"))
    csv_files = [f for f in csv_files if 'seed' not in f]  # baseline_5seeds.csvは除外

print(f"Found CSV files: {csv_files}")

if not csv_files:
    print("Error: No training CSV files found")
    exit(1)

# データを集約
all_data = {}

for csv_file in csv_files:
    label = os.path.basename(csv_file).replace('.csv', '')
    print(f"Reading: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        
        # 列のチェック
        if 'Step' not in df.columns:
            df.rename(columns={'Step ': 'Step'}, inplace=True)
        
        all_data[label] = df
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

if not all_data:
    print("Error: No valid data loaded")
    exit(1)

# プロット
fig, axes = plt.subplots(1, len(all_data), figsize=(5*len(all_data), 4))
if len(all_data) == 1:
    axes = [axes]

for idx, (label, df) in enumerate(all_data.items()):
    ax = axes[idx]
    
    # データをプロット
    if 'Step' in df.columns and 'Value' in df.columns:
        ax.plot(df['Step'], df['Value'], linewidth=1.5, alpha=0.7)
        ax.scatter(df['Step'], df['Value'], s=20, alpha=0.5)
        
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel('Reward', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 統計情報を表示
        mean_reward = df['Value'].mean()
        max_reward = df['Value'].max()
        min_reward = df['Value'].min()
        
        stats_text = f"Mean: {mean_reward:.1f}\nMax: {max_reward:.1f}\nMin: {min_reward:.1f}"
        ax.text(0.98, 0.05, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_path = r'c:\Users\masan\Documents\vertical_hopper_2\learning_curves_from_csv.png'
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"✓ グラフを保存: {output_path}")
plt.show()
