"""報酬遷移のグラフをプロット"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# baseline_5seeds.csvを読み込み
csv_path = r'c:\Users\masan\Documents\vertical_hopper_2\RL\paper_experiments\baseline_5seeds.csv'
df = pd.read_csv(csv_path)

print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nData preview:")
print(df.head(10))

# グループ化（ステップ数ごと）
step_groups = df.groupby('steps')

# 図を作成
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. final_reward の推移
ax = axes[0]
for steps_val, group in step_groups:
    x = np.arange(len(group))
    rewards = group['final_reward'].values
    ax.plot(x, rewards, marker='o', label=f'Steps: {steps_val}', linewidth=2, markersize=6)

ax.set_xlabel('Seed Index', fontsize=12)
ax.set_ylabel('Final Reward', fontsize=12)
ax.set_title('Final Reward by Seed', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. avg_eval_reward の推移
ax = axes[1]
for steps_val, group in step_groups:
    x = np.arange(len(group))
    rewards = group['avg_eval_reward'].values
    ax.plot(x, rewards, marker='s', label=f'Steps: {steps_val}', linewidth=2, markersize=6)

ax.set_xlabel('Seed Index', fontsize=12)
ax.set_ylabel('Average Evaluation Reward', fontsize=12)
ax.set_title('Average Evaluation Reward by Seed', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = r'c:\Users\masan\Documents\vertical_hopper_2\reward_transition.png'
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"\n✓ グラフを保存: {output_path}")

# 統計情報を表示
print("\n=== 統計情報 ===")
print("\n250,000ステップの結果:")
data_250k = df[df['steps'] == 250000]
print(f"  Final Reward:  平均={data_250k['final_reward'].mean():.3f}, 標準偏差={data_250k['final_reward'].std():.3f}")
print(f"  Eval Reward:   平均={data_250k['avg_eval_reward'].mean():.3f}, 標準偏差={data_250k['avg_eval_reward'].std():.3f}")

print("\n50,000ステップの結果:")
data_50k = df[df['steps'] == 50000]
print(f"  Final Reward:  平均={data_50k['final_reward'].mean():.3f}, 標準偏差={data_50k['final_reward'].std():.3f}")
print(f"  Eval Reward:   平均={data_50k['avg_eval_reward'].mean():.3f}, 標準偏差={data_50k['avg_eval_reward'].std():.3f}")

plt.show()
