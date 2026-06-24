"""Utility for plotting training/evaluation statistics saved to CSV by train_TH_PPO_v3.py."""
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_csv(csv_path, save_fig=None):
    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty")
        return
    seeds = df['seed'].unique()
    plt.figure(figsize=(6,4))
    # column may be named avg_eval or avg_eval_reward depending on run script version
    if 'avg_eval' in df.columns:
        col = 'avg_eval'
    elif 'avg_eval_reward' in df.columns:
        col = 'avg_eval_reward'
    else:
        raise ValueError('CSV missing avg_eval column')
    for s in seeds:
        sub = df[df['seed']==s]
        plt.plot(sub[col], label=f'seed{s}')
    plt.xlabel('run index')
    plt.ylabel('avg eval reward')
    plt.title('Evaluation rewards per seed')
    plt.legend()
    plt.grid(True)
    if save_fig:
        plt.savefig(save_fig)
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='csv file produced by training script')
    parser.add_argument('--out', '-o', help='output figure file')
    args = parser.parse_args()
    plot_csv(args.csv, args.out)
