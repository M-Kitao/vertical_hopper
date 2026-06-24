"""TensorBoardのeventファイルから学習曲線を抽出してプロット"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.compat.tensorflow_stub import io as tb_io

def extract_scalar_data(log_dir):
    """
    TensorBoardのeventファイルからスカラーデータを抽出
    """
    data = {}
    
    # ログディレクトリ内のすべてのeventファイルを取得
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        print(f"Warning: No event files found in {log_dir}")
        return data
    
    for event_file in event_files:
        print(f"Reading: {event_file}")
        try:
            for event in tb_io.tf_record_iterator(event_file):
                event = tf.compat.v1.Event.FromString(event)
                
                if event.summary.value:
                    for value in event.summary.value:
                        tag = value.tag
                        step = event.step
                        
                        # スカラー値を取得
                        if value.HasField('simple_value'):
                            scalar_value = value.simple_value
                            
                            if tag not in data:
                                data[tag] = {'steps': [], 'values': []}
                            
                            data[tag]['steps'].append(step)
                            data[tag]['values'].append(scalar_value)
        except Exception as e:
            print(f"Error reading {event_file}: {e}")
    
    return data

def plot_learning_curves(log_dirs, labels, output_path=None):
    """
    複数のログディレクトリから学習曲線をプロット
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # プロット対象のメトリクス
    metrics = [
        'rollout/ep_rew_mean',
        'train/value_loss',
        'train/policy_loss',
        'train/entropy_loss'
    ]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for log_dir, label in zip(log_dirs, labels):
            data = extract_scalar_data(log_dir)
            
            if metric in data:
                steps = np.array(data[metric]['steps'])
                values = np.array(data[metric]['values'])
                
                # ステップでソート
                sort_idx = np.argsort(steps)
                steps = steps[sort_idx]
                values = values[sort_idx]
                
                ax.plot(steps, values, marker='o', label=label, alpha=0.7)
            else:
                print(f"Metric '{metric}' not found in {log_dir}")
        
        ax.set_xlabel('Steps', fontsize=11)
        ax.set_ylabel(metric.split('/')[-1], fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ グラフを保存: {output_path}")
    plt.show()

if __name__ == '__main__':
    import tensorflow as tf
    
    # ログディレクトリ
    base_dir = r'c:\Users\masan\Documents\vertical_hopper_2\RL\tensorboard_logs'
    
    # 使用可能なログディレクトリを探す
    log_dirs = []
    labels = []
    
    for i in range(1, 6):
        log_dir = os.path.join(base_dir, f'{i}')
        if os.path.exists(log_dir):
            log_dirs.append(log_dir)
            labels.append(f'Seed {i}')
            print(f"Found: {log_dir}")
    
    if not log_dirs:
        print(f"No log directories found in {base_dir}")
        print("Checking PPO_logs instead...")
        ppo_base = r'c:\Users\masan\Documents\vertical_hopper_2\PPO_logs'
        for folder in glob.glob(os.path.join(ppo_base, 'PPO_vertical_hopper_*')):
            if os.path.isdir(folder):
                log_dirs.append(folder)
                labels.append(os.path.basename(folder))
    
    if log_dirs:
        output_path = r'c:\Users\masan\Documents\vertical_hopper_2\learning_curves.png'
        plot_learning_curves(log_dirs, labels, output_path)
    else:
        print("Error: No log directories found!")
