"""
plot_results.py
================
実験結果の可視化スクリプト

使い方:
    python plot_results.py --result-dir results/TH_PPO_v3_improved

生成される図:
    1. learning_curves.png  - 複数シードの学習曲線（平均±標準偏差）
    2. baseline_bar.png     - ベースライン比較棒グラフ（平均±標準偏差）
    3. robustness_heatmap.png - 堅牢性テストヒートマップ
    4. ablation_bar.png     - アブレーション解析棒グラフ（もし ablation ディレクトリがあれば）
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUIなし環境でも動く
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

# ----- カラーパレット (カラーバインドフレンドリー) -----
PALETTE = {
    "default":  "#0072B2",  # 青
    "gainonly": "#E69F00",  # オレンジ
    "torque":   "#009E73",  # 緑
    "ablation": "#CC79A7",  # ピンク
}
ALPHA_BAND = 0.25


# ============================================================
# 1. 学習曲線プロット (TensorBoardログ読み込み→npz経由)
# ============================================================
def plot_learning_curves(result_dir: str, out_path: str):
    """
    result_dir/models/<baseline>/seed_<N>/progress.csv を探して描画。
    SB3は models_dir にはなく tensorboard に書くので、
    代わりに monitor.csv (Monitor wrapper が書く) を利用する。
    """
    result_dir = Path(result_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for baseline_dir in sorted((result_dir / "models").iterdir()):
        if not baseline_dir.is_dir():
            continue
        baseline = baseline_dir.name
        color    = PALETTE.get(baseline, "#333333")

        all_curves = []
        for seed_dir in sorted(baseline_dir.iterdir()):
            monitor_csv = seed_dir / "monitor.csv"
            if not monitor_csv.exists():
                # 1階層下を探す
                cands = list(seed_dir.rglob("monitor.csv"))
                if not cands:
                    continue
                monitor_csv = cands[0]

            df = pd.read_csv(monitor_csv, skiprows=1)  # skip comment line
            if 'r' not in df.columns or 't' not in df.columns:
                continue
            # 時間軸を累積ステップへ
            df = df.sort_values('t').reset_index(drop=True)
            df['cumsteps'] = df['l'].cumsum()
            all_curves.append(df[['cumsteps', 'r']].values)

        if not all_curves:
            continue

        # 同じ cumsteps 軸に補間してから平均を取る
        max_steps = max(c[-1, 0] for c in all_curves)
        xs = np.linspace(0, max_steps, 200)
        ys = []
        for curve in all_curves:
            ys.append(np.interp(xs, curve[:, 0], curve[:, 1]))
        ys = np.array(ys)
        mean_y = ys.mean(axis=0)
        std_y  = ys.std(axis=0)

        ax.plot(xs, mean_y, color=color, label=baseline, linewidth=2)
        ax.fill_between(xs, mean_y - std_y, mean_y + std_y,
                        color=color, alpha=ALPHA_BAND)
        plotted = True

    if not plotted:
        print("[plot_learning_curves] monitor.csv が見つかりませんでした。スキップ。")
        plt.close(fig)
        return

    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel("Episode Reward", fontsize=13)
    ax.set_title("Learning Curves (mean ± 1 std)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# 2. ベースライン比較棒グラフ
# ============================================================
def plot_baseline_bar(summary_csv: str, out_path: str):
    df = pd.read_csv(summary_csv)
    if df.empty:
        print("[plot_baseline_bar] summary.csv が空です。スキップ。")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(df))
    colors = [PALETTE.get(c, "#888888") for c in df['Condition']]

    bars = ax.bar(x, df['Mean_reward'], yerr=df['Std_reward'],
                  color=colors, capsize=6, edgecolor='black', linewidth=0.8)

    # 値アノテーション
    for bar, mean, std in zip(bars, df['Mean_reward'], df['Std_reward']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 20,
                f"{mean:.0f}±{std:.0f}",
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(df['Condition'], fontsize=11)
    ax.set_ylabel("Mean Episode Reward", fontsize=13)
    ax.set_title("Baseline Comparison (mean ± 1 std, n seeds)", fontsize=13)
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# 3. 堅牢性ヒートマップ
# ============================================================
def plot_robustness_heatmap(result_dir: str, out_path: str):
    result_dir = Path(result_dir)
    rob_files = list(result_dir.glob("robustness_*.json"))
    if not rob_files:
        print("[plot_robustness_heatmap] robustness_*.json が見つかりません。スキップ。")
        return

    baselines   = []
    conditions  = None
    means_table = []

    for rob_file in sorted(rob_files):
        baseline = rob_file.stem.replace("robustness_", "")
        with open(rob_file) as f:
            data = json.load(f)
        baselines.append(baseline)
        conds = list(data.keys())
        if conditions is None:
            conditions = conds
        means_table.append([data[c]['mean'] for c in conditions])

    mat = np.array(means_table)  # shape (n_baselines, n_conditions)

    fig, ax = plt.subplots(figsize=(max(7, len(conditions) * 1.2),
                                    max(3, len(baselines) * 1.0 + 1)))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label="Cumulative Episode Reward")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha='right', fontsize=18)
    ax.set_yticks(range(len(baselines)))
    ax.set_yticklabels(baselines, fontsize=18)
    ax.set_title("Robustness Test — Mean Reward Heatmap", fontsize=16)

    # セル内に数値
    for i in range(len(baselines)):
        for j in range(len(conditions)):
            ax.text(j, i, f"{mat[i, j]:.2f}",
                    ha='center', va='center', fontsize=20,
                    color='white' if mat[i, j] < 15 else 'black')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# 4. アブレーション棒グラフ
# ============================================================
def plot_ablation(ablation_csv: str, out_path: str):
    """
    ablation_csv の形式:
    Condition,Mean_reward,Std_reward
    full_model,3200,150
    no_reaction,2800,200
    ...
    """
    if not os.path.exists(ablation_csv):
        print(f"[plot_ablation] {ablation_csv} が見つかりません。スキップ。")
        return

    df = pd.read_csv(ablation_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))

    bars = ax.bar(x, df['Mean_reward'], yerr=df['Std_reward'],
                  color=PALETTE["ablation"], capsize=6,
                  edgecolor='black', linewidth=0.8)
    for bar, mean, std in zip(bars, df['Mean_reward'], df['Std_reward']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 20,
                f"{mean:.0f}",
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(df['Condition'], rotation=20, ha='right', fontsize=18)
    ax.set_ylabel("Mean Episode Reward", fontsize=13)
    ax.set_title("Ablation Study (mean ± 1 std)", fontsize=13)
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", "-r", required=True,
                        help="例: results/TH_PPO_v3_improved")
    parser.add_argument("--ablation-csv", default=None,
                        help="アブレーション結果CSV (なければスキップ)")
    args = parser.parse_args()

    out_dir = os.path.join(args.result_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 学習曲線
    plot_learning_curves(
        result_dir=args.result_dir,
        out_path=os.path.join(out_dir, "learning_curves.png"),
    )

    # 2. ベースライン比較
    summary_csv = os.path.join(args.result_dir, "summary.csv")
    if os.path.exists(summary_csv):
        plot_baseline_bar(
            summary_csv=summary_csv,
            out_path=os.path.join(out_dir, "baseline_bar.png"),
        )

    # 3. 堅牢性ヒートマップ
    plot_robustness_heatmap(
        result_dir=args.result_dir,
        out_path=os.path.join(out_dir, "robustness_heatmap.png"),
    )

    # 4. アブレーション
    if args.ablation_csv:
        plot_ablation(
            ablation_csv=args.ablation_csv,
            out_path=os.path.join(out_dir, "ablation_bar.png"),
        )

    print(f"\n全図を {out_dir} に保存しました。")


if __name__ == "__main__":
    main()