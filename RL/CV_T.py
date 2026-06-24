"""
CV_T.py
==========
evaluate_robustness.py が出力した JSON 結果ファイルを読み込み、
CV_Tの比較グラフを生成するスクリプト。

- baseline は default と Gain-only と Torque の 3 つを対象
- evaluate_robustness.py の plot_results と同等の実装方針に統一

使用方法:
  # デフォルト（results/robustness_eval 以下の JSON を自動検出）
  python CV_T.py --input-dir results/robustness_eval

  # JSON ファイルを明示指定
  python CV_T.py \
    --json results/robustness_eval/robustness_default.json \
            results/robustness_eval/robustness_gainonly.json \
            results/robustness_eval/robustness_torque.json \
    --output-dir results/robustness_eval
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# ユーティリティ
# =============================================================================
BASELINE_LABELS = {
    "default":  "Tegotae-RL",
    "gainonly": "Gain-only",
    "torque":   "Torque",
}


def load_results_from_json(json_path: str) -> dict:
    """JSON ファイルから evaluate_robustness.py の結果を読み込む。"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_results_from_dir(input_dir: str,
                              baselines: list[str]) -> dict[str, dict]:
    """
    input_dir 以下にある robustness_<baseline>.json を自動検出して読み込む。

    Returns:
        {baseline: {condition: {metric: value, ...}, ...}, ...}
    """
    all_results = {}
    for bl in baselines:
        candidate = os.path.join(input_dir, f"robustness_{bl}.json")
        if os.path.isfile(candidate):
            all_results[bl] = load_results_from_json(candidate)
            print(f"✓ Loaded: {candidate}")
        else:
            print(f"⚠ Not found (skipped): {candidate}")
    return all_results


# =============================================================================
# プロット
# =============================================================================
def plot_CV_T_results(all_results: dict, output_dir: str) -> None:
    """
    CV_T の棒グラフを生成して保存する。

    plot_results (evaluate_robustness.py) と同等の実装:
      - 条件名の取得・ソート
      - width を baselines 数に合わせて動的に計算
      - yerr に std_cot_inv を使用
      - x 軸ラベルは中央揃え
    """
    os.makedirs(output_dir, exist_ok=True)

    # 条件名を全ベースラインにわたって収集・ソート
    all_conditions: set[str] = set()
    for baseline_results in all_results.values():
        all_conditions.update(baseline_results.keys())
    conditions = sorted(all_conditions, reverse=True)  # plot_results と同じ reverse=True

    conditions_raw = sorted(all_conditions, reverse=True)
    conditions = [f"{i+1}. {c}" for i, c in enumerate(conditions_raw)]

    baselines = list(all_results.keys())
    n_baselines = len(baselines)

    metric = "cv_t"
    std_key = "std_cv_t"        # evaluate_robustness.py の格納キーに合わせる
    title = "CV_T (Efficiency)"

    # ── plot_results と同じ width 計算 ──────────────────────────
    # plot_results では固定 0.2 を使用しているが、ここでは
    # ベースライン数に応じてグループ幅 0.8 を均等分割する方式に統一
    group_width = 0.8
    width = group_width / max(n_baselines, 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(conditions))

    for i, baseline in enumerate(baselines):
        values = []
        errors = []
        for cond_raw in conditions_raw:
            if cond_raw in all_results[baseline]:
                v = all_results[baseline][cond_raw].get(metric, 0.0)
                e = all_results[baseline][cond_raw].get(std_key, 0.0)
            else:
                v, e = 0.0, 0.0
            values.append(v)
            errors.append(e)

        label = BASELINE_LABELS.get(baseline, baseline)
        offset = (i - (n_baselines - 1) / 2) * width
        ax.bar(x + offset, values, width, label=label,
               yerr=errors, capsize=3)

    ax.set_xlabel("Condition")
    ax.set_ylabel(title)
    ax.set_title(title)
    # 中央揃え（plot_results と同じ計算式）
    ax.set_xticks(x + width * (n_baselines - 1) / 2 - (n_baselines - 1) / 2 * width)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "CV_T_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved: {plot_path}")
    plt.close()


# =============================================================================
# メイン
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CV_T 比較グラフ生成（default / gainonly / torque のみ）"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-dir", "-i",
        help="robustness_<baseline>.json が存在するディレクトリ"
    )
    group.add_argument(
        "--json", "-j", nargs="+",
        metavar="JSON_PATH",
        help="JSON ファイルを直接指定（複数可）。"
             "ファイル名から baseline 名を自動推定する。"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="results/robustness_eval",
        help="グラフの出力先ディレクトリ"
    )
    args = parser.parse_args()

    # 対象ベースライン
    target_baselines = ["default", "gainonly", "torque"]

    # --- 結果を収集 ---
    all_results: dict[str, dict] = {}

    if args.input_dir:
        all_results = collect_results_from_dir(args.input_dir, target_baselines)

    else:  # --json で直接指定
        for json_path in args.json:
            # ファイル名から baseline 名を推定: robustness_<baseline>.json
            stem = os.path.splitext(os.path.basename(json_path))[0]  # robustness_default
            bl = stem.replace("robustness_", "")
            if bl not in target_baselines:
                print(f"⚠ '{bl}' は対象ベースライン外のためスキップ: {json_path}")
                continue
            all_results[bl] = load_results_from_json(json_path)
            print(f"✓ Loaded: {json_path}")

    if not all_results:
        print("エラー: 読み込める JSON ファイルが見つかりませんでした。")
        return

    print(f"\n対象ベースライン: {list(all_results.keys())}")
    plot_CV_T_results(all_results, args.output_dir)

    print("\n完了!")


if __name__ == "__main__":
    main()