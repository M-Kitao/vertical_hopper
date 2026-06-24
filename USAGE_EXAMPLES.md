# 使用例＆実行コマンド

## 📋 概要

改修したスクリプトで以下を実現できます：

1. **複数シードでの統計的再現性** (P0)
2. **異なるベースラインとの比較**（既存手法・トルク直接出力）(P0)
3. **ロバストネステスト** - ノイズ・外乱・質量変化 (P1)
4. **結果の自動保存・可視化** (P0)

---

## 🚀 実行方法

> **注意 (v4 修正事項)**
> `train_TH_PPO_v4.py` からモデル保存ディレクトリの扱いを整理し，
> `--experiment` で生成される `results/<experiment>/models/<baseline>/seed_<N>`
> の直下に `final_model.zip` が置かれるようになりました。
> 旧バージョンでは `seed_<N>/seed_<N>` と二重になっていたため、
> 後続の堅牢性テストがファイルを見つけられずエラーになることがありました。
> このドキュメントのコマンドはすべて v4 に対応しています。

### 基本例 1：単一シードでの標準学習

```powershell
cd C:\Users\masan\Documents\vertical_hopper_2\RL
python train_TH_PPO_v3.py --experiment "PPO_baseline_v1" --timesteps 50000
```

**効果：**
- 実験名 `PPO_baseline_v1` で結果を保存
- 50,000 タイムステップ学習
- シード 0（デフォルト）で実行
- `models/PPO_baseline_v1/` 以下にモデル・正規化統計を保存

---

### 基本例 2：複数シード（5回実行・平均±σ計算）

```powershell
python train_TH_PPO_v3.py `
  --experiment "PPO_multi_seed_v1" `
  --timesteps 50000 `
  --seeds 0 1 2 3 4 `
  --output-csv "results_multi_seed.csv"
```

**効果：**
- シード 0～4 で 5 回独立学習
- 各シードの評価報酬を CSV に記録
- 終了時に「mean±std」を表示：
  ```
  Summary over seeds (5 runs):
    avg_eval mean±std = 2500.34 ± 120.56
    max_eval mean±std = 3100.21 ± 150.43
  ```

---

### 基本例 3：ベースライン比較（Tegotae vs 単純RL vs トルク直接）

```powershell
# --- Tegotae（CPG + 手応え） ---
python train_TH_PPO_v3.py `
  --experiment "compare_Tegotae" `
  --baseline default `
  --seeds 0 1 2 `
  --output-csv "compare_results.csv"

# --- 手応えなし（ゲインのみ） ---
python train_TH_PPO_v3.py `
  --experiment "compare_GainOnly" `
  --baseline gainonly `
  --seeds 0 1 2 `
  --output-csv "compare_results.csv"

# --- トルク直接出力（ベースライン） ---
python train_TH_PPO_v3.py `
  --experiment "compare_DirectTorque" `
  --baseline torque `
  --seeds 0 1 2 `
  --output-csv "compare_results.csv"
```

**結果：**
- `compare_results.csv` に全ベースラインの結果を集約
- 論文の「既存手法との比較表」を作成可能

---

### 基本例 4：ロバストネステスト（ノイズ・外乱・質量変化）

#### ノイズ耐性テスト
```powershell
python train_TH_PPO_v3.py `
  --experiment "robust_noise_0.1" `
  --baseline default `
  --noise 0.1 `
  --seeds 0 1 2 `
  --output-csv "robust_noise.csv"
```
- センサ値に標準偏差 0.1 のガウスノイズを付加

#### 質量変化テスト
```powershell
python train_TH_PPO_v3.py `
  --experiment "robust_mass_1.2x" `
  --baseline default `
  --mass-scale 1.2 `
  --seeds 0 1 2 `
  --output-csv "robust_mass.csv"
```
- 全体の質量・摩擦を 1.2 倍にスケール

#### 外乱（ランダムな左右パルス）
```powershell
python train_TH_PPO_v3.py `
  --experiment "robust_disturbance" `
  --baseline default `
  --ext-force 5.0 `
  --seeds 0 1 2 `
  --output-csv "robust_disturb.csv"
```
- 各ステップで 1% の確率に最大 5.0 N の外力パルス

#### 組み合わせ：複合ロバストテスト
```powershell
python train_TH_PPO_v3.py `
  --experiment "robust_combined" `
  --baseline default `
  --noise 0.05 `
  --mass-scale 0.9 `
  --ext-force 3.0 `
  --seeds 0 1 2 4 5 `
  --output-csv "robust_combined.csv"
```
- ノイズ + 質量 90% + 外乱を同時に適用

---

### 実践例 5：論文用・複合実験セット

```powershell
# 出力ディレクトリを作成
mkdir paper_experiments

# 1. 基準：標準環境 (5 シード)
python train_TH_PPO_v3.py `
  --experiment "paper_baseline" `
  --baseline default `
  --seeds 0 1 2 3 4 `
  --output-csv "paper_experiments/baseline_5seeds.csv" `
  --timesteps 50000

# 2. ベースライン比較
python train_TH_PPO_v3.py `
  --experiment "paper_gainonly" `
  --baseline gainonly `
  --seeds 0 1 2 3 4 `
  --output-csv "paper_experiments/gainonly_5seeds.csv" `
  --timesteps 50000

python train_TH_PPO_v3.py `
  --experiment "paper_torque" `
  --baseline torque `
  --seeds 0 1 2 3 4 `
  --output-csv "paper_experiments/torque_5seeds.csv" `
  --timesteps 50000

# 3. ロバストネステスト
python train_TH_PPO_v3.py `
  --experiment "paper_robust_noise" `
  --baseline default `
  --noise 0.1 `
  --seeds 0 1 2 `
  --output-csv "paper_experiments/robust_noise.csv"

python train_TH_PPO_v3.py `
  --experiment "paper_robust_mass_0.8" `
  --baseline default `
  --mass-scale 0.8 `
  --seeds 0 1 2 `
  --output-csv "paper_experiments/robust_mass_0.8.csv"

python train_TH_PPO_v3.py `
  --experiment "paper_robust_mass_1.2" `
  --baseline default `
  --mass-scale 1.2 `
  --seeds 0 1 2 `
  --output-csv "paper_experiments/robust_mass_1.2.csv"
```

---

## 📊 結果の集計・可視化

### 結果 CSV の構造

実行完了後、`--output-csv` で指定したファイルは以下の構造：

```
seed,final_reward,avg_eval_reward,steps,notes
0,3150.45,2850.32,50000,
1,3200.10,2900.45,50000,
2,3050.78,2750.12,50000,
```

### Python で統計処理

```python
import pandas as pd
import numpy as np

# CSV を読み込み
df = pd.read_csv("results_multi_seed.csv")

# グループ化（複数実験を混在させた場合）
for exp in df['experiment'].unique():
    sub = df[df['experiment'] == exp]
    print(f"{exp}:")
    print(f"  avg_eval: {sub['avg_eval_reward'].mean():.2f} ± {sub['avg_eval_reward'].std():.2f}")
    print(f"  max_eval: {sub['final_reward'].mean():.2f} ± {sub['final_reward'].std():.2f}")
```

### 可視化（プロット）

```powershell
# plot_results.py を使用
python plot_results.py "paper_experiments/baseline_5seeds.csv" --out baseline_plot.png
```

---

## 🔍 ハイパーパラメータ確認

各実験実行後、モデルディレクトリ内に `hyperparams.json` が保存されます：

```
models/PPO_baseline_v1/hyperparams.json
```

内容例：
```json
{
  "learning_rate": 0.0003,
  "n_steps": 2048,
  "batch_size": 64,
  "n_epochs": 10,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_range": 0.2,
  "noise": 0.0,
  "mass_scale": 1.0,
  "ext_force": 0.0,
  "total_timesteps": 50000,
  "baseline": "default"
}
```

→ 論文に掲載する「実験設定表」の各行はこれをコピペしてください。

---

## 📝 論文記述パターン

### 表1：実験設定・ハイパーパラメータ

| 項目 | 値 |
|------|-----|
| Learning Rate | 3e-4 |
| Batch Size | 64 |
| N Steps | 2048 |
| Epochs | 10 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| Total Timesteps | 50,000 |
| Seeds | 5 |

### 表2：ベースライン比較結果

| ベースライン | avg_eval (mean±std) | max_eval (mean±std) |
|--------|------------|------------|
| Tegotae (提案手法) | 2850±120 | 3100±150 |
| Gain Only | 2500±150 | 2800±180 |
| Direct Torque | 2200±200 | 2500±220 |

### 表3：ロバストネステスト

| 条件 | Reward (mean±std) |
|------|--------|
| Baseline | 2850±120 |
| Noise (σ=0.1) | 2750±140 |
| Mass ×0.8 | 2380±180 |
| Mass ×1.2 | 2420±160 |
| Disturbance (5N) | 2600±150 |

---

## 🎯 推奨実行順序

1. **P0 優先（短期）**
   - 複数シード実験（例4）
   - ベースライン比較（例3）
   - 結果の CSV 集計・表化

2. **P1 推奨（中期）**
   - ロバストネステスト各種（例4）
   - 結果をヒートマップ化

3. **P2 オプション（時間あれば）**
   - より多くのシード（7～10）
   - より多くの条件組み合わせ

---

## ⚠️ トラブルシューティング

### エラー：`Policy/Environment not found`
```
→ GymEnv/Tegotae_Policy.py の位置を確認
→ sys.path.append() で正しくパスが通っているか確認
```

### エラー：`CUDA out of memory`
```
→ device="auto" を device="cpu" に変更
→ batch_size を 32 に減らす
```

### 実行が遅い
```
→ timesteps を 10000 に減らして試す（デバッグ用）
→ render_mode="human" を外す（描画なし）
```

---

以上が主な使用例です。ご質問があればお答えします！
