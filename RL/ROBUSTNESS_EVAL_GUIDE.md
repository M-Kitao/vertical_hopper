# ロバスト性評価スクリプト使用ガイド

## 概要

`evaluate_robustness.py` は学習済みPPOモデルをさまざまな環境条件下でテストし、ロバスト性を評価するスタンドアロンスクリプトです。

**主な機能:**
- 複数の堅牢性条件をテスト（ノイズ、質量変化、外乱など）
- CSV、JSON、Markdown、matplotlib グラフで結果出力
- 複数ベースラインの一括評価・比較
- CoT⁻¹、跳躍高さ、姿勢誤差など複合指標を集計

---

## インストール

前提条件:
- Python 3.9+
- Stable-Baselines3
- gymnasium
- pandas
- matplotlib
- numpy

依存パッケージは `requirements.txt` からインストール:
```bash
pip install -r requirements.txt
```

---

## 使用方法

### 1. 単一ベースラインの評価

```bash
python evaluate_robustness.py \
  --baseline default \
  --model-dir /path/to/TH_PPO_models/default/seed_0 \
  --output-dir results/robustness_eval \
  --trials 5
```

**オプション:**
- `--baseline`: ベースライン選択 (default, gainonly, torque, nofeedback)
- `--model-dir`: 学習済みモデルのディレクトリ（`final_model.zip` と `vec_normalize.pkl` が含まれる）
- `--output-dir`: 結果保存先（デフォルト: `results/robustness_eval`）
- `--trials`: 各条件でのテスト試行回数（デフォルト: 5）

### 2. 複数ベースラインの比較評価

```bash
python evaluate_robustness.py \
  --baseline default gainonly torque \
  --compare \
  --output-dir results/robustness_comparison \
  --trials 5
```

`--compare` フラグを使用すると、スクリプト内の `baseline_dirs` 辞書で定義されたパスから自動的にモデルを探します。

---

## 出力形式

評価完了後、出力ディレクトリに以下の形式で結果が保存されます:

### CSV (`robustness_{baseline}.csv`)
```
Condition,mean,std,min,max,mean_cot_inv,std_cot_inv,mean_max_height,...
nominal,234.5,12.3,205.6,258.2,0.8421,0.0234,0.45,0.015,...
noise_low,231.2,14.1,198.3,255.0,0.8341,0.0251,0.443,0.018,...
...
```

### JSON (`robustness_{baseline}.json`)
```json
{
  "nominal": {
    "mean": 234.5,
    "std": 12.3,
    "mean_cot_inv": 0.8421,
    "std_cot_inv": 0.0234,
    "mean_max_height": 0.45,
    ...
  },
  ...
}
```

### Markdown テーブル (`robustness_{baseline}.md`)
Summary テーブル（主要指標）と Detailed テーブル（全指標）が含まれます。

### グラフ (`robustness_comparison.png`) ※複数ベースライン比較時
複数ベースラインの結果を以下の指標で比較するグラフを生成:
- 累積報酬（mean）
- CoT⁻¹（効率性）
- 最大跳躍高さ
- 高さ変動係数（CV_h）
- 周期変動係数（CV_T）
- Peak GRF（地面反力）
- 早期終了率（termination rate）

---

## テストされる堅牢性条件

| 条件名 | ノイズ | 質量 | 外乱 | 説明 |
|---|---|---|---|---|
| nominal | 0% | 100% | 0N | 基準条件 |
| noise_low | 1% | 100% | 0N | 低ノイズ |
| noise_high | 5% | 100% | 0N | 高ノイズ |
| mass_light | 0% | 80% | 0N | 軽量化（-20%） |
| mass_heavy | 0% | 120% | 0N | 重量化（+20%） |
| ext_force_moderate | 0% | 100% | 10N | 中程度の外乱 |
| ext_force_large | 0% | 100% | 30N | 大きな外乱 |
| combined | 2% | 110% | 10N | 複合条件（ノイズ+質量+外乱） |

---

## 出力指標の説明

### 基本指標
- **mean**: エピソード当たりの平均累積報酬
- **std**: 報酬の標準偏差
- **min / max**: 報酬の最小・最大値

### 効率性指標
- **mean_cot_inv**: Cost of Transport の逆数（高いほど効率的）
  - $\text{CoT}^{-1} = \frac{m g h}{E}$（m=質量, g=重力加速度, h=跳躍高さ, E=エネルギー消費）

### 安定性指標
- **mean_max_height**: 平均最大跳躍高さ [m]
- **cv_h**: 跳躍高さの変動係数（低いほど安定）
  - $\text{CV}_h = \frac{\sigma(h)}{\bar{h}}$
- **cv_t**: 跳躍周期の変動係数（低いほど周期が安定）
  - $\text{CV}_T = \frac{\sigma(T)}{\bar{T}}$

### 物理指標
- **mean_pose_error**: 目標姿勢との平均誤差
- **mean_interval**: 着地間隔の平均 [s]
- **mean_peak_grf**: 最大地面反力の平均 [N]

### 早期終了
- **term_rate**: 早期終了（転倒など）の発生率（0=なし, 1=常に転倒）

---

## 具体的な実行例

### 例1: default ベースラインの単一評価

```bash
cd /mnt/ssd1/MKitao/vertical_hopper_3/RL

python evaluate_robustness.py \
  --baseline default \
  --model-dir results/TH_v7_tegotae/models/default/seed_0 \
  --output-dir results/robustness_default \
  --trials 5
```

### 例2: 全ベースラインの一括比較

```bash
cd /mnt/ssd1/MKitao/vertical_hopper_3/RL

python evaluate_robustness.py \
  --baseline default gainonly torque nofeedback \
  --compare \
  --output-dir results/robustness_comparison \
  --trials 10
```

### 例3: gainonly ベースラインのみ詳細評価（20試行）

```bash
python evaluate_robustness.py \
  --baseline gainonly \
  --model-dir results/TH_v7_tegotae/models/gainonly/seed_0 \
  --output-dir results/robustness_gainonly_detailed \
  --trials 20
```

---

## トラブルシューティング

### エラー: `Model not found: ...final_model.zip`
- モデルディレクトリが正しいか確認してください
- `train_TH_PPO_v4.py` で学習を完了していることを確認してください

### エラー: `Stats not found: ...vec_normalize.pkl`
- VecNormalize の統計ファイルが学習時に保存されたことを確認してください
- 必要に応じて `train_TH_PPO_v4.py` を再実行してください

### メモリ不足エラー
- `--trials` の値を減らしてください（例: `--trials 3`）
- または複数のプロセスで分割実行してください

### 結果がすべて0または NaN
- 環境のインポートが正しいか確認してください（GymEnv, NN, CPG フォルダ）
- モデルと環境が一致しているか確認してください

---

## カスタマイズ

### 堅牢性条件の追加・変更

[evaluate_robustness.py](evaluate_robustness.py) の `evaluate_robustness()` 関数内で `conditions` 辞書を編集:

```python
conditions = {
    "nominal":            dict(noise_std=0.0,  mass_scale=1.0,  ext_force=0.0),
    # 新しい条件を追加
    "custom_condition":   dict(noise_std=0.03, mass_scale=0.9,  ext_force=5.0),
    ...
}
```

### グラフ出力の修正

`plot_results()` 関数内の `metrics_to_plot` リストを変更して、プロットする指標をカスタマイズできます。

---

## 参考

- 元のトレーニングスクリプト: [train_TH_PPO_v4.py](train_TH_PPO_v4.py)
- ロバスト性テスト実装: `train_TH_PPO_v4.py` の `robustness_test()` 関数を参考に作成
