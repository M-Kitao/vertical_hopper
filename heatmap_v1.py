import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# 1. データの読み込み
filename = "rewards.csv"
df = pd.read_csv(filename)

# 2. 軸データとプロット対象データの特定
# omegaを数値に変換（'pi'や'1.5pi'などを数値に変換）
def convert_omega(x):
    if isinstance(x, str):
        x = x.strip()
        if x == 'pi':
            return np.pi
        elif 'pi' in x:
            return float(x.replace('pi', '')) * np.pi
    return float(x)

df['omega'] = df['omega'].apply(convert_omega)

# 軸に使用するデータ
x_data = df['omega']
y_data = df['init_height']

# プロット対象の列（報酬データ）を特定
# num, init_height, omega, nofeedback 以外の列をすべて抽出
target_columns = [c for c in df.columns if c not in ['num', 'init_height', 'omega', 'nofeedback']]

# 3. サブプロットのレイアウト計算
num_plots = len(target_columns)
cols = 3  # 横に並べるグラフの数（お好みで変更可能）
rows = math.ceil(num_plots / cols)

# 4. グラフの生成
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
axes = axes.flatten() # 1次元配列にしてループしやすくする

for i, col_name in enumerate(target_columns):
    ax = axes[i]
    z_data = df[col_name]
    
    # tricontourfを使って滑らかな地形図（ヒートマップ）を描画
    # levels=100 にすることで色の階調を細かくし、「写真のような」滑らかさを出す
    contour = ax.tricontourf(x_data, y_data, z_data, levels=100, cmap='jet')
    
    # タイトル設定（条件1：列名 + " timesteps"）
    ax.set_title(f"{col_name} timesteps", fontsize=12, fontweight='bold')
    
    # 軸ラベル設定（条件2：init_heightとomega）
    ax.set_xlabel('omega')
    ax.set_ylabel('init_height')
    
    # カラーバーの追加
    fig.colorbar(contour, ax=ax, shrink=0.9)

# 余ったサブプロット領域を非表示にする（グラフ数が3の倍数でない場合など）
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()