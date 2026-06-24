import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker

# 1. データの読み込み
filename = "rewards.csv"
df = pd.read_csv(filename)

# 2. データの整形
# omegaを数値に変換する処理（ソート用）
def convert_omega_to_float(x):
    if isinstance(x, str):
        x = x.strip()
        if x == 'pi':
            return np.pi
        elif 'pi' in x:
            try:
                return float(x.replace('pi', '')) * np.pi
            except ValueError:
                return 0.0
    return float(x)

# 数値をπを用いた文字列表記に戻す関数（ラベル表示用）
def format_omega_label(val):
    # πで割った値を計算
    coef = val / np.pi
    
    # 誤差を考慮して近い整数や半整数か判定
    if np.isclose(coef, 1.0):
        return r'$\pi$'
    elif np.isclose(coef, 0.0):
        return '0'
    elif np.isclose(coef % 1, 0): # 整数倍（例：2π, 3π）
        return f'{int(coef)}$\pi$'
    elif np.isclose(coef % 0.5, 0): # 0.5刻み（例：0.5π, 1.5π）
        return f'{coef:.1f}$\pi$'
    else:
        # それ以外は小数第2位まで表示 + πをつけるか、そのまま表示するか
        # ここでは単純に数値として表示（必要なら変更可）
        return f'{val:.2f}'

# データフレームのomegaを数値化
df['omega_val'] = df['omega'].apply(convert_omega_to_float)

# 指定した timesteps を含む列名を削除（例: '100000timesteps', '150000timesteps' を含む列を削除）
# 列名が数値や文字列のどちらでも対応できるように文字列化してチェックする
cols_to_drop = [c for c in df.columns if ('100000' in str(c) or '200000' in str(c))]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"Dropped columns: {cols_to_drop}")

# プロット対象の列を特定（num列などは除外）
# 'num/timesteps', 'init_height', 'omega', 'nofeedback', 'omega_val' 以外を対象
base_cols = ['num/timesteps', 'init_height', 'omega', 'nofeedback', 'omega_val']
target_columns = [c for c in df.columns if c not in base_cols]

# 列を2つのグループに分ける
group_normal = [c for c in target_columns if 'gainonly' not in c]
group_gainonly = [c for c in target_columns if 'gainonly' in c]

# グループごとのリストを作成
groups = [
    ('Environment 1', group_normal),
    ('Environment 2', group_gainonly)
]

# 3. 描画関数
def plot_heatmap_group(group_name, columns):
    if not columns:
        return
    
    cols_to_calc_range = columns + ['nofeedback']
    subset_data = df[cols_to_calc_range]
    g_min = subset_data.min().min()
    g_max = subset_data.max().max()

    # 行・列数の決定 (2行2列のレイアウト)
    cols_count = 2
    rows_count = 2

    fig, axes = plt.subplots(rows_count, cols_count, figsize=(14, 12))
    axes = axes.flatten()
    
    # 全体のタイトル
    #fig.suptitle(f'{group_name}', fontsize=24, fontweight='bold', y=0.98)

    # nofeedback を先頭のサブプロットに割り当てる（index 0）
    nofeedback_idx = 0
    used_indices = set()

    for i, col_name in enumerate(columns):
        ax_idx = i + 1  # reserve 0 for nofeedback
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        
        # ピボットテーブル作成
        # 行: init_height, 列: omega_val, 値: 対象列
        pivot_data = df.pivot_table(index='init_height', columns='omega_val', values=col_name)
        
        # 縦軸(height)は下から上へ増えるようにソート（デフォルトは昇順）
        pivot_data = pivot_data.sort_index(ascending=True)
        
        # 横軸のラベル用テキストを作成
        xticklabels = [format_omega_label(v) for v in pivot_data.columns]
        
        # ヒートマップ描画（カラーバーは後で共有）
        ax_h = sns.heatmap(pivot_data,
             ax=ax,
             cmap='jet',
             annot=True,
             fmt=".1f",
             annot_kws={"size": 28},
             cbar=False,
             xticklabels=xticklabels,
             yticklabels=True,
             vmin=g_min,
             vmax=g_max)

        # 軸の向きを反転（高さ方向を下から上へ）
        ax.invert_yaxis()

        # タイトルと軸ラベル
        ax.set_title(f"{col_name} timesteps", fontsize=16, fontweight='bold')
        ax.set_xlabel(r"$\omega$ [rad/s]", fontsize=13)
        ax.set_ylabel("init_height [m]", fontsize=13)

        # 横軸ラベルを回転させずに全て表示、フォントサイズを大きく
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
        ax.tick_params(axis='both', labelsize=16)

        used_indices.add(ax_idx)

    ax_nf = axes[nofeedback_idx]
    pivot_nf = df.pivot_table(index='init_height', columns='omega_val', values='nofeedback')
    pivot_nf = pivot_nf.sort_index(ascending=True)
    xticklabels_nf = [format_omega_label(v) for v in pivot_nf.columns]

    ax_h_nf = sns.heatmap(pivot_nf, ax=ax_nf, cmap='jet', annot=True, fmt=".1f", annot_kws={"size": 28},
                cbar=False, xticklabels=xticklabels_nf, yticklabels=True,
                vmin=g_min, vmax=g_max)

    ax_nf.invert_yaxis()
    ax_nf.set_title("nofeedback", fontsize=16, fontweight='bold', color='darkblue')
    ax_nf.set_xlabel(r"$\omega$ [rad/s]", fontsize=13)
    ax_nf.set_ylabel("init_height [m]", fontsize=13)
    ax_nf.set_xticklabels(ax_nf.get_xticklabels(), rotation=0, fontsize=12)
    ax_nf.tick_params(axis='both', labelsize=12)
    used_indices.add(nofeedback_idx)

    # 共有カラーバーを1本だけ追加（専用の軸を作成して配置）
    # Seaborn returns an Axes; get the QuadMesh mappable from the heatmap Axes collections
    try:
        mappable = ax_h_nf.collections[0]
    except Exception:
        mappable = None
    if mappable is not None:
        # Figure内の右側にカラーバー用の専用軸を追加（図の座標系で指定）
        cax = fig.add_axes([0.91, 0.12, 0.02, 0.83])  # [left, bottom, width, height]
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.ax.tick_params(labelsize=12)

    for j in range(len(axes)):
        if j not in used_indices:
            fig.delaxes(axes[j])
    
    # タイトルと被らないように調整（右側にカラーバー領域を確保）
        plt.subplots_adjust(top=0.95, bottom=0.12, left=0.05, right=0.9, hspace=0.40, wspace=0.18)

    # グループごとにSVGで保存
    safe_name = group_name.replace(' ', '_')
    out_fname = f"heatmap_{safe_name}.svg"
    fig.savefig(out_fname, format='svg', bbox_inches='tight')
    print(f"Saved {out_fname}")

# 4. 実行（2つのウィンドウを表示）
for name, cols in groups:
    plot_heatmap_group(name, cols)

plt.show()