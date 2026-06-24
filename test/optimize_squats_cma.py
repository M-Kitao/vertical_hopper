import cma
import time
import math
import mujoco
import numpy as np

# --- 最適化対象のパラメータ定義 ---
# [0]: 周期 (freq) Hz
# [1]: 振幅 (amp) m
# [2]: 中心高さ (base_h) m
# [3]: キックの非対称性 (kick_ratio) 0.5=対称, 0.8=急激なキック
# [4]: スムージング率 (blend)

# 初期値 (人間が設定した「まあまあ動く」値)
initial_params = [1.5, 0.25, 0.60, 0.6, 0.1]
# パラメータの標準偏差 (探索範囲の広さ)
sigma0 = 0.2

# --- シミュレーション設定 ---
xml_path = "vertical_hopper.xml"
# 毎回モデルを読み込むコストを下げるため、バイト列としてロード
model_spec = mujoco.MjModel.from_xml_path(xml_path)

# IK関数 (前回の修正版)
def inverse_kinematics(h, L1=0.5, L2=0.4):
    h = min(max(h, 0.3), L1 + L2 - 0.001)
    val = (L1**2 + L2**2 - h**2) / (2 * L1 * L2)
    gamma = math.acos(np.clip(val, -1.0, 1.0))
    q_knee = gamma - math.pi
    val2 = (L1**2 + h**2 - L2**2) / (2 * L1 * h)
    q_hip = math.acos(np.clip(val2, -1.0, 1.0))
    return q_hip, q_knee

def get_trajectory(t_cycle, p_amp, p_base, p_ratio):
    """パラメータに基づいて、その瞬間の目標高さhを返す"""
    # 0.0 ~ 1.0 の t_cycle に応じて波形を作る
    if t_cycle < p_ratio:
        # 縮むフェーズ (ゆっくり)
        r = t_cycle / p_ratio
        # cos波の上半分を使って滑らかに下げる
        # 1 -> -1 にマッピング
        wave = math.cos(r * math.pi) 
        # wave は 1 から -1 へ。これを高さに変換
        # base + amp -> base - amp
        h = p_base + p_amp * wave
    else:
        # 伸びるフェーズ (急激に)
        r = (t_cycle - p_ratio) / (1.0 - p_ratio)
        # -1 -> 1 にマッピング (急上昇)
        wave = -math.cos(r * math.pi)
        h = p_base + p_amp * wave
    return h

# --- 評価関数 (ここをCMA-ESが呼び出す) ---
def evaluate_hopper(params):
    """
    パラメータを受け取り、シミュレーションを実行し、
    「コスト(最小化したい値)」を返す。
    高く飛びたいので、(マイナスの最大到達高度) を返す。
    """
    # パラメータの分解と制限 (物理的にありえない値をクリップ)
    freq = np.clip(params[0], 0.5, 3.0)
    amp  = np.clip(params[1], 0.05, 0.4)
    base = np.clip(params[2], 0.4, 0.7)
    ratio= np.clip(params[3], 0.1, 0.9)
    blend= np.clip(params[4], 0.01, 0.5)

    # シミュレーション初期化
    # 注意: 高速化のため viewer は使いません (ヘッドレス)
    data = mujoco.MjData(model_spec)
    
    # 安定化設定
    knee_jid = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint")
    hip_jid = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint")
    model_spec.dof_damping[model_spec.jnt_dofadr[knee_jid]] = 10.0
    model_spec.dof_damping[model_spec.jnt_dofadr[hip_jid]] = 10.0
    
    hip_act_id = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_joint")
    knee_act_id = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_joint")
    foot_site_id = mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_SITE, "foot_site")
    
    # ループ変数
    current_target_h = base
    max_z_reached = 0.0
    start_time = 0.0
    duration = 3.0  # 3秒間テストする
    
    # --- シミュレーションループ ---
    while data.time < duration:
        now = data.time
        
        # 1. 軌道生成
        cycle_pos = (now * freq) % 1.0 # 0.0 -> 1.0
        
        # 空中判定
        foot_z = data.site_xpos[foot_site_id][2]
        
        if foot_z > 0.05:
            desired_h = 0.85 # 空中では伸ばしきる
        else:
            # 接地中はパラメータに基づいた軌道
            desired_h = get_trajectory(cycle_pos, amp, base, ratio)
            
        # 2. スムージング
        current_target_h = (1 - blend) * current_target_h + blend * desired_h
        
        # 3. 制御
        th, tk = inverse_kinematics(current_target_h)
        data.ctrl[hip_act_id] = math.degrees(th)
        data.ctrl[knee_act_id] = math.degrees(tk)
        
        # X軸固定 (XMLにEqualityがない場合の保険)
        # data.qpos[model_spec.jnt_qposadr[mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_JOINT, "rootz")]] は自由
        
        mujoco.mj_step(model_spec, data)
        
        # 最大高さの記録 (Center bodyのZ座標)
        # ルートジョイントのZ位置を取得
        root_z = data.qpos[model_spec.jnt_qposadr[mujoco.mj_name2id(model_spec, mujoco.mjtObj.mjOBJ_JOINT, "rootz")]]
        if root_z > max_z_reached:
            max_z_reached = root_z

    # CMA-ESは「最小化」を行うため、高く飛んだらマイナスの値を返す
    # ペナルティ: もし高さが低すぎる(転倒/失敗)なら、大きな値を返す
    if max_z_reached < 0.25:
        return 0.0 # 悪いスコア
    
    # ★ペナルティ2: 1.0m を超えてしまった場合
    if max_z_reached > 1.0:
        # 1mを超えた分だけ、強烈なペナルティを与える
        # これによりAIは「1.0mギリギリ」を目指すようになります
        over_height = max_z_reached - 1.0
        return 5.0 + over_height * 10.0
        
    return -max_z_reached

# --- CMA-ESの実行 ---
if __name__ == "__main__":
    print("Optimization Start...")
    print(f"Initial params: {initial_params}")
    
    # 最適化設定
    es = cma.CMAEvolutionStrategy(initial_params, sigma0, {'popsize': 8, 'maxiter': 20})
    
    # 最適化ループ
    es.optimize(evaluate_hopper)
    
    # 結果表示
    best_params = es.result.xbest
    best_score = -es.result.fbest # マイナスを戻す
    
    print("\n--------------------------------")
    print("Optimization Done!")
    print(f"Best Jump Height: {best_score:.3f} m")
    print("Best Parameters:")
    print(f"  Frequency  : {best_params[0]:.2f} Hz")
    print(f"  Amplitude  : {best_params[1]:.2f} m")
    print(f"  Base Height: {best_params[2]:.2f} m")
    print(f"  Kick Ratio : {best_params[3]:.2f} (Speed balance)")
    print(f"  Smoothing  : {best_params[4]:.3f}")
    
    # --- 最良の結果で可視化実行 ---
    print("\nVisualizing best result...")
    # (ここで再度 best_params を使って viewer ありで実行するコードを書くと良い)
    # 簡易的にパラメータを表示して終了