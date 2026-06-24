"""
CPG_orbit_eddited.csv に合わせた修正版
"""

import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class KuramotoCPG_gainonly(torch.nn.Module):
    def __init__(self,
                 csv_path,
                 omega,#=2 *torch.pi / 1.35625,
                 dt=0.01,
                 device='cpu'):
        """
        Args:
            csv_path (str): 軌道データ(Phase, Hip, Knee)のCSVパス
            omega (float): 基本角振動数 (rad/s)
            dt (float): タイムステップ
            device (str): torchのデバイス ('cpu' or 'cuda')
        """
        self.omega = omega
        self.dt = dt
        self.device = device
        self.phi = 0.0  # 初期位相
        self.phi_dot = self.omega  # 位相速度の初期値

        # 軌道データの読み込み
        self.load_orbit_data(csv_path)

    def load_orbit_data(self, csv_path):
        try:
            df = pd.read_csv(csv_path)

            # 列名の揺らぎ吸収
            # 想定: [Phase, Hip, Knee] または [Time, Smoothed_Hip, ...]
            cols = df.columns
            phase_col = 'Phase' if 'Phase' in cols else cols[0]
            # Hip/Kneeの列特定 (列名にHip/Kneeが含まれるか、なければ2,3列目)
            hip_col = next((c for c in cols if 'Hip' in c), cols[1])
            knee_col = next((c for c in cols if 'Knee' in c), cols[2])

            # データをNumpy配列化
            phase_vals = df[phase_col].values
            hip_vals = df[hip_col].values
            knee_vals = df[knee_col].values

            # 位相の正規化 (0 ~ 2pi に収まっているか確認)
            # CSVが0~2piでない場合でも、最後の値を2piとみなして正規化
            if phase_vals[-1] > 2 * np.pi + 0.1: 
                 # もし時間がそのまま入っていたら位相に変換
                phase_vals = (phase_vals / phase_vals[-1]) * 2 * np.pi

            # データ末尾(2pi)と先頭(0)をつなげるためにデータを拡張
            phase_extended = np.concatenate(([phase_vals[-1] - 2*np.pi], phase_vals, [phase_vals[0] + 2*np.pi]))
            hip_extended = np.concatenate(([hip_vals[-1]], hip_vals, [hip_vals[0]]))
            knee_extended = np.concatenate(([knee_vals[-1]], knee_vals, [knee_vals[0]]))

            # 補間関数作成 (cubicスプラインで滑らかに)
            self.func_hip = interp1d(phase_extended, hip_extended, kind='cubic', bounds_error=False, fill_value="extrapolate")
            self.func_knee = interp1d(phase_extended, knee_extended, kind='cubic', bounds_error=False, fill_value="extrapolate")

            print(f"Kuramoto Loaded: {len(df)} points from {csv_path}")

        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise e
        
    def reset(self, phi0=0.0):
        self.phi = phi0
        self.phi_dot = self.omega

    def get_fixed_action(self, phi):
        """
        位相に応じた固定のアクション A'(phi) を返す
        例: sin波。着地(phi=pi付近)で負になるように調整するなど。
        """
        
        return -np.cos(phi) 
        
        # もし「着地タイミング(pi)で足を縮めたい(-1にしたい)」なら cos(phi) など
        # return np.cos(phi)

    def get_fixed_reaction(self, grf):
        """
        床反力に応じた固定のリアクション R(grf) を返す
        """
        # GRFが大きいほど強く反応する (-1.0 ~ 1.0)
        # 正規化定数 1000.0 はロボットの重量に合わせて調整
        norm_grf = grf / 1000.0
        
        # 地面があるときは「負（減速）」にしたい場合:
        return np.arcsinh(norm_grf)

    def step(self, gain, grf):
        """
        Args:
            gain (float): フィードバックゲイン (K)
            grf (float): 床反力 (Ground Reaction Force)
        
        Returns:
            phase (Tensor): 現在の位相 [rad]
            cmd_hip (Tensor): 目標股関節角度 [rad]
            cmd_knee (Tensor): 目標膝関節角度 [rad]
        """
        # 入力をTensorからfloatへ (計算はスカラで行うため)
        if isinstance(gain, torch.Tensor): gain = gain.item()
        if isinstance(grf, torch.Tensor): grf = grf.item()
        #if isinstance(action, torch.Tensor): action = action.item()
        #if isinstance(reaction, torch.Tensor): reaction = reaction.item()

        action = self.get_fixed_action(self.phi)
        reaction = self.get_fixed_reaction(grf)

        # クラマトー方程式に基づく位相更新
        tegotae_feedback = action * reaction
        dphi_dt = self.omega + gain * tegotae_feedback

        self.phi_dot = dphi_dt  # 位相速度を保存
        self.phi_dot = np.clip(self.phi_dot, -2*np.pi, 2*np.pi)  # 位相速度の上限を設定 (必要に応じて調整)

        # 位相を更新
        self.phi += dphi_dt * self.dt

        # 位相を0~2piの範囲に正規化
        self.phi = self.phi % (2 * np.pi)

        # 補間関数から目標角度を取得
        target_hip = self.func_hip(self.phi)
        target_knee = self.func_knee(self.phi)

        # Tensorとして返す (勾配計算が必要な場合は別途考慮が必要だが、通常は指令値なので不要)
        return (
            torch.tensor(self.phi, dtype=torch.float32, device=self.device),
            torch.tensor(target_hip, dtype=torch.float32, device=self.device),
            torch.tensor(target_knee, dtype=torch.float32, device=self.device),
            action,
            reaction
        )

    def get_trajectory(self, phase):
        """位相(0~2pi)を指定して角度を取得する（デバッグ・可視化用）"""
        h = self.func_hip(phase)
        k = self.func_knee(phase)
        return h, k