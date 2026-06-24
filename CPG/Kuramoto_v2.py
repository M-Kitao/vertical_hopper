"""
NN参照軌道ベースのKuramoto CPG
（CSVデータの代わりにニューラルネット参照軌道を使用）
"""

import torch
import numpy as np
from NN.ref_traj_loader import get_reference_trajectory_nn, create_phase_array

class KuramotoCPG(torch.nn.Module):
    def __init__(self,
                 ref_traj_model,
                 ref_traj_meta,
                 omega,
                 dt=0.01,
                 device='cpu'):
        """
        Args:
            ref_traj_model: NN参照軌道モデル（RefTrajNet）
            ref_traj_meta (dict): ref_traj_modelのメタデータ (K, f_min, f_max, Y_mean, Y_std)
            omega (float): 基本角振動数 (rad/s)
            dt (float): タイムステップ
            device (str): torchのデバイス ('cpu' or 'cuda')
        """
        super().__init__()  # torch.nn.Module.__init__() を先に呼び出す
        self.ref_traj_model = ref_traj_model
        self.ref_traj_meta = ref_traj_meta
        self.omega = omega
        self.dt = dt
        self.device = device
        self.phi = 0.0  # 初期位相
        self.phi_dot = 0.0  # 位相速度の初期値
        self.current_freq_hz = omega / (2 * np.pi)  # 現在の周波数 [Hz]

    
    def reset(self, phi0=0.0):
        self.phi = phi0
        self.phi_dot = self.omega

    def update_frequency(self, omega):
        """周波数を更新"""
        self.omega = omega
        self.phi_dot = omega
        self.current_freq_hz = omega / (2 * np.pi)

    def step(self, gain, action, reaction):
        """
        Args:
            gain (float): フィードバックゲイン (K)
            action (float): ロボットの行動 (例: モーター指令や速度)
            reaction (float): 環境からの反作用 (例: 床反力、加速度)
        
        Returns:
            phase (Tensor): 現在の位相 [rad]
            cmd_hip (Tensor): 目標股関節角度 [rad]
            cmd_knee (Tensor): 目標膝関節角度 [rad]
        """
        # 入力をTensorからfloatへ (計算はスカラで行うため)
        if isinstance(gain, torch.Tensor): gain = gain.item()
        if isinstance(action, torch.Tensor): action = action.item()
        if isinstance(reaction, torch.Tensor): reaction = reaction.item()

        # Kuramoto方程式に基づく位相更新
        tegotae_feedback = action * reaction
        dphi_dt = self.omega + gain * tegotae_feedback

        self.phi_dot = dphi_dt  # 位相速度を保存

        # 位相を更新
        self.phi += dphi_dt * self.dt

        # 位相を0~2piの範囲に正規化
        self.phi = self.phi % (2 * np.pi)

        # NN参照軌道から目標角度を取得
        target_hip, target_knee = get_reference_trajectory_nn(
            phase_values=np.array([self.phi]),
            freq_hz=self.current_freq_hz,
            model=self.ref_traj_model,
            meta=self.ref_traj_meta,
            device=self.device
        )
        # 配列の最初の要素を取り出す
        target_hip = target_hip[0]
        target_knee = target_knee[0]

        # Tensorとして返す
        return (
            torch.tensor(self.phi, dtype=torch.float32, device=self.device),
            torch.tensor(target_hip, dtype=torch.float32, device=self.device),
            torch.tensor(target_knee, dtype=torch.float32, device=self.device)
        )

    def get_trajectory(self, phase):
        """位相(0~2pi)を指定して角度を取得する（デバッグ・可視化用）"""
        h, k = get_reference_trajectory_nn(
            phase_values=np.array([phase]),
            freq_hz=self.current_freq_hz,
            model=self.ref_traj_model,
            meta=self.ref_traj_meta,
            device=self.device
        )
        return h[0], k[0]