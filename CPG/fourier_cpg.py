import numpy as np

class FourierCPG:
    def __init__(self, omega, weights_hip, weights_knee):
        """
        Args:
            omega (float): 基本角振動数 (rad/s)
            weights_hip (list/array): Hip関節の学習済み重み [w0, w_cos1, w_sin1, w_cos2, ...]
            weights_knee (list/array): Knee関節の学習済み重み
        """
        self.omega = omega
        self.weights_hip = np.array(weights_hip)
        self.weights_knee = np.array(weights_knee)
        
        # 状態変数
        self.phase = 0.0  # 現在の位相 [0, 2pi)
        
        # 重みの次元チェック
        # w[0]はバイアス、以降はcos, sinのペアなので、長さは奇数になるはず
        assert len(self.weights_hip) % 2 == 1, "Weights length must be odd (bias + cos/sin pairs)."
        self.order = (len(self.weights_hip) - 1) // 2

    def update(self, dt, enable_progress=True):
        """
        CPGの位相を1ステップ進める
        
        Args:
            dt (float): 経過時間
            enable_progress (bool): Trueなら位相を進める（接地中のみTrueにするなど）
        """
        if enable_progress:
            self.phase += self.omega * dt
            
            # 位相を 0~2pi に正規化（必須ではないが数値安定性のため）
            self.phase = self.phase % (2 * np.pi)

    def get_target_angles(self):
        """
        現在の位相に対応する関節角度を計算する
        Returns:
            (float, float): (target_hip_angle, target_knee_angle)
        """
        theta_hip = self._calculate_output(self.weights_hip)
        theta_knee = self._calculate_output(self.weights_knee)
        return theta_hip, theta_knee

    def _calculate_output(self, weights):
        """
        フーリエ級数の式に基づいて値を計算
        y = w0 + Σ [w_cos_k * cos(k*phi) + w_sin_k * sin(k*phi)]
        """
        phi = self.phase
        
        # バイアス項 (w0)
        y = weights[0]
        
        # 高調波成分の加算
        for k in range(1, self.order + 1):
            idx_cos = 2 * k - 1
            idx_sin = 2 * k
            
            y += weights[idx_cos] * np.cos(k * phi)
            y += weights[idx_sin] * np.sin(k * phi)
            
        return y