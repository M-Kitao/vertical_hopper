import torch

class Kuramoto_Oscillator:
    def __init__(self,
                 omega=torch.pi,
                 dt=0.01,):
        self.omega = omega  # 固有振動数
        self.dt = dt        # タイムステップ
        self.phi = 0.0    # 初期位相

    def reset(self, phi0=0.0):
        self.phi = phi0
        self.phi_dot = self.omega

    def step(self, gain, action, reaction): #gain, action, reactionはtensorでもfloatでもOK
        # クラマトー方程式に基づく位相更新
        dphi_dt = self.omega + gain * action * reaction
        self.phi += dphi_dt * self.dt
        self.phi = self.phi % (2 * torch.pi)  # 位相を0から2πの範囲に保つ
        return torch.tensor(self.phi)