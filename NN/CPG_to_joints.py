import torch
import torch.nn as nn
import torch.nn.functional as F

class CPGToJointsMLP(nn.Module):
    """
    入力: sin(phi), cos(phi)  -> shape (batch, 2)
    出力: hip_angle, knee_angle, hip_vel, knee_vel -> shape (batch, 4)
    最終層は tanh にして適当なスケールで角度・速度に変換する想定。(by chatGPT)
    """

    def __init__(self,
                hip_range,
                knee_range,
                vel_max=10.0,
                hidden_sizes=(16, 16),
                ):
        super().__init__()

        in_dim = 2
        out_dim = 4

        self.hip_min,  self.hip_max  = hip_range
        self.knee_min, self.knee_max = knee_range
        self.vel_max = vel_max

        self.fc1 = nn.Linear(in_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], out_dim)  # 4 outputs: hip_angle, knee_angle, hip_vel, knee_vel

    def _scale_to_range(self, r, low, high):
        """ r in [-1, 1] を [low, high] にスケーリング """
        return low + (r + 1.0) * 0.5 * (high - low)

    def forward(self, phi):
        if phi.dim() == 1:
            # raw phase
            phi = phi.unsqueeze(-1)  # (batch,1)
        if phi.size(-1) == 1:
            phi = torch.cat([torch.sin(phi), torch.cos(phi)], dim=-1)  # (batch,2)

        x = F.relu(self.fc1(phi))
        x = F.relu(self.fc2(x))
        out = torch.tanh(self.fc3(x))  # 範囲 [-1,1]

        hip_angle  = self._scale_to_range(out[:,0], self.hip_min,  self.hip_max)
        knee_angle = self._scale_to_range(out[:,1], self.knee_min, self.knee_max)

        hip_vel  = out[:,2] * self.vel_max
        knee_vel = out[:,3] * self.vel_max

        return torch.stack([hip_angle, knee_angle, hip_vel, knee_vel], dim=-1)