import torch
import torch.nn as nn
import torch.nn.functional as F

"""
入力：CPG位相Φをsin およびcosで表現したもの (shape: (batch, 2))
出力：A(Φ) 行動ネットワーク [-1, 1] の範囲で出力
"""

class ActionNetMLP(nn.Module):
    def __init__(self,
                 output_dim,
                 hidden_sizes=(64, 64),
                 ):
        super().__init__()

        in_dim = 2  # CPG位相Φ

        self.fc1 = nn.Linear(in_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_dim)  # output_dim outputs: action values

    def forward(self, phi):
        if phi.dim() == 1:
            phi = phi.unsqueeze(-1)  # (batch,1)
        if phi.size(-1) == 1:
            phi = torch.cat([torch.sin(phi), torch.cos(phi)], dim=-1)  # (batch,2)

        x = F.relu(self.fc1(phi))
        x = F.relu(self.fc2(x))
        Tegotae_Action = torch.tanh(self.fc3(x))  # 範囲 [-1,1]

        return Tegotae_Action