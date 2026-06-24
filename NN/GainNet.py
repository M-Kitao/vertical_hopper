import torch
import torch.nn as nn
import torch.nn.functional as F

class GainNetMLP(nn.Module):
    """
    入力: CPG位相Φ、センサー情報のベクトル -> shape (batch, 2 + sensor_dim)
    出力: ゲイン値 -> shape (batch, output_dim)
    """

    def __init__(self,
                 sensor_dim,
                 output_dim,
                 hidden_sizes=(64, 64),
                 ):
        super().__init__()

        in_dim = 2 + sensor_dim  # CPG位相Φ + センサー情報

        self.fc1 = nn.Linear(in_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_dim)  # output_dim outputs: gain values

    def forward(self, phi, sensor_input):
        if phi.dim() == 1:
            phi = phi.unsqueeze(-1)  # (batch,1)
        if phi.size(-1) == 1:
            phi = torch.cat([torch.sin(phi), torch.cos(phi)], dim=-1)  # (batch,2)

        x = torch.cat([phi, sensor_input], dim=-1)  # (batch, 2 + sensor_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # 範囲 [-1,1]
        Tegotae_Gain = 2.5 * x + 2.5 # スケーリングして [0,5] に変換

        return Tegotae_Gain