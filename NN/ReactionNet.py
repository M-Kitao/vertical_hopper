import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ReactionNetMLP(nn.Module):
    """
    入力: センサー情報のベクトル -> shape (batch, input_dim)
    出力: 反応力 -> shape (batch, output_dim)
    最終層は tanh にして適当なスケールで反応力に変換する想定。
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 force_max=1000.0,
                 hidden_sizes=(64, 64),
                 ):
        super().__init__()

        self.force_max = force_max

        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_dim)  # output_dim outputs: reaction forces

    def forward(self, sensor_input):
        x = F.relu(self.fc1(sensor_input))  # 入力をスケーリングして安定化
        x = F.relu(self.fc2(x))
        #out = torch.tanh(self.fc3(x))  # 範囲 [-1,1]
        #out = torch.arcsinh(self.fc3(x))  # 範囲 (-∞, ∞)
        out = F.elu(self.fc3(x))  # 範囲 (-∞, ∞)

        Tegotae_Reaction = out #* self.force_max

        return Tegotae_Reaction