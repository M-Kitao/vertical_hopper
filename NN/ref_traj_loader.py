"""
ref_traj_loader.py
参照軌道ニューラルネットワーク (ref_traj_nn.pt) を読み込み、参照軌道を生成するユーティリティ
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn


class RefTrajNet(nn.Module):
    """参照軌道ネットワーク（recurrent_learning.py と同一のアーキテクチャ）"""
    def __init__(self, K, hidden, layers):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(2*K+1, hidden), nn.SiLU())
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, 4)
    
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


class ResBlock(nn.Module):
    """ResidualBlock（recurrent_learning.py と同一）"""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(x + self.net(x))


def load_ref_traj_model(model_path, device='cpu'):
    """
    ref_traj_nn.pt を読み込む
    
    Parameters
    ----------
    model_path : str
        ref_traj_nn.pt へのパス
    device : str
        'cpu' or 'cuda'
    
    Returns
    -------
    model : RefTrajNet
        読み込まれたニューラルネットワーク
    meta : dict
        メタデータ {K_fourier, f_min, f_max, Y_mean, Y_std, hidden, layers}
    """
    # 新しいPyTorchでは'auto'は対応していないため'cpu'に変換
    if device == 'auto':
        device = 'cpu'
    checkpoint = torch.load(model_path, map_location=device)
    meta = checkpoint['meta']
    
    K = meta['K_fourier']
    hidden = meta['hidden']
    layers = meta['layers']
    
    model = RefTrajNet(K, hidden, layers)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    return model, meta


def fourier_embed(phi_arr, K):
    """フーリエ基底展開"""
    return np.stack([f(k*phi_arr) for k in range(1, K+1) for f in [np.cos, np.sin]], axis=1)


def get_reference_trajectory_nn(phase_values, freq_hz, model, meta, device='cpu'):
    """
    位相と周波数から参照軌道を生成
    
    Parameters
    ----------
    phase_values : np.ndarray
        位相の配列 [0, 2π]
    freq_hz : float
        周波数 [Hz]
    model : RefTrajNet
        読み込まれたニューラルネットワーク
    meta : dict
        メタデータ
    device : str
        'cpu' or 'cuda'
    
    Returns
    -------
    hip_angles : np.ndarray
        Hip関節角度 [rad]
    knee_angles : np.ndarray
        Knee関節角度 [rad]
    """
    K = meta['K_fourier']
    f_min = meta['f_min']
    f_max = meta['f_max']
    Y_mean = np.array(meta['Y_mean']).reshape(1, 4)
    Y_std = np.array(meta['Y_std']).reshape(1, 4)
    
    # 周波数を正規化
    freq_norm = (freq_hz - f_min) / (f_max - f_min)
    
    # フーリエ基底展開
    phi_fd = fourier_embed(phase_values, K).astype(np.float32)
    
    # ネットワーク入力を作成
    X = np.concatenate([
        phi_fd,
        np.full((len(phase_values), 1), freq_norm, dtype=np.float32)
    ], axis=1)
    
    # 予測
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        Y_pred_norm = model(X_tensor).cpu().numpy()
    
    # 逆正規化
    Y_pred = Y_pred_norm * Y_std + Y_mean
    
    hip_angles = Y_pred[:, 0]
    knee_angles = Y_pred[:, 1]
    
    return hip_angles, knee_angles


def create_phase_array(n_points=1000):
    """0 から 2π までの位相配列を作成"""
    return np.linspace(0, 2*np.pi, n_points, dtype=np.float32)
