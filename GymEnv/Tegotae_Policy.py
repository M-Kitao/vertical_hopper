import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
import os
import sys

# このファイルの場所 (GymEnv/) を取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 一つ上の階層 (vertical_hopper/ ルート) を取得
root_dir = os.path.dirname(current_dir)

# Pythonがモジュールを探すパスに、ルート、CPG、NNフォルダを追加
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'CPG'))
sys.path.append(os.path.join(root_dir, 'NN'))

from CPG.Kuramoto_v2 import KuramotoCPG
from NN.ActionNet import ActionNetMLP
from NN.ReactionNet import ReactionNetMLP
from NN.GainNet import GainNetMLP

class Tegotae_Actor(nn.Module):
    """
    PPOのActorネットワーク
    """
    def __init__(self, 
                 observation_space,
                 action_space,
                 sensor_dim):
        super().__init__()

        self.action_out_dim = 1
        self.reaction_in_dim = 1
        self.reaction_out_dim = 1
        self.gain_out_dim = 1

        self.action_net = ActionNetMLP(output_dim=self.action_out_dim)
        # ReactionNet の入力は地面反力（GRF）だけに固定
        self.reaction_net = ReactionNetMLP(input_dim=self.reaction_in_dim, output_dim=self.reaction_out_dim)
        # GainNet を sin(phi), cos(phi) と GRF のみ入力にする
        self.gain_net = GainNetMLP(sensor_dim=self.reaction_in_dim, output_dim=self.gain_out_dim)
        
        # PPOの確率分布用 (log_std) のパラメータ
        total_action_dim = action_space.shape[0]
        #self.log_std = nn.Parameter(torch.zeros(total_action_dim))

    def forward(self, obs):
        """
        Args:
            obs (torch.Tensor): 観測値 (batch, obs_dim)
        Returns:
            action_mean (torch.Tensor): アクションの平均 (batch, action_dim)
            log_std (torch.Tensor): アクションの対数標準偏差 (action_dim,)
        """
        # 観測値を分割
        # obs = [CPG位相Φ, センサー情報]
        phi = obs[:, 0:2]  # (batch,2)
        sensor_input = obs[:, 2:]  # (batch, sensor_dim)

        # アクションとゲインを計算
        action_mean = self.action_net(phi)  # (batch, action_out_dim)
        reaction_mean = self.reaction_net(sensor_input[:, 0:1])  # (batch, reaction_out_dim)
        # GainNetにはGRFのみを渡す（sin,cosはphiで渡す）
        gain_mean = self.gain_net(phi, sensor_input[:, 0:1])  # (batch, gain_out_dim)

        # 最終的なアクション平均を結合
        full_action_mean = torch.cat([gain_mean, action_mean, reaction_mean], dim=1)  # (batch, action_dim)

        return full_action_mean
    
    def get_distribution(self, obs):
        """
        PPOの確率分布オブジェクトを取得
        Returns:
            DiagGaussianDistribution: 対角ガウス分布オブジェクト
        """
        mean =self.forward(obs)
        return DiagGaussianDistribution(mean.size(1)).proba_distribution(mean, self.log_std)
    
class ValueNetMLP(nn.Module):
    """
    Criticネットワーク (価値関数)
    入力: 全観測情報 [sin, cos, sensor1, sensor2, ...]
    出力: 状態価値 V(s) (スカラー)
    """
    def __init__(self, input_dim, hidden_sizes=(64, 64)):
        super().__init__()
        
        # 第1層
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        # 第2層
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        # 出力層 (価値V)
        self.value_head = nn.Linear(hidden_sizes[1], 1)

    def forward(self, obs):
        # SB3の標準実装に合わせて Tanh を使用 (ReLUでも可)
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        v = self.value_head(x)
        return v
    
    
class Tegotae_Policy(ActorCriticPolicy):
    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 **kwargs):
        
        # 親クラスの初期化 (ここで self.observation_space 等がセットされる)
        # net_arch などを指定して標準ネットワークが作られないように空リストを渡す手もありますが、
        # ここでは後で self.action_net 等を上書きする形で実装します。
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # --- カスタムネットワークの構築 ---
        
        # 観測次元の取得
        obs_dim = observation_space.shape[0]
        # センサー次元 (CPGのsin/cos分である2を引く)
        # ※ 環境が [sin, cos, sensors...] を返している前提
        self.sensor_dim = obs_dim - 2 

        # 1. Actor (既存のもの)
        # self.action_net という名前で保存 (SB3がパラメータ探索する際に重要)
        self.action_net = Tegotae_Actor(observation_space, action_space, self.sensor_dim)
        
        # 2. Critic (今回追加)
        # Criticは全観測情報を見るため input_dim = obs_dim
        self.value_net = ValueNetMLP(input_dim=obs_dim, hidden_sizes=(64, 64))

        # 【重要】親クラスが作成した log_std を上書きして、アクション次元に合わせる
        # 親クラスは observation_space を見て log_std を初期化するため、不正な次元になる場合がある
        action_dim = action_space.shape[0]
        #self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),  **self.optimizer_kwargs)
        #print(f"[DEBUG] __init__ 完了時 log_std.shape={self.log_std.shape}")

    def _build(self, lr_schedule):
        """
        オプティマイザを作成し、ネットワークパラメータを登録する
        """
        # 親クラスの _build を呼ぶと、self.parameters() に含まれる全パラメータが登録される
        # action_net と value_net は nn.Module なので自動的に含まれます
        super()._build(lr_schedule)
        action_dim = self.action_space.shape[0]
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        #print(f"[DEBUG] _build 完了時 log_std.shape={self.log_std.shape}")

    def forward(self, obs, deterministic=False):
        """
        順伝播: 行動、価値、対数確率を返す (学習時に呼ばれる)
        """
        # 1. Actorの計算 (確率分布を取得)
        # Tegotae_Actor は mean を返すが、ここでは分布が必要
        action_mean = self.action_net(obs)
        distribution = self.get_distribution_from_mean(action_mean)
        
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        # 2. Criticの計算
        values = self.value_net(obs)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs, actions):
        """
        【重要】学習(Train)時に呼ばれるメソッド
        デフォルトの実装では mlp_extractor を通した特徴量を使うため、
        次元不一致エラーが起きます。
        ここで obs を直接ネットワークに通すようにオーバーライドします。
        """
        # Actorの計算
        action_mean = self.action_net(obs)
        distribution = self.get_distribution_from_mean(action_mean)
        
        # ログ確率とエントロピーの計算
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Critic (Value) の計算
        values = self.value_net(obs)
        
        return values, log_prob, entropy

    def _predict(self, observation, deterministic=False):
        """
        推論用: 行動だけを返す
        """
        action_mean = self.action_net(observation)
        distribution = self.get_distribution_from_mean(action_mean)
        return distribution.get_actions(deterministic=deterministic)

    def predict_values(self, obs):
        """
        価値だけを返す (学習時の価値関数の更新に使用)
        """
        return self.value_net(obs)

    def get_distribution_from_mean(self, mean):
        """
        平均値から確率分布を作成するヘルパー
        """
        # log_std は Tegotae_Policy が持っている学習可能パラメータ（正しいサイズ）
        action_dim = mean.shape[-1]  # 常に行動次元(=3)を取得
        if self.log_std.shape[0] != action_dim:
            self.log_std = nn.Parameter(
                torch.zeros(action_dim, device=mean.device),
                requires_grad=True
        )
        #print(f"[DEBUG] mean.shape={mean.shape}, log_std.shape={self.log_std.shape}, log_std={self.log_std}")
        return DiagGaussianDistribution(action_dim).proba_distribution(mean, self.log_std)