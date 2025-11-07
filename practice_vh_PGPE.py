import torch
import evotorch
from evotorch import Problem
from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger
from evotorch.neuroevolution import GymNE, VecGymNE
import gymnasium as gym
import os
import numpy as np
import vh_env  # 作成したカスタム環境をインポート

log_dir = './PGPE_logs/'
model_dir = './PGPE_models/'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

LOAD_MODEL = True

def make_env(**kwargs):
    import vh_env  # カスタム環境をインポート
    return gym.make('vh-v0')

#ネットワークの定義
def my_policy_network(obs_dim, act_dim): 
    return torch.nn.Sequential(
    torch.nn.Linear(obs_dim, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, act_dim),
    torch.nn.Tanh()  # 行動空間が-1から1の範囲の場合
)

#ネットワークと環境の準備
temp_env = make_env()
obs_space_dim = temp_env.observation_space.shape[0] #観測空間の次元数
act_space_dim = temp_env.action_space.shape[0]   #行動空間の次元数
temp_env.close()

policy_network_template = my_policy_network(obs_space_dim, act_space_dim)
num_net_params = sum(p.numel() for p in policy_network_template.parameters()) #ネットワークのパラメータ数

def evaluate_net(params: torch.Tensor) -> float:
    import vh_env  # カスタム環境をインポート
    #ローカル環境とネットワークの作成
    local_net = my_policy_network(obs_space_dim, act_space_dim)
    local_env = make_env()
    #パラメータの設定
    offset = 0
    for p in local_net.parameters():
        param_length = p.numel()
        p.data = params[offset:offset + param_length].view_as(p).data
        offset += param_length
    
    total_reward = 0.0
    obs, info = local_env.reset()

    # エピソードの実行
    max_steps = getattr(local_env.spec, 'max_episode_steps', 1000) or 1000
    for _ in range(max_steps):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = local_net(obs_tensor).numpy()
        obs, reward, terminated, truncated, info = local_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    local_env.close()
    return total_reward

# --- 1. 問題の定義 ---
problem = Problem(
    "max",
    objective_func=evaluate_net,
    solution_length=num_net_params,
    initial_bounds=(-0.1, 0.1),
    num_actors=4,
    device='cpu'
)

# --- 2. アルゴリズムの初期化 ---
seracher = PGPE(
    problem,
    center_learning_rate=0.05,
    stdev_learning_rate=0.1,
    stdev_init=0.1,
    popsize=32
)

# --- 3. ロガーの設定 ---
_ = StdOutLogger(seracher)
#_ = evotorch.logging.FileLogger(pgpe, log_dir) <- 使えないんでしょう、多分。

# --- 4. 最適化の実行 ---
print('学習始めっぞ!')
num_generations = 100
seracher.run(num_generations)
print('学習終わり！閉廷！')

# --- 5. 最適な解の保存 ---
best_params = seracher._center_learning_rate
save_path = os.path.join(model_dir, 'PGPE_vertical_hopper_params.pt')   

#torch.save(best_params, save_path)
torch.save(save_path, best_params)
print(f'大松「最適なパラメータを{save_path}に保存したぞ」')
