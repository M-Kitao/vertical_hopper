import gymnasium as gym
from stable_baselines3 import PPO 
from stable_baselines3.common.env_util import make_vec_env
import os
import vh_env  # 作成したカスタム環境をインポート

#ログandモデル保存先
log_dir = './PPO_logs/'
model_dir = './PPO_models/'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

#学習環境の準備
vec_env = make_vec_env('vh-v0', n_envs=4, seed=0)

#モデルの準備
model = PPO(
    'MlpPolicy',
    vec_env,
    verbose=1,
    tensorboard_log=log_dir,
    device="auto"# 自動でGPUを検出して使用
) 

#学習の実行
time_steps = 100000
model_name = 'PPO_vertical_hopper'

print('学習始めっぞ!')
model.learn(total_timesteps=time_steps, tb_log_name=model_name)
print('学習終わり！閉廷！')

#モデルの保存
save_path = os.path.join(model_dir, model_name)
model.save(save_path)
print(f'大松「モデルを{save_path}.zipに保存したぞ」')

#学習環境の解放
vec_env.close()