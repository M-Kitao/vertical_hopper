from Tegotae_Hopper_PPO_V3 import Tegotae_Hopper_PPO_v2_Env

# インスタンス化
env = Tegotae_Hopper_PPO_v2_Env(render_mode="human")
obs, info = env.reset()

# シミュレーションループ
for _ in range(10000):
    # アクションは現在使っていませんが、形式的に渡します
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # ログ出力例
    if _ % 100 == 0:
        print(f"Step: {_}, Phi: {info['phase']:.2f}, Height: {obs[0]:.2f}, Gain: {info['gain']:.2f}, grf: {info['grf']:.2f}")
        
    if terminated:
        obs, info = env.reset()

env.close()