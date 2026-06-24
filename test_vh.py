import gymnasium as gym
import vh_env  # 同ファイル内にクラスと登録がある

env = gym.make("vh-v0", render_mode="human")
obs, info = env.reset()

for _ in range(300):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    if done:
        break

env.close()