import matplotlib.pyplot as plt
import pandas as pd

file_path1 = 'PPO_JumpReward_v12_1.csv'
file_path2 = 'PPO_JumpReward_gainonly_v2_1.csv'
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

step_data1 = df1['Step']
value_data1 = df1['Value']
step_data2 = df2['Step']
value_data2 = df2['Value']

plt.figure(figsize=(10, 5))
plt.plot(step_data1, value_data1, label='condition 1', color='blue')
plt.plot(step_data2, value_data2, label='condition 2', color='red')
#plt.title('ep_rew_mean')
plt.xlabel('Timesteps') 
plt.ylabel('Average Episode Reward')
plt.legend()
plt.grid(True)
plt.show()