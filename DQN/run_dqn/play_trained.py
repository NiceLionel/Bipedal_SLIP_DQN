import pickle

import gym
import numpy as np
from DQN import DQNAgent

with open('/home/jarvis/UPennROBO/Courses/Semester1/MEAM5170/Final/DQN/flipV2/agent_balance.obj', 'rb') as f:
    agent = pickle.load(f)
    
actual_actions = np.arange(-1., 1.1, 0.1, dtype=np.float32)
env1 = gym.make('SLIP', render_mode="human")
terminated = False
observation = np.float32(env1.reset()[0])
steps = 0
for _ in range(100000):

    if(terminated):
        observation = np.float32(env1.reset()[0])

    action = agent.choose_action(observation[:-2])

    observation, reward, terminated, _, _ = env1.step(np.array([actual_actions[action]]))
    observation = np.float32(observation)

    #print(observation[-1])
    steps += 1
    env1.render()
env1.close()
print(steps)
