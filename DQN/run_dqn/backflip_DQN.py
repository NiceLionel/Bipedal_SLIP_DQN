import pickle

import gym
import numpy as np
from DQN import DQNAgent

with open('/home/jarvis/UPennROBO/Courses/Semester1/MEAM5170/Final/DQN/flipV2/agent_flip.obj', 'rb') as f1:
    agent_flip = pickle.load(f1)

with open('/home/jarvis/UPennROBO/Courses/Semester1/MEAM5170/Final/DQN/flipV2/agent_balance.obj', 'rb') as f2:
    agent_balance = pickle.load(f2)
   
actual_actions = np.arange(-1., 1.1, 0.1, dtype=np.float32)
env1 = gym.make('SLIP', render_mode="human")
terminated = False
observation = np.float32(env1.reset()[0])
steps = 0

num_ground_contact = 0
first_ground_contact = False
first_backflip = False
switch_backflip_agent = False
wrap_angle = False

for _ in range(100000):

    if(terminated):
        observation = np.float32(env1.reset()[0])
        first_ground_contact = False
        first_backflip = False
        switch_backflip_agent = False

    if(observation[-1]): num_ground_contact += 1
    if (not first_ground_contact) and observation[-1]: first_ground_contact = True

    if(observation[1] > -0.6 and (abs(observation[4]) < 0.06) and first_ground_contact and (abs(observation[2]) < 6.) and (not first_backflip)): 
        switch_backflip_agent = True
        first_backflip = True
    if(abs(observation[2]) > 5.9 and switch_backflip_agent and observation[-1]): 
        switch_backflip_agent = False
        observation[2] = observation[2] - (2 * np.pi)

    if(switch_backflip_agent):
        action = agent_flip.choose_action(observation[:-2])
        print("backflip_agent!")
        print(observation[2])
    else:
        action = agent_balance.choose_action(observation[:-2])
        print("balance_agent!")
        print(observation[2])

    #action = agent.choose_action(observation)

    observation, reward, terminated, _, _ = env1.step(np.array([actual_actions[action]]))
    observation = np.float32(observation)
    if (not switch_backflip_agent) and first_backflip and (observation[2] > 5.9 or wrap_angle):
        observation[2] = observation[2] - 2 * np.pi
        wrap_angle = True


    #print(observation[2])
    steps += 1
    env1.render()
env1.close()
print(steps)
