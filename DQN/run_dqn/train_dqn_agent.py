import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from DQN import DQNAgent

import os
import datetime

import pickle

comments = "_flipv2"
reward_function_used = """
alive_bonus = 5.
reward = 0.
reward += 30 * (angle_after - angle_previous) / self.dt
reward += alive_bonus
reward -= 12 * np.square(a)
s = self.state_vector()
terminated = not (
    np.isfinite(s).all()
    and (height > -2.38)
    and (np.abs(s[[0,1]]) < 25).all()
)

if terminated: reward -= 90000

if(self.sim.data.ncon): 
    if (abs(angle_after) % 2*np.pi < 0.27): reward += 750
"""

workspace_folder = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + comments
os.makedirs(workspace_folder)

m_p = {
    'gamma':0.996,
    'epsilon':1.0,
    'batch_size':128,
    'n_actions':21,
    'eps_end':0.01,
    'input_dims':6,
    'lr':1e-4,
    'reward_function_comments':reward_function_used
}

with open(workspace_folder+"/config.txt", 'w') as data:
    data.write(str(m_p))

if __name__ == "__main__":
    env = gym.make('SLIP')
    agent = DQNAgent(gamma=m_p['gamma'], epsilon=m_p['epsilon'], batch_size=m_p['batch_size'], n_actions=m_p['n_actions'], eps_end=m_p['eps_end'], input_dims=[m_p['input_dims'],], lr=m_p['lr'])
    scores, avg_scores, eps_history, step_history = [], [], [], []
    n_games = 5000

    actual_actions = np.arange(-1., 1.1, 0.1, dtype=np.float32)

    for i in range(n_games):
        score = 0
        done = False
        observation = np.float32(env.reset()[0])
        steps = 0
        while not done:
            if(steps > 10000): break
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(np.array([actual_actions[action]]))
            observation_ = np.float32(observation_)
            score += reward

            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()

            observation=observation_
            steps += 1
        scores.append(score)
        eps_history.append(agent.epsilon)
        step_history.append(steps)

        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        print('episode ', i, 'score %.2f' % score, 'average_score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)

        if(i % 25 == 0 and i>0):
            save_path = workspace_folder+"/"+str(i)
            os.makedirs(save_path)

            x = [j+1 for j in range(i+1)]

            plt.figure(1)
            plt.plot(x, scores, "r", x, avg_scores, "k--")
            plt.xlabel("Iterations")
            plt.ylabel("Reward accrued")
            plt.title("Average scores and raw scores")
            plt.legend(["Avg score", "score"])
            plt.grid(True)
            plt.savefig(save_path + "/scores.png", dpi=600)

            plt.figure(2)
            plt.plot(x, eps_history)
            plt.xlabel("Iterations")
            plt.ylabel("epsilon")
            plt.title("Epsilon decay")
            plt.grid(True)
            plt.savefig(save_path + "/eps.png", dpi=600)

            with open(save_path + '/agent.obj', 'wb') as f1:
                pickle.dump(agent, f1)

            with open(save_path + '/sim_data.pkl', 'wb') as f2:
                #pickle.dump([scores, eps_history, obs_history, action_history, step_history], f2)
                pickle.dump([scores, eps_history, step_history], f2)

            plt.close()

    env.close()
    x = [i+1 for i in range(n_games)]
    
    plt.figure(1)
    plt.plot(x, scores, "r", x, avg_scores, "k--")
    plt.xlabel("Iterations")
    plt.ylabel("Reward accrued")
    plt.title("Average scores and raw scores")
    plt.legend(["Avg score", "score"])
    plt.grid(True)
    plt.savefig(workspace_folder + "/scores.png", dpi=900)

    plt.figure(2)
    plt.plot(x, eps_history)
    plt.xlabel("Iterations")
    plt.ylabel("epsilon")
    plt.title("Epsilon decay")
    plt.grid(True)
    plt.savefig(workspace_folder + "/eps.png", dpi=900)

    with open(workspace_folder + '/agent.obj', 'wb') as f:
        pickle.dump(agent, f)

    with open(workspace_folder + '/sim_data.pkl', 'wb') as f2:
        pickle.dump([scores, eps_history, step_history], f2)

    
    env1 = gym.make('SLIP', render_mode="human")
    terminated = False
    observation = np.float32(env1.reset()[0])
    for ijk in range(10000):

        if(terminated): env1.reset()

        action = agent.choose_action(observation)

        observation, reward, terminated, _, _ = env1.step(np.array([actual_actions[action]]))
        observation = np.float32(observation)

        env1.render()
    env1.close()
    