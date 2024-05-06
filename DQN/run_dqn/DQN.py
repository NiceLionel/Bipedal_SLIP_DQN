import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class DQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=2048, fc2_dims=2048, fc3_dims=2048, n_actions=21):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)

        return actions
    

class DQNAgent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=10000000, eps_end=0.01, eps_dec=5e-6):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0

        self.Q_eval = DQN(lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=2048, fc2_dims=2048, fc3_dims=2048)
        #self.Q_targ = DQN(lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=1024, fc2_dims=1024)
        #self.Q_targ.load_state_dict(self.Q_eval.state_dict())

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        #print(self.state_memory.shape)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        #print(state)
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # with torch.no_grad():
            #     state = torch.tensor([observation]).to(self.Q_eval.device)
            #     actions = self.Q_eval.forward(state)
            #     action = torch.argmax(actions).item()
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        # q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        # with torch.no_grad():
        #     q_next = self.Q_targ.forward(new_state_batch)
        # q_next[terminal_batch] = 0.0

        # q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        # Q_targ_params = self.Q_targ.state_dict()
        # Q_eval_params = self.Q_eval.state_dict()
        # for key in Q_eval_params:
        #     Q_targ_params[key] = Q_eval_params[key] * self.tau + Q_targ_params[key] * (1-self.tau)
        # self.Q_targ.load_state_dict(Q_targ_params)

