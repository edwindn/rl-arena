import torch
import torch.nn as nn
import numpy as np
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.data.replay_buffers import LazyTensorStorage
from tensordict import TensorDict

"""
use caching and recall to train on random previous actions
separates learning from experience gaining
DQN :  state -> action quality values (for the whole action space)

added second memory for episode level learning
-> move from distance to episode reward ~100 episodes
"""

WEIGHTS = 'grid_agent.pth'

class Agent():
    def __init__(self, state_dim, action_dim):
        self.explore_rate = 1.0
        self.explore_decay = 0.9995
        self.min_explore = 0.0
        self.gamma = 0.9
        self.sync_every = 10

        self.batch_size = 1000
        self.lr = 0.001

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.game_count = 0

        self.memory = TensorDictReplayBuffer(storage=LazyTensorStorage(1000000))

        self.online_net = AgentNet(state_dim, action_dim, self.lr) # contains .model and .optimizer
        self.target_net = AgentNet(state_dim, action_dim, self.lr)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.loss_fn = torch.nn.MSELoss()

    
    def choose_action(self, state):
        if np.random.uniform() < self.explore_rate:
            action = np.random.randint(0, self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            Q_vals = self.online_net(state)
            action = torch.argmax(Q_vals).item()
        return action
    
    def cache(self, state, reward, action, next_state, done):
        self.memory.add(TensorDict({
            'state': torch.tensor(state, dtype=torch.float32),
            'reward': torch.tensor(reward, dtype=torch.float32),
            'action': torch.tensor(action, dtype=torch.int32),
            'next_state': torch.tensor(next_state, dtype=torch.float32),
            'done': torch.tensor(done, dtype=torch.int32)
        }))

    def long_train(self):
        if len(self.memory) < self.batch_size:
            return
        sample = self.memory.sample(self.batch_size)
        state, reward, action, next_state, done = (sample.get(key) for key in ('state', 'reward', 'action', 'next_state', 'done'))
        # here each of the above are batched
        self.train_step(state, reward, action, next_state, done)
        print('training ...')

    def short_train(self, state, reward, action, next_state, done):
        state, reward, action, next_state, done = torch.tensor(state, dtype=torch.float32), torch.tensor(reward,  dtype=torch.float32), torch.tensor(action,  dtype=torch.int32), torch.tensor(next_state,  dtype=torch.float32), torch.tensor(done,  dtype=torch.int32)
        state, next_state = state, next_state
        self.train_step(state, reward, action, next_state, done)

    def train_step(self, state, reward, action, next_state, done):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            reward = reward.unsqueeze(0)
            action = action.unsqueeze(0)
            done = done.unsqueeze(0)
        pred = self.online_net(state)
        target = pred.clone().detach()
        with torch.no_grad():
            next_Qs = self.target_net(next_state)
            max_Qs = torch.max(next_Qs, axis=1).values
        for idx in range(len(done)):
            if done[idx]:
                target[idx, action[idx]] = reward[idx]
            else:
                target[idx, action[idx]] = reward[idx] + self.gamma * max_Qs[idx]

        self.online_net.optimizer.zero_grad()
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.online_net.optimizer.step()
        return loss.item()
    
    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_model(self):
        torch.save(self.online_net.model, WEIGHTS)

    def load_model(self):
        self.online_net.model = torch.load(WEIGHTS)
        print(f'--- Loaded weights from {WEIGHTS} ---')
    

class AgentNet(nn.Module):
    def __init__(self, input_size, output_size, lr):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = 256
        self.model = self.ffwd()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def ffwd(self):
        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_size),
        )    

    def forward(self, x):
        return self.model(x)



