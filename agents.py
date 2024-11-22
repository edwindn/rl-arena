import torch
import torch.nn as nn
import numpy as np
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict

"""
use caching and recall to train on random previous actions
separates learning from experience gaining
DQN :  state -> action quality values (for the whole action space)

added second memory for episode level learning
"""

class Agent:
    def __init__(self, state_dim, action_dim):
        # actions are UP, RIGHT, DOWN, LEFT
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = 64

        self.burnin = 10000
        self.burnin_count = 0

        self.explore_rate = 1
        self.explore_decay = 0.995
        self.min_explore_rate = 0.1
        self.discount = 0.9 # gamma

        self.update_every = 1
        self.sync_every = 5 # sync online to target

        self.net = AgentNet(state_dim, action_dim)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.0005)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.memory2 = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
    
    def select_action(self, state):
        if np.random.rand() < self.explore_rate:
            action = np.random.randint(self.action_dim)
            method = 'random'
        else:
            next_Q_vals = self.net(torch.tensor(state, dtype=torch.float32).unsqueeze(0), model='online')
            action = torch.argmax(next_Q_vals).item()
            method = 'model'

        if self.burnin_count > self.burnin:
            self.explore_rate *= self.explore_decay
            self.explore_rate = max(self.min_explore_rate, self.explore_rate)
        else:
            self.burnin_count += 1
        return action, method

    def get_theta_est(self, state, action): # estimate current Q value using online model
        return self.net(torch.tensor(state, dtype=torch.float32).unsqueeze(0), model='online')[np.arange(self.batch_size), action]
        
    @torch.no_grad()
    def get_theta_target(self, next_state, reward, done):
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        next_state_Qs = self.net(next_state, model='online')
        action = torch.argmax(next_state_Qs, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(self.batch_size), action]
        return (reward + (1 - done.float()) * self.discount * next_Q).float()

    def sync_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
    
    def update_online(self, theta_est, theta_target):
        loss = self.loss_fn(theta_est, theta_target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()
    
    def recall(self, memory=1):
        if memory == 1 or (memory == 2 and len(self.memory2.storage) < self.batch_size):
            batch = self.memory.sample(self.batch_size)
        elif memory == 2:
            batch = self.memory2.sample(self.batch_size)
        state, next_state, action, reward, done = (batch.get(key) for key in ('state', 'next_state', 'action', 'reward', 'done'))
        return state, next_state, action, reward, done

    def cache(self, state, next_state, action, reward, done, memory=1):
        if memory == 1:
            self.memory.add(TensorDict({
                    "state": torch.tensor(state),
                    "next_state": torch.tensor(next_state),
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "done": torch.tensor(done)
            }))
        elif memory == 2:
            self.memory2.add(TensorDict({
                    "state": torch.tensor(state),
                    "next_state": torch.tensor(next_state),
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "done": torch.tensor(done)
            }))
    
    def learn(self, memory=1):
        if self.burnin_count < self.burnin:
            return
        state, next_state, action, reward, done = self.recall(memory)
        theta_est = self.get_theta_est(state, action)
        theta_target = self.get_theta_target(next_state, reward, done)
        loss = self.update_online(theta_est, theta_target)
        return loss
    
    def save_models(self, path='agent_net.pth'):
        torch.save({
            "online": self.net.online.state_dict(),
            "target": self.net.target.state_dict()
        }, path)
        print(f'Saved weights to {path}')

    def load_models(self, path='agent_net.pth'):
        models = torch.load(path)
        self.net.online.load_state_dict(models['online'])
        self.net.target.load_state_dict(models['target'])
    

class AgentNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.size = state_dim[0]
        self.output = action_dim

        self.online = self.cnn(1, self.output) # 1 is the number of channels we have
        self.target = self.cnn(1, self.output)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = torch.transpose(x, 1, 0) # batch first

        if model == 'online':
            return self.online(x)
        elif model == 'target':
            return self.target(x)

    def cnn(self, input_dim, output_dim): # can update to take in a few consecutive frames (learns velocity)
        return nn.Sequential(
            nn.Conv2d(input_dim, 8, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

if __name__ == '__main__':
    agent1 = Agent((256, 256), 2)
    agent2 = Agent((256, 256), 2)



# ------------

class CNNAgent:
    """
    takes in snapshot and gives the next move
    the loss is just the distance between the two agents
    """
    def __init__(self, state_dim, action_dim):
        # actions are UP, RIGHT, DOWN, LEFT
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = 1 # can run multiple training simulations in parallel

        self.net = CNNNet(action_dim)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.05)
        self.loss_fn = torch.nn.SmoothL1Loss()
    
    def select_action(self, state):
        self.net.eval()
        Q_vals = self.net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        action = torch.argmax(Q_vals).item()
        return action
    
    def reward_fn(self, state, action): # state is the grid, update using action
        moves = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1),
        }
        # assuming we are using agent 1 and state is the grid
        grid_size = state.shape[0]
        pos = np.argwhere(state==1)
        pos = pos[0] + moves[action]
        pos = np.clip(pos, 0, grid_size-1)
        pos_target = np.argwhere(state==2)[0]
        dist = tuple(pos - pos_target)
        dist = dist[0]**2 + dist[1]**2
        return -dist
    
    def theta_estimate(self, state):
        Q_vals = self.net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        # simplest reward - at each step
        rewards = []
        for action in range(self.action_dim):
            rewards.append(self.reward_fn(state, action))
        action = np.argmax(np.array(rewards))
        Q_opt = [0]*self.action_dim
        Q_opt[action] = 1
        return Q_vals, torch.tensor(Q_opt, dtype=torch.float32)
    
    def train_step(self, state):
        self.net.train()
        Q_est, Q_opt = self.theta_estimate(state)
        loss = self.loss_fn(Q_est, Q_opt)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

class CNNNet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.model = self.cnn(action_dim)
    
    def forward(self, x):
        return self.model(x.unsqueeze(0)) # if no batch dim

    def cnn(self, output_dim):
        return nn.Sequential(
            nn.Conv2d(1, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 16, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim)
        )
    