import torch
import torch.nn as nn
import pygame
import matplotlib.pyplot as plt
from IPython import display
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage
from tensordict import TensorDict

import numpy as np
from snake_game import SnakeGame, Point, Direction

plt.ion()

"""
takes in a vectorised compression of the state (eg. direction in which danger / reward is + direction of motion)

try using CNN + image (or convert to a matrix grid)

"""

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

class SnakeAgent():
    def __init__(self):
        self.explore_rate = 1.0
        self.explore_decay = 0.995
        self.min_explore = 0.0
        self.gamma = 0.9
        self.sync_every = 5 # games

        self.batch_size = 1000
        self.lr = 0.001

        self.state_dim = 11
        self.action_dim = 3
        self.game_count = 0

        self.memory = TensorDictReplayBuffer(storage=LazyTensorStorage(100000))

        self.online_net = AgentNet(self.state_dim, self.action_dim, self.lr) # contains .model and .optimizer
        self.target_net = AgentNet(self.state_dim, self.action_dim, self.lr)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.loss_fn = torch.nn.MSELoss()
        
    def get_state(self, game): # calculates state vectorl
        goal = game.food
        head = game.head
        direction = game.direction

        direction_map = {
            1: [1, 0, 0, 0],
            2: [0, 1, 0, 0],
            3: [0, 0, 1, 0],
            4: [0, 0, 0, 1]
        }

        direction_array = direction_map[direction]
        direction_array = [bool(d) for d in direction_array]

        pt_r = Point(head.x+20, head.y)
        pt_d = Point(head.x, head.y-20)
        pt_l = Point(head.x-20, head.y)
        pt_u = Point(head.x, head.y+20)

        coll_r = game._is_collision(pt_r)
        coll_d = game._is_collision(pt_d)
        coll_l = game._is_collision(pt_l)
        coll_u = game._is_collision(pt_u)

        danger = [
            # straight
            (direction == Direction.RIGHT.value and coll_r) or
            (direction == Direction.DOWN.value and coll_d) or
            (direction == Direction.LEFT.value and coll_l) or
            (direction == Direction.UP.value and coll_u),

            # right
            (direction == Direction.UP.value and coll_r) or
            (direction == Direction.RIGHT.value and coll_d) or
            (direction == Direction.DOWN.value and coll_l) or
            (direction == Direction.LEFT.value and coll_u),

            # left
            (direction == Direction.DOWN.value and coll_r) or
            (direction == Direction.LEFT.value and coll_d) or
            (direction == Direction.UP.value and coll_l) or
            (direction == Direction.RIGHT.value and coll_u),
        ]

        goal_direction = [
            goal.x > head.x,
            goal.y < head.y,
            goal.x < head.x,
            goal.y > head.y
        ]

        return danger + goal_direction + direction_array
    
    def choose_action(self, state):
        if np.random.uniform() < self.explore_rate:
            action = np.random.randint(0, self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            Q_vals = self.online_net(state)
            action = torch.argmax(Q_vals)
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
        self.train_step(state, reward, action, next_state, done)

    def train_step(self, state, reward, action, next_state, done):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            reward = reward.unsqueeze(0)
            action = action.unsqueeze(0)
            done = done.unsqueeze(0)
        """
        pred = self.online_net(state)
        pred_actions = torch.argmax(self.online_net(next_state), axis=1)
        target = pred.clone()
        next_Q = self.net(next_state) # where we would use the target model
        for idx, a in enumerate(pred_actions.detach().tolist()):
            if done[idx]:
                target_Q = reward[idx]
            else:
                target_Q = reward[idx] + torch.max(next_Q[idx]) # just uses max again?
            target[idx, a] = target_Q
        """
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
        torch.save(self.online_net.model, 'snake_model.pth')

    def load_model(self):
        self.online_net.model = torch.load('snake_model.pth')
    

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
            #nn.Sigmoid()
        )    

    def forward(self, x):
        return self.model(x)
    


def train():
    scores = []
    mean_scores = []
    tot_score = 0
    record = 0

    agent = SnakeAgent()
    game = SnakeGame()
    while True:
        state = agent.get_state(game)
        action = agent.choose_action(state) # note here action is a single int
        reward, done, score = game.play_step(action)
        next_state = agent.get_state(game)
        agent.short_train(state, reward, action, next_state, done)
        agent.cache(state, reward, action, next_state, done)

        if done:
            agent.long_train() # experience replay
            game.reset()
            agent.game_count += 1

            agent.explore_rate = max(agent.min_explore, agent.explore_rate*agent.explore_decay)

            if agent.game_count % agent.sync_every == 0:
                agent.sync_target()

            if score > record:
                agent.save_model()
            record = max(record, score)

            scores.append(score)
            tot_score += score
            mean_scores.append(tot_score / agent.game_count)
            plot(scores, mean_scores)
            print('\n')
            print(f'--- Game {agent.game_count + 1} ---')
            print(f'Score: {score}')
            print(f'Explore rate: {agent.explore_rate}')


def play():
    agent = SnakeAgent()
    game = SnakeGame()
    for p in agent.online_net.model.parameters():
        p.requires_grad = False
    agent.load_model()
    agent.explore_rate = 0
    record = 0

    while True:
        state = agent.get_state(game)
        action = agent.choose_action(state)
        _, done, score = game.play_step(action)

        if done:
            game.reset()
            agent.game_count += 1
            record = max(record, score)
            print('\n')
            print(f'--- Game {agent.game_count + 1} ---')
            print(f'Score: {score}')
            print(f'Record: {record}')

if __name__ == '__main__':
    train()
    #play()