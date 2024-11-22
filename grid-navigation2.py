import sys
import random
from agents import Agent, CNNAgent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import torch

"""
agent learns to approach target but not how to complete the task (move to episode-level rewards)

single agent

!!! push many articial states to learn to hit the target

"""

agent = Agent((10, 10), 4)
grid_size = 10
game_time = 20
game_time = 50
num_episodes = 5000

O, X1, X2 = 0, 1, 2
"""
USED TOKENS:
0, 1, 2, 3, 6
"""

def reset_grid():
    grid = np.ones([grid_size, grid_size], dtype=np.int32) * O
    while True:
        x1_pos = (random.randint(0,grid_size-1), random.randint(0,grid_size-1))
        x2_pos = (random.randint(0,grid_size-1), random.randint(0,grid_size-1))
        #x1_pos = (0, 0)
        #x2_pos = (grid_size-1, grid_size-1)
        if x1_pos != x2_pos:
            break
    grid[x1_pos] = X1
    grid[x2_pos] = X2
    return grid

def update_grid(grid, action):
    moves = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
        999: (0, 0),
    }

    pos_old = np.argwhere(grid == X1)
    pos_target = np.argwhere(grid == X2)[0]

    assert pos_old.size != 0, 'agent is missing from the grid'

    pos = pos_old[0] + moves[action]

    pos = np.clip(pos, 0, grid_size-1)

    if np.all(pos == pos_target):
        grid.fill(0)
        grid[tuple(pos)] = X1
        done = 1
    
    else:
        grid.fill(0)
        grid[tuple(pos)] = X1
        grid[tuple(pos_target)] = X2
        done = 0
    
    return grid, done

def plot_grid(grid, prev_grid=None):
    plt.figure(figsize=(6,6))
    cmap = ListedColormap(["white", "blue", "red", "lightblue"])
    grid_vals = [0, 1, 2, 3]
    if prev_grid is not None:
        grid = grid + prev_grid*3 # 1->3, 2->6
        val_map = {4:1, 7:1, 5:2, 8:2}
        for k, v in val_map.items():
            grid[grid == k] = v

    norm = BoundaryNorm(grid_vals + [4], cmap.N)

    plt.imshow(grid, cmap=cmap, norm=norm, extent=[0, 10, 0, 10])
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

def get_dist(grid):
    dist = np.argwhere(grid == X1)[0] - np.argwhere(grid == X2)[0]
    dist = tuple(dist)
    dist = np.sqrt(dist[0]**2 + dist[1]**2)
    return dist

def get_reward(state, next_state):
    dist = get_dist(state)
    next_dist = get_dist(next_state)
    r = dist - next_dist
    return r * 10 # try amplification

def step(grid):
    action, _ = agent.select_action(grid)
    grid, done = update_grid(grid, action)
    return grid, action, done

def stage1_episode():
    grid = reset_grid()

    for i in range(game_time):
        prev_grid = grid.copy()
        grid, action, done = step(grid)

        if done == 1:
            reward = 100.0 # for completing the task
            agent.cache(prev_grid, grid, action, reward, done)
            print('-----------------------------')
            print(f'Game ended at iteration {i} (agent 1 WINS)')
            print('-----------------------------')
            return
        
        reward = get_reward(prev_grid, grid)

        if i % agent.update_every == 0:
            agent.learn()

        if i % agent.sync_every == 0:
            agent.sync_target()
        
        agent.cache(prev_grid, grid, action, reward, done)

    print('-----------------------------')
    print('Game did not end (agent 1 LOSES)')
    print('-----------------------------')

def stage2_episode(ep_idx): # switch to episode-level rewards
                            # we only log the first and final steps (can use different method)
    grid = reset_grid()
    first_grid = grid.copy()

    for i in range(game_time):
        prev_grid = grid.copy()
        grid, action, done = step(grid)
        if i == 0:
            second_grid = grid.copy()
            first_action = action

        if done == 1:
            reward = 10.0
            #agent.cache(first_grid, second_grid, first_action, reward, 0, memory=2)
            agent.cache(prev_grid, grid, action, reward, 1, memory=2)
            print('-----------------------------')
            print(f'Game ended at iteration {i} (agent 1 WINS)')
            print('-----------------------------')
            return

        if ep_idx % agent.update_every == 0:
            agent.learn(memory=2)

        if ep_idx % agent.sync_every == 0:
            agent.sync_target()
        
        reward = -10.0
        #agent.cache(first_grid, second_grid, first_action, reward, 0, memory=2)
        agent.cache(prev_grid, grid, action, reward, 1, memory=2)

    print('-----------------------------')
    print('Game did not end (agent 1 LOSES)')
    print('-----------------------------')

def plotted_episode():
    total_reward = 0
    grid = reset_grid()

    for i in range(game_time):
        prev_grid = grid.copy()
        grid, action, done = step(grid)
        if done == 1:
            print('-----------------------------')
            print(f'Game ended at iteration {i} (agent 1 WINS)')
            print('-----------------------------')
            return total_reward
        plot_grid(grid, prev_grid)

    print('-----------------------------')
    print('Game did not end (agent 1 LOSES)')
    print('-----------------------------')
    return total_reward


def main():


    for iter in range(num_episodes):
        print(f'{iter+1}/{num_episodes}')
        stage1_episode()
    agent.save_models('top_to_bottom.pth')
    plotted_episode()
    return
    agent.save_models('agent_net_stage1.pth')
    for iter in range(num_episodes):
        print(f'{iter+1}/{num_episodes}')
        stage2_episode(iter)
    agent.save_models('agent_net_stage2.pth')
    for _ in range(3):
        plotted_episode()

if __name__ == '__main__':
    main()
