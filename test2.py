import sys
import random
from agents import Agent, CNNAgent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import torch

"""
! teach the agents they can't move in the direction of an obstacle -> use walls as padding all round

use step reward as pre-training then move to episode reward

add large reward when agent reaches target

add wall behaviour
"""

# ---
agent1 = Agent((10, 10), 4)
agent2 = Agent((10, 10), 4)
grid_size = 20
game_time = 30
num_episodes = 1000
# ---
W, O, X1, X2 = -1, 0, 1, 2
"""
USED TOKENS:
-1, 0, 1, 2, 3, 6
"""

grid = np.ones([grid_size, grid_size], dtype=np.int32) * O
grid[5,6] = X1
grid[14,19] = X2
grid[:,0] = W
grid[:,grid_size-1] = W
grid[0,:] = W
grid[grid_size-1,:] = W

def reset_grid():
    global grid
    grid = np.ones([grid_size, grid_size], dtype=np.int32) * O
    #grid[:,0] = W
    #grid[:,grid_size-1] = W
    #grid[0,:] = W
    #grid[grid_size-1,:] = W
    grid[5,6] = X1
    grid[random.randint(10,19),random.randint(10,19)] = X2

def blank_grid():
    grid = np.ones([grid_size, grid_size], dtype=np.int32) * O
    return grid
    grid[:,0] = W
    grid[:,grid_size-1] = W
    grid[0,:] = W
    grid[grid_size-1,:] = W
    return grid

# turn the grid + reward etc. into an environmnet class

def update_grid(grid, action1, action2):
    moves = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
        999: (0, 0),
    }

    pos1_old = np.argwhere(grid == X1)
    pos2_old = np.argwhere(grid == X2)

    assert pos1_old.size != 0, 'agent 1 is missing from the grid'
    assert pos2_old.size != 0, 'agent 2 is missing from the grid'

    pos1 = pos1_old[0] + moves[action1]
    pos2 = pos2_old[0] + moves[action2]

    pos1 = np.clip(pos1, 0, grid_size-1)
    pos2 = np.clip(pos2, 0, grid_size-1)

    if np.all(pos1 == pos2):
        grid = blank_grid()
        grid[tuple(pos1)] = X1
        done = 1
    
    else:
        grid = blank_grid()
        grid[tuple(pos1)] = X1
        grid[tuple(pos2)] = X2
        done = 0
    
    return grid, done

def plot_grid(grid, prev_grid=None):
    plt.figure(figsize=(6,6))
    cmap = ListedColormap(["grey", "white", "blue", "red", "lightblue", "lightcoral"])
    grid_vals = [-1, 0, 1, 2, 3, 6]
    if prev_grid is not None:
        grid = grid + prev_grid*3 # 1->3, 2->6
        val_map = {4:1, 7:1, 5:2, 8:2, -4:-1} # since the walls don't move between steps
        for k, v in val_map.items():
            grid[grid == k] = v

    norm = BoundaryNorm(grid_vals + [7], cmap.N)

    plt.imshow(grid, cmap=cmap, norm=norm, extent=[0, 10, 0, 10])
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def get_dist(grid):
    dist = np.argwhere(grid == X1)[0] - np.argwhere(grid == X2)[0]
    dist = tuple(dist)
    dist = np.sqrt(dist[0]**2 + dist[1]**2)
    return dist # reward is -dist for agent1, +dist for agent2

def get_rewards(state, next_state):
    dist = get_dist(state)
    next_dist = get_dist(next_state)
    reward1 = dist - next_dist # reward agent 1 for moving closer to agent 2
    return reward1, -reward1

def step():
    global grid
    assert grid is not None, '------ no grid found ------'
    action1, _ = agent1.select_action(grid)
    action2, _ = agent2.select_action(grid)
    grid, done = update_grid(grid, action1, action2)
    return action1, action2, done

def episode():
    reset_grid()

    for i in range(game_time):
        prev_grid = grid.copy()
        action1, action2, done = step()

        if done == 1:
            reward1 = 10.0 # for completing the task
            reward2 = -10.0
            agent1.cache(prev_grid, grid, action1, reward1, done)
            agent2.cache(prev_grid, grid, action2, reward2, done)
            print('-----------------------------')
            print(f'Game ended at iteration {i} (agent 1 WINS)')
            print('-----------------------------')
            return
        
        reward1, reward2 = get_rewards(prev_grid, grid)

        if i % agent1.update_every == 0:
            agent1.learn()

        if i % agent1.update_every == 0:
            agent1.sync_target()

        if i % agent2.update_every == 0:
            agent2.learn()

        if i % agent2.update_every == 0:
            agent2.sync_target()
        
        agent1.cache(prev_grid, grid, action1, reward1, done)
        agent2.cache(prev_grid, grid, action2, reward2, done)

    print('-----------------------------')
    print('Game did not end (agent 1 LOSES)')
    print('-----------------------------')



def plotted_episode():
    total_reward = 0
    reset_grid()

    for i in range(game_time):
        prev_grid = grid.copy()
        step()
        if grid is None:
            print('-----------------------------')
            print(f'Game ended at iteration {i} (agent 1 WINS)')
            print('-----------------------------')
            return total_reward
        plot_grid(grid, prev_grid)

    print('-----------------------------')
    print('Game did not end (agent 1 LOSES)')
    print('-----------------------------')
    return total_reward



if __name__ == '__main__':
    for iter in range(num_episodes):
        print(f'{iter+1}/{num_episodes}')
        episode()
    plotted_episode()
