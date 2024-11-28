import sys
import random
from agents import Agent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import torch

agent = Agent(state_dim=8, action_dim=4)

grid_size = 20
game_time = 30
num_episodes = 5000
W, O, X1, X2 = -1, 0, 1, 2 # wall, grid, agent, goal

grid = None

def reset_grid():
    global grid
    grid = np.ones([grid_size, grid_size], dtype=np.int32) * O
    grid[5,6] = X1
    grid[random.randint(10,19),random.randint(10,19)] = X2

def blank_grid():
    grid = np.ones([grid_size, grid_size], dtype=np.int32) * O
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
    pos_goal = np.argwhere(grid == X2)[0]

    assert pos_old.size != 0, 'agent is missing from the grid'
    assert pos_goal.size != 0, 'goal is missing from the grid'

    pos = pos_old[0] + moves[action]

    pos = np.clip(pos, 0, grid_size-1)

    if np.all(pos == pos_goal):
        grid = blank_grid()
        grid[tuple(pos)] = X1
        done = 1
    
    else:
        grid = blank_grid()
        grid[tuple(pos)] = X1
        grid[tuple(pos_goal)] = X2
        done = 0
    
    return grid, done

def get_state(grid):
    pos = np.argwhere(grid == X1)[0]

    walls = [
        pos[1] == 0,             # UP
        pos[0] == grid_size - 1, # RIGHT
        pos[1] == grid_size - 1, # DOWN
        pos[0] == 0              # LEFT
    ]

    pos_goal = np.argwhere(grid == X2)

    if len(pos_goal) == 0:
        goal = [0, 0, 0, 0]
        return goal + walls

    pos_goal = pos_goal[0]

    goal = [
        pos[1] < pos_goal[1], # UP
        pos[0] < pos_goal[0], # RIGHT
        pos[1] > pos_goal[1], # DOWN
        pos[0] > pos_goal[0]  # LEFT
    ]

    return goal + walls

def get_dist(grid):
    pos = np.argwhere(grid == X1)[0]
    pos_goal = np.argwhere(grid == X2)
    if len(pos_goal) == 0:
        return 0
    pos_goal = pos_goal[0]
    return abs(pos[0] - pos_goal[0]) + abs(pos[1] - pos_goal[1]) # Manhattan distance

def plot_grid(grid, prev_grid=None):
    plt.figure(figsize=(6,6))
    cmap = ListedColormap(["white", "blue", "red", "lightblue", "lightcoral"])
    grid_vals = [0, 1, 2, 3, 6]
    if prev_grid is not None:
        grid = grid + prev_grid*3 # 1->3, 2->6
        val_map = {4:1, 7:1, 5:2, 8:2, -4:-1}
        for k, v in val_map.items():
            grid[grid == k] = v

    norm = BoundaryNorm(grid_vals + [7], cmap.N)

    plt.imshow(grid, cmap=cmap, norm=norm, extent=[0, 10, 0, 10])
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def step():
    global grid
    state = get_state(grid)
    action = agent.choose_action(state)
    grid, done = update_grid(grid, action)
    return action, done

def episode():
    reset_grid()

    for i in range(game_time):
        prev_grid = grid.copy()
        action, done = step()
        prev_state = get_state(prev_grid)
        state = get_state(grid)
        distance_reward = get_dist(prev_grid) - get_dist(grid)
        agent.short_train(prev_state, distance_reward, action, state, done)
        agent.cache(prev_state, distance_reward, action, state, done)

        if done == 1:
            reward = 10
            agent.cache(prev_state, reward, action, state, done)
            agent.long_train()
            agent.explore_rate = max(agent.min_explore, agent.explore_rate*agent.explore_decay)
            agent.game_count += 1
            print('-----------------------------')
            print(f'Game ended at iteration {i} (agent WINS)')
            print(f'Explore rate: {agent.explore_rate}')
            print('-----------------------------')
            return
        
        if agent.game_count % agent.sync_every == 0:
            agent.sync_target()

    reward = -10
    agent.cache(prev_state, reward, action, state, done)
    agent.long_train()
    agent.explore_rate = max(agent.min_explore, agent.explore_rate*agent.explore_decay)
    agent.game_count += 1
    print('-----------------------------')
    print('Game did not end (agent LOSES)')
    print(f'Explore rate: {agent.explore_rate}')
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
        print('\n')
        print(f'{iter+1}/{num_episodes}')
        episode()
    agent.explore_rate = 0
    plotted_episode()
