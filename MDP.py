# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 20:29:02 2023

@author: Aric
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.xkcd()

class GridWorld:
    """
        2  表示障碍物
        1  表示奖励
        -1 game over
        0   0   0   1
        0   2   0   -1
        0   0   0   0

    """
    def __init__(self):
        self.grid = np.zeros((3, 4)) # 传入一个元组， 创建一个矩阵
        self.grid[0][3] = 1
        self.grid[1][1] = 2
        self.grid[1][3] = -1
        self.row, self.col = self.grid.shape
        self.state = self.row * self.col
        self.action = 4
        self.reward_table = self.get_reward_table()
        self.transition_table = self.get_transition_table()
            
        
    def get_index_from_grid(self, pos):
        return pos[0] * self.col + pos[1]
    

    def get_pos_from_state(self, state):
        return state // self.col, state % self.col
    
    
    def get_reward_table(self):
        reward_table = np.zeros(self.state) # 创建一个向量
        for i in range(self.row):
            for j in range(self.col):
                index = self.get_index_from_grid((i, j))
                reward_table[index] = self.grid[i][j];
        return reward_table
        
    
    def get_transition_table(self, random_rate = 0.2):
        transition_table = np.zeros(
                # 12 * 4 * 12
                # 在当前的（状态）和（行动）下，通往下一个（状态）的概率
                (self.state, self.action, self.state)
            )
        for i in range(self.row):
            for j in range(self.col):
                now_state = self.get_index_from_grid((i, j))
                # 当前的状态(i, j) 采取 0 ~ 3 类行动可以到达的新状态
                nxt_state = np.zeros(self.action)
                if self.grid[i][j] == 0: # 空白格子，可以继续前进
                    for a in range(self.action):
                        r, c = i, j
                        if a == 0:
                            r = max(0, i - 1)
                        elif a == 1:
                            c = min(self.col - 1, j + 1)
                        elif a == 2:
                            r = min(self.row - 1, i + 1)
                        elif a == 3:
                            c = max(0, j - 1)
                        if self.grid[r][c] == 2: # 障碍无法停留
                            r, c = i, j
                        nxt_state[a] = self.get_index_from_grid((r, c))
                else:
                    nxt_state = np.ones(self.action) * now_state
                
                for a in range(self.action):
                    transition_table[now_state][a][int(nxt_state[a])] += 1 - random_rate
                    transition_table[now_state][a][int(nxt_state[(a + 1) % self.action])] += random_rate / 2
                    transition_table[now_state][a][int(nxt_state[a - 1])] += random_rate / 2
                    
        return transition_table
        
    """
    回到初始位置  (2, 0)
    """
    def reset(self):
        self.start_pos = (2, 0)
        self.now_state = self.get_index_from_grid(self.start_pos)
        self.reward = 0
        return self.now_state
        

    def step(self, action):
        p = self.transition_table[self.now_state, action]
        nxt_state = np.random.choice(self.state, p=p)
        self.reward = self.reward_table[nxt_state]
        self.now_state = nxt_state
        done = False
        if self.reward != 0:
            done = True
        return self.now_state, self.reward, done, []
     
        
    def render(self):
        # 设置画布的大小为 8 * 6
        unit = 2
        fig_size = (self.col * unit, self.row * unit)
        fig, ax = plt.subplots(1, 1, figsize = fig_size)
        ax.axis('off')
        
        """
        画布坐标
        ^
        |
        |
        |
     (0,0)---------->    
        
        在 y 轴画一条长度为 6 的线
        ax.plot([0, 0], [0, 6])
        """
        
        for i in range(self.col + 1):
            if i == 0 or i == self.col:
                ax.plot([i * unit, i * unit], [0, self.row * unit], color = 'black')
            else:
                ax.plot([i * unit, i * unit], [0, self.row * unit], color = 'grey', 
                                            alpha = 0.7, linestyle = 'dashed')
        for i in range(self.row + 1):
            if i == 0 or i == self.row:
                ax.plot([0, unit * self.col], [i * unit, i * unit], color = 'black')
            else:
                ax.plot([0, unit * self.col], [i * unit, i * unit], color = 'grey', 
                                            alpha = 0.7, linestyle = 'dashed')
        
        for i in range(self.row):
            for j in range(self.col):
                x = j * unit
                y = (self.row - 1 - i) * unit
                if self.grid[i][j] == 2:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor = 'none', 
                                             facecolor = 'black', alpha = 0.6)
                    ax.add_patch(rect)
                elif self.grid[i][j] == 1:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='red',
                                             alpha = 0.6)
                    ax.add_patch(rect)
                elif self.grid[i][j] == -1:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='green',
                                             alpha = 0.6)
                    ax.add_patch(rect)
        
        i, j = self.get_pos_from_state(self.now_state)
        y = (self.row - 1 - i) * unit
        x = j * unit
        ax.plot([x + 0.5 * unit], [y + 0.5 * unit], marker = "o",
                linestyle = 'none', markersize = max(fig_size) * unit, color = 'blue')
        wait_time = 0.1
        if self.grid[i][j] != 0:
            ax.text(fig_size[0] / 3, fig_size[1] * 2 / 3,
                    s = "episode ends, reward: {:.2f}".format(self.reward))
            wait_time += 2

        plt.show(block = False)
        plt.pause(wait_time)
        plt.show()
        


class Robot:
    def __init__(self, action = 4):
        self.action = action
        
        
    def choose_action(self):
        # 随机返回一个 [0, action) 区间的数
        return np.random.randint(self.action)


if "__main__" == __name__:
    env = GridWorld()
    agent = Robot(4)
    EPISODE = 10
    for episode in range(EPISODE):
        now_state = env.reset()
        while 1:
            choose_action = agent.choose_action()
            now_state, reward, done, info = env.step(choose_action)
            env.render()
            if done:
                break 


