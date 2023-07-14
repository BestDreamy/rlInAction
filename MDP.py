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
    def __init__(self):
        """
        2  表示障碍物
        1  表示奖励
        -1 game over
        0   0   0   1
        0   2   0   -1
        0   0   0   0

        """
        self.grid = np.zeros((3, 4)) # 传入一个元组， 创建一个矩阵
        self.grid[0][3] = 1
        self.grid[1][1] = 2
        self.grid[1][3] = -1
        self.row, self.col = self.grid.shape
        self.state = self.row * self.col
        self.action = 4
        self.reward_table = self.get_reward_table()
            
        
    def get_index_from_grid(self, pos):
        return pos[0] * self.col + pos[1]
        
    
    def get_reward_table(self):
        reward_table = np.zeros(self.state) # 创建一个向量
        for i in range(self.row):
            for j in range(self.col):
                index = self.get_index_from_grid((i, j))
                reward_table[index] = self.grid[i][j];
        return reward_table
        
    
    def get_transition_model(self):
        transition_model = np.zeros(
                # 12 * 4 * 12
                # 在当前的（状态）和（行动）下，通往下一个（状态）的概率
                self.state, self.action, self.state
            )
        for i in range(self.row):
            for j in range(self.col):
                now_state = self.get_index_from_grid(i, j)
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
                        nxt_state[a] = self.get_index_from_grid(r, c)
                else:
                    nxt_state = np.ones(self.action) * now_state
                
                # for a in range(self.action):
                    
                    
                
        return transition_model
        
    
    def dbg(self):
        print("\r{}\n{}\n".format(self.reward_table, self.reward_table.shape))
        
        
if "__main__" == __name__:
    # env = GridWorld()
    # env.dbg()
    l = [1, 2, 3]
    # print('{}\n{}\n{}\n{}'.format(l[-1], l[-4], l[3], l[6]))
    print(l[3], l[-3])
    
    
        