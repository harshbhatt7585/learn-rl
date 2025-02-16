import numpy as np
import random


class GridWorld:
    def __init__(self, size=4, stochasticity=0.1):
        self.size = size
        self.state = (0, 0)
        self.goal = (size-1, size-1)
        self.actions = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        self.stochasticity = stochasticity
        self.walls = [(1, 1), (2, 2)]

    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        
        if random.random() < self.stochasticity:
            action = random.choice([0, 1, 2, 3])

        x, y = self.state
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and self.size - 1:
            y += 1
        