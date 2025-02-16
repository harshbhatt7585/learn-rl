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
        
        if (x, y) in self.walls:
            reward = -1
            new_state = self.state
        else:
            new_state = (x, y)
            reward = 10 if new_state == self.goal else -0.1
        
        self.state = new_state
        done = new_state == self.goal

        return new_state, reward, done
    
    def render(self):
        grid = np.full((self.size, self.size), ".", dtype=str)
        for wall in self.walls:
            grid[wall] = "W"
        grid[self.goal] = "G"
        x, y = self.state
        grid[x, y] = "A"
        print("\n".join([" ".join(row) for row in grid]) + "\n")


    
env = GridWorld()
env.reset()
env.render()

def value_iteration(env, gamma=0.9, threshold=0.01):
    states = [(x, y) for x in range(env.size) for y in range(env.size)]
    V = {s: 0 for s in states}

    while True:
        delta = 0
        V_new = V.copy()

        for state in states:
            if state == env.goal:
                continue # skip the goal state
            
            action_values = []
            for action in range(4):
                env.state = state
                new_state = reward, _ = env.step(action)
                action_values.append(reward + gamma * V[new_state])
            
            V_new[state] = max(action_values)
            delta = max(delta, abs(V_new[state] - V[state]))

        V = V_new
        if delta < threshold:
            break

    return V




# for _ in range(10):
#     action = np.random.choice([0,1,2,3])
#     state, reward, done = env.step(action)
#     env.render()
#     if done:
#         print("Goal Reached")
#         break


        