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
        elif action == 3 and y < self.size - 1:
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


    


def q_learning(env, alpha=0.1, gaama=0.9, epsilon=0.2, episodes=1000):
    states = [(x, y) for x in env.size for y in env.size]
    Q = {(s, a): 0 for s in states for a in range(4)}

    for episode in range(episodes):
        state = env.reset()
        Q_new = Q.copy()
        while True:
            if random.random() < epsilon:
                action = random.choice(range(4)) # Explore (random action)
            else:
                action = max(range(4), key=lambda a: Q_new[state, a])   # Exploit (best action)

            new_state, reward, done = env.step(action)

            # update bellman equation
            best_next_action = max(Q[(new_state, a)] for a in range(4))
            Q[(state, action)] = alpha * (reward + (gaama * (best_next_action - Q[(state, action)])))

            state = new_state

            if done:
                break
    
    return Q   # return learned Q-values

