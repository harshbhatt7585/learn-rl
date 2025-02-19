import gym
import random
import numpy as np

env = gym.make('CartPole-v1', render_mode="human")  # Use "human" for normal rendering
state = env.reset()[0]


gamma = 0.9
alpha = 0.2
epsilon = 0.1


# V = np.zeros([4, len(range(0.1, -4.8, 4.8))])
# V = np.zeros([env.observation_space.n])

# for episode in range(1000):
#     for _ in range(500): 
#         env.render()  
#         delta = 0
#         action_values = []
#         for action in range(env.action_space.n):
#             new_state, reward, done, _, _ = env.step(action)
#             action_values.append(reward + gamma * V[new_state])
        
#         old_value = V[state].copy()
#         V[state] = max(action_values)
#         delta = max(delta, abs(V[state], old_value))

#         if delta < 0.01:
#             break
        
# env.close()  