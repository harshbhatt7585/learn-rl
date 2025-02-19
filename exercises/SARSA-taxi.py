import gym 
import numpy as np
import random

env = gym.make('Taxi-v3')
state, _ = env.reset()

alpha = 0.2
gamma = 0.9
epsilon = 0.1

Q = np.zeros([env.observation_space.n, env.action_space.n])

for episode in range(1000):
    state, _ = env.reset()
    done = False
    total_reward = 0

    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])

    while not done:
        next_state, reward, done, _, _ = env.step(action)

        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q[next_state])

        old_value = Q[state, action]
        next_value = Q[next_state, next_action]
        new_value = old_value + alpha * (reward + gamma * next_value - old_value)
        
        Q[state, action] = new_value
        total_reward += reward
        state, action = next_state, next_action

    if episode % 100 == 0:
        print("Episode {} Total Reward: {}".format(episode, total_reward))