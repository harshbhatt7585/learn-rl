import gym
import numpy as np
import random

# Create Taxi-v3 environment
env = gym.make('Taxi-v3')
state, _ = env.reset()  # Unpack (state, info) if using Gymnasium
# env.render()

# Initialize Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
gamma = 0.7  # Discount factor
alpha = 0.2  # Learning rate
epsilon = 0.1  # Exploration probability

# Training loop
for episode in range(1000):
    done = False
    total_reward = 0
    state, _ = env.reset() 

    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])  # Exploit
        
        # Take action and observe result
        next_state, reward, done, _, _ = env.step(action)

        # Q-learning update rule
        next_max = np.max(Q[next_state])
        old_value = Q[state, action]
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)

        Q[state, action] = new_value
        total_reward += reward
        state = next_state  # Move to next state

    # Print progress every 100 episodes
    if episode % 100 == 0:
        print("Episode {} Total Reward: {}".format(episode, total_reward))