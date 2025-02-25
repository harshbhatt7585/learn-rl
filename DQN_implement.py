import gym
import numpy as np
import random

def dqn(env, epsilon=0, gamma=0.9):
    replay_buffer = []
    Q_net = np.ndarray((env.obervation_space.n, env.action_space.n))
    Q_target_net = Q_net.copy()
    
    state = env.reset()[0]
    done = False
    while not done:
        optim.zero_grad()
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_target_net[state])
        
        new_state, reward, done, _, _ = env.step(action)
        replay_buffer.append((state, reward, action, new_state))
        sample_buffer = np.random.choice(replay_buffer)

        pred_value = Q_net[state, action]

        if done:
            target_value = Q_target_net[sample_buffer[0], sample_buffer[1]]
        else:
            new_action = np.argmax(Q_target_net[new_state])
            target_value = Q_target_net[new_state, new_action]
        
        loss =  ( (reward + gamma * target_value) - pred_value )**2
        loss.backward()
        optim.step()

    




