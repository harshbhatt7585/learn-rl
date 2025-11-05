from collections import deque
from traceback import print_tb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random


class DQN(nn.Module):
    def __init__(self, hidden_dim=256, num_actions=4):
        super(DQN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.hidden_layer = nn.Linear(32*9*9, 256)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(32*9*9)
        x = self.hidden_layer(x)
        x = self.action_head(x)
        return x




class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


def process_state(state):
    state = torch.from_numpy(state).float()
    state = state.mean(dim=2)
    state = state[16:100, 16:100]
    state = state.unsqueeze(0)
    return state

def train():
    replay_buffer = ReplayBuffer(10000)
    num_episodes = 100
    max_steps = 100
    epsilon = 0.2
    
    dqn = DQN(num_actions=env.action_space.n)
    
    for episode in range(num_episodes):
        # initialize sequence
        state, _ = env.reset()

        state = process_state(state)
        for t in range(max_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
                print(action)
            else:
                action = dqn(state)
                # print(action.shape)

                # convert action logits to probs
                # action_probs = nn.Softmax()(action)
                # print(action_probs.shape)
                # action = action_probs.argmax()
                # print(action)




            
    

        


if __name__ == "__main__":
    import gymnasium as gym
    from ale_py import ALEInterface
    ale = ALEInterface()

    env = gym.make("ALE/Assault-v5")
    # state, _ = env.reset()

    # # convert to grayscale
    # state = torch.from_numpy(state).float()
    # state = state.mean(dim=2)
    # # crop it to 84x84 on the center
    # state = state[16:100, 16:100]
    # state = state.unsqueeze(0)
    # print(state.shape)

    # qdn = DQN(num_actions=env.action_space.n)
    # print(qdn(state))


    train()