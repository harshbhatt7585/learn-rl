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
        x = x.view(-1, 32*9*9)
        x = self.hidden_layer(x)
        x = self.action_head(x)
        return x




class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
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
    batch_size = 32
    gamma = 0.99
    
    
    dqn = DQN(num_actions=env.action_space.n)
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    
    for episode in range(num_episodes):
        # initialize sequence
        state, _ = env.reset()

        state = process_state(state)
        for t in range(max_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = dqn(state)

                # convert action logits to probs
                action_probs = nn.Softmax()(action)
                action = action_probs.argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = process_state(next_state)
            done = terminated or truncated
            replay_buffer.push(state.numpy(), action, reward, next_state.numpy(), done)
            state = next_state
        

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                s, a, r, s_, d = zip(*batch)


                s = torch.tensor(s)
                a = torch.tensor(a)
                r = torch.tensor(r)
                s_ = torch.tensor(s_)
                d = torch.tensor(d)

                q_values = dqn(s).gather(1, a.unsqueeze(1))
                next_q_value = dqn(s_).max(dim=1).values    
                
                expected_q = r + gamma * next_q_value 

                loss = F.mse_loss(q_values, expected_q.detach())

                print(loss)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        
            if done:
                break

    
        
            




            
    

        


if __name__ == "__main__":
    import gymnasium as gym
    from ale_py import ALEInterface
    ale = ALEInterface()

    env = gym.make("ALE/Assault-v5", render_mode="human") 
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