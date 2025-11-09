import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import gymnasium as gym
import wandb


# -------------------------------
# Initialize Weights & Biases
# -------------------------------
wandb.init(
    project="dqn-assault",
    config={
        "num_episodes": 200,
        "max_steps": 500,
        "epsilon": 0.2,
        "batch_size": 32,
        "gamma": 0.99,
        "lr": 1e-3,
        "buffer_capacity": 10000,
    },
)

config = wandb.config


# -------------------------------
# DQN Model
# -------------------------------
class DQN(nn.Module):
    def __init__(self, hidden_dim=256, num_actions=4):
        super(DQN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.hidden_layer = nn.Linear(32 * 9 * 9, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.hidden_layer(x))
        x = self.action_head(x)
        return x


# -------------------------------
# Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# -------------------------------
# Preprocess function
# -------------------------------
def process_state(state):
    state = torch.from_numpy(state).float()
    state = state.mean(dim=2)  # grayscale
    state = state[16:100, 16:100]  # crop to 84x84
    state = state.unsqueeze(0)  # add channel dim
    return state / 255.0


# -------------------------------
# Training Loop
# -------------------------------
def train(env):
    replay_buffer = ReplayBuffer(config.buffer_capacity)
    dqn = DQN(num_actions=env.action_space.n)
    optimizer = optim.Adam(dqn.parameters(), lr=config.lr)

    epsilon = config.epsilon

    for episode in range(config.num_episodes):
        state, _ = env.reset()
        state = process_state(state).unsqueeze(0)
        episode_reward = 0
        episode_loss = 0

        for t in range(config.max_steps):
            # ε-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = dqn(state)
                    action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            next_state = process_state(next_state).unsqueeze(0)
            replay_buffer.push(state.numpy(), action, reward, next_state.numpy(), done)
            state = next_state

            # Training step
            if len(replay_buffer) > config.batch_size:
                batch = replay_buffer.sample(config.batch_size)
                s, a, r, s_, d = zip(*batch)

                s = torch.tensor(np.array(s), dtype=torch.float32).squeeze(1)
                a = torch.tensor(a, dtype=torch.long)
                r = torch.tensor(r, dtype=torch.float32)
                s_ = torch.tensor(np.array(s_), dtype=torch.float32).squeeze(1)
                d = torch.tensor(d, dtype=torch.float32)

                q_values = dqn(s).gather(1, a.unsqueeze(1)).squeeze(1)
                next_q_value = dqn(s_).max(dim=1).values
                expected_q = r + config.gamma * next_q_value * (1 - d)

                loss = F.mse_loss(q_values, expected_q.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_loss += loss.item()

            if done:
                break

        # ε-decay
        epsilon = max(0.01, epsilon * 0.995)

        # Log metrics to wandb
        wandb.log({
            "episode": episode,
            "reward": episode_reward,
            "loss": episode_loss / max(1, t),
            "epsilon": epsilon,
        })

        print(f"Episode {episode+1} | Reward: {episode_reward:.2f} | Loss: {episode_loss:.4f} | ε: {epsilon:.3f}")

    wandb.finish()


# -------------------------------
# Run Training
# -------------------------------
if __name__ == "__main__":
    from ale_py import ALEInterface
    ale = ALEInterface()
    env = gym.make("ALE/Assault-v5", render_mode="human")
    train(env)
