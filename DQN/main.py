import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, hidden_dim=256, num_actions=4):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(81, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    import gymnasium as gym
    from ale_py import ALEInterface
    ale = ALEInterface()

    env = gym.make("ALE/Assault-v5")
    state, _ = env.reset()

    # convert to grayscale
    state = torch.from_numpy(state).float()
    state = state.mean(dim=2)
    # crop it to 84x84 on the center
    state = state[16:100, 16:100]
    state = state.unsqueeze(0)
    print(state.shape)

    qdn = DQN(num_actions=env.action_space.n)
    print(qdn(state))
