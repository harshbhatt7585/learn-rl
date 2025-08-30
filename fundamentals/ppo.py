from basic_environment import BasicGridWorld
import torch
import torch.nn as nn


class PPOTrainer:
    def __init__(self, env, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.has_continuous_action_space = has_continuous_action_space

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)


    
    def train(self):
        for epoch in range(self.K_epochs):
            
            # sample action from policy


            # compute value from critic


            # take step and get reward


            # store (state, action, reward)


            # calculate advantage (GAE)


            # calculate loss


            # optimize (update policy and value network)




if __name__ == "__main__":
    env = BasicGridWorld()
    trainer = PPOTrainer(env, state_dim=1, action_dim=1, lr_actor=0.001, lr_critic=0.001, gamma=0.99, K_epochs=10, eps_clip=0.2, has_continuous_action_space=False)
    trainer.train()
    
