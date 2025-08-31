from basic_environment import BasicGridWorld
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F


class PPOTrainer:
    def __init__(self, env, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, lambda_gae=0.95, action_std_init=0.6):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.has_continuous_action_space = has_continuous_action_space
        self.lambda_gae = lambda_gae

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

        self.buffer = []

        

    
    def train(self):
        self.actor.train()
        self.critic.train()

        state = self.env.reset()

        for epoch in range(self.K_epochs):
            
            # sample action from policy
            logits = self.actor(state)  # raw scores from actor network
            action_probs = torch.softmax(logits, dim=-1) # convert scores to probabilities distribution
            dist = Categorical(action_probs) # create catergorical distribution 
            action = dist.sample() # sample an action from the distribution
            log_prob = dist.log_prob(action) # log probability of the action

            # compute value from critic
            state_value = self.critic(state)
            state_value = state_value.squeeze(0)


            # take step and get reward
            new_state, reward, done = self.env.step(action)


            # store (state, action, reward)
            self.buffer.append((state, action, log_prob, state_value, reward, done))


            # calculate advantage (GAE)
            states, actions, log_probs, values, returns, advantages = self.compute_gae() 
            


            # calculate loss
            # get new log probs 
            logits = self.actor(states)
            action_probs = torch.softmax(logits, dim=-1)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            # Value
            new_values = self.critic(states).squeeze(-1)
            value_loss = F.mse_loss(new_values, returns)

            # Entropy
            entropy = dist.entropy().mean()

            # actor loss
            actor_loss = policy_loss + 0.01 * entropy
            # critic loss
            critic_loss = value_loss

            # update policy
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # clear buffer
        self.buffer.clear()

            
        
    def compute_gae(self):
        states, actions, log_probs, values, rewards, dones = zip(*self.buffer)

        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)

        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            # mask is used for eliminating the last value
            mask = 1.0 - float(dones[t])
            next_value = values[t + 1] if t < len(rewards) - 1 else 0
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lambda_gae * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        
        return states, actions, log_probs, values, returns, advantages






if __name__ == "__main__":
    env = BasicGridWorld()
    trainer = PPOTrainer(env, state_dim=1, action_dim=1, lr_actor=0.001, lr_critic=0.001, gamma=0.99, K_epochs=10, eps_clip=0.2, has_continuous_action_space=False)
    trainer.train()
    
