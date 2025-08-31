from basic_environment import BasicGridWorld
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F


class PPOTrainer:
    def __init__(self, env, state_dim, action_dim, lr_actor, lr_critic, gamma, rollout_steps, eps_clip, has_continuous_action_space, lambda_gae=0.95, action_std_init=0.6):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.rollout_steps = rollout_steps
        self.eps_clip = eps_clip
        self.has_continuous_action_space = has_continuous_action_space
        self.lambda_gae = lambda_gae

        self.actor = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = []

    def train(self):
        self.actor.train()
        self.critic.train()

        # Reset environment and get initial state
        _, state = self.env.reset()
        state = torch.tensor([float(state)], dtype=torch.float32).unsqueeze(0)  # shape: (1, 1)

        # 1. Collect experiences (rollout)
        for step in range(self.rollout_steps):
            # Sample action from policy
            logits = self.actor(state)                   # Actor network output
            action_probs = torch.softmax(logits, dim=-1) # Convert to probabilities
            dist = Categorical(action_probs)            # Create categorical distribution
            action = dist.sample()                       # Sample action
            log_prob = dist.log_prob(action).detach()            # Log probability of the action (detached)

            # Compute value from critic
            state_value = self.critic(state).squeeze().detach()

            # Take step in environment
            new_state, reward, done = self.env.step(action.item())

            # Store experience in buffer
            self.buffer.append((state.squeeze(), action, log_prob, state_value, reward, done))

            # Update current state
            state = torch.tensor([float(new_state)], dtype=torch.float32).unsqueeze(0)
            
            # Reset if episode is done
            if done:
                _, state = self.env.reset()
                state = torch.tensor([float(state)], dtype=torch.float32).unsqueeze(0)

        # 2. Compute advantages and returns (GAE)
        states, actions, log_probs, values, returns, advantages = self.compute_gae()

        # 3. Calculate new log probabilities
        logits = self.actor(states.unsqueeze(1))  # Add feature dimension
        action_probs = torch.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)

        # 4. Compute PPO surrogate loss
        ratio = torch.exp(new_log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 5. Compute value loss
        new_values = self.critic(states.unsqueeze(1)).squeeze()
        value_loss = F.mse_loss(new_values, returns)

        # 6. Compute entropy bonus
        entropy = dist.entropy().mean()

        # 7. Separate actor and critic updates
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss = policy_loss - 0.01 * entropy  # subtract entropy to encourage exploration
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss = 0.5 * value_loss  # scale value loss
        critic_loss.backward()
        self.critic_optimizer.step()

        # Clear buffer for next rollout
        self.buffer.clear()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

    def compute_gae(self):
        states, actions, log_probs, values, rewards, dones = zip(*self.buffer)

        # Compute targets without tracking gradients to avoid graph reuse
        with torch.no_grad():
            states = torch.stack(states)
            actions = torch.stack(actions)
            log_probs = torch.stack(log_probs)
            values = torch.stack(values)
            
            advantages = torch.zeros_like(values)
            returns = torch.zeros_like(values)

            gae = 0
            next_value = 0

            for t in reversed(range(len(rewards))):
                # Mask for terminal states
                mask = 1.0 - float(dones[t])
                next_value = values[t + 1] if t < len(rewards) - 1 else 0
                delta = rewards[t] + self.gamma * next_value * mask - values[t]
                gae = delta + self.gamma * self.lambda_gae * mask * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, log_probs, values, returns, advantages


if __name__ == "__main__":
    env = BasicGridWorld()
    trainer = PPOTrainer(
        env=env, 
        state_dim=1,  # Using current_place as single state feature
        action_dim=len(env.actions), 
        lr_actor=0.001, 
        lr_critic=0.001, 
        gamma=0.99, 
        rollout_steps=20,  # Increased for better learning
        eps_clip=0.2, 
        has_continuous_action_space=False
    )
    
    # Train for multiple iterations
    for i in range(100):
        metrics = trainer.train()
        if i % 10 == 0:
            print(f"Iteration {i}: Policy Loss: {metrics['policy_loss']:.4f}, "
                  f"Value Loss: {metrics['value_loss']:.4f}, Entropy: {metrics['entropy']:.4f}")