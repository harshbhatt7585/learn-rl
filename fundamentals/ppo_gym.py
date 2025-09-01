import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F
from plotter import LivePlotter
import numpy as np


class PPOTrainer:
    def __init__(self, env, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, rollout_steps=2048, 
                 eps_clip=0.2, lambda_gae=0.95, k_epochs=4, entropy_coef=0.001, max_grad_norm=0.5):
        self.env = env
        self.gamma = gamma
        self.rollout_steps = rollout_steps
        self.eps_clip = eps_clip
        self.lambda_gae = lambda_gae
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Get environment dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n  # Assuming discrete action space
        
        # Device placement (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
        # Networks - smaller networks for CartPole
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dim),
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.buffer = []

    def train(self):
        self.actor.train()
        self.critic.train()
        
        # Reset environment and get initial state
        state, _ = self.env.reset()  # Gymnasium returns (obs, info)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        episode_return = 0.0
        episode_returns = []
        episode_lengths = []
        episode_length = 0

        # 1. Collect experiences (rollout)
        with torch.inference_mode():
            for step in range(self.rollout_steps):
                # Sample action from policy using logits directly
                logits = self.actor(state)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # Compute value from critic
                state_value = self.critic(state).squeeze()

                # Take step in environment
                next_state, reward, terminated, truncated, info = self.env.step(int(action.item()))
                done = terminated or truncated
                    
                episode_return += reward
                episode_length += 1

                # Store experience in buffer
                self.buffer.append((
                    state.squeeze().cpu(), 
                    action.cpu(), 
                    log_prob.cpu(), 
                    state_value.cpu(), 
                    reward, 
                    done
                ))

                # Update current state
                state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                # Handle episode termination
                if done:
                    episode_returns.append(episode_return)
                    episode_lengths.append(episode_length)
                    
                    # Reset environment
                    state, _ = self.env.reset()
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    episode_return = 0.0
                    episode_length = 0

        # 2. Compute advantages and returns (GAE) with last state bootstrap if needed
        with torch.no_grad():
            last_value = self.critic(state).squeeze()
        
        states, actions, log_probs, values, returns, advantages = self.compute_gae(last_value)
        
        policy_loss = torch.tensor(0.0, device=self.device)
        value_loss = torch.tensor(0.0, device=self.device)
        entropy = torch.tensor(0.0, device=self.device)

        # 3. PPO update for multiple epochs
        for _ in range(self.k_epochs):
            logits = self.actor(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = self.critic(states).squeeze()
            value_loss = F.mse_loss(new_values, returns)

            entropy = dist.entropy().mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss = policy_loss - self.entropy_coef * entropy
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss = 0.5 * value_loss
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        # Clear buffer for next rollout
        self.buffer.clear()

        avg_return = float(np.mean(episode_returns)) if len(episode_returns) > 0 else 0.0
        avg_length = float(np.mean(episode_lengths)) if len(episode_lengths) > 0 else 0.0
        
        return {
            'policy_loss': float(policy_loss.detach().item()),
            'value_loss': float(value_loss.detach().item()),
            'entropy': float(entropy.detach().item()),
            'avg_return': avg_return,
            'avg_length': avg_length,
            'num_episodes': len(episode_returns)
        }

    def compute_gae(self, last_value):
        states, actions, log_probs, values, rewards, dones = zip(*self.buffer)

        # Compute targets without tracking gradients
        with torch.no_grad():
            states = torch.stack(states).to(self.device)
            actions = torch.stack(actions).to(self.device)
            log_probs = torch.stack(log_probs).to(self.device)
            values = torch.stack(values).to(self.device)
            
            advantages = torch.zeros_like(values)
            returns = torch.zeros_like(values)

            gae = torch.tensor(0.0, device=self.device)

            for t in reversed(range(len(rewards))):
                mask_val = 1.0 - float(dones[t])
                mask = torch.tensor(mask_val, device=self.device)
                
                if t == len(rewards) - 1:
                    next_value = last_value
                else:
                    next_value = values[t + 1]
                    
                reward_t = torch.tensor(rewards[t], device=self.device, dtype=values.dtype)
                delta = reward_t + self.gamma * next_value * mask - values[t]
                gae = delta + self.gamma * self.lambda_gae * mask * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, log_probs, values, returns, advantages


if __name__ == "__main__":
    # Test with CartPole-v1
    env = gym.make('CartPole-v1')
    
    trainer = PPOTrainer(
        env=env,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        rollout_steps=2048,
        eps_clip=0.2,
        k_epochs=4,
        entropy_coef=0.001
    )
    
    print(f"Training PPO on {env.spec.id}")
    print(f"State dim: {trainer.state_dim}, Action dim: {trainer.action_dim}")
    print(f"Device: {trainer.device}")
    
    # Train for multiple iterations
    num_iters = 200
    history = {"policy_loss": [], "value_loss": [], "entropy": [], "avg_return": [], "avg_length": []}
    live = LivePlotter()
    
    plot_every = 5
    for i in range(num_iters):
        metrics = trainer.train()
        
        history["policy_loss"].append(metrics['policy_loss'])
        history["value_loss"].append(metrics['value_loss'])
        history["entropy"].append(metrics['entropy'])
        history["avg_return"].append(metrics['avg_return'])
        history["avg_length"].append(metrics['avg_length'])
        
        if (i + 1) % plot_every == 0:
            # Update plot data for live plotting
            plot_history = {k: v for k, v in history.items() if k != 'avg_length'}
            live.update(plot_history)
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}: Policy Loss: {metrics['policy_loss']:.4f}, "
                  f"Value Loss: {metrics['value_loss']:.4f}, Entropy: {metrics['entropy']:.4f}, "
                  f"Avg Return: {metrics['avg_return']:.2f}, Avg Length: {metrics['avg_length']:.1f}, "
                  f"Episodes: {metrics['num_episodes']}")

    # Save final figure
    live.fig.savefig("ppo_gym_training_curves.png")
    print("Saved training curves to ppo_gym_training_curves.png")
    
    env.close()
