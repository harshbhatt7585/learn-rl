from basic_environment import BasicGridWorld
from gridworld import GridWorld
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F
import matplotlib.pyplot as plt


class PPOTrainer:
    def __init__(self, env, state_dim, action_dim, lr_actor, lr_critic, gamma, rollout_steps, eps_clip, has_continuous_action_space, lambda_gae=0.95, action_std_init=0.6, k_epochs=4, entropy_coef=0.01, max_grad_norm=0.5, normalize_state=True):
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
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_state = normalize_state

        # Scale for normalizing discrete states like BasicGridWorld (0..size-1)
        self.state_scale = float(getattr(self.env, "size", 1) - 1) if self.normalize_state else 1.0
        if self.state_scale <= 0:
            self.state_scale = 1.0

        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 16),
            nn.Tanh(),
            nn.Linear(16, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 16),
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
        state_value_raw = self.env.reset()
        state_value_norm = float(state_value_raw) / self.state_scale
        state = torch.tensor([state_value_norm], dtype=torch.float32).unsqueeze(0)  # shape: (1, state_dim)
        episode_return = 0.0
        episode_returns = []

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
            new_state, reward, done, step_count = self.env.step(action.item())
            episode_return += reward

            # Store experience in buffer
            self.buffer.append((state.squeeze(), action, log_prob, state_value, reward, done))

            # Update current state
            new_state_norm = float(new_state) / self.state_scale
            state = torch.tensor([new_state_norm], dtype=torch.float32).unsqueeze(0)
            
            # Reset if episode is done
            if done:
                episode_returns.append(episode_return)
                print(f"Reached goal in {step_count} steps with reward {episode_return}")
                reset_val = self.env.reset()
                state = torch.tensor([float(reset_val) / self.state_scale], dtype=torch.float32).unsqueeze(0)
                episode_return = 0.0

        # 2. Compute advantages and returns (GAE) with last state bootstrap if needed
        with torch.no_grad():
            last_value = self.critic(state).squeeze()
        states, actions, log_probs, values, returns, advantages = self.compute_gae(last_value)
        policy_loss = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)
        entropy = torch.tensor(0.0)

        # 3-7. PPO update for multiple epochs (full-batch)
        for _ in range(self.k_epochs):
            logits = self.actor(states.unsqueeze(1))
            action_probs = torch.softmax(logits, dim=-1)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = self.critic(states.unsqueeze(1)).squeeze()
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

        avg_return = float(torch.tensor(episode_returns).mean().item()) if len(episode_returns) > 0 else float(episode_return)
        return {
            'policy_loss': float(policy_loss.detach().item()),
            'value_loss': float(value_loss.detach().item()),
            'entropy': float(entropy.detach().item()),
            'avg_return': avg_return
        }

    def compute_gae(self, last_value):
        states, actions, log_probs, values, rewards, dones = zip(*self.buffer)

        # Compute targets without tracking gradients to avoid graph reuse
        with torch.no_grad():
            states = torch.stack(states)
            actions = torch.stack(actions)
            log_probs = torch.stack(log_probs)
            values = torch.stack(values)
            
            advantages = torch.zeros_like(values)
            returns = torch.zeros_like(values)

            gae = 0.0

            for t in reversed(range(len(rewards))):
                mask = 1.0 - float(dones[t])
                if t == len(rewards) - 1:
                    next_value = last_value if mask == 1.0 else 0.0
                else:
                    next_value = values[t + 1]
                delta = rewards[t] + self.gamma * float(next_value) * mask - values[t]
                gae = delta + self.gamma * self.lambda_gae * mask * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, log_probs, values, returns, advantages


if __name__ == "__main__":
    # Switch between BasicGridWorld() and GridWorld()
    # env = BasicGridWorld()
    env = GridWorld(width=5, height=5, step_limit=100, slip_prob=0.1)
    trainer = PPOTrainer(
        env=env, 
        state_dim=1,  # Using current_place as single state feature
        action_dim=len(env.actions), 
        lr_actor=0.001, 
        lr_critic=0.001, 
        gamma=0.99, 
        rollout_steps=50,  # Increased for better learning
        eps_clip=0.2, 
        has_continuous_action_space=False
    )
    
    # Train for multiple iterations
    num_iters = 200
    history = {"policy_loss": [], "value_loss": [], "entropy": [], "avg_return": []}
    for i in range(num_iters):
        metrics = trainer.train()
        history["policy_loss"].append(metrics['policy_loss'])
        history["value_loss"].append(metrics['value_loss'])
        history["entropy"].append(metrics['entropy'])
        history["avg_return"].append(metrics['avg_return'])
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}: Policy Loss: {metrics['policy_loss']:.4f}, "
                  f"Value Loss: {metrics['value_loss']:.4f}, Entropy: {metrics['entropy']:.4f}, "
                  f"AvgReturn: {metrics['avg_return']:.2f}")

    # Plot and save training curves
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].plot(history["policy_loss"])
    axs[0, 0].set_title("Policy Loss")
    axs[0, 1].plot(history["value_loss"])
    axs[0, 1].set_title("Value Loss")
    axs[1, 0].plot(history["entropy"])
    axs[1, 0].set_title("Entropy")
    axs[1, 1].plot(history["avg_return"])
    axs[1, 1].set_title("Average Return")
    for ax in axs.flat:
        ax.set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig("ppo_training_curves.png")
    print("Saved training curves to ppo_training_curves.png")