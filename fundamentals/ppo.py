from basic_environment import BasicGridWorld
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F
from plotter import LivePlotter
import copy

# Set up device (MPS if available, otherwise CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) for GPU acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA for GPU acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values.squeeze(), dist_entropy

class PPOTrainer:
    def __init__(self, env, state_dim, action_dim, lr_actor, lr_critic, gamma, rollout_steps, eps_clip, lambda_gae=0.95, k_epochs=10, entropy_coef=0.01, max_grad_norm=0.5, batch_size=64):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.rollout_steps = rollout_steps
        self.eps_clip = eps_clip
        self.lambda_gae = lambda_gae
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.MseLoss = nn.MSELoss()
        self.buffer = []

    def train(self):
        self.policy.train()

        # Reset environment and get initial state (one-hot)
        state_value_raw = int(self.env.reset())
        state = F.one_hot(torch.tensor([state_value_raw]), num_classes=self.state_dim).float().to(device)
        episode_return = 0.0
        episode_returns = []

        # 1. Collect experiences (rollout) using old policy
        for step in range(self.rollout_steps):
            # Use old policy for rollout
            action, log_prob, state_value = self.policy_old.act(state)

            # Take step in environment
            new_state, reward, done, step_count = self.env.step(action.item())
            episode_return += reward

            # Store experience in buffer
            self.buffer.append((state.squeeze(), action, log_prob, state_value, reward, done))

            # Update current state (one-hot)
            state = F.one_hot(torch.tensor([int(new_state)]), num_classes=self.state_dim).float().to(device)

            # Reset if episode is done
            if done:
                episode_returns.append(episode_return)
                print(f"Reached goal in {step_count} steps with reward {episode_return}")
                reset_val = int(self.env.reset())
                state = F.one_hot(torch.tensor([reset_val]), num_classes=self.state_dim).float().to(device)
                episode_return = 0.0

        # 2. Compute advantages and returns (GAE) with last state bootstrap if needed
        with torch.no_grad():
            _, _, last_value = self.policy_old.act(state)  # Use old policy for bootstrap
        states, actions, log_probs, values, returns, advantages = self.compute_gae(last_value)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 3. PPO update for multiple epochs with mini-batches
        dataset_size = states.size(0)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0
        
        for epoch in range(self.k_epochs):
            # Shuffle indices for each epoch
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluating with current policy
                new_log_probs, new_values, entropy = self.policy.evaluate(batch_states, batch_actions)

                # Ratio
                ratio = torch.exp(new_log_probs - batch_log_probs.detach())

                # Surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.MseLoss(new_values, batch_returns.squeeze())

                # Combined loss (like reference, but with GAE)
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy.mean()

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate losses for reporting
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

        avg_return = float(torch.tensor(episode_returns).mean().item()) if len(episode_returns) > 0 else float(episode_return)
        return {
            'policy_loss': total_policy_loss / num_batches if num_batches > 0 else 0.0,
            'value_loss': total_value_loss / num_batches if num_batches > 0 else 0.0,
            'entropy': total_entropy / num_batches if num_batches > 0 else 0.0,
            'avg_return': avg_return
        }

    def compute_gae(self, last_value):
        states, actions, log_probs, values, rewards, dones = zip(*self.buffer)

        with torch.no_grad():
            states = torch.stack(states).to(device)
            actions = torch.stack(actions).to(device)
            log_probs = torch.stack(log_probs).to(device)
            values = torch.stack(values).to(device)

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
    env = BasicGridWorld()
    trainer = PPOTrainer(
        env=env,
        state_dim=env.size,  # one-hot over all cells
        action_dim=len(env.actions),
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        rollout_steps=200,
        eps_clip=0.2,
        k_epochs=10,
        entropy_coef=0.01,
        batch_size=64
    )

    # Train for multiple iterations
    num_iters = 400
    history = {"policy_loss": [], "value_loss": [], "entropy": [], "avg_return": []}
    live = LivePlotter()
    for i in range(num_iters):
        metrics = trainer.train()
        history["policy_loss"].append(metrics['policy_loss'])
        history["value_loss"].append(metrics['value_loss'])
        history["entropy"].append(metrics['entropy'])
        history["avg_return"].append(metrics['avg_return'])
        live.update(history)
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}: Policy Loss: {metrics['policy_loss']:.4f}, "
                  f"Value Loss: {metrics['value_loss']:.4f}, Entropy: {metrics['entropy']:.4f}, "
                  f"AvgReturn: {metrics['avg_return']:.2f}")

    # Save final figure
    live.fig.savefig("ppo_training_curves.png")
    print("Saved training curves to ppo_training_curves.png")