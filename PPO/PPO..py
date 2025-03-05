import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

device = torch.device("mps") if torch.mps.is_available() else torch.device('cpu')

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminal = []
    
    def clear(self):
        del self.actions
        del self.states
        del self.logprobs
        del self.rewards
        del self.state_values
        del self.is_terminal
    


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continous_action_space = has_continous_action_space

        if has_continous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

            if has_continous_action_space:
                self.actor = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, action_dim),
                    nn.Tanh()
                )
            else:
                self.actor = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
            
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )


    def set_action_std(self, new_action_std):
        if self.has_continous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    
    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    

    def evaluate(self, state, action):
        if self.has_continous_action_space:
            action_mean = self.actor(state)
        
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy




