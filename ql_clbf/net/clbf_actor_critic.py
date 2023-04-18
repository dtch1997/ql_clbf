import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

class CLBFActorCritic(nn.Module):
    
    def __init__(self, 
                observation_dim: int,
                action_dim: int,
                hidden_dim: int, 
                r_max: float,
                gamma: float):
        super(CLBFActorCritic, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.r_max = r_max
        self.gamma = gamma

        self.actor_mean = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(self.action_dim))

        self.critic = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def get_value(self, x: torch.Tensor):
        """ Return Q-values """
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def get_h_values(self, x: torch.Tensor):
        """ Return barrier function value """
        state_values = self.get_value(x)
        barrier_values = - state_values + self.r_max / (1 - self.gamma)
        return barrier_values
    
    def compute_cbf_losses(self, x, is_x_unsafe):
        h_values = self.get_h_values(x)                
        # x_unsafe loss: penalize h(x) <= 0 where is_unsafe(x) = 1
        x_unsafe_loss = (F.relu(-h_values) * is_x_unsafe.float()).mean()
        # x_safe loss: penalize h(x) >= 0 where is_unsafe(x) = 0
        x_safe_loss = (F.relu(h_values) * (1 - is_x_unsafe.float())).mean()
        return {
            'x_unsafe': x_unsafe_loss,
            'x_safe': x_safe_loss,
        }