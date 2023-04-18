import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F

class QCLBFNet(nn.Module):
    
    def __init__(self, 
                observation_dim: int,
                action_dim: int,
                hidden_dim: int, 
                r_max: float,
                gamma: float):
        super(QCLBFNet, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.r_max = r_max
        self.gamma = gamma

        self.net = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
    
    def get_q_values(self, x: torch.Tensor):
        """ Return Q-values """
        return self.net(x)
    
    def get_h_values(self, x: torch.Tensor):
        """ Return barrier function value """
        q_values = self.get_q_values(x)
        state_values = torch.max(q_values, dim=1)[0]
        barrier_values = - state_values + self.r_max / (1 - self.gamma)
        return barrier_values
    
    def compute_cbf_losses(self, x, is_x_unsafe, u, x_next):
        h_values = self.get_h_values(x)                
        # x_unsafe loss: penalize h(x) <= 0 where is_unsafe(x) = 1
        x_unsafe_loss = F.relu(-h_values[is_x_unsafe == 1]).mean()
        # x_safe loss: penalize h(x) >= 0 where is_unsafe(x) = 0
        x_safe_loss = F.relu(h_values[is_x_unsafe == 0]).mean()
        return {
            'x_unsafe': x_unsafe_loss,
            'x_safe': x_safe_loss,
        }

    def get_safety_prediction(self, x: torch.Tensor):
        """ Return safety prediction 
        1 means safe, 0 means unsafe """
        h_values = self.get_h_values(x)
        preds = (h_values <= 0).float()
        log_probs = F.logsigmoid(-h_values)
        return preds, log_probs