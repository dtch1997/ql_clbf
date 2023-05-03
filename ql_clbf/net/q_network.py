
import numpy as np

import torch 
import torch.nn as nn

from typing import List

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ELU(),
            nn.Linear(120, 84),
            nn.ELU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)
    
    def predict(self, state: np.ndarray):
        # Get the optimal action
        state = np.array(state)
        state_th = torch.Tensor(state)
        return torch.argmax(self.forward(state_th), dim=-1).detach().numpy()
 
    def predict_value(self, state: np.ndarray, action: np.ndarray):
        # Get the state value
        state = np.array(state)
        state_th = torch.Tensor(state)
        return torch.max(self.forward(state_th), dim=-1)[0].detach().numpy()
    
class QNetworkEnsemble(nn.Module):
    def __init__(self, envs, models: List[nn.Module]):
        super(QNetworkEnsemble, self).__init__()
        self.envs = envs
        self.models = nn.ModuleList(models)

    def get_num_models(self):
        return len(self.models)

    def forward(self, x, reduction: str ='min'):
        assert reduction in ['min', 'max', 'mean']
        q_values = torch.stack([model(x) for model in self.models], dim=0)
        if reduction == 'min':
            return torch.min(q_values, dim=0)[0]
        elif reduction == 'max':
            return torch.max(q_values, dim=0)[0]
        elif reduction == 'mean':
            return torch.mean(q_values, dim=0)
        
    def predict(self, state: np.ndarray, reduction:str ='min'):
        # Get the optimal action
        state = np.array(state)
        state_th = torch.Tensor(state)
        return torch.argmax(self.forward(state_th, reduction), dim=-1).detach().numpy()
 
    def predict_value(self, state: np.ndarray, action: np.ndarray, reduction:str ='min'):
        # Get the state value
        state = np.array(state)
        state_th = torch.Tensor(state)
        return torch.max(self.forward(state_th, reduction), dim=-1)[0].detach().numpy()