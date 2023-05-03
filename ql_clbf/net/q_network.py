
import numpy as np

import torch 
import torch.nn as nn

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