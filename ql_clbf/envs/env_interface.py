
import abc
import dataclasses
import numpy as np

@dataclasses.dataclass
class EnvHParams:
    observation_dim: int # Flattened dimension of observation
    action_dim: int # Flattened dimension of actions
    r_max: float # Maximum reward

class SafetyEnv(abc.ABC):

    @abc.abstractmethod
    def get_hparams(self) -> EnvHParams:
        pass

    @abc.abstractmethod
    def is_unsafe(self, states: np.ndarray) -> np.ndarray :
        """ 
        Input: states (batch_size, observation_dim)
        Output: is_unsafe (batch_size, 1)
        """
        pass

    @abc.abstractmethod
    def sample_states(self, n_samples: int) -> np.ndarray:
        """
        Input: n_samples (int)
        Output: states (n_samples, observation_dim)
        """
        pass


