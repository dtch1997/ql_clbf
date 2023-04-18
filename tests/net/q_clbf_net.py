import torch

from ql_clbf.net.q_clbf_net import QCLBFNet

def test_q_clbf_net():

    observation_dim = 4
    action_dim = 2
    hidden_dim = 64
    r_max = 1.0
    gamma = 0.99

    net = QCLBFNet(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        r_max=r_max,
        gamma=gamma
    )

    assert net.observation_dim == observation_dim
    assert net.action_dim == action_dim
    assert net.hidden_dim == hidden_dim
    assert net.r_max == r_max
    assert net.gamma == gamma

    x = torch.randn(32, observation_dim)
    q_values = net.get_q_values(x)
    assert q_values.shape == (32, action_dim)
    h_values = net.get_h_values(x)
    assert h_values.shape == (32,)

