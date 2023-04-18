import torch

from ql_clbf.net.clbf_actor_critic import CLBFActorCritic

def test_clbf_actor_critic():

    observation_dim = 4
    action_dim = 2
    hidden_dim = 64
    r_max = 1.0
    gamma = 0.99

    net = CLBFActorCritic(
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
    action, log_prob, entropy, value = net.get_action_and_value(x)
    assert action.shape == (32, action_dim)
    assert log_prob.shape == (32,)
    assert entropy.shape == (32,)
    assert value.shape == (32, 1)

    h_values = net.get_h_values(x)
    assert h_values.shape == (32, 1)

    losses = net.compute_cbf_losses(x, torch.zeros(32))
    assert losses['x_unsafe'] == 0
    
    losses = net.compute_cbf_losses(x, torch.ones(32))
    assert losses['x_safe'] == 0
