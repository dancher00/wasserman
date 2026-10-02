import pytest
import torch
from tensordict import TensorDict

from wasman.controllers.valve_policy import ValveResidualActor

pytestmark = pytest.mark.unit


def test_button_observations_are_rejected():
    obs = TensorDict({"policy": torch.zeros(1, 37)}, batch_size=[1])
    with pytest.raises(ValueError, match="50-D"):
        ValveResidualActor(obs, {"actor": ["policy"]}, "actor", 11, hidden_dims=[16])


def test_increment_actor_and_export_use_only_public_observation():
    obs = TensorDict({"policy": torch.randn(4, 50)}, batch_size=[4])
    actor = ValveResidualActor(obs, {"actor": ["policy"]}, "actor", 11, hidden_dims=[16], obs_normalization=True)
    actor.eval()
    with torch.no_grad():
        for parameter in actor.mlp.parameters():
            parameter.zero_()
        previous = obs["policy"][:, 27:38].clone()
        previous[:, 3:6] = 0
        assert torch.equal(actor(obs), previous)
        list(actor.mlp.modules())[-1].bias.fill_(0.5)
        expected = obs["policy"][:, 27:38] + 0.5 * actor.increment_scale
        expected[:, 3:6] = 0
        assert torch.allclose(actor(obs), expected)
        assert torch.allclose(actor.as_jit()(obs["policy"]), expected)
        assert torch.allclose(torch.jit.script(actor.as_jit())(obs["policy"]), expected)
        assert torch.allclose(actor.as_onnx()(obs["policy"]), expected)


def test_stochastic_actor_distribution_is_in_absolute_action_units():
    obs = TensorDict({"policy": torch.randn(4, 50)}, batch_size=[4])
    actor = ValveResidualActor(
        obs,
        {"actor": ["policy"]},
        "actor",
        11,
        hidden_dims=[16],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.02},
    )
    mean = actor(obs)
    sampled = actor(obs, stochastic_output=True)
    assert torch.allclose(actor.output_mean, mean)
    assert torch.allclose(actor.output_std, torch.full_like(mean, 0.02))
    assert torch.isfinite(actor.get_output_log_prob(sampled)).all()
