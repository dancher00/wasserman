import pytest
import torch

from wasman.controllers.valve_recurrent_policy import ValveRecurrentPolicy


def test_streaming_matches_full_sequence_and_reset_is_per_environment():
    torch.manual_seed(12)
    model = ValveRecurrentPolicy(torch.zeros(50), torch.ones(50), memory_size=16).eval()
    observations = torch.randn(3, 12, 50)
    with torch.no_grad():
        expected, _ = model.forward_sequence(observations)
        actual = torch.stack([model({"policy": observations[:, i]}) for i in range(12)], dim=1)
        torch.testing.assert_close(actual, expected)
        saved = model.hidden.clone()
        model.reset(torch.tensor([1]))
        assert torch.equal(model.hidden[:, [0, 2]], saved[:, [0, 2]])
        assert torch.count_nonzero(model.hidden[:, 1]) == 0
        model.reset()
        torch.testing.assert_close(model({"policy": observations[:, 0]}), expected[:, 0])


def test_phase_head_does_not_control_actions():
    model = ValveRecurrentPolicy(torch.zeros(50), torch.ones(50), memory_size=16).eval()
    observations = torch.randn(2, 5, 50)
    with torch.no_grad():
        before, _ = model.forward_sequence(observations)
        model.phase_auxiliary.weight.fill_(100)
        model.phase_auxiliary.bias.fill_(-100)
        after, _ = model.forward_sequence(observations)
    assert torch.equal(before, after)
    assert torch.count_nonzero(after[..., 3:6]) == 0


def test_zero_normalization_is_rejected():
    with pytest.raises(ValueError, match="positive effective"):
        ValveRecurrentPolicy(torch.zeros(50), torch.zeros(50))
