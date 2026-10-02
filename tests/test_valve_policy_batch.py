import pytest
import torch

from wasman.controllers.valve_policy_batch import ValvePolicyBatch


class Branch(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.last_reset = None

    def forward(self, observations):
        return observations["policy"][:, :11] + self.value

    def reset(self, env_ids=None):
        self.last_reset = env_ids


def test_candidates_receive_only_their_environment_slice_and_local_resets():
    branches = [Branch(1), Branch(2)]
    model = ValvePolicyBatch(branches, 3)
    assert torch.equal(model({"policy": torch.zeros(6, 50)})[:, 0], torch.tensor([1, 1, 1, 2, 2, 2]))
    model.reset(torch.tensor([1, 4, 5]))
    assert torch.equal(branches[0].last_reset, torch.tensor([1]))
    assert torch.equal(branches[1].last_reset, torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="environment count"):
        model({"policy": torch.zeros(5, 50)})
