import torch

from wasman.controllers.valve_mixture_policy import ValveMixturePolicy


class CounterPolicy(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, observations):
        self.calls += 1
        return torch.full((len(observations["policy"]), 11), self.value)

    def reset(self, env_ids=None):
        pass


def test_mixture_updates_both_branches_and_retains_observed_history():
    local, recurrent = CounterPolicy(0.0), CounterPolicy(1.0)
    model = ValveMixturePolicy(local, recurrent, threshold=0.7)
    x = torch.zeros(2, 50)
    assert not model({"policy": x}).any()
    x[1, 38] = 0.8
    assert torch.equal(model({"policy": x})[:, 0], torch.tensor([0.0, 1.0]))
    x[:, 38] = 0
    assert torch.equal(model({"policy": x})[:, 0], torch.tensor([0.0, 1.0]))
    assert local.calls == recurrent.calls == 3
    model.reset(torch.tensor([1]))
    assert not model({"policy": x}).any()
