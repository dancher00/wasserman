"""Learn actuator-target increments; no expert, phase or task-space solver inside."""

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories


class ValveResidualActor(MLPModel):
    """RSL-compatible Gaussian actor with an explicit previous-action skip.

    Absolute-target BC can obtain a small MSE simply by holding the previous
    action forever. Increment-normalized fitting makes slow wrist motion a
    substantial target. The input remains exactly the public 50-D state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.obs_dim != 50:
            raise ValueError("ValveResidualActor requires the 50-D valve observation")
        self.register_buffer("increment_scale", torch.tensor([0.005] * 9 + [0.002, 0.03]))

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        latent = self.get_latent(obs)
        mean = obs["policy"][:, 27:38] + self.increment_scale * self.mlp(latent)
        # Every demonstration requests level station keeping. Enforce those
        # three known-constant targets exactly, rather than integrating tiny
        # imitation errors into a drifting vehicle attitude reference.
        mean = torch.cat((mean[:, :3], torch.zeros_like(mean[:, 3:6]), mean[:, 6:]), -1)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        return mean

    def as_jit(self):
        return ValvePolicyExport(self)

    def as_onnx(self, verbose=False):
        exported = ValvePolicyExport(self)
        exported.verbose = verbose
        return exported


class ValvePolicyExport(torch.nn.Module):
    is_recurrent = False

    def __init__(self, model):
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.register_buffer("increment_scale", model.increment_scale.detach().clone())
        self.verbose = False
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else torch.nn.Identity()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean = obs[:, 27:38] + self.increment_scale * self.mlp(self.obs_normalizer(obs))
        mean = torch.cat((mean[:, :3], torch.zeros_like(mean[:, 3:6]), mean[:, 6:]), -1)
        return self.deterministic_output(mean)

    @torch.jit.export
    def reset(self) -> None:
        pass

    def get_dummy_inputs(self):
        return (torch.zeros(1, 50),)

    @property
    def input_names(self):
        return ["obs"]

    @property
    def output_names(self):
        return ["actions"]
