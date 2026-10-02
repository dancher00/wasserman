"""Nonparametric, state-only local-linear imitation policy for RotateValve.

Fits a local action map to stored training demonstrations, with either an
increment or absolute-target regression prior. An optional classifier learned
from demonstrations partitions moving/stopped examples using angle history.
No live expert, simulator access or IK at inference; input remains public 50-D
state and output remains the original 11 normalized actuator commands.
"""

import torch


class ValveLocalPolicy(torch.nn.Module):
    def __init__(
        self,
        observations,
        targets,
        *,
        neighbors=64,
        ridge=0.1,
        match_held=False,
        progress_weight=1.0,
        absolute_targets=False,
        stopped_modes=None,
        learned_stop_threshold=None,
        post_stop_blend=1.0,
        regression_float64=False,
        anchor_at_stop=False,
        exclude_previous_commands=False,
        exclude_commands_after_stop=False,
        latch_release_references=False,
        stage_ages=None,
        incremental_hold=False,
    ):
        super().__init__()
        if observations.ndim != 2 or observations.shape[1] != 50 or targets.shape != (len(observations), 11):
            raise ValueError("Expected public valve observations and 11 action targets")
        if not 2 <= neighbors <= len(observations) or ridge <= 0 or progress_weight <= 0:
            raise ValueError("Invalid local regression parameters")
        self.neighbors, self.ridge = neighbors, ridge
        self.match_held = match_held
        self.absolute_targets = absolute_targets
        self.regression_float64 = regression_float64
        self.exclude_previous_commands = exclude_previous_commands
        self.exclude_commands_after_stop = exclude_commands_after_stop
        self.latch_release_references = latch_release_references
        self.release_reference = self.release_latched = None
        self.uses_stage_age = stage_ages is not None
        self.incremental_hold = incremental_hold
        if incremental_hold and not absolute_targets:
            raise ValueError("Mixed hold increments require absolute references in other stages")
        if exclude_commands_after_stop and (stopped_modes is None or anchor_at_stop or exclude_previous_commands):
            raise ValueError("Late command exclusion requires a binary partition and no other command transform")
        if exclude_previous_commands and anchor_at_stop:
            raise ValueError("Command exclusion and reference anchoring are separate experiments")
        if anchor_at_stop and (not absolute_targets or stopped_modes is None):
            raise ValueError("Anchored references require absolute targets and a learned binary partition")
        self.anchor_at_stop = anchor_at_stop
        self.anchor = self.anchored = None
        self.learned_stop_threshold = learned_stop_threshold
        if not 0 < post_stop_blend <= 1:
            raise ValueError("Output blend must be in (0,1]")
        self.post_stop_blend = post_stop_blend
        self.angle_history = None
        self.mode_router = None
        self.register_buffer("prototype_modes", None)
        if stopped_modes is not None and learned_stop_threshold is None:
            raise ValueError("Mode prototypes require a learned threshold")
        self.register_buffer("stopped_modes", None)
        if stopped_modes is not None:
            if stopped_modes.shape != (len(observations),) or not torch.isfinite(torch.tensor(learned_stop_threshold)):
                raise ValueError("Invalid learned mode partition")
            self.stopped_modes = stopped_modes.bool()
        held = observations[:, 48] > 0.5
        if match_held and min(int(held.sum()), int((~held).sum())) < neighbors:
            raise ValueError("Need enough training neighbors in each observed held category")
        feature_observations = observations.clone()
        if stage_ages is not None:
            if stage_ages.shape != (len(observations),):
                raise ValueError("One stage-age feature per prototype required")
            feature_observations = torch.cat((feature_observations, stage_ages[:, None]), -1)
        if exclude_previous_commands:
            feature_observations[:, 27:38] = 0
        mean = feature_observations.mean(0)
        scale = feature_observations.std(0).clamp_min(0.05)
        # Avoid chasing high-frequency velocity/contact noise in neighbor search.
        weight = torch.ones(feature_observations.shape[1], device=observations.device)
        weight[6:12] = 0.25
        weight[19:23] = 0.25
        weight[41:44] = 0.25
        scale = scale / weight
        scale[[38, 47]] /= progress_weight
        features = (feature_observations - mean) / scale
        if exclude_commands_after_stop:
            features[stopped_modes.bool(), 27:38] = 0
        increment_scale = observations.new_tensor([0.005] * 9 + [0.002, 0.03])
        self.register_buffer("mean", mean)
        self.register_buffer("held", held)
        self.register_buffer("scale", scale)
        self.register_buffer("features", features)
        self.register_buffer("squared_norm", features.square().sum(-1))
        self.register_buffer(
            "increments", (targets if absolute_targets else targets - observations[:, 27:38]) / increment_scale
        )
        self.register_buffer("increment_scale", increment_scale)
        # Same reference speeds as the demonstration controller, not direct
        # object forces or a contact override. Zero attitude commands stay zero.
        self.register_buffer(
            "max_delta",
            observations.new_tensor(
                [
                    0.1 / 30 / 0.5,
                    0.1 / 30 / 0.35,
                    0.1 / 30 / 0.35,
                    0,
                    0,
                    0,
                    0.35 / 30 / 0.8,
                    0.35 / 30,
                    0.35 / 30,
                    0.35 / 30 / 3.2,
                    0.8 / 30,
                ]
            ),
        )

    def forward(self, obs):
        x = obs["policy"]
        original_previous = x[:, 27:38]
        if self.anchor_at_stop:
            if self.anchor is None or len(self.anchor) != len(x):
                self.anchor = torch.zeros_like(original_previous)
                self.anchored = torch.zeros(len(x), device=x.device, dtype=torch.bool)
            crossing = (x[:, 38] >= self.learned_stop_threshold) & ~self.anchored
            self.anchor[crossing] = original_previous[crossing]
            self.anchored |= crossing
            x = x.clone()
            x[:, 27:38] -= self.anchor
        mode = self.mode_router(x) if self.mode_router is not None else None
        features = x
        if self.uses_stage_age:
            if self.mode_router is None or not hasattr(self.mode_router, "age"):
                raise ValueError("Stage-age features require the learned sequential router")
            features = torch.cat((x, self.mode_router.age[:, None] / 30), -1)
        query = (features - self.mean) / self.scale
        if self.exclude_previous_commands:
            query[:, 27:38] = 0
        elif self.exclude_commands_after_stop:
            stopped_now = x[:, 38] >= self.learned_stop_threshold
            if self.angle_history is not None and len(self.angle_history) == len(x):
                stopped_now |= self.angle_history >= self.learned_stop_threshold
            query[stopped_now, 27:38] = 0
        distance = (query.square().sum(-1, keepdim=True) + self.squared_norm - 2 * query @ self.features.T).clamp_min(0)
        if self.match_held:
            # This public measured Boolean is categorical, not an interpolated
            # hidden expert phase. All actions still come from learned examples.
            distance = distance.masked_fill(self.held[None] != (x[:, 48:49] > 0.5), float("inf"))
        if self.learned_stop_threshold is not None:
            if self.angle_history is None or len(self.angle_history) != len(x):
                self.angle_history = x[:, 38].clone()
            self.angle_history = torch.maximum(self.angle_history, x[:, 38])
            stopped = self.angle_history >= self.learned_stop_threshold
            if self.stopped_modes is not None:
                partitioned = distance.masked_fill(self.stopped_modes[None] != stopped[:, None], float("inf"))
                # Unsupported combinations use the ordinary observation metric;
                # never invent labels or solve a system with infinite distances.
                supported = torch.isfinite(partitioned).sum(-1) >= self.neighbors
                distance = torch.where(supported[:, None], partitioned, distance)
        if self.mode_router is not None:
            stopped = mode >= getattr(self.mode_router, "stopping_mode", 1)
            partitioned = distance.masked_fill(self.prototype_modes[None] != mode[:, None], float("inf"))
            supported = torch.isfinite(partitioned).sum(-1) >= self.neighbors
            distance = torch.where(supported[:, None], partitioned, distance)
        distances, indices = distance.topk(self.neighbors, largest=False)
        local = self.features[indices] - query[:, None]
        design = torch.cat((torch.ones_like(local[..., :1]), local), -1)
        weights = torch.exp(-distances / distances[:, -1:].clamp_min(1e-4))
        if self.regression_float64:
            design, weights = design.double(), weights.double()
        transposed = design.transpose(1, 2) * weights[:, None]
        gram = transposed @ design
        regularizer = torch.eye(design.shape[-1], device=x.device, dtype=design.dtype) * self.ridge
        regularizer[0, 0] = 1e-5
        coefficients = torch.linalg.solve(gram + regularizer, transposed @ self.increments[indices].to(design.dtype))
        prediction = (coefficients[:, 0] * self.increment_scale).to(x.dtype)
        if self.incremental_hold:
            if mode is None or getattr(self.mode_router, "stopping_mode", None) != 4:
                raise ValueError("Hold increments require a learned sequential router")
            prediction += torch.where((mode == 4)[:, None], original_previous, 0)
        if self.anchor_at_stop:
            prediction += self.anchor
        delta = (prediction - original_previous if self.absolute_targets else prediction).clamp(
            -self.max_delta, self.max_delta
        )
        if self.learned_stop_threshold is not None or self.mode_router is not None:
            delta = delta * torch.where(stopped[:, None], self.post_stop_blend, 1.0)
        action = (original_previous + delta).clamp(-1, 1)
        if self.latch_release_references:
            if self.mode_router is None or getattr(self.mode_router, "stopping_mode", None) != 4:
                raise ValueError("Release-reference latching requires the learned sequential router")
            if self.release_reference is None or len(self.release_reference) != len(x):
                self.release_reference = torch.zeros_like(original_previous)
                self.release_latched = torch.zeros(len(x), device=x.device, dtype=torch.bool)
            releasing = mode >= 5
            entering = releasing & ~self.release_latched
            self.release_reference[entering] = original_previous[entering]
            self.release_latched |= entering
            # Explicit control constraint: preserve the achieved arm and
            # lateral station-keeping references while opening/withdrawing.
            # Mode decisions and gripper/retreat commands remain learned.
            action[releasing, 6:10] = self.release_reference[releasing, 6:10]
            action[releasing, 1:3] = self.release_reference[releasing, 1:3]
            action[mode == 5, 0] = self.release_reference[mode == 5, 0]
            limited = original_previous + (action - original_previous).clamp(-self.max_delta, self.max_delta)
            action = torch.where(releasing[:, None], limited, action)
        return torch.cat((action[:, :3], torch.zeros_like(action[:, 3:6]), action[:, 6:]), -1)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.release_reference = self.release_latched = None
        elif self.release_reference is not None:
            self.release_reference[env_ids] = 0
            self.release_latched[env_ids] = False
        if env_ids is None:
            self.anchor = self.anchored = None
        elif self.anchor is not None:
            self.anchor[env_ids] = 0
            self.anchored[env_ids] = False
        if self.mode_router is not None:
            self.mode_router.reset(env_ids)
        if env_ids is None:
            self.angle_history = None
        elif self.angle_history is not None:
            self.angle_history[env_ids] = -float("inf")

    @classmethod
    def from_checkpoint(cls, checkpoint, device):
        observations, targets = checkpoint["observations"], checkpoint["targets"]
        count = checkpoint.get("normalization_count", len(observations))
        if not 1 <= count <= len(observations):
            raise ValueError("Invalid reference-normalization prefix")
        mode_options = {}
        if "learned_stop_threshold" in checkpoint:
            mode_options = {
                "stopped_modes": checkpoint["stopped_modes"][:count] if "stopped_modes" in checkpoint else None,
                "learned_stop_threshold": checkpoint["learned_stop_threshold"],
            }
        if "prototype_stage_age" in checkpoint:
            if count != len(observations):
                raise ValueError("Stage-age prototypes require full normalization")
            mode_options["stage_ages"] = checkpoint["prototype_stage_age"]
        policy = cls(observations[:count], targets[:count], **checkpoint["parameters"], **mode_options)
        if "feature_normalization" in checkpoint:
            normalizer = checkpoint["feature_normalization"]
            policy.mean = normalizer["mean"].clone()
            policy.scale = normalizer["scale"].clone()
            if (
                policy.mean.shape != (50,)
                or policy.scale.shape != (50,)
                or not torch.isfinite(policy.mean).all()
                or not torch.isfinite(policy.scale).all()
                or not (policy.scale > 0).all()
            ):
                raise ValueError("Invalid stored feature normalization")
            policy.features = (observations[:count] - policy.mean) / policy.scale
            policy.squared_norm = policy.features.square().sum(-1)
        if "mode_tree" in checkpoint:
            from wasman.controllers.valve_mode_router import ValveModeRouter

            policy.mode_router = ValveModeRouter(checkpoint["mode_tree"])
            policy.prototype_modes = checkpoint["prototype_modes"].long()
        if "transition_trees" in checkpoint:
            from wasman.controllers.valve_sequential_router import ValveSequentialRouter

            policy.mode_router = ValveSequentialRouter(checkpoint["transition_trees"])
            policy.prototype_modes = checkpoint["prototype_modes"].long()
        if count < len(observations):
            extra = (observations[count:] - policy.mean) / policy.scale
            policy.features = torch.cat((policy.features, extra))
            policy.squared_norm = policy.features.square().sum(-1)
            policy.increments = torch.cat(
                (
                    policy.increments,
                    (targets[count:] if policy.absolute_targets else targets[count:] - observations[count:, 27:38])
                    / policy.increment_scale,
                )
            )
            policy.held = torch.cat((policy.held, observations[count:, 48] > 0.5))
            if "stopped_modes" in checkpoint:
                policy.stopped_modes = checkpoint["stopped_modes"].bool()
        if policy.exclude_previous_commands:
            policy.features[:, 27:38] = 0
            policy.squared_norm = policy.features.square().sum(-1)
        elif policy.exclude_commands_after_stop:
            policy.features[policy.stopped_modes, 27:38] = 0
            policy.squared_norm = policy.features.square().sum(-1)
        return policy.to(device).eval()
