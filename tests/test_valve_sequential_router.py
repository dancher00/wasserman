import numpy as np
import torch

from wasman.controllers.valve_sequential_router import ValveSequentialRouter, previous_stages_and_ages
from wasman.learning.valve_mode_tree import fit_mode_tree


def test_training_age_matches_prediction_before_action_timing():
    phases = torch.tensor([[0], [0], [1], [1], [2]])
    previous, ages = previous_stages_and_ages(phases)
    assert previous[:, 0].tolist() == [0, 0, 0, 1, 1]
    assert ages[:, 0].tolist() == [0, 1, 2, 0, 1]


def test_sequential_router_advances_at_most_one_and_resets_independently():
    features = np.zeros((100, 40))
    features[:, 0] = np.linspace(0, 1, 100)
    tree = fit_mode_tree(features, (features[:, 0] >= 0.5).astype(int), num_classes=2, min_leaf=4)
    router = ValveSequentialRouter([tree] * 7)
    obs = torch.zeros(2, 50)
    obs[1, 0] = 1
    for step in range(10):
        assert router(obs).tolist() == [0, min(step + 1, 7)]
    router.reset(torch.tensor([1]))
    obs[:, 27:38] = 100
    assert router(obs).tolist() == [0, 1]
    assert router.age.tolist() == [11, 0]
    router.reset()
    assert router.mode is None


def test_merged_completion_has_no_inferred_success_state():
    tree = {
        "left": torch.tensor([-1]),
        "right": torch.tensor([-1]),
        "feature": torch.tensor([-1]),
        "threshold": torch.tensor([0.0], dtype=torch.float64),
        "prediction": torch.tensor([1]),
        "max_depth": 0,
    }
    router = ValveSequentialRouter([tree] * 6)
    for _ in range(10):
        router(torch.zeros(1, 50))
    assert router.mode.item() == 6


def test_sequential_policy_blends_only_after_turn_not_after_standoff():
    from wasman.controllers.valve_local_policy import ValveLocalPolicy

    tree = {
        "left": torch.tensor([-1]),
        "right": torch.tensor([-1]),
        "feature": torch.tensor([-1]),
        "threshold": torch.tensor([0.0], dtype=torch.float64),
        "prediction": torch.tensor([1]),
        "max_depth": 0,
    }
    obs, targets = torch.zeros(64, 50), torch.zeros(64, 11)
    targets[:, 0] = 0.002
    policy = ValveLocalPolicy(obs, targets, neighbors=16, post_stop_blend=0.25)
    policy.mode_router = ValveSequentialRouter([tree] * 6)
    policy.prototype_modes = torch.ones(64, dtype=torch.long)
    action = policy({"policy": obs[:1]})
    assert abs(action[0, 0].item() - 0.002) < 1e-6
    policy.mode_router.mode[:] = 3
    policy.prototype_modes[:] = 4
    action = policy({"policy": obs[:1]})
    assert abs(action[0, 0].item() - 0.0005) < 1e-6


def test_explicit_release_constraint_freezes_arm_but_keeps_learned_grip_and_retreat():
    from wasman.controllers.valve_local_policy import ValveLocalPolicy

    tree = {
        "left": torch.tensor([-1]),
        "right": torch.tensor([-1]),
        "feature": torch.tensor([-1]),
        "threshold": torch.tensor([0.0], dtype=torch.float64),
        "prediction": torch.tensor([1]),
        "max_depth": 0,
    }
    obs, targets = torch.zeros(64, 50), torch.full((64, 11), 0.001)
    policy = ValveLocalPolicy(obs, targets, neighbors=16, latch_release_references=True)
    policy.mode_router = ValveSequentialRouter([tree] * 6)
    policy.mode_router.mode = torch.tensor([4])
    policy.mode_router.age = torch.tensor([0])
    policy.prototype_modes = torch.full((64,), 5, dtype=torch.long)
    action = policy({"policy": obs[:1]})
    assert not action[:, :10].any()
    assert action[0, 10] > 0
    policy.prototype_modes[:] = 6
    action = policy({"policy": obs[:1]})
    assert action[0, 0] > 0
    assert not action[:, 1:10].any()
    policy.reset(torch.tensor([0]))
    assert not policy.release_latched.any()
    policy.reset()
    assert policy.release_reference is None


def test_stage_age_regression_and_hold_increment_checkpoint_roundtrip():
    from wasman.controllers.valve_local_policy import ValveLocalPolicy

    tree = {
        "left": torch.tensor([-1]),
        "right": torch.tensor([-1]),
        "feature": torch.tensor([-1]),
        "threshold": torch.tensor([0.0], dtype=torch.float64),
        "prediction": torch.tensor([0]),
        "max_depth": 0,
    }
    checkpoint = {
        "observations": torch.zeros(64, 50),
        "targets": torch.zeros(64, 11),
        "parameters": {"neighbors": 16, "absolute_targets": True, "incremental_hold": True},
        "prototype_stage_age": torch.zeros(64),
        "prototype_modes": torch.full((64,), 4),
        "transition_trees": [tree] * 6,
    }
    policy = ValveLocalPolicy.from_checkpoint(checkpoint, "cpu")
    policy.mode_router.mode = torch.tensor([4])
    policy.mode_router.age = torch.tensor([0])
    query = torch.zeros(1, 50)
    query[:, 27] = 0.003
    result = policy({"policy": query})
    assert policy.features.shape == (64, 51)
    assert abs(result[0, 0].item() - 0.003) < 1e-8
    assert policy.mode_router.age.item() == 1
