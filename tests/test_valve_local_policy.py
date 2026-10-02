import pytest
import torch

from wasman.controllers.valve_local_policy import ValveLocalPolicy


def test_local_policy_uses_public_state_and_respects_reference_rate_limits():
    obs = torch.zeros(128, 50)
    obs[:, 0] = torch.linspace(-0.1, 0.1, 128)
    targets = torch.zeros(128, 11)
    targets[:, 0] = 0.001 + 0.01 * obs[:, 0]
    targets[:, 10] = -0.026
    model = ValveLocalPolicy(obs, targets, neighbors=32, ridge=0.01)
    query = torch.zeros(2, 50)
    query[:, 0] = torch.tensor([-0.025, 0.025])
    result = model({"policy": query})
    assert torch.allclose(result[:, 0], 0.001 + 0.01 * query[:, 0], atol=2e-5)
    assert torch.allclose(result[:, 10], torch.full((2,), -0.026), atol=1e-5)
    assert torch.equal(result[:, 3:6], torch.zeros(2, 3))
    assert (result.abs() <= model.max_delta + 1e-6).all()
    assert torch.equal(result, model({"policy": query}))


def test_local_policy_rejects_wrong_observation_interface():
    with pytest.raises(ValueError):
        ValveLocalPolicy(torch.zeros(64, 49), torch.zeros(64, 11))


def test_observed_held_category_is_not_blended_across_task_states():
    obs, target = torch.zeros(128, 50), torch.zeros(128, 11)
    obs[64:, 48] = 1
    target[:64, 10], target[64:, 10] = -0.02, 0.02
    model = ValveLocalPolicy(obs, target, neighbors=16, match_held=True, progress_weight=4)
    result = model({"policy": obs[[0, 64]]})
    assert torch.allclose(result[:, 10], torch.tensor([-0.02, 0.02]), atol=1e-5)
    with pytest.raises(ValueError, match="each observed held category"):
        ValveLocalPolicy(obs[:64], target[:64], neighbors=16, match_held=True)


def test_extra_prototypes_can_retain_original_training_normalization():
    obs, target = torch.zeros(128, 50), torch.zeros(128, 11)
    obs[64:, 0] = 10
    model = ValveLocalPolicy.from_checkpoint(
        {
            "observations": obs,
            "targets": target,
            "normalization_count": 64,
            "parameters": {"neighbors": 16},
        },
        "cpu",
    )
    assert model.features.shape == (128, 50)
    assert model.mean[0] == 0
    assert model.scale[0] == 0.05
    assert torch.isfinite(model({"policy": obs[:2]})).all()


def test_absolute_fit_corrects_reference_drift_instead_of_integrating_it():
    obs, target = torch.zeros(128, 50), torch.zeros(128, 11)
    # All demonstrated references are stationary; query the same measured
    # physical state with an erroneous previous command.
    query = torch.zeros(1, 50)
    query[:, 27] = 0.003
    model = ValveLocalPolicy(obs, target, neighbors=32, absolute_targets=True)
    assert abs(float(model({"policy": query})[0, 0])) < 1e-6
    incremental = ValveLocalPolicy(obs, target, neighbors=32)
    assert abs(float(incremental({"policy": query})[0, 0]) - 0.003) < 1e-6


def test_learned_history_partition_does_not_revert_when_angle_decreases():
    obs, target = torch.zeros(128, 50), torch.zeros(128, 11)
    modes = torch.arange(128) >= 64
    target[:64, 10], target[64:, 10] = -0.02, 0.02
    model = ValveLocalPolicy(obs, target, neighbors=16, stopped_modes=modes, learned_stop_threshold=0.9)
    query = torch.zeros(2, 50)
    query[1, 38] = 1
    assert (model({"policy": query})[:, 10] * torch.tensor([-1, 1]) > 0).all()
    query[:, 38] = 0
    assert (model({"policy": query})[:, 10] * torch.tensor([-1, 1]) > 0).all()
    model.reset(torch.tensor([1]))
    assert (model({"policy": query})[:, 10] < 0).all()


def test_optional_blending_does_not_change_commands_before_learned_stop():
    obs, target = torch.zeros(128, 50), torch.zeros(128, 11)
    target[:, 10] = -0.02
    model = ValveLocalPolicy(obs, target, neighbors=16, learned_stop_threshold=0.9, post_stop_blend=0.25)
    query = torch.zeros(2, 50)
    query[1, 38] = 1
    result = model({"policy": query})
    unfiltered = ValveLocalPolicy(obs, target, neighbors=16, learned_stop_threshold=0.9)
    expected = unfiltered({"policy": query})[:, 10] * torch.tensor([1.0, 0.25])
    torch.testing.assert_close(result[:, 10], expected)


def test_double_precision_regression_matches_ill_conditioned_analytic_solution():
    import math

    obs, target = torch.zeros(512, 50), torch.zeros(512, 11)
    target[:, 0] = 0.002
    model = ValveLocalPolicy(obs, target, neighbors=512, ridge=0.01, regression_float64=True)
    query = torch.zeros(1, 50)
    query[:, 38] = 1
    result = model({"policy": query})
    # Identical shifted neighbors: intercept ridge competes with one constant
    # feature (-20). The exact regularized solution is available in closed form.
    expected = 0.002 / (1 + 1e-5 / (512 / math.e) + 1e-5 * 400 / 0.01)
    assert result.dtype == torch.float32
    assert abs(float(result[0, 0]) - expected) < 1e-8


def test_anchored_reference_is_latched_once_and_reset_per_environment():
    obs, target = torch.zeros(128, 50), torch.zeros(128, 11)
    modes = torch.arange(128) >= 64
    obs[64:, 38] = 1
    model = ValveLocalPolicy(
        obs,
        target,
        neighbors=16,
        absolute_targets=True,
        stopped_modes=modes,
        learned_stop_threshold=0.9,
        anchor_at_stop=True,
    )
    query = torch.zeros(2, 50)
    query[:, 38] = 1
    query[:, 27] = torch.tensor([0.2, -0.3])
    result = model({"policy": query})
    torch.testing.assert_close(result[:, 0], query[:, 27])
    query[:, 27] += 0.001
    result = model({"policy": query})
    torch.testing.assert_close(result[:, 0], torch.tensor([0.2, -0.3]))
    model.reset(torch.tensor([0]))
    result = model({"policy": query})
    torch.testing.assert_close(result[:, 0], torch.tensor([0.201, -0.3]))
    model.reset()
    assert model.anchor is None


def test_stored_feature_normalization_is_used_and_validated():
    obs, target = torch.zeros(128, 50), torch.zeros(128, 11)
    checkpoint = {
        "observations": obs,
        "targets": target,
        "parameters": {"neighbors": 16},
        "feature_normalization": {"mean": torch.ones(50), "scale": torch.full((50,), 2.0)},
    }
    model = ValveLocalPolicy.from_checkpoint(checkpoint, "cpu")
    torch.testing.assert_close(model.features, torch.full((128, 50), -0.5))
    checkpoint["feature_normalization"]["mean"][0] = float("nan")
    with pytest.raises(ValueError, match="normalization"):
        ValveLocalPolicy.from_checkpoint(checkpoint, "cpu")


def test_command_exclusion_keeps_physical_feedback_and_rate_limits():
    obs = torch.zeros(128, 50)
    obs[:, 0] = torch.linspace(-0.1, 0.1, 128)
    obs[:, 27] = obs[:, 0] * 0.02
    target = obs[:, 27:38].clone()
    model = ValveLocalPolicy(obs, target, neighbors=32, absolute_targets=True, exclude_previous_commands=True)
    query = torch.zeros(2, 50)
    query[:, 0] = 0.02
    query[:, 27] = torch.tensor([0.0, 0.002])
    result = model({"policy": query})
    assert (result[:, 0] - 0.0004).abs().max() < 1e-5
    assert not model.features[:, 27:38].any()
    assert ((result - query[:, 27:38]).abs() <= model.max_delta + 1e-6).all()


def test_late_command_exclusion_preserves_acquisition_and_remembers_stop():
    obs, target = torch.zeros(128, 50), torch.zeros(128, 11)
    modes = torch.arange(128) >= 64
    obs[:, 27] = torch.linspace(-0.003, 0.003, 128)
    obs[modes, 38] = 1
    target[:, 0] = obs[:, 27]
    options = dict(neighbors=16, absolute_targets=True, stopped_modes=modes, learned_stop_threshold=0.9)
    baseline = ValveLocalPolicy(obs, target, **options)
    model = ValveLocalPolicy(obs, target, exclude_commands_after_stop=True, **options)
    query = obs[:4].clone()
    torch.testing.assert_close(baseline({"policy": query}), model({"policy": query}))
    query[:, 38] = 1
    model({"policy": query})
    query[:, 38] = 0.8
    result = model({"policy": query})
    assert torch.isfinite(result).all()
    assert (model.angle_history == 1).all()
    assert not model.features[modes, 27:38].any()
    assert model.features[~modes, 27:38].abs().sum() > 0
