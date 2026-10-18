import numpy as np
import pytest
import torch

from scripts.experiment_1 import (
    ConditionSpec,
    ExperimentConfig,
    TemporalAttractorSettings,
    _fit_projection,
    _record_condition,
    _record_step,
    _resolved_conditions,
    _state_feature,
    _trajectory_legend_handles,
    _transform_with_residual,
    load_experiment_config,
)


def test_state_feature_uses_all_channels_in_center_crop():
    state = torch.arange(16 * 6 * 8, dtype=torch.float32).reshape(1, 16, 6, 8)

    feature = _state_feature(state, crop_size=4)

    expected = state[0, :, 1:5, 2:6].reshape(-1).numpy()
    assert feature.shape == (16 * 4 * 4,)
    assert np.array_equal(feature, expected)


def test_record_schedule_is_dense_then_periodic_and_includes_horizon():
    settings = TemporalAttractorSettings(horizon=13, dense_until=3, record_every=5)

    steps = [step for step in range(14) if _record_step(step, settings)]

    assert steps == [0, 1, 2, 3, 5, 10, 13]


def test_temporal_attractor_legend_explains_all_markers():
    labels = {
        handle.get_label()
        for handle in _trajectory_legend_handles(
            "Control trajectory",
            reference_as_points=False,
        )
    }

    assert labels == {
        "Control trajectory",
        "Trajectory state (color encodes step)",
        "H* (intervention start)",
        "Last recorded state",
        "First sustained morphology failure",
    }


def test_standardized_projection_returns_finite_coordinates_and_residuals():
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(20, 12)).astype(np.float32)

    projection = _fit_projection(matrix, components=3, random_state=2)
    coordinates, residuals = _transform_with_residual(matrix, projection)

    assert coordinates.shape == (20, 3)
    assert residuals.shape == (20,)
    assert np.isfinite(coordinates).all()
    assert np.isfinite(residuals).all()
    assert (residuals >= 0.0).all()


def test_full_state_recording_retains_visible_dynamics_under_hidden_freeze():
    class AddOneModel:
        def __call__(self, state, update_probability):
            del update_probability
            return state + 1.0

    mature = torch.zeros(1, 16, 4, 4)
    condition = ConditionSpec("hidden_freeze", "Freeze", "freeze", "beta", 1.0)
    settings = TemporalAttractorSettings(
        horizon=2,
        dense_until=2,
        record_every=1,
        crop_size=4,
    )

    trajectory = _record_condition(
        AddOneModel(),
        mature,
        condition,
        ExperimentConfig(),
        settings,
        continuation_seed=1,
    )
    states = trajectory.features.reshape(3, 16, 4, 4)

    assert np.array_equal(states[:, 0, 0, 0], np.asarray([0.0, 1.0, 2.0]))
    assert np.array_equal(states[:, 4:, :, :], np.zeros((3, 12, 4, 4)))


def test_default_attractor_conditions_include_both_freezes_and_development_references():
    config = load_experiment_config("configs/experiment_1.yaml")
    conditions = _resolved_conditions(config, None)
    assert len(conditions) == 8
    assert {condition.name for condition in conditions} >= {
        "hidden_freeze",
        "hidden_freeze_morphology",
        "hidden_freeze_first_6",
        "restore_development_step_16_period_150",
    }
    assert not any(condition.reference_step == 48 for condition in conditions)


def test_full_grid_representation_includes_rectangular_background():
    state = torch.zeros(1, 16, 6, 8)
    state[:, 0, 0, 0] = 7
    feature = _state_feature(state, crop_size=0)
    assert feature.size == 16 * 6 * 8
    assert feature[0] == 7
    assert _state_feature(state, crop_size=4).sum() == 0


def test_nonfinite_state_is_detected_between_recording_steps():
    class TransientNonfinite:
        def __init__(self):
            self.calls = 0

        def __call__(self, state, update_probability):
            self.calls += 1
            return (
                torch.full_like(state, float("nan")) if self.calls == 2 else torch.zeros_like(state)
            )

    model = TransientNonfinite()
    trajectory = _record_condition(
        model,
        torch.zeros(1, 16, 4, 4),
        ConditionSpec("baseline", "Control", "baseline"),
        ExperimentConfig(),
        TemporalAttractorSettings(horizon=5, dense_until=0, record_every=5, crop_size=0),
        1,
    )
    assert trajectory.divergence_step == 2
    assert trajectory.steps.tolist() == [0]
    assert model.calls == 2


def test_disk_backed_pca_matches_in_memory_pca_and_keeps_input_unchanged(tmp_path):
    matrix = np.random.default_rng(8).normal(size=(140, 10)).astype(np.float32)
    before = matrix.copy()
    direct = _fit_projection(matrix, 3, 3)
    mapped = _fit_projection((matrix[:70], matrix[70:]), 3, 3, tmp_path)
    direct_scores, direct_residuals = _transform_with_residual(matrix, direct)
    mapped_scores, mapped_residuals = _transform_with_residual(matrix, mapped)
    assert np.array_equal(matrix, before)
    np.testing.assert_allclose(mapped_scores, direct_scores, atol=1e-10)
    np.testing.assert_allclose(mapped_residuals, direct_residuals, atol=1e-10)


def test_three_constant_states_have_three_finite_pca_axes():
    projection = _fit_projection(np.ones((3, 8)), 12, 0)
    assert projection.coordinates.shape == (3, 3)
    assert np.isfinite(projection.coordinates).all()
    assert np.array_equal(projection.pca.explained_variance_ratio_, np.zeros(3))


def test_attractor_cli_dispatches_all_four_morphologies_and_conditions(tmp_path, monkeypatch):
    from scripts import experiment_1 as cli

    calls = []
    config = load_experiment_config("configs/experiment_1.yaml")
    monkeypatch.setattr(
        cli,
        "_load_population",
        lambda folder: (
            config,
            cli.build_conditions(config.interventions),
            [{"metadata": {"replicate": i}} for i in (0, 1)],
            [],
        ),
    )

    def capture(*args, **kwargs):
        calls.append(args)
        return args[1]

    monkeypatch.setattr(cli, "run_temporal_attractor_analysis", capture)
    args = cli.build_parser().parse_args(
        ["attractor", "--results-root", str(tmp_path), "--all-replicates"]
    )
    args.handler(args)
    assert len(calls) == 8
    assert {call[0].name for call in calls} == set(cli.MORPHOLOGIES)
    assert all(
        len(call[3]) == 8
        and "hidden_freeze_morphology" in call[3]
        and "hidden_freeze" in call[3]
        and "hidden_freeze_first_6" in call[3]
        for call in calls
    )
    assert all(call[4].crop_size == 0 for call in calls)
    with pytest.raises(ValueError, match="baseline"):
        cli._resolved_conditions(config, ("hidden_freeze",))
