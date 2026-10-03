import numpy as np
import torch

from latent_nca.experiment_1.conditions import ConditionSpec
from latent_nca.experiment_1.config import ExperimentConfig
from latent_nca.experiment_1.temporal_attractor import (
    TemporalAttractorSettings,
    _fit_projection,
    _record_condition,
    _record_step,
    _state_feature,
    _trajectory_legend_handles,
    _transform_with_residual,
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
