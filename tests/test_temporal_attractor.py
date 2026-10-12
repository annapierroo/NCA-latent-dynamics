import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from latent_nca.experiment_1.conditions import ConditionSpec
from latent_nca.experiment_1.config import (
    ExperimentConfig,
    load_experiment_config,
)
from latent_nca.experiment_1.metrics import TargetRegime
from latent_nca.experiment_1.temporal_attractor import (
    TemporalAttractorSettings,
    _fit_projection,
    _record_condition,
    _record_step,
    _resolved_conditions,
    _state_feature,
    _trajectory_legend_handles,
    _transform_with_residual,
    run_temporal_attractor_analysis,
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
    assert len(conditions) == 9
    assert {condition.name for condition in conditions} >= {
        "hidden_freeze",
        "hidden_freeze_first_6",
        "restore_development_step_48_period_150",
    }


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
    for morphology in cli.MORPHOLOGIES:
        folder = tmp_path / morphology
        folder.mkdir()
        config = load_experiment_config("configs/experiment_1.yaml")
        (folder / "resolved_config.json").write_text(json.dumps(config.to_dict()))
        with (folder / "per_run_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["replicate", "condition", "survival_time"])
            writer.writeheader()
            writer.writerows(
                [{"replicate": i, "condition": "baseline", "survival_time": 10 + i} for i in (0, 1)]
            )

    def capture(*args, **kwargs):
        calls.append(args)
        return args[1]

    monkeypatch.setattr(cli, "run_temporal_attractor_analysis", capture)
    args = cli.build_parser().parse_args(
        ["attractor", "--results-root", str(tmp_path), "--all-replicates", "--crop-size", "0"]
    )
    args.handler(args)
    assert len(calls) == 8
    assert {call[0].name for call in calls} == set(cli.MORPHOLOGIES)
    assert all(
        len(call[3]) == 9 and "hidden_freeze" in call[3] and "hidden_freeze_first_6" in call[3]
        for call in calls
    )
    assert all(call[4].crop_size == 0 for call in calls)
    with pytest.raises(ValueError, match="baseline"):
        cli.analyze_attractor(
            cli.build_parser().parse_args(
                [
                    "attractor",
                    "--results-dir",
                    str(tmp_path / "gecko"),
                    "--conditions",
                    "hidden_freeze",
                ]
            )
        )


def test_small_full_pipeline_recomputes_failures_and_records_all_conditions(tmp_path, monkeypatch):
    from latent_nca.experiment_1 import temporal_attractor as analysis

    class ToyModel:
        def __call__(self, state, update_probability):
            return state + 0.02 * (torch.rand_like(state[:, :1]) <= update_probability)

    source = tmp_path / "source"
    source.mkdir()
    checkpoint, target_path = source / "model.pt", source / "target.png"
    checkpoint.write_bytes(b"synthetic checkpoint")
    target_path.write_bytes(b"synthetic target")
    cfg = replace(
        load_experiment_config("configs/experiment_1.yaml"),
        checkpoint=str(checkpoint),
        target=str(target_path),
    )
    (source / "resolved_config.json").write_text(json.dumps(cfg.to_dict()))
    regime = TargetRegime(0.00025, 0.9, 0.9, 1.1, 0.95)
    (source / "target_regime.json").write_text(json.dumps(regime.to_dict()))
    with (source / "per_run_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "replicate", "first_failure_step"])
        writer.writeheader()
        writer.writerows(
            [
                {"condition": c.name, "replicate": 0, "first_failure_step": 999}
                for c in _resolved_conditions(cfg, None)
            ]
        )
    mature = torch.ones(16, 4, 4)
    torch.save(
        {
            i: {
                "state": mature,
                "mature_step": 68,
                "development_seed": 20000 + i,
                "development_states": {16: mature * 0.2, 48: mature * 0.8},
            }
            for i in range(3)
        },
        source / "mature_states.pt",
    )
    monkeypatch.setattr(analysis, "load_model", lambda *args: ToyModel())
    monkeypatch.setattr(analysis, "load_distill_target", lambda *args: mature[None, :4])
    output = tmp_path / "pca"
    settings = TemporalAttractorSettings(
        horizon=15, dense_until=2, record_every=3, crop_size=0, pca_components=3
    )
    run_temporal_attractor_analysis(source, output, 0, None, settings, "cpu")
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(metadata["conditions"]) == 9
    assert metadata["representation"]["features"] == 16 * 4 * 4
    assert metadata["source_first_failure_steps"]["baseline"] == 999
    assert 0 < metadata["replayed_first_failure_steps"]["baseline"] < 15
    with np.load(output / "full_state_features__hidden_freeze.npz") as archive:
        frames = archive["features"].reshape(-1, 16, 4, 4)
        assert np.all(frames[:, 4:] == 1)
        assert np.any(frames[-1, :4] != 1)
    assert (output / "pairwise_pca_3d.png").is_file()
    assert (output / "reference_pca_trajectories.png").is_file()
