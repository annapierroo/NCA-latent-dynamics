import json

import numpy as np
import pytest
import torch

from latent_nca.artifacts import file_sha256
from latent_nca.experiment_1.metrics import MorphologyEvaluator, TargetRegime
from scripts import experiment_1 as experiment
from scripts.experiment_1 import (
    ConditionSpec,
    TemporalAttractorSettings,
    _record_condition,
    apply_post_step_intervention,
    load_experiment_config,
)


def test_masked_freeze_preserves_rgba_and_background_on_rectangular_grid():
    state = torch.arange(16 * 4 * 6, dtype=torch.float32).reshape(1, 16, 4, 6)
    before = state.clone()
    reference = state * 0.5
    mask = torch.zeros(1, 1, 4, 6, dtype=torch.bool)
    mask[:, :, 1:3, 2:4] = True
    condition = ConditionSpec("hidden_freeze_morphology", "Target freeze", "morphology_freeze")

    result = apply_post_step_intervention(state, reference, condition, 1, target_mask=mask)

    assert torch.equal(state, before)
    assert torch.equal(result[:, :4], state[:, :4])
    assert torch.equal(result[:, 4:, 1:3, 2:4], reference[:, 4:, 1:3, 2:4])
    outside = (~mask).expand_as(state[:, 4:])
    assert torch.equal(result[:, 4:][outside], state[:, 4:][outside])
    with pytest.raises(ValueError, match="fixed target"):
        apply_post_step_intervention(state, reference, condition, 1)
    with pytest.raises(ValueError, match="boolean"):
        apply_post_step_intervention(state, reference, condition, 1, target_mask=mask.float())


def test_pca_recording_uses_same_fixed_mask_and_keeps_background():
    class AddOne:
        def __call__(self, state, update_probability):
            return state + 1

    mature = torch.ones(1, 16, 4, 6)
    target = torch.zeros(1, 4, 4, 6)
    target[:, 3, 1:3, 2:4] = 1
    evaluator = MorphologyEvaluator(target)
    condition = ConditionSpec("hidden_freeze_morphology", "Target freeze", "morphology_freeze")
    trajectory = _record_condition(
        AddOne(),
        mature,
        condition,
        load_experiment_config("configs/experiment_1.yaml"),
        TemporalAttractorSettings(horizon=2, dense_until=2, crop_size=0),
        continuation_seed=1,
        evaluator=evaluator,
    )
    frames = trajectory.features.reshape(3, 16, 4, 6)
    assert np.all(frames[:, 4:, 1:3, 2:4] == 1)
    assert np.all(frames[2, 4:, 0, :] == 3)
    assert np.all(frames[2, :4] == 3)


@pytest.fixture
def small_run(tmp_path, monkeypatch):
    checkpoint, target_path = tmp_path / "model.pt", tmp_path / "target.png"
    checkpoint.write_bytes(b"checkpoint")
    target_path.write_bytes(b"target")
    target = torch.zeros(1, 4, 6, 8)
    target[:, :3, 2:4, 2:6] = 0.5
    target[:, 3, 2:4, 2:6] = 1

    class ToyModel:
        def __init__(self):
            self.calls = 0

        def seed(self, batch, height, width, device):
            state = torch.full((batch, 16, height, width), 0.1, device=device)
            state[:, :4] = target.to(device)
            return state

        def __call__(self, state, update_probability):
            self.calls += 1
            delta = 0.01 * (torch.rand_like(state[:, :1]) < update_probability)
            result = state + delta
            result[:, :4] = torch.where(
                target[:, 3:4].to(state.device) > 0.1, result[:, :4], state[:, :4]
            )
            return result

    model = ToyModel()
    monkeypatch.setattr(experiment, "load_model", lambda *args: model)
    monkeypatch.setattr(experiment, "load_distill_target", lambda *args: target)
    calibration_calls = []

    def calibrate(*args):
        calibration_calls.append(args[2].calibration.replicates)
        return TargetRegime(100, 0.01, 0.01, 100, 0.95), {
            name: np.ones(2) for name in experiment.MORPHOLOGY_METRIC_NAMES
        }

    monkeypatch.setattr(experiment, "_calibrate_regime", calibrate)
    config = experiment.ExperimentConfig(
        checkpoint=str(checkpoint),
        target=str(target_path),
        output_dir=str(tmp_path / "gecko"),
        device="cpu",
        evaluation=experiment.EvaluationConfig(replicates=2, horizon=4),
        analysis=experiment.AnalysisConfig(
            snapshot_steps=(0, 2, 4), hidden_channel_record_every=2, visible_channel_record_every=2
        ),
    )
    return config, model, calibration_calls


def test_fresh_run_records_all_conditions_and_resumes_without_repeating(small_run, monkeypatch):
    config, model, calibration_calls = small_run
    original_develop = experiment._develop_to_target
    interrupted = False

    def develop(*args):
        nonlocal interrupted
        if args[-1] == config.evaluation.seed_start + 1 and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted")
        return original_develop(*args)

    monkeypatch.setattr(experiment, "_develop_to_target", develop)
    with pytest.raises(RuntimeError, match="interrupted"):
        experiment.run_experiment(config)
    folder = experiment.Path(config.output_dir)
    first = folder / "replicate_000" / "trajectories.npz"
    digest = file_sha256(first)
    experiment.run_experiment(config)
    assert file_sha256(first) == digest
    assert calibration_calls == [32]
    calls = model.calls
    experiment.run_experiment(config)
    assert model.calls == calls
    payload = experiment.load_payload(first.parent)
    assert payload["condition_names"].tolist() == [
        "baseline",
        "hidden_freeze",
        "hidden_freeze_morphology",
        "hidden_freeze_first_6",
        "restore_beta_0p1",
        "periodic_reset_150",
        "restore_development_step_16_beta_0p1",
        "restore_development_step_16_period_150",
    ]
    assert payload["metrics"].shape == (8, 5, len(experiment.METRIC_NAMES))
    assert payload["snapshots"].shape == (8, 3, 4, 6, 8)
    assert payload["hidden_channels"].shape == (8, 3, 12, 3)
    assert payload["visible_channels"].shape == (8, 3, 4, 3)
    assert payload["development_reference_steps"].tolist() == [16]
    assert np.all(payload["hidden_channels"][1, ..., 0] == 0)
    assert np.any(payload["hidden_channels"][2, 1:, ..., 0] > 0)
    # No duplicate aggregate trajectory archives or import/migration caches.
    assert not (folder / "trajectory_metrics.npz").exists()
    assert not (folder / "morphology_freeze_runs").exists()
    assert (folder / "per_run_summary.csv").is_file()
    experiment.Path(config.checkpoint).write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="inputs differ"):
        experiment.run_experiment(config)


def test_all_snapshot_figures_use_saved_frames_without_replay(small_run, monkeypatch):
    config, _, _ = small_run
    folder = experiment.run_experiment(config)
    originals = {path: file_sha256(path) for path in folder.glob("replicate_*/trajectories.npz")}
    monkeypatch.setattr(experiment, "load_model", lambda *args: pytest.fail("No snapshot replay"))
    monkeypatch.setattr(
        experiment,
        "_run_branch",
        lambda *args, **kwargs: pytest.fail("No simulation during plotting"),
    )
    experiment.plot_folder(folder)
    output = folder / "analysis"
    for name in (
        "core_morphology_metrics.png",
        "hidden_channel_dynamics.png",
        "visible_channel_dynamics.png",
        "morphology_snapshots.png",
    ):
        assert (output / name).stat().st_size > 1000
    assert len(list(output.glob("final_t4__*.png"))) == 8
    assert originals == {path: file_sha256(path) for path in originals}


def test_full_state_pca_pipeline_uses_new_archives_and_does_not_save_raw_states(small_run):
    config, _, _ = small_run
    folder = experiment.run_experiment(config)
    output = folder / "analysis" / "pca"
    settings = experiment.TemporalAttractorSettings(
        horizon=4, dense_until=2, record_every=2, crop_size=0, pca_components=3
    )
    experiment.run_temporal_attractor_analysis(folder, output, 0, None, settings, "cpu")
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(metadata["conditions"]) == 8
    assert metadata["representation"]["features"] == 16 * 6 * 8
    assert metadata["raw_features_saved"] is False
    assert not list(output.glob("full_state_features*"))
    assert (output / "pairwise_pca_3d.png").is_file()
    assert (output / "reference_pca_trajectories.png").is_file()
    with np.load(output / "reference_pca_coordinates.npz", allow_pickle=False) as archive:
        assert np.isfinite(archive["hidden_freeze_morphology__coordinates"]).all()


def test_cli_defaults_are_four_morphologies_and_30_replicas():
    args = experiment.build_parser().parse_args(["run"])
    assert args.replicates == 30
    assert set(args.morphologies) == {"gecko", "butterfly", "planaria", "spider"}
    args = experiment.build_parser().parse_args(["attractor"])
    assert args.crop_size == 0
    with pytest.raises(SystemExit):
        experiment.build_parser().parse_args(["run-morphology-freeze"])


def test_nonfinite_hidden_states_are_rejected_before_target_attainment():
    class BrokenModel:
        def seed(self, batch, height, width, device):
            state = torch.ones(batch, 16, height, width)
            state[:, 4:] = float("inf")
            return state

    target = torch.ones(1, 4, 4, 4)
    config = experiment.ExperimentConfig()
    with pytest.raises(ValueError, match="non-finite calibration"):
        experiment._calibrate_regime(BrokenModel(), target, config, torch.device("cpu"))
    result = experiment._develop_to_target(
        BrokenModel(), target, TargetRegime(1, 0.5, 0.5, 2, 0.95), config, torch.device("cpu"), 1
    )
    assert result[0] is None
    assert result[-1] == "nonfinite_during_development"
