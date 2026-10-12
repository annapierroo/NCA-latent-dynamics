import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from latent_nca.artifacts import file_sha256
from latent_nca.experiment_1.conditions import ConditionSpec
from latent_nca.experiment_1.config import (
    AnalysisConfig,
    EvaluationConfig,
    ExperimentConfig,
)
from scripts import experiment_1 as experiment


def test_snapshot_selection_uses_all_conditions_and_penalizes_missing_values():
    constant = {name: np.ones((3, 5)) for name in ("target_region_mse", "alpha_iou", "area_ratio")}
    varying = {
        "target_region_mse": np.repeat([[0.01], [0.02], [0.5]], 5, axis=1),
        "alpha_iou": np.repeat([[0.2], [0.7], [0.9]], 5, axis=1),
        "area_ratio": np.repeat([[0.4], [1.0], [5.0]], 5, axis=1),
    }
    for values in varying.values():
        values[:, -1] = np.nan
        values[0, 2:4] = np.nan
    scores = experiment._snapshot_replica_scores({"baseline": constant, "restore": varying})
    assert np.isfinite(scores).all()
    assert np.argmin(scores) == 1
    assert scores[0] > scores[1]


def test_snapshot_replay_preserves_seed_and_checks_saved_metrics():
    class NoisyRule:
        calls = 0

        def __call__(self, state, update_probability):
            self.calls += 1
            return state + torch.rand_like(state) * 0.01

    config = replace(ExperimentConfig(), evaluation=EvaluationConfig(horizon=4))
    condition = ConditionSpec("hidden_freeze", "Freeze", "freeze", "beta", 1.0)
    mature = torch.ones(1, 16, 4, 4)
    target = mature[:, :4].clone()
    evaluator = experiment.MorphologyEvaluator(target, 0.1)
    names = ("target_mse", "target_region_mse", "alpha_iou", "area_ratio")
    saved = {name: np.full(5, np.nan) for name in names}
    state = mature.clone()
    experiment.seed_everything(123)
    for step in range(3):
        values = evaluator.evaluate(state)
        for name in names:
            saved[name][step] = values[name]
        if step < 2:
            state = experiment.apply_post_step_intervention(
                NoisyRule()(state, 0.5), mature, condition, step + 1
            )
    model = NoisyRule()
    frames = experiment._replay_snapshot_frames(
        model, target, mature, mature, condition, config, 123, (0, 2, 4), saved
    )
    assert model.calls == 2 and frames[4] is None
    assert frames[2].shape == (4, 4, 4)
    torch.testing.assert_close(frames[2], state[0, :4])
    model = NoisyRule()
    assert experiment._replay_snapshot_frames(
        model, target, mature, mature, condition, config, 123, (4,), saved
    ) == {4: None}
    assert model.calls == 0
    saved["target_mse"][2] += 1
    with pytest.raises(experiment._SnapshotReplayMismatch, match="replay differs") as caught:
        experiment._replay_snapshot_frames(
            NoisyRule(), target, mature, mature, condition, config, 123, (2,), saved
        )
    difference = caught.value.differences[0]
    assert difference["metric"] == "target_mse" and difference["step"] == 2
    assert difference["saved_value"] == saved["target_mse"][2]
    assert difference["absolute_difference"] == pytest.approx(1)
    assert difference["allowed_difference"] == pytest.approx(1e-6 + 1e-4 * saved["target_mse"][2])


def test_snapshot_gallery_only_requests_t10000_and_draws_divergent_replicas(tmp_path, monkeypatch):
    args = experiment.build_parser().parse_args(["plot-final-gallery", "--morphologies", "gecko"])
    assert args.gallery and args.snapshot_steps == (10000,)
    rgba = torch.zeros(4, 8, 8)
    rgba[1, 2:6, 2:6] = 0.7
    rgba[3, 2:6, 2:6] = 1
    path = tmp_path / "final_gallery.png"
    from matplotlib.axes import Axes

    labels = []
    original_text = Axes.text

    def record_text(axis, x, y, text, *args, **kwargs):
        labels.append(text)
        return original_text(axis, x, y, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", record_text)
    experiment._plot_final_gallery(
        path, {10: rgba, 20: None, 30: None}, "Synthetic", 10000, unverified={20}
    )
    assert "Replay not\nverified" in labels and "Numerically\ndiverged" in labels
    assert path.is_file()
    with Image.open(path) as image:
        assert image.width > image.height
    experiment.render_snapshot_grid(
        tmp_path / "representative.png",
        {"baseline": {0: rgba, 2: rgba}, "hidden_freeze": {0: rgba, 2: None}},
        (
            ConditionSpec("baseline", "Control", "baseline"),
            ConditionSpec("hidden_freeze", "Freeze", "freeze", "beta", 1.0),
        ),
        (0, 2),
        include_all=True,
        title="Synthetic paired replica 010",
    )


@pytest.mark.parametrize("unrelated_error", (False, True))
def test_gallery_records_mismatches_and_continues_only_for_replay_errors(
    tmp_path, monkeypatch, unrelated_error
):
    folder = tmp_path / "gecko"
    folder.mkdir()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    target = tmp_path / "target.png"
    target.write_bytes(b"target")
    config = replace(
        ExperimentConfig(),
        checkpoint=str(checkpoint),
        target=str(target),
        evaluation=EvaluationConfig(horizon=4),
        analysis=AnalysisConfig(snapshot_steps=(0, 4)),
    )
    (folder / "resolved_config.json").write_text(json.dumps(config.to_dict()))
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "device": "cpu",
                "checkpoint_sha256": file_sha256(checkpoint),
                "target_sha256": file_sha256(target),
            }
        )
    )
    np.savez(folder / "trajectory_metrics.npz", replicate_ids=np.arange(4))
    original_hash = file_sha256(folder / "trajectory_metrics.npz")
    condition = ConditionSpec("baseline", "Control", "baseline")
    metrics = {
        condition.name: {
            name: np.ones((4, 5))
            for name in ("target_mse", "target_region_mse", "alpha_iou", "area_ratio")
        }
    }
    mature = torch.ones(16, 4, 4)
    torch.save(
        {replicate: {"state": mature} for replicate in range(4)}, folder / "mature_states.pt"
    )
    torch.save({"intervention": {"baseline": {4: mature[:4]}}}, folder / "snapshots.pt")
    monkeypatch.setattr(experiment, "_load_core_metrics", lambda _: (metrics, (condition,), None))
    monkeypatch.setattr(experiment, "get_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(experiment, "load_model", lambda *_: object())
    monkeypatch.setattr(experiment, "load_distill_target", lambda *_: mature[:4].unsqueeze(0))
    replayed_ids = []

    def replay(model, target, state, reference, condition, config, seed, steps, saved):
        replicate = seed - config.evaluation.seed_start - config.evaluation.continuation_seed_offset
        replayed_ids.append(replicate)
        if replicate == 1:
            if unrelated_error:
                raise ValueError("Invalid reference state")
            raise experiment._SnapshotReplayMismatch(
                condition.name,
                4,
                [{"step": 4, "metric": "target_mse", "saved_value": 1, "replayed_value": 2}],
            )
        return {4: None if replicate == 3 else mature[:4]}

    monkeypatch.setattr(experiment, "_replay_snapshot_frames", replay)
    galleries = []

    def plot(path, frames, title, step, *, unverified):
        galleries.append((frames, unverified))
        path.write_bytes(b"gallery")

    monkeypatch.setattr(experiment, "_plot_final_gallery", plot)
    args = experiment.build_parser().parse_args(
        ["plot-final-gallery", "--morphologies", "gecko", "--results-root", str(tmp_path)]
    )
    args.snapshot_steps = (4,)
    if unrelated_error:
        with pytest.raises(ValueError, match="Invalid reference state"):
            experiment.plot_population_snapshots(args)
        assert replayed_ids == [1] and not galleries
    else:
        experiment.plot_population_snapshots(args)
        assert replayed_ids == [1, 2, 3]
        frames, unverified = galleries[0]
        assert set(frames) == {0, 1, 2, 3} and unverified == {1}
        assert frames[1] is None and frames[3] is None
        torch.testing.assert_close(frames[2], mature[:4])
        diagnostics = folder / "final_t4__baseline.replay_mismatches.csv"
        with diagnostics.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["replicate"] == "1" and rows[0]["condition"] == "baseline"
        assert rows[0]["saved_value"] == "1" and rows[0]["replayed_value"] == "2"
        experiment.plot_population_snapshots(args)
        assert replayed_ids == [1, 2, 3] and len(galleries) == 1
    assert file_sha256(folder / "trajectory_metrics.npz") == original_hash
    assert not (folder / "population_snapshots").exists()


@pytest.mark.parametrize("command", ("tables", "analyze"))
def test_removed_analysis_commands_are_unavailable(command):
    with pytest.raises(SystemExit) as caught:
        experiment.build_parser().parse_args([command])
    assert caught.value.code == 2
