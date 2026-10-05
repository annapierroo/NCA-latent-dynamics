"""Protocol checks on small synthetic states, without training a model."""

import sys
from argparse import Namespace
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
import torch

spec = spec_from_file_location(
    "experiment_2_comparison", Path(__file__).parents[1] / "scripts" / "experiment_2.py"
)
experiment = module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


class StochasticRule(torch.nn.Module):
    def forward(self, state, update_probability=0.5):
        mask = torch.rand_like(state[:, :1]) <= update_probability
        return state + mask * 0.0001


def test_mature_restore_does_not_change_rgba_or_reference():
    cfg = experiment.Settings()
    model = StochasticRule()
    reference = torch.ones(1, 16, 4, 4)
    state = reference * 2
    torch.manual_seed(4)
    natural = model(state)
    torch.manual_seed(4)
    restored = experiment.update(model, state, reference, "growing_restore", cfg)
    assert torch.equal(restored[:, :4], natural[:, :4])
    assert torch.allclose(restored[:, 4:], 0.9 * natural[:, 4:] + 0.1 * reference[:, 4:])
    assert torch.all(reference == 1)


def test_branch_masks_are_paired_and_independent_control_is_separate():
    device = torch.device("cpu")
    experiment.seed_nca(20, device)
    common = experiment.capture_rng(device)
    experiment.seed_nca(30, device)
    independent = experiment.capture_rng(device)
    reference = torch.ones(1, 16, 16, 16)
    states = [reference.clone() for _ in experiment.BRANCHES]
    for _ in range(3):
        states, common, independent = experiment.paired_updates(
            StochasticRule(),
            states,
            reference,
            "persistent",
            experiment.Settings(),
            common,
            independent,
        )
        for state in states[2:]:
            assert torch.equal(state, states[0])
    assert not torch.equal(states[1], states[0])


def test_noise_and_erase_have_different_channel_scopes_and_do_not_consume_torch_rng():
    anchor = torch.ones(1, 16, 8, 8)
    target = anchor[0, :4].numpy().copy()
    target[3, :2] = 0
    history = np.ones((24, 16, 8, 8), dtype=np.float32)
    saved_rng = torch.random.get_rng_state()
    states, metadata, disks = experiment.perturbations(anchor, target, history, 8, 0.1)
    assert torch.equal(torch.random.get_rng_state(), saved_rng)
    for state in states[2:5]:
        assert torch.equal(state[:, :4], anchor[:, :4])
        assert torch.equal(state[:, 4:, :2], anchor[:, 4:, :2])
        assert not torch.equal(state[:, 4:], anchor[:, 4:])
    for state, disk in zip(states[5:], disks, strict=True):
        assert torch.count_nonzero(state[:, :, disk]) == 0
        assert torch.equal(state[:, :, ~disk], anchor[:, :, ~disk])
    assert torch.all(anchor == 1)
    assert metadata["noise_0p1"]["relative_sigma"] == 0.1
    assert 0 < metadata["erase_0p1"]["actual_target_fraction"] < 1


def test_local_pca_fit_does_not_use_validation_states_and_detects_out_of_plane_motion():
    rng = np.random.default_rng(1)
    frames = rng.normal(size=(40, 4, 3, 3))
    altered = frames.copy()
    altered[20:] += 100
    first = experiment.fit_local_pca(frames, 3, 5)
    second = experiment.fit_local_pca(altered, 3, 5)
    assert np.array_equal(first.components, second.components)
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.scale, second.scale)
    direction = rng.normal(size=first.mean.shape)
    direction -= first.components.T @ (first.components @ direction)
    moved = (first.mean + direction * 100).reshape(4, 3, 3) * first.scale[:, None, None]
    scores, residual, _, _ = first.project(moved)
    assert np.linalg.norm(scores) < 1e-8
    assert residual > first.residual_limit


def test_constant_reference_has_finite_pca_and_no_invented_spectral_peak():
    local = experiment.fit_local_pca(np.ones((40, 4, 3, 3)), 3, 2)
    scores, residual, cloud, radius = local.project(np.ones((4, 3, 3)))
    assert np.allclose(scores, 0)
    assert residual == cloud == radius == 0
    summary, _ = experiment.spectral_diagnostics(np.zeros((128, 3)), 64)
    assert np.isnan(summary["dominant_period"])
    assert summary["dominant_power_fraction"] == 0


def test_fft_resolves_a_known_period_and_windows_exclude_the_impulse():
    scores = np.sin(2 * np.pi * np.arange(1024) / 32)[:, None]
    summary, _ = experiment.spectral_diagnostics(scores, 128)
    assert summary["dominant_period"] == pytest.approx(32)
    assert summary["at_least_three_cycles"]
    windows = experiment.analysis_windows(experiment.Settings())
    assert windows[0] == ("pre_pulse", 250, 499)
    assert windows[1] == ("t501_1000", 501, 1000)
    assert windows[-1] == ("t8001_10000", 8001, 10000)
    assert all(not start <= 500 <= stop for _, start, stop in windows)


def test_circular_damage_masks_match_across_systems_despite_different_hidden_scales():
    target = np.ones((4, 12, 12))
    a = experiment.noise_and_disks(target, 42, 0.1)
    b = experiment.noise_and_disks(target, 42, 0.1)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_cli_defaults_are_the_new_protocol():
    args = experiment.build_parser().parse_args(["run"])
    assert args.perturb_step == 500
    assert args.horizon == 10_000
    assert tuple(args.morphologies) == ("gecko", "butterfly")
    assert experiment.build_parser().parse_args(["train"]).systems == ("persistent",)


def test_short_synthetic_pipeline_and_offline_analysis(tmp_path):
    cfg = replace(
        experiment.Settings(),
        perturb_step=30,
        reference_start=8,
        horizon=70,
        pca_components=3,
        snapshot_replicates=1,
    )
    cfg.validate()
    target = torch.ones(1, 4, 4, 4)
    mature = torch.ones(1, 16, 4, 4)
    regime = {
        "target_mse_max": 10.0,
        "alpha_iou_min": 0.0,
        "area_ratio_min": 0.0,
        "area_ratio_max": 2.0,
    }
    payload, metadata = experiment.simulate(
        StochasticRule(), target, mature, "growing_restore", 0, regime, cfg
    )
    assert payload["metrics"].shape == (7, 71, len(experiment.METRICS))
    assert np.all(payload["metrics"][:, :30, 8] == 0)
    assert np.all(payload["metrics"][2:5, 30, 9] == 0)
    assert np.all(payload["metrics"][5:, 30, 9] > 0)
    assert np.all(payload["divergence_step"] == -1)
    assert payload["channel_temporal_variance"].shape == (7, 2, 16)
    assert np.isfinite(payload["channel_temporal_variance"]).all()
    folder = tmp_path / "gecko" / "default"
    replica = folder / "growing_restore" / "replicate_000"
    replica.mkdir(parents=True)
    experiment.save_json(folder / "manifest.json", {"settings": experiment.asdict(cfg)})
    experiment.save_json(replica / "metadata.json", metadata)
    experiment.save_npz(replica / "trajectories.npz", **payload)
    args = Namespace(
        output_root=str(tmp_path),
        morphologies=["gecko"],
        checkpoint_tag="default",
        bootstrap_samples=20,
    )
    experiment.analyze(args)
    assert (replica / "window_metrics.csv").is_file()
    assert (replica / "window_diagnostics.npz").is_file()
    assert (folder / "condition_metrics.csv").is_file()
    assert (folder / "analysis_settings.json").is_file()


def test_numerically_failed_window_is_not_analyzed_as_a_shorter_successful_window():
    cfg = replace(
        experiment.Settings(),
        perturb_step=30,
        reference_start=8,
        horizon=70,
        pca_components=3,
        snapshot_replicates=0,
    )
    shape = (len(experiment.BRANCHES), 71, 3)
    scores = np.zeros(shape)
    scores[2, 60:] = np.nan
    payload = {
        "steps": np.arange(71),
        "metrics": np.zeros((7, 71, len(experiment.METRICS))),
        "hidden_scores": scores,
        "rgba_scores": scores,
        "in_target": np.ones((7, 71), bool),
        "in_reference": np.ones((7, 71), bool),
        "channel_temporal_variance": np.zeros((7, 2, 16)),
    }
    rows, _ = experiment.window_rows(payload, cfg)
    failed = next(
        row
        for row in rows
        if row["branch"] == "noise_0p01" and row["group"] == "hidden" and row["window"] == "t31_70"
    )
    assert not failed["complete"]
    assert "dominant_period" not in failed


def test_streaming_temporal_variance_matches_a_known_linear_trajectory():
    cfg = replace(
        experiment.Settings(),
        perturb_step=30,
        reference_start=8,
        horizon=70,
        pca_components=3,
        snapshot_replicates=0,
        fire_rate=1,
    )
    regime = {"target_mse_max": 10, "alpha_iou_min": 0, "area_ratio_min": 0, "area_ratio_max": 2}
    payload, _ = experiment.simulate(
        StochasticRule(),
        torch.ones(1, 4, 4, 4),
        torch.ones(1, 16, 4, 4),
        "persistent",
        0,
        regime,
        cfg,
    )
    increment = float((torch.tensor(1.0) + 0.0001) - 1)
    for index, (_, start, stop) in enumerate(experiment.analysis_windows(cfg)):
        expected = increment**2 * ((stop - start + 1) ** 2 - 1) / 12
        np.testing.assert_allclose(
            payload["channel_temporal_variance"][0, index], expected, rtol=1e-6
        )


def test_recovery_waits_for_delayed_visible_damage_before_counting_return():
    cfg = replace(experiment.Settings(), perturb_step=30, reference_start=8, horizon=70)
    flags = np.ones((7, 71), dtype=bool)
    flags[2, 35:45] = False
    payload = {
        "metrics": np.zeros((7, 71, len(experiment.METRICS))),
        "in_target": flags,
        "in_reference": np.ones_like(flags),
        "divergence_step": np.full(7, -1),
    }
    metadata = {
        "in_target_before_pulse": True,
        "local_pca": {
            group: {"approximately_stationary_heuristic": True} for group in experiment.GROUPS
        },
    }
    rows = experiment.recovery_rows(payload, metadata, [], cfg)
    assert rows[0]["never_left_target"]
    assert rows[0]["first_recovery_delay"] == 0
    assert rows[2]["first_target_exit_delay"] == 5
    assert rows[2]["first_recovery_delay"] == 15
    assert rows[2]["recovery_observed"]
    assert not rows[2]["never_left_target"]


def test_resume_skips_recorded_replicas_and_rejects_changed_checkpoints(tmp_path, monkeypatch):
    calls = []
    args = experiment.build_parser().parse_args(
        [
            "run",
            "--morphologies",
            "gecko",
            "--replicates",
            "1",
            "--device",
            "cpu",
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--data-dir",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "reports"),
        ]
    )
    paths = []
    for system in experiment.SYSTEMS:
        path = experiment.checkpoint_path(Path(args.checkpoint_root), "gecko", system, "default")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"synthetic checkpoint")
        paths.append(path)
    Path(args.data_dir).mkdir()
    (Path(args.data_dir) / "gecko.png").write_bytes(b"synthetic target")
    monkeypatch.setattr(experiment, "load_growing_checkpoint", lambda *a, **kw: StochasticRule())
    monkeypatch.setattr(experiment, "load_distill_target", lambda *a, **kw: torch.ones(1, 4, 4, 4))
    monkeypatch.setattr(experiment, "calibrate", lambda *a: ({"synthetic": True}, np.zeros((2, 2))))
    monkeypatch.setattr(experiment, "develop", lambda *a: (torch.ones(1, 16, 4, 4), 70, None))

    def fake_simulate(model, target, mature, system, replicate, regime, cfg):
        calls.append((system, replicate))
        return {"synthetic": np.zeros(1)}, {"status": "complete"}

    monkeypatch.setattr(experiment, "simulate", fake_simulate)
    experiment.run(args)
    experiment.run(args)
    assert calls == [("persistent", 0), ("growing_restore", 0)]
    paths[0].write_bytes(b"changed synthetic checkpoint")
    with pytest.raises(ValueError, match="changed"):
        experiment.run(args)
