"""Protocol checks on small synthetic states, without training a model."""

import errno
import json
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


def synthetic_erasure_plan(size=4, morphology="gecko"):
    plan = {}
    positions = ((size // 2, 1), (size // 2, size - 2), (size - 1, 1), (size - 1, size - 2))
    for part, (y, x) in zip(experiment.PRIMITIVE_REGIONS[morphology], positions, strict=True):
        mask = np.zeros((size, size), dtype=bool)
        mask[y, x] = True
        plan[f"erase_{part}"] = {
            "mask": mask,
            "primitive": part,
            "removed_target_pixels": 1,
            "requested_target_fraction": 0.05,
            "actual_target_fraction": 1 / size**2,
        }
    return plan


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


def test_all_branch_masks_are_paired():
    device = torch.device("cpu")
    experiment.seed_nca(20, device)
    common = experiment.capture_rng(device)
    reference = torch.ones(1, 16, 16, 16)
    states = [reference.clone() for _ in experiment.condition_names(synthetic_erasure_plan())]
    for _ in range(3):
        states, common = experiment.paired_updates(
            StochasticRule(),
            states,
            reference,
            "persistent",
            experiment.Settings(),
            common,
        )
        for state in states[1:]:
            assert torch.equal(state, states[0])


def test_noise_and_erase_have_different_channel_scopes_and_do_not_consume_torch_rng():
    anchor = torch.ones(1, 16, 8, 8)
    target = anchor[0, :4].numpy().copy()
    target[3, :2] = 0
    history = np.ones((24, 16, 8, 8), dtype=np.float32)
    saved_rng = torch.random.get_rng_state()
    states, metadata, disks = experiment.perturbations(
        anchor, target, history, 8, 0.1, synthetic_erasure_plan(8)
    )
    assert torch.equal(torch.random.get_rng_state(), saved_rng)
    for state in states[1:3]:
        assert torch.equal(state[:, :4], anchor[:, :4])
        assert torch.equal(state[:, 4:, :2], anchor[:, 4:, :2])
        assert not torch.equal(state[:, 4:], anchor[:, 4:])
    for state, disk in zip(states[3:], disks, strict=True):
        assert torch.count_nonzero(state[:, :, disk]) == 0
        assert torch.equal(state[:, :, ~disk], anchor[:, :, ~disk])
    assert torch.all(anchor == 1)
    assert metadata["noise_0p1"]["relative_sigma"] == 0.1
    assert metadata["noise_0p25"]["relative_sigma"] == 0.25
    assert 0 < metadata["erase_leg"]["actual_target_fraction"] < 1
    np.testing.assert_allclose(
        states[2][:, 4:].numpy() - 1, 2.5 * (states[1][:, 4:].numpy() - 1), atol=3e-7
    )


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
    scores, residual = first.project(moved)
    assert np.linalg.norm(scores) < 1e-8
    assert residual > first.residual_limit


@pytest.mark.parametrize("morphology", experiment.MORPHOLOGIES)
def test_anatomical_masks_are_equal_area_connected_and_confined_to_target(morphology):
    from scipy.ndimage import label

    target = experiment.as_array(
        experiment.load_distill_target(Path(__file__).parents[1] / "data" / f"{morphology}.png")
    )
    plan = experiment.primitive_erasure_plan(target, morphology, experiment.Settings())
    masks = np.stack([entry["mask"] for entry in plan.values()])
    counts = masks.sum(axis=(1, 2))
    if morphology == "gecko":
        assert plan["erase_leg"]["removed_primitive_fraction"] == 1
        assert np.all(counts == plan["erase_leg"]["primitive_target_pixels"])
    else:
        assert set(plan) == {
            "erase_upper_left_wing",
            "erase_upper_right_wing",
            "erase_lower_left_wing",
            "erase_central_body",
        }
        assert np.all(counts == round(0.10 * (target[3] > 0.1).sum()))
        central_y, central_x = np.argwhere(plan["erase_central_body"]["mask"]).mean(axis=0)
        assert abs(central_x - target.shape[-1] / 2) < 3
        assert target.shape[-2] / 3 < central_y < 2 * target.shape[-2] / 3
    assert np.all(counts > round(0.05 * (target[3] > 0.1).sum()))
    assert not masks[:, target[3] <= 0.1].any()
    assert masks.sum(axis=0).max() == 1
    for mask in masks:
        assert label(mask, structure=np.ones((3, 3)))[1] == 1
    with pytest.raises(ValueError, match="decrease --erase-fraction"):
        experiment.primitive_erasure_plan(
            target, morphology, replace(experiment.Settings(), erase_fraction=0.9)
        )


def test_cli_defaults_are_the_new_protocol():
    args = experiment.build_parser().parse_args(["run"])
    assert args.perturb_step == 500
    assert args.horizon == 10_000
    assert args.erase_fraction is None
    assert args.output_root == "reports/experiment_2"
    assert experiment.NOISE_BRANCHES == ("noise_0p1", "noise_0p25")
    assert experiment.condition_names(synthetic_erasure_plan()) == (
        "control",
        "noise_0p1",
        "noise_0p25",
        "erase_leg",
        "erase_head",
        "erase_tail",
        "erase_torso",
    )
    assert tuple(args.morphologies) == ("gecko", "butterfly")
    assert experiment.build_parser().parse_args(["train"]).systems == ("persistent",)
    attractor = experiment.build_parser().parse_args(["attractor", "--replicates", "0", "2"])
    assert attractor.handler is experiment.analyze_attractor
    assert attractor.replicates == [0, 2]
    assert tuple(attractor.morphologies) == ("gecko", "butterfly")
    with pytest.raises(SystemExit):
        experiment.build_parser().parse_args(["replace-erasure"])
    comparison = experiment.build_parser().parse_args(["compare-control"])
    assert comparison.handler is experiment.compare_control
    assert tuple(comparison.morphologies) == experiment.MORPHOLOGIES


@pytest.mark.parametrize("morphology", experiment.MORPHOLOGIES)
def test_short_synthetic_pipeline_and_offline_analysis(tmp_path, morphology):
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
    plan = synthetic_erasure_plan(morphology=morphology)
    folder = tmp_path / morphology / "default"
    hashes = {}
    for system in experiment.SYSTEMS:
        payload, metadata = experiment.simulate(
            StochasticRule(), target, mature, system, 0, regime, cfg, plan
        )
        assert payload["metrics"].shape == (7, 71, 5)
        assert payload["control_morphology"].shape == (7, 71, 2)
        np.testing.assert_array_equal(payload["control_morphology"][..., :30, 0], 0)
        np.testing.assert_array_equal(payload["control_morphology"][..., :30, 1], 1)
        np.testing.assert_array_equal(payload["control_morphology"][0, :, 0], 0)
        np.testing.assert_array_equal(payload["control_morphology"][0, :, 1], 1)
        assert np.all(payload["control_morphology"][3:, 30, 0] > 0)
        assert np.all(payload["control_morphology"][3:, 30, 1] < 1)
        assert tuple(payload["branch_names"]) == experiment.condition_names(plan)
        np.testing.assert_array_equal(
            payload["erase_masks"], np.stack([entry["mask"] for entry in plan.values()])
        )
        pulse_index = payload["snapshot_steps"].tolist().index(cfg.perturb_step)
        for branch, entry in enumerate(plan.values(), start=3):
            frame = payload["snapshots"][branch, pulse_index]
            assert np.all(frame[:, entry["mask"]] == 0)
            np.testing.assert_array_equal(
                frame[:, ~entry["mask"]], payload["pre_pulse_state"][:4, ~entry["mask"]]
            )
        assert tuple(payload["metric_names"]) == experiment.METRICS
        assert np.all(payload["metrics"][:, :30, experiment.METRIC_INDEX["hidden_pair_rms"]] == 0)
        assert np.all(payload["divergence_step"] == -1)
        assert set(metadata["local_pca"]) == {"hidden"}
        assert "cloud_limit" not in metadata["local_pca"]["hidden"]
        assert "hidden_reference_scores" not in payload and "in_reference" not in payload
        assert not any(key.startswith(("rgba_", "channel_")) for key in payload)
        assert metadata["fit_steps_inclusive"] == [8, 18]
        assert metadata["validation_steps_inclusive"] == [19, 29]
        replica = folder / system / "replicate_000"
        replica.mkdir(parents=True)
        payload["in_target"].fill(False)
        metadata["in_target_before_pulse"] = False
        experiment.save_json(replica / "metadata.json", metadata)
        experiment.save_npz(replica / "trajectories.npz", **payload)
        hashes[replica] = experiment.file_sha256(replica / "trajectories.npz")
    experiment.save_json(folder / "manifest.json", {"settings": experiment.asdict(cfg)})
    args = Namespace(
        output_root=str(tmp_path),
        morphologies=[morphology],
        checkpoint_tag="default",
        bootstrap_samples=20,
        replicates=[0],
    )
    experiment.analyze_attractor(args)
    output = folder / "analysis"
    for name in (
        "recovery_by_replica.csv",
        "recovery_summary.csv",
        "paired_model_comparison.csv",
        "analysis_settings.json",
        "exclusions.json",
        "figure_replicas.json",
        "morphological_stability_recovery.png",
        "hidden_divergence_from_control.png",
        "morphological_deviation_from_control.png",
    ):
        assert (output / name).is_file()
    assert not (output / "hidden_cloud_distance.png").exists()
    for replica, original_hash in hashes.items():
        assert experiment.file_sha256(replica / "trajectories.npz") == original_hash
        assert not (replica / "window_metrics.csv").exists()
        assert not (replica / "window_diagnostics.npz").exists()
        assert not (folder / "attractor_analysis").exists()
        plots = output / replica.parent.name / replica.name
        assert (plots / "hidden_pca_2d.png").is_file()
        assert (plots / "hidden_pca_3d.png").is_file()
    import csv

    with (output / "recovery_by_replica.csv").open() as handle:
        rows = list(csv.DictReader(handle))
        assert all(row["in_target_before_pulse"] == "True" for row in rows)
        assert not any(row["branch"] == "independent_control" for row in rows)


def test_visible_comparison_uses_current_rgba_only_and_handles_empty_masks():
    control = np.ones((16, 4, 4))
    state = control.copy()
    state[4:] += 100
    np.testing.assert_array_equal(
        experiment.control_morphology_metrics(state, control, 0.1), [0, 1]
    )
    state[0] += 0.2
    values = experiment.control_morphology_metrics(state, control, 0.1)
    np.testing.assert_allclose(values, [0.01, 1])
    changed_control = state.copy()
    np.testing.assert_array_equal(
        experiment.control_morphology_metrics(state, changed_control, 0.1), [0, 1]
    )
    state[3, :2] = 0
    assert experiment.control_morphology_metrics(state, control, 0.1)[1] == 0.5
    control[3] = state[3] = 0
    assert experiment.control_morphology_metrics(state, control, 0.1)[1] == 1
    assert np.isnan(experiment.control_morphology_metrics(state, None, 0.1)).all()


def test_control_comparison_backfills_all_steps_without_changing_saved_run(tmp_path, monkeypatch):
    cfg = replace(
        experiment.Settings(),
        perturb_step=30,
        reference_start=8,
        horizon=70,
        pca_components=3,
        snapshot_replicates=1,
    )
    target, mature = torch.ones(1, 4, 4, 4), torch.ones(1, 16, 4, 4)
    regime = {
        "target_mse_max": 10.0,
        "alpha_iou_min": 0.0,
        "area_ratio_min": 0.0,
        "area_ratio_max": 2.0,
    }
    folder = tmp_path / "butterfly" / "default"
    folder.mkdir(parents=True)
    checkpoint, target_path = tmp_path / "model.pt", tmp_path / "butterfly.png"
    checkpoint.write_bytes(b"synthetic checkpoint")
    target_path.write_bytes(b"synthetic target")
    experiment.save_json(
        folder / "manifest.json",
        {
            "settings": experiment.asdict(cfg),
            "device": "cpu",
            "python": experiment.platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "target_path": str(target_path),
            "target_hash": experiment.file_sha256(target_path),
            "checkpoint_paths": {system: str(checkpoint) for system in experiment.SYSTEMS},
            "checkpoint_hashes": {
                system: experiment.file_sha256(checkpoint) for system in experiment.SYSTEMS
            },
        },
    )
    expected = {}
    for system in experiment.SYSTEMS:
        payload, metadata = experiment.simulate(
            StochasticRule(),
            target,
            mature,
            system,
            0,
            regime,
            cfg,
            synthetic_erasure_plan(morphology="butterfly"),
        )
        expected[system] = payload.pop("control_morphology")
        replica = folder / system / "replicate_000"
        replica.mkdir(parents=True)
        experiment.save_npz(replica / "trajectories.npz", **payload)
        experiment.save_json(replica / "metadata.json", metadata)
    output = folder / "analysis"
    output.mkdir()
    for name in ("morphological_stability_recovery.png", "hidden_divergence_from_control.png"):
        (output / name).write_bytes(b"original figure")
    originals = {path: experiment.file_sha256(path) for path in folder.rglob("*") if path.is_file()}
    monkeypatch.setattr(experiment, "load_distill_target", lambda *a, **kw: target)
    monkeypatch.setattr(experiment, "load_growing_checkpoint", lambda *a, **kw: StochasticRule())

    def unexpected_fit(*args):
        pytest.fail("Control comparison must not fit PCA or rerun calibration/development")

    for name in ("fit_local_pca", "calibrate", "develop"):
        monkeypatch.setattr(experiment, name, unexpected_fit)
    args = experiment.build_parser().parse_args(
        ["compare-control", "--morphologies", "butterfly", "--output-root", str(tmp_path)]
    )
    args.handler(args)
    for system in experiment.SYSTEMS:
        replica = folder / system / "replicate_000"
        actual = experiment.load_control_comparison(replica, experiment.load_payload(replica))
        np.testing.assert_array_equal(actual, expected[system])
    assert (output / "morphological_deviation_from_control.png").is_file()
    assert json.loads((output / "control_comparison_settings.json").read_text())["steps"] == [0, 70]
    assert all(experiment.file_sha256(path) == value for path, value in originals.items())
    monkeypatch.setattr(experiment, "replay_snapshots", unexpected_fit)
    args.handler(args)
    assert all(experiment.file_sha256(path) == value for path, value in originals.items())
    archive_path = folder / "persistent" / "replicate_000" / "trajectories.npz"
    archive_path.write_bytes(archive_path.read_bytes() + b"changed source")
    with pytest.raises(ValueError, match="does not match the saved run"):
        args.handler(args)


@pytest.mark.parametrize("bad_step", (2, 32))
def test_control_comparison_checks_replay_at_steps_without_snapshots(bad_step):
    cfg = replace(
        experiment.Settings(),
        perturb_step=30,
        reference_start=8,
        horizon=70,
        pca_components=3,
        snapshot_replicates=1,
    )
    target, mature = torch.ones(1, 4, 4, 4), torch.ones(1, 16, 4, 4)
    regime = {
        "target_mse_max": 10.0,
        "alpha_iou_min": 0.0,
        "area_ratio_min": 0.0,
        "area_ratio_max": 2.0,
    }
    plan = synthetic_erasure_plan()
    payload, _ = experiment.simulate(
        StochasticRule(), target, mature, "persistent", 0, regime, cfg, plan
    )
    payload["metrics"][0, bad_step, experiment.METRIC_INDEX["target_mse"]] += 1
    with pytest.raises(ValueError, match=f"t={bad_step} differs from saved metrics"):
        experiment.replay_snapshots(
            StochasticRule(),
            target,
            mature,
            "persistent",
            0,
            cfg,
            payload["pre_pulse_state"],
            payload["metrics"],
            payload["branch_names"],
            plan,
            control_metrics=np.full_like(payload["control_morphology"], np.nan),
        )


@pytest.mark.parametrize("system", experiment.SYSTEMS)
def test_snapshot_replay_matches_snapshots_recorded_during_run(system):
    cfg = replace(
        experiment.Settings(),
        perturb_step=30,
        reference_start=8,
        horizon=70,
        pca_components=3,
        snapshot_replicates=1,
    )
    target = torch.ones(1, 4, 4, 4)
    mature = torch.ones(1, 16, 4, 4)
    regime = {
        "target_mse_max": 10.0,
        "alpha_iou_min": 0.0,
        "area_ratio_min": 0.0,
        "area_ratio_max": 2.0,
    }
    plan = synthetic_erasure_plan()
    payload, _ = experiment.simulate(StochasticRule(), target, mature, system, 0, regime, cfg, plan)
    steps, snapshots = experiment.replay_snapshots(
        StochasticRule(),
        target,
        torch.as_tensor(payload["mature_state"]).unsqueeze(0),
        system,
        0,
        cfg,
        payload["pre_pulse_state"],
        payload["metrics"],
        payload["branch_names"],
        plan,
    )
    assert np.array_equal(steps, payload["snapshot_steps"])
    np.testing.assert_array_equal(snapshots, payload["snapshots"])


@pytest.mark.parametrize("legacy", (False, True))
def test_snapshots_command_fills_missing_image_without_changing_trajectory(
    tmp_path, monkeypatch, legacy
):
    cfg = replace(
        experiment.Settings(),
        perturb_step=30,
        reference_start=8,
        horizon=70,
        pca_components=3,
        snapshot_replicates=0,
    )
    target = torch.ones(1, 4, 4, 4)
    mature = torch.ones(1, 16, 4, 4)
    regime = {
        "target_mse_max": 10.0,
        "alpha_iou_min": 0.0,
        "area_ratio_min": 0.0,
        "area_ratio_max": 2.0,
    }
    payload, metadata = experiment.simulate(
        StochasticRule(),
        target,
        mature,
        "growing_restore",
        3,
        regime,
        cfg,
        synthetic_erasure_plan(),
    )
    folder = tmp_path / "gecko" / "default"
    replica = folder / "growing_restore" / "replicate_003"
    replica.mkdir(parents=True)
    checkpoint = tmp_path / "model.pt"
    target_path = tmp_path / "gecko.png"
    checkpoint.write_bytes(b"test checkpoint")
    target_path.write_bytes(b"test target")
    experiment.save_json(
        folder / "manifest.json",
        {
            "settings": experiment.asdict(cfg),
            "device": "cpu",
            "checkpoint_paths": {"growing_restore": str(checkpoint)},
            "checkpoint_hashes": {"growing_restore": experiment.file_sha256(checkpoint)},
            "target_path": str(target_path),
            "target_hash": experiment.file_sha256(target_path),
        },
    )
    experiment.save_json(replica / "metadata.json", metadata)
    archive = replica / "trajectories.npz"
    recorded = dict(payload)
    if legacy:
        recorded.pop("control_morphology")
        recorded["branch_names"] = np.insert(
            payload["branch_names"].astype("U32"), 1, "independent_control"
        )
        for key in ("metrics", "hidden_scores", "divergence_step", "snapshots", "in_target"):
            recorded[key] = np.insert(payload[key], 1, payload[key][0], axis=0)
    experiment.save_npz(archive, **recorded)
    original_hash = experiment.file_sha256(archive)
    monkeypatch.setattr(experiment, "load_growing_checkpoint", lambda *a, **kw: StochasticRule())
    monkeypatch.setattr(experiment, "load_distill_target", lambda *a, **kw: target)
    args = experiment.build_parser().parse_args(
        [
            "snapshots",
            "--morphologies",
            "gecko",
            "--systems",
            "growing_restore",
            "--output-root",
            str(tmp_path),
        ]
    )
    args.handler(args)
    assert (replica / "perturbation_snapshots.png").is_file()
    assert (replica / "snapshot_replay.npz").is_file()
    assert experiment.file_sha256(archive) == original_hash
    with np.load(replica / "snapshot_replay.npz") as replay:
        np.testing.assert_array_equal(replay["branch_names"], payload["branch_names"])
        if legacy:
            cached_steps = replay["snapshot_steps"]
            cached_snapshots = np.insert(replay["snapshots"], 1, replay["snapshots"][0], axis=0)
    if legacy:
        experiment.save_npz(
            replica / "snapshot_replay.npz",
            snapshot_steps=cached_steps,
            snapshots=cached_snapshots,
        )
    image = replica / "perturbation_snapshots.png"
    image.write_bytes(b"old snapshot figure")
    args.overwrite = True

    def unexpected_replay(*args, **kwargs):
        pytest.fail("Existing snapshots must be reused without simulation")

    monkeypatch.setattr(experiment, "replay_snapshots", unexpected_replay)
    args.handler(args)
    assert image.read_bytes().startswith(b"\x89PNG")
    assert experiment.file_sha256(archive) == original_hash


def test_recovery_waits_for_delayed_visible_damage_before_counting_return():
    cfg = replace(experiment.Settings(), perturb_step=30, reference_start=8, horizon=270)
    payload, metadata = synthetic_recovery_payload(cfg)
    payload["in_target"][1, 35:45] = False
    rows = experiment.recovery_rows(payload, metadata, cfg)
    assert rows[0]["morphology_never_left_target"]
    assert rows[0]["first_morphology_recovery_delay"] == ""
    assert not rows[0]["morphology_recovered_after_exit"]
    assert rows[1]["first_target_exit_delay"] == 5
    assert rows[1]["first_morphology_recovery_delay"] == 15
    assert rows[1]["morphology_recovered_after_exit"]
    assert not rows[1]["morphology_never_left_target"]


def test_save_npz_rejects_low_disk_space_without_changing_archive(tmp_path, monkeypatch):
    path = tmp_path / "trajectories.npz"
    path.write_bytes(b"existing archive")
    monkeypatch.setattr(experiment.shutil, "disk_usage", lambda _: Namespace(free=0))
    with pytest.raises(OSError, match="Insufficient disk space") as caught:
        experiment.save_npz(path, values=np.zeros(4))
    assert caught.value.errno == errno.ENOSPC
    assert path.read_bytes() == b"existing archive"
    assert not path.with_suffix(".tmp.npz").exists()


def test_save_npz_cleans_partial_write_and_preserves_archive(tmp_path, monkeypatch):
    path = tmp_path / "trajectories.npz"
    path.write_bytes(b"existing archive")
    monkeypatch.setattr(experiment.shutil, "disk_usage", lambda _: Namespace(free=1024**3))

    def interrupted_save(temporary, **payload):
        temporary.write_bytes(b"partial write")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(experiment.np, "savez_compressed", interrupted_save)
    with pytest.raises(OSError, match="unsaved replica must be repeated") as caught:
        experiment.save_npz(path, values=np.zeros(4))
    assert caught.value.errno == errno.ENOSPC
    assert path.read_bytes() == b"existing archive"
    assert not path.with_suffix(".tmp.npz").exists()


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
    monkeypatch.setattr(experiment, "primitive_erasure_plan", lambda *a: synthetic_erasure_plan())
    monkeypatch.setattr(experiment, "plot_erasure_plan", lambda *a: None)

    def fake_simulate(model, target, mature, system, replicate, regime, cfg, erasure_plan):
        calls.append((system, replicate))
        return {"synthetic": np.zeros(1)}, {"status": "complete"}

    monkeypatch.setattr(experiment, "simulate", fake_simulate)
    experiment.run(args)
    manifest_path = Path(args.output_root) / "gecko" / "default" / "manifest.json"
    recorded = json.loads(manifest_path.read_text())
    recorded["implementation_sha256"] = "previous script with different plotting code"
    recorded["provenance"] = {"source": "archived run", "verified": True}
    experiment.save_json(manifest_path, recorded)
    experiment.run(args)
    assert calls == [("persistent", 0), ("growing_restore", 0)]
    assert json.loads(manifest_path.read_text()) == recorded

    args.seed += 1
    with pytest.raises(ValueError, match=r"changed \(settings\)"):
        experiment.run(args)
    args.seed -= 1
    recorded["protocol"] = "different simulation protocol"
    experiment.save_json(manifest_path, recorded)
    with pytest.raises(ValueError, match=r"changed \(protocol\)"):
        experiment.run(args)
    recorded["protocol"] = "persistent_vs_mature_restore_primitive_erasure_v3"
    experiment.save_json(manifest_path, recorded)

    original_region = recorded["erasure_regions"]["erase_leg"]["removed_target_pixels"]
    recorded["erasure_regions"]["erase_leg"]["removed_target_pixels"] += 1
    experiment.save_json(manifest_path, recorded)
    with pytest.raises(ValueError, match=r"changed \(erasure_regions\)"):
        experiment.run(args)
    recorded["erasure_regions"]["erase_leg"]["removed_target_pixels"] = original_region
    experiment.save_json(manifest_path, recorded)

    paths[0].write_bytes(b"changed synthetic checkpoint")
    with pytest.raises(ValueError, match=r"changed \(checkpoint_hashes\)"):
        experiment.run(args)


def test_constant_reference_has_finite_pca():
    local = experiment.fit_local_pca(np.ones((40, 12, 3, 3)), 3, 2)
    scores, residual = local.project(np.ones((12, 3, 3)))
    assert np.allclose(scores, 0)
    assert residual == 0
    assert local.speed_ratio == 1


def synthetic_recovery_payload(cfg):
    branches = experiment.condition_names(synthetic_erasure_plan())
    shape = (len(branches), cfg.horizon + 1)
    metrics = np.zeros((*shape, len(experiment.METRICS)))
    metrics[..., experiment.METRIC_INDEX["alpha_iou"]] = 1
    metrics[..., experiment.METRIC_INDEX["area_ratio"]] = 1
    payload = {
        "branch_names": branches,
        "metrics": metrics,
        "in_target": np.ones(shape, bool),
        "divergence_step": np.full(len(branches), -1),
    }
    metadata = {
        "local_pca": {
            "hidden": {
                "approximately_stationary_heuristic": True,
                "residual_limit": 1.0,
                "explained_variance": 1.0,
                "reached_95_percent": True,
            }
        }
    }
    return payload, metadata


def test_morphology_and_paired_return_are_distinct_without_regime_classification():
    cfg = replace(experiment.Settings(), perturb_step=30, reference_start=8, horizon=270)
    payload, metadata = synthetic_recovery_payload(cfg)
    metrics = payload["metrics"]
    metrics[1:, 30, experiment.METRIC_INDEX["hidden_pair_rms"]] = 1
    metrics[1:, 31:, experiment.METRIC_INDEX["hidden_pair_rms"]] = 0.01
    metrics[1, 31:, experiment.METRIC_INDEX["hidden_pair_rms"]] = 0.4
    metrics[2, 31:, experiment.METRIC_INDEX["hidden_pca_residual"]] = 2
    payload["in_target"][2, 31:] = False
    metrics[3, 100:] = np.nan
    payload["divergence_step"][3] = 100
    payload["in_target"][3, 100:] = False
    rows = experiment.recovery_rows(payload, metadata, cfg)
    assert rows[1]["morphology_retained_or_recovered"]
    assert not rows[1]["paired_hidden_return"]
    assert not rows[2]["morphology_retained_or_recovered"]
    assert rows[2]["paired_hidden_return"]
    assert all("hidden_reference_compatible" not in row for row in rows)
    assert not rows[0]["paired_hidden_return_eligible"]
    assert rows[2]["first_paired_hidden_return_delay"] == 1
    assert not rows[3]["morphology_retained_or_recovered"]
    assert not rows[3]["paired_hidden_return"]
    summaries, comparisons = experiment.aggregate(
        [
            {"system": system, "replicate": 0, **row}
            for system in experiment.SYSTEMS
            for row in rows
        ],
        20,
        5,
    )
    failure = next(
        r
        for r in summaries
        if r["system"] == "persistent"
        and r["branch"] == "erase_leg"
        and r["endpoint"] == "paired_hidden_return"
    )
    assert failure["n_eligible"] == failure["n_numerical_failures"] == 1
    assert failure["successes"] == 0
    assert all(row["endpoint"] != "hidden_reference_compatible" for row in summaries + comparisons)
    assert all(row["growing_minus_persistent_fraction"] == 0 for row in comparisons)
    metadata["local_pca"]["hidden"]["approximately_stationary_heuristic"] = False
    rows = experiment.recovery_rows(payload, metadata, cfg)
    assert not any(row["reference_stationary_heuristic"] for row in rows)
    assert rows[2]["paired_hidden_return"]


def test_morphology_recovery_does_not_depend_on_a_finite_paired_control():
    cfg = replace(experiment.Settings(), perturb_step=30, reference_start=8, horizon=270)
    payload, metadata = synthetic_recovery_payload(cfg)
    payload["metrics"][0, 100:] = np.nan
    payload["divergence_step"][0] = 100
    payload["metrics"][1:, 100:, experiment.METRIC_INDEX["hidden_pair_rms"]] = np.nan
    payload["in_target"][0, 100:] = False
    rows = experiment.recovery_rows(payload, metadata, cfg)
    assert rows[2]["morphology_retained_or_recovered"]
    assert not rows[2]["paired_hidden_return_eligible"]


def test_measure_and_reference_use_only_hidden_channels():
    cfg = experiment.Settings()
    rng = np.random.default_rng(7)
    history = rng.normal(size=(40, 12, 3, 3))
    local = experiment.fit_local_pca(history, 3, 4)
    state = np.ones((16, 3, 3))
    state[4:] = history[0]
    target = np.ones((4, 3, 3))
    first, scores = experiment.measure(state, state, target, cfg, local)
    altered = state.copy()
    altered[:4] = 100
    second, second_scores = experiment.measure(altered, state, target, cfg, local)
    np.testing.assert_array_equal(scores, second_scores)
    for name in ("hidden_pair_rms", "hidden_pca_residual"):
        assert first[experiment.METRIC_INDEX[name]] == second[experiment.METRIC_INDEX[name]]
    assert (
        second[experiment.METRIC_INDEX["target_mse"]] > first[experiment.METRIC_INDEX["target_mse"]]
    )


def test_legacy_metric_columns_are_read_by_name_and_never_overwritten(tmp_path):
    names = ("hidden_cloud_distance", "obsolete_metric", *reversed(experiment.METRICS))
    branches = ("control", "independent_control", "noise_0p1")
    values = np.broadcast_to(np.arange(len(names)), (3, 3, len(names))).copy()
    values[1] = -99
    scores = np.arange(27).reshape(3, 3, 3)
    snapshots = np.arange(48).reshape(3, 1, 4, 2, 2)
    path = tmp_path / "trajectories.npz"
    experiment.save_npz(
        path,
        metric_names=np.asarray(names),
        metrics=values,
        branch_names=np.asarray(branches),
        hidden_scores=scores,
        hidden_variance_ratio=np.asarray([0.5, 0.3, 0.2]),
        snapshots=snapshots,
        divergence_step=np.asarray([-1, 22, 15]),
        rgba_scores=np.ones((2, 3, 3)),
        channel_metrics=np.ones((2, 3, 2, 16)),
        hidden_reference_scores=np.ones((2, 3)),
        in_reference=np.ones((2, 3), dtype=bool),
    )
    before = experiment.file_sha256(path)
    payload = experiment.load_payload(tmp_path)
    assert tuple(payload["metric_names"]) == experiment.METRICS
    assert payload["branch_names"].tolist() == ["control", "noise_0p1"]
    assert payload["metrics"].shape == (2, 3, len(experiment.METRICS))
    np.testing.assert_array_equal(payload["hidden_scores"], scores[[0, 2]])
    np.testing.assert_array_equal(payload["snapshots"], snapshots[[0, 2]])
    np.testing.assert_array_equal(payload["divergence_step"], [-1, 15])
    np.testing.assert_array_equal(payload["hidden_variance_ratio"], [0.5, 0.3, 0.2])
    np.testing.assert_array_equal(
        payload["metrics"][0, 0], [names.index(name) for name in experiment.METRICS]
    )
    assert "rgba_scores" not in payload and "channel_metrics" not in payload
    assert "hidden_reference_scores" not in payload and "in_reference" not in payload
    assert experiment.file_sha256(path) == before


def test_paired_statistics_use_only_matching_eligible_seeds_and_count_failures():
    cfg = replace(experiment.Settings(), perturb_step=30, reference_start=8, horizon=270)
    payload, metadata = synthetic_recovery_payload(cfg)
    base = experiment.recovery_rows(payload, metadata, cfg)[2]
    rows = []
    for system in experiment.SYSTEMS:
        for replicate in range(3):
            rows.append(
                {
                    **base,
                    "system": system,
                    "replicate": replicate,
                    "morphology_retained_or_recovered": system == "persistent",
                    "in_target_before_pulse": not (system == "growing_restore" and replicate == 2),
                }
            )
    summaries, comparisons = experiment.aggregate(rows, 100, 8)
    comparison = next(
        row for row in comparisons if row["endpoint"] == "morphology_retained_or_recovered"
    )
    assert comparison["n_matched_evaluation_seeds"] == 2
    assert comparison["growing_minus_persistent_fraction"] == -1
    assert comparison["paired_bootstrap_ci95_high"] == -1
    assert comparison["mcnemar_exact_p_value"] == 0.5
    growing = next(
        row
        for row in summaries
        if row["system"] == "growing_restore"
        and row["endpoint"] == "morphology_retained_or_recovered"
    )
    assert growing["n_recorded"] == 3 and growing["n_eligible"] == 2
