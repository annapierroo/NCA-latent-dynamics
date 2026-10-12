from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from latent_nca.artifacts import file_sha256, write_csv
from latent_nca.data.device import get_device, seed_everything
from latent_nca.data.images import load_distill_target
from latent_nca.experiment_1 import load_experiment_config, run_experiment
from latent_nca.experiment_1.channel_dynamics import render_clear_channel_dynamics
from latent_nca.experiment_1.conditions import (
    ConditionSpec,
    apply_post_step_intervention,
    build_conditions,
    resolve_intervention_reference,
)
from latent_nca.experiment_1.metrics import (
    MORPHOLOGY_METRIC_NAMES,
    MorphologyEvaluator,
    TargetRegime,
)
from latent_nca.experiment_1.model import load_model
from latent_nca.experiment_1.temporal_attractor import (
    TemporalAttractorSettings,
    run_temporal_attractor_analysis,
    select_representative_replicate,
)
from latent_nca.experiment_1.visualization import (
    plot_core_morphology_metrics,
    plot_core_morphology_metrics_robust,
)
from latent_nca.experiment_1.visualization import (
    plot_snapshots as render_snapshot_grid,
)

plt.switch_backend("Agg")

MORPHOLOGIES = ("gecko", "butterfly", "spider", "planaria")


def run(args: argparse.Namespace) -> None:
    base = load_experiment_config(args.config)
    for morphology in args.morphologies:
        config = replace(
            base,
            checkpoint=str(Path(args.checkpoint_root) / morphology / "model.pt"),
            target=str(Path(args.data_dir) / f"{morphology}.png"),
            output_dir=str(Path(args.output_root) / morphology),
            device=base.device if args.device is None else args.device,
        )
        output_dir = run_experiment(config)
        print(f"Experiment 1 completed for {morphology}: {output_dir}")


def analyze_attractor(args: argparse.Namespace) -> None:
    settings = TemporalAttractorSettings(
        horizon=args.horizon,
        dense_until=args.dense_until,
        record_every=args.record_every,
        crop_size=args.crop_size,
        pca_components=args.pca_components,
        pca_seed=args.pca_seed,
        update_probability=args.update_probability,
    )
    settings.validate()
    folders = (
        [Path(args.results_dir)]
        if args.results_dir
        else [Path(args.results_root) / name for name in args.morphologies]
    )
    selection = "selected_conditions" if args.conditions else "all_conditions"
    region = "full_grid" if args.crop_size == 0 else f"crop{args.crop_size}"
    analysis_name = (
        args.analysis_name or f"temporal_full_state_{selection}_{region}_t0_{args.horizon}"
    )
    if Path(analysis_name).name != analysis_name or analysis_name in {".", ".."}:
        raise ValueError("analysis-name must be a single folder name")
    jobs = []
    for results_dir in folders:
        config = load_experiment_config(results_dir / "resolved_config.json")
        available = tuple(condition.name for condition in build_conditions(config.interventions))
        conditions = tuple(args.conditions) if args.conditions else available
        if len(set(conditions)) != len(conditions) or set(conditions) - set(available):
            raise ValueError(f"invalid or repeated conditions for {results_dir}: {conditions}")
        if "baseline" not in conditions:
            raise ValueError("conditions must include baseline as the shared reference")
        with (results_dir / "per_run_summary.csv").open(newline="") as handle:
            recorded = sorted(
                {
                    int(row["replicate"])
                    for row in csv.DictReader(handle)
                    if row["condition"] == "baseline"
                }
            )
        replicas = (
            recorded
            if args.all_replicates
            else (
                [args.replicate]
                if args.replicate is not None
                else [select_representative_replicate(results_dir)]
            )
        )
        if not replicas or set(replicas) - set(recorded):
            raise ValueError(f"requested replicas are absent from {results_dir}: {replicas}")
        for replicate in replicas:
            output_dir = (
                Path(args.output_dir)
                if args.output_dir
                else (
                    results_dir
                    / "attractor_analysis"
                    / analysis_name
                    / f"replicate_{replicate:03d}"
                )
            )
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(
                    f"output directory is not empty: {output_dir}; use a new --analysis-name"
                )
            jobs.append((results_dir, output_dir, replicate, conditions))
    if args.output_dir and len(jobs) != 1:
        raise ValueError(
            "output-dir requires a single morphology and replica; use analysis-name for a batch"
        )
    for results_dir, output_dir, replicate, conditions in jobs:
        print(
            f"Analyzing {results_dir.name}, replica {replicate}: {', '.join(conditions)}",
            flush=True,
        )
        result = run_temporal_attractor_analysis(
            results_dir,
            output_dir,
            replicate,
            conditions,
            settings,
            device_name=args.device,
        )
        print(f"Temporal attractor analysis completed. Results: {result}")


def plot_channels(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root)
    for morphology in args.morphologies:
        results_dir = results_root / morphology
        result = render_clear_channel_dynamics(
            results_dir,
            results_dir / args.output_name,
            early_stop=args.early_stop,
        )
        print(f"Channel-dynamics plots completed for {morphology}: {result}")


def _load_core_metrics(
    results_dir: Path,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    tuple[ConditionSpec, ...],
    TargetRegime,
]:
    config = load_experiment_config(results_dir / "resolved_config.json")
    available = {condition.name: condition for condition in build_conditions(config.interventions)}
    with np.load(results_dir / "trajectory_metrics.npz") as archive:
        condition_names = tuple(str(name) for name in archive["condition_names"])
        missing = [name for name in condition_names if name not in available]
        if missing:
            raise ValueError(f"resolved configuration is missing conditions: {missing}")
        metrics = {
            condition: {
                metric: np.asarray(archive[f"{condition}__{metric}"])
                for metric in MORPHOLOGY_METRIC_NAMES
                if f"{condition}__{metric}" in archive
            }
            for condition in condition_names
        }
    regime = TargetRegime(**json.loads((results_dir / "target_regime.json").read_text()))
    return metrics, tuple(available[name] for name in condition_names), regime


def plot_core(args: argparse.Namespace) -> None:
    robust = args.summary == "median-quantiles"
    plotter = plot_core_morphology_metrics_robust if robust else plot_core_morphology_metrics
    output_name = args.output_name or (
        "core_morphology_metrics_robust.png" if robust else "core_morphology_metrics.png"
    )
    results_root = Path(args.results_root)
    for morphology in args.morphologies:
        results_dir = results_root / morphology
        metrics, conditions, regime = _load_core_metrics(results_dir)
        output = results_dir / output_name
        plotter(output, metrics, conditions, regime)
        print(f"Updated {morphology}: {output}")


def _display_rgb(state: torch.Tensor) -> np.ndarray:
    rgba = state[:4].detach().cpu().permute(1, 2, 0).numpy()
    alpha = np.clip(rgba[..., 3:4], 0.0, 1.0)
    return np.clip(rgba[..., :3] + 1.0 - alpha, 0.0, 1.0)


def _condition_labels(results_dir: Path) -> dict[str, str]:
    path = results_dir / "conditions.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["condition"]: row["label"] for row in csv.DictReader(handle)}


def plot_snapshots(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    payload = torch.load(results_dir / "snapshots.pt", map_location="cpu", weights_only=True)
    snapshots = payload["intervention"]
    missing = [name for name in args.conditions if name not in snapshots]
    if missing:
        available = ", ".join(sorted(snapshots))
        raise ValueError(f"unknown conditions: {missing}; available: {available}")

    steps = args.steps or sorted(snapshots[args.conditions[0]])
    for name in args.conditions:
        missing_steps = [step for step in steps if step not in snapshots[name]]
        if missing_steps:
            raise ValueError(f"condition {name} has no snapshots at steps {missing_steps}")

    labels = _condition_labels(results_dir)
    figure, axes = plt.subplots(
        len(args.conditions),
        len(steps),
        figsize=(2.4 * len(steps), 2.2 * len(args.conditions)),
        squeeze=False,
    )
    for row, name in enumerate(args.conditions):
        for column, step in enumerate(steps):
            axis = axes[row, column]
            state = snapshots[name][step]
            if state is None:
                axis.text(
                    0.5,
                    0.5,
                    "Numerically\ndiverged",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            else:
                axis.imshow(_display_rgb(state))
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"t={step}")
            if column == 0:
                axis.set_ylabel(labels.get(name, name))

    output = Path(args.output) if args.output else results_dir / "selected_snapshots.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


def _snapshot_replica_scores(metrics: dict) -> np.ndarray:
    """Equal-weight, IQR-scaled distance to temporal medians across all conditions."""
    scores = np.zeros(next(iter(metrics.values()))["alpha_iou"].shape[0])
    for condition in metrics.values():
        for name in ("target_region_mse", "alpha_iou", "area_ratio"):
            values = np.asarray(condition[name], dtype=np.float64)
            finite = np.isfinite(values)
            values = np.where(finite, values, np.nan)
            if name != "alpha_iou":
                values = np.log10(np.maximum(values, 1e-12))
            # All-missing times carry no ranking information. Missing individual values
            # receive the maximum distance, so divergence cannot improve a replica's rank.
            values = values[:, finite.any(axis=0)]
            if not values.shape[1]:
                continue
            q25, median, q75 = np.nanquantile(values, (0.25, 0.5, 0.75), axis=0)
            scale = np.maximum(q75 - q25, 0.05 if name == "alpha_iou" else 0.1)
            distance = np.abs(values - median) / scale
            scores += np.where(np.isfinite(distance), np.minimum(distance, 10), 10).mean(axis=1)
    return scores / (3 * len(metrics))


class _SnapshotReplayMismatch(ValueError):
    def __init__(self, condition: str, step: int, differences: list[dict]):
        self.differences = differences
        super().__init__(
            f"Snapshot replay differs from saved metrics: {condition}, t={step}. "
            "Check the original device, checkpoint and numerical environment."
        )


@torch.no_grad()
def _replay_snapshot_frames(
    model, target, mature, reference, condition, config, seed, steps, saved
):
    seed_everything(seed)
    state = mature.clone()
    evaluator = MorphologyEvaluator(target, config.model.alpha_threshold)
    missing = np.flatnonzero(~np.isfinite(saved["target_mse"]))
    stop = min(max(steps), int(missing[0]) - 1) if missing.size else max(steps)
    frames = {step: None for step in steps}
    if not any(step <= stop for step in steps):
        return frames
    for step in range(stop + 1):
        if step in frames:
            values = evaluator.evaluate(state)
            names = tuple(saved)
            replayed = np.array([values[name] for name in names], dtype=np.float64)
            expected = np.array([saved[name][step] for name in names], dtype=np.float64)
            matches = np.isclose(replayed, expected, rtol=1e-4, atol=1e-6)
            if not matches.all():
                raise _SnapshotReplayMismatch(
                    condition.name,
                    step,
                    [
                        {
                            "step": step,
                            "metric": names[index],
                            "saved_value": expected[index],
                            "replayed_value": replayed[index],
                            "absolute_difference": abs(replayed[index] - expected[index]),
                            "allowed_difference": 1e-6 + 1e-4 * abs(expected[index]),
                        }
                        for index in np.flatnonzero(~matches)
                    ],
                )
            frames[step] = state[0, :4].detach().cpu().clone()
        if step < stop:
            state = model(state, update_probability=config.model.update_probability)
            state = apply_post_step_intervention(state, reference, condition, step + 1)
    return frames


def _plot_final_gallery(
    path: Path, frames: dict, title: str, step: int, *, unverified: set[int] | None = None
) -> None:
    columns = min(6, len(frames))
    rows = math.ceil(len(frames) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, (replicate, state) in zip(axes.flat, frames.items()):
        if state is None:
            axis.text(
                0.5,
                0.5,
                "Replay not\nverified"
                if unverified and replicate in unverified
                else "Numerically\ndiverged",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=8,
            )
        else:
            axis.imshow(_display_rgb(state))
        axis.set_title(f"Replica {replicate:03d}", fontsize=9)
    figure.suptitle(f"{title} | t={step} | n={len(frames)}")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_population_snapshots(args: argparse.Namespace) -> None:
    steps = tuple(sorted(set(args.snapshot_steps)))
    for morphology in args.morphologies:
        folder = Path(args.results_root) / morphology
        config = load_experiment_config(folder / "resolved_config.json")
        if not steps or steps[0] < 0 or steps[-1] > config.evaluation.horizon:
            raise ValueError("Snapshot steps must fit the saved experiment horizon")
        metrics, conditions, _ = _load_core_metrics(folder)
        with np.load(folder / "trajectory_metrics.npz", allow_pickle=False) as archive:
            ids = archive["replicate_ids"].astype(int).tolist()
        scores = _snapshot_replica_scores(metrics)
        selected = int(np.argmin(scores))
        indices = range(len(ids)) if args.gallery else (selected,)
        metadata = json.loads((folder / "metadata.json").read_text())
        for name in ("checkpoint", "target"):
            if file_sha256(getattr(config, name)) != metadata[name + "_sha256"]:
                raise ValueError(f"{name} changed since the original experiment")
        device = get_device(args.device or metadata["device"])
        if str(device) != metadata["device"]:
            raise ValueError("Snapshot replay must use the original experiment device")
        states = torch.load(folder / "mature_states.pt", map_location="cpu", weights_only=True)
        original = torch.load(folder / "snapshots.pt", map_location="cpu", weights_only=True)
        output = folder
        output.mkdir(exist_ok=True)
        if not args.gallery:
            write_csv(
                output / "representative_selection.csv",
                [
                    {
                        "replicate": replicate,
                        "distance_score": float(score),
                        "selected": index == selected,
                    }
                    for index, (replicate, score) in enumerate(zip(ids, scores, strict=True))
                ],
            )
        target = model = None
        snapshots = {}
        for condition in conditions:
            gallery_path = output / f"final_t{steps[-1]}__{condition.name}.png"
            if args.gallery and gallery_path.exists():
                print(f"Already plotted: {gallery_path}", flush=True)
                continue
            gallery = {}
            unverified = set()
            replay_mismatches = []
            for index in indices:
                replicate = ids[index]
                print(f"Snapshots: {morphology}/{condition.name}, replica {replicate}", flush=True)
                existing = original["intervention"].get(condition.name, {})
                if index == 0 and all(step in existing for step in steps):
                    frames = {step: existing[step] for step in steps}
                else:
                    if model is None:
                        model = load_model(config, device)
                        target = load_distill_target(
                            config.target,
                            config.model.target_size,
                            config.model.target_padding,
                            device,
                        )
                    record = states[replicate]
                    mature = record["state"].unsqueeze(0).to(device)
                    reference = resolve_intervention_reference(
                        mature,
                        condition,
                        {
                            step: value.unsqueeze(0).to(device)
                            for step, value in record.get("development_states", {}).items()
                        },
                    )
                    seed = (
                        config.evaluation.seed_start
                        + config.evaluation.continuation_seed_offset
                        + replicate
                    )
                    try:
                        with threadpool_limits(limits=1):
                            frames = _replay_snapshot_frames(
                                model,
                                target,
                                mature,
                                reference,
                                condition,
                                config,
                                seed,
                                steps,
                                {
                                    name: values[index]
                                    for name, values in metrics[condition.name].items()
                                },
                            )
                    except _SnapshotReplayMismatch as error:
                        if not args.gallery:
                            raise
                        gallery[replicate] = None
                        unverified.add(replicate)
                        replay_mismatches.extend(
                            {
                                "morphology": morphology,
                                "condition": condition.name,
                                "replicate": replicate,
                                **difference,
                            }
                            for difference in error.differences
                        )
                        diagnostics = gallery_path.with_suffix(".replay_mismatches.csv")
                        write_csv(diagnostics, replay_mismatches)
                        print(
                            f"Replay not verified: replica {replicate}; see {diagnostics}",
                            flush=True,
                        )
                        continue
                if args.gallery:
                    gallery[replicate] = frames[steps[-1]]
                else:
                    snapshots[condition.name] = frames
            if args.gallery:
                _plot_final_gallery(
                    gallery_path,
                    gallery,
                    f"{morphology}: {condition.label}",
                    steps[-1],
                    unverified=unverified,
                )
                print(f"Saved {gallery_path}", flush=True)
        if not args.gallery:
            path = output / f"representative_replica_{ids[selected]:03d}.png"
            render_snapshot_grid(
                path,
                snapshots,
                conditions,
                steps,
                include_all=True,
                title=f"{morphology}: paired replica {ids[selected]:03d}, nearest median trajectories",
            )
            print(f"Saved {path}", flush=True)


def _add_morphologies(parser: argparse.ArgumentParser, *, restrict: bool = False) -> None:
    options: dict[str, object] = {
        "nargs": "+",
        "default": MORPHOLOGIES,
    }
    if restrict:
        options["choices"] = MORPHOLOGIES
    parser.add_argument("--morphologies", **options)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and analyze all components of Growing NCA Experiment 1."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Run all paired intervention conditions.")
    run_parser.add_argument("--config", default="configs/experiment_1.yaml")
    run_parser.add_argument("--data-dir", default="data")
    run_parser.add_argument("--checkpoint-root", default="checkpoints/growing")
    run_parser.add_argument("--output-root", default="reports/experiment_1")
    _add_morphologies(run_parser, restrict=True)
    run_parser.add_argument("--device")
    run_parser.set_defaults(handler=run)

    attractor_parser = commands.add_parser(
        "attractor",
        help="Generate full-state reference and pairwise temporal PCA trajectories.",
    )
    sources = attractor_parser.add_mutually_exclusive_group()
    sources.add_argument("--results-dir", help="Analyze one morphology directory.")
    sources.add_argument("--results-root", default="reports/experiment_1")
    _add_morphologies(attractor_parser, restrict=True)
    attractor_parser.add_argument("--output-dir")
    replicas = attractor_parser.add_mutually_exclusive_group()
    replicas.add_argument("--replicate", type=int)
    replicas.add_argument(
        "--all-replicates",
        action="store_true",
        help="Analyze every included replica; default: representative control replica.",
    )
    attractor_parser.add_argument(
        "--conditions",
        nargs="+",
        help="Default: every condition in the saved experiment configuration, including freezes.",
    )
    attractor_parser.add_argument("--analysis-name")
    attractor_parser.add_argument("--horizon", type=int, default=10_000)
    attractor_parser.add_argument("--dense-until", type=int, default=250)
    attractor_parser.add_argument("--record-every", type=int, default=10)
    attractor_parser.add_argument(
        "--crop-size",
        type=int,
        default=40,
        help="Central crop side; 0 uses the full grid including external background.",
    )
    attractor_parser.add_argument("--pca-components", type=int, default=12)
    attractor_parser.add_argument("--pca-seed", type=int, default=40_000)
    attractor_parser.add_argument("--update-probability", type=float)
    attractor_parser.add_argument("--device")
    attractor_parser.set_defaults(handler=analyze_attractor)

    channels_parser = commands.add_parser(
        "plot-channels",
        help="Regenerate hidden- and visible-channel dynamics plots.",
    )
    channels_parser.add_argument("--results-root", default="reports/experiment_1")
    _add_morphologies(channels_parser)
    channels_parser.add_argument("--output-name", default="channel_dynamics_clear")
    channels_parser.add_argument("--early-stop", type=int, default=1000)
    channels_parser.set_defaults(handler=plot_channels)

    core_parser = commands.add_parser(
        "plot-core",
        help="Regenerate core morphology metric plots.",
    )
    core_parser.add_argument("--results-root", default="reports/experiment_1")
    _add_morphologies(core_parser)
    core_parser.add_argument(
        "--summary",
        choices=("mean-sd", "median-quantiles"),
        default="mean-sd",
    )
    core_parser.add_argument("--output-name")
    core_parser.set_defaults(handler=plot_core)

    snapshots_parser = commands.add_parser(
        "plot-snapshots",
        help="Plot selected conditions and saved time points.",
    )
    snapshots_parser.add_argument("--results-dir", required=True)
    snapshots_parser.add_argument("--conditions", nargs="+", required=True)
    snapshots_parser.add_argument("--steps", nargs="+", type=int)
    snapshots_parser.add_argument("--output")
    snapshots_parser.set_defaults(handler=plot_snapshots)
    for command, gallery in (("plot-representative", False), ("plot-final-gallery", True)):
        command_parser = commands.add_parser(command, help="Replay paired morphology snapshots.")
        command_parser.add_argument("--results-root", default="reports/experiment_1")
        _add_morphologies(command_parser, restrict=True)
        command_parser.add_argument("--device")
        if not gallery:
            command_parser.add_argument(
                "--steps",
                dest="snapshot_steps",
                nargs="+",
                type=int,
                default=(0, 500, 2000, 5000, 10000),
            )
        command_parser.set_defaults(handler=plot_population_snapshots, gallery=gallery)
        if gallery:
            command_parser.set_defaults(snapshot_steps=(10000,))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
