from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_nca.experiment_1 import load_experiment_config, run_experiment
from latent_nca.experiment_1.channel_dynamics import render_clear_channel_dynamics
from latent_nca.experiment_1.conditions import ConditionSpec, build_conditions
from latent_nca.experiment_1.metrics import MORPHOLOGY_METRIC_NAMES, TargetRegime
from latent_nca.experiment_1.temporal_attractor import (
    TemporalAttractorSettings,
    run_temporal_attractor_analysis,
    select_representative_replicate,
)
from latent_nca.experiment_1.visualization import (
    plot_core_morphology_metrics,
    plot_core_morphology_metrics_robust,
)

plt.switch_backend("Agg")

MORPHOLOGIES = ("gecko", "butterfly", "spider", "planaria")
ATTRACTOR_CONDITIONS = (
    "baseline",
    "restore_beta_0p1",
    "periodic_reset_150",
    "restore_development_step_16_beta_0p1",
    "restore_development_step_48_beta_0p1",
    "restore_development_step_16_period_150",
    "restore_development_step_48_period_150",
)


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
    results_dir = Path(args.results_dir)
    replicate = (
        args.replicate
        if args.replicate is not None
        else select_representative_replicate(results_dir)
    )
    analysis_name = args.analysis_name or (
        f"temporal_full_state_development_restore_t0_{args.horizon}"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else results_dir / "attractor_analysis" / analysis_name / f"replicate_{replicate:03d}"
    )
    settings = TemporalAttractorSettings(
        horizon=args.horizon,
        dense_until=args.dense_until,
        record_every=args.record_every,
        crop_size=args.crop_size,
        pca_components=args.pca_components,
        pca_seed=args.pca_seed,
        update_probability=args.update_probability,
    )
    result = run_temporal_attractor_analysis(
        results_dir,
        output_dir,
        replicate,
        ATTRACTOR_CONDITIONS,
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
    attractor_parser.add_argument("--results-dir", required=True)
    attractor_parser.add_argument("--output-dir")
    attractor_parser.add_argument("--replicate", type=int)
    attractor_parser.add_argument("--analysis-name")
    attractor_parser.add_argument("--horizon", type=int, default=10_000)
    attractor_parser.add_argument("--dense-until", type=int, default=250)
    attractor_parser.add_argument("--record-every", type=int, default=10)
    attractor_parser.add_argument("--crop-size", type=int, default=40)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
