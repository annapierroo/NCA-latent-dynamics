from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latent_nca.experiment_1.conditions import build_conditions
from latent_nca.experiment_1.config import load_experiment_config
from latent_nca.experiment_1.metrics import MORPHOLOGY_METRIC_NAMES, TargetRegime
from latent_nca.experiment_1.visualization import (
    plot_core_morphology_metrics,
    plot_core_morphology_metrics_robust,
)

DEFAULT_MORPHOLOGIES = ("gecko", "butterfly", "spider", "planaria")


def _load_metrics(
    results_dir: Path,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    tuple,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate Experiment 1 core morphology plots from saved metrics."
    )
    parser.add_argument(
        "--results-root",
        default="reports/experiment_1",
    )
    parser.add_argument(
        "--morphologies",
        nargs="+",
        default=DEFAULT_MORPHOLOGIES,
    )
    parser.add_argument(
        "--summary",
        choices=("mean-sd", "median-quantiles"),
        default="mean-sd",
    )
    parser.add_argument("--output-name")
    args = parser.parse_args()

    robust = args.summary == "median-quantiles"
    plotter = plot_core_morphology_metrics_robust if robust else plot_core_morphology_metrics
    output_name = args.output_name or (
        "core_morphology_metrics_robust.png" if robust else "core_morphology_metrics.png"
    )
    results_root = Path(args.results_root)
    for morphology in args.morphologies:
        results_dir = results_root / morphology
        metrics, conditions, regime = _load_metrics(results_dir)
        output = results_dir / output_name
        plotter(output, metrics, conditions, regime)
        print(f"Updated {morphology}: {output}")


if __name__ == "__main__":
    main()
