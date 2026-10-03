from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .conditions import ConditionSpec, build_conditions
from .config import load_experiment_config
from .visualization import (
    plot_hidden_channel_dynamics,
    plot_visible_channel_dynamics,
)


def _load_channel_metrics(
    path: Path,
) -> tuple[list[int], tuple[str, ...], dict[str, dict[str, np.ndarray]]]:
    with np.load(path) as archive:
        steps = archive["steps"].astype(np.int64).tolist()
        condition_names = tuple(archive["condition_names"].astype(str).tolist())
        metric_names = tuple(archive["metric_names"].astype(str).tolist())
        metrics = {
            condition: {
                metric: np.asarray(archive[f"{condition}__{metric}"]) for metric in metric_names
            }
            for condition in condition_names
        }
    return steps, condition_names, metrics


def _ordered_conditions(
    results_dir: Path,
    condition_names: tuple[str, ...],
) -> tuple[ConditionSpec, ...]:
    config = load_experiment_config(results_dir / "resolved_config.json")
    available = {item.name: item for item in build_conditions(config.interventions)}
    missing = [name for name in condition_names if name not in available]
    if missing:
        raise ValueError(f"conditions are absent from the resolved experiment: {missing}")
    return tuple(available[name] for name in condition_names)


def _visible_metrics_path(results_dir: Path) -> Path:
    direct = results_dir / "visible_channel_metrics.npz"
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"visible-channel metrics not found: {direct}")


def render_clear_channel_dynamics(
    results_dir: str | Path,
    output_dir: str | Path,
    early_stop: int = 1_000,
) -> Path:
    """Render robust full-horizon and early-window hidden/RGBA heatmaps."""
    if early_stop <= 0:
        raise ValueError("early_stop must be positive")
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_path = results_dir / "hidden_channel_metrics.npz"
    visible_path = _visible_metrics_path(results_dir)
    hidden_steps, hidden_names, hidden_metrics = _load_channel_metrics(hidden_path)
    visible_steps, visible_names, visible_metrics = _load_channel_metrics(visible_path)
    hidden_conditions = _ordered_conditions(results_dir, hidden_names)
    visible_conditions = _ordered_conditions(results_dir, visible_names)

    plot_hidden_channel_dynamics(
        output_dir / "hidden_channel_dynamics_full.png",
        hidden_metrics,
        hidden_conditions,
        hidden_steps,
    )
    plot_hidden_channel_dynamics(
        output_dir / f"hidden_channel_dynamics_t0_{early_stop}.png",
        hidden_metrics,
        hidden_conditions,
        hidden_steps,
        max_step=early_stop,
    )
    plot_visible_channel_dynamics(
        output_dir / "visible_channel_dynamics_full.png",
        visible_metrics,
        visible_conditions,
        visible_steps,
    )
    plot_visible_channel_dynamics(
        output_dir / f"visible_channel_dynamics_t0_{early_stop}.png",
        visible_metrics,
        visible_conditions,
        visible_steps,
        max_step=early_stop,
    )

    metadata = {
        "source_results": str(results_dir.resolve()),
        "hidden_metrics": str(hidden_path.resolve()),
        "visible_metrics": str(visible_path.resolve()),
        "early_stop": early_stop,
        "replicate_aggregation": "median",
        "relative_l2_scale": {
            "transform": "log10",
            "lower_clip": 1e-4,
            "upper_clip": 10.0,
        },
        "step_rms_scale": {
            "transform": "log10",
            "lower_clip": 1e-6,
            "upper_clip": 1e-1,
        },
        "spatial_scope": "Complete NCA substrate, including background.",
        "simulation_replayed": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output_dir


__all__ = ["render_clear_channel_dynamics"]
