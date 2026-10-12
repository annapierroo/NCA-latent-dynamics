from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latent_nca.artifacts import file_sha256, write_csv
from latent_nca.data.device import get_device, seed_everything
from latent_nca.data.images import load_distill_target
from latent_nca.models import GrowingNCA

from .channel_artifacts import (
    save_hidden_channel_metrics,
    save_visible_channel_metrics,
    summarize_hidden_channels,
    summarize_visible_channels,
)
from .conditions import (
    ConditionSpec,
    apply_post_step_intervention,
    build_conditions,
    resolve_intervention_reference,
)
from .config import ExperimentConfig
from .metrics import (
    HIDDEN_CHANNEL_METRIC_NAMES,
    METRIC_NAMES,
    MORPHOLOGY_METRIC_NAMES,
    VISIBLE_CHANNEL_METRIC_NAMES,
    MorphologyEvaluator,
    TargetRegime,
    hidden_channel_metrics,
    survival_outcome,
    visible_channel_metrics,
)
from .model import load_model
from .statistics import summarize_conditions
from .visualization import (
    plot_core_morphology_metrics,
    plot_development,
    plot_hidden_channel_dynamics,
    plot_snapshots,
    plot_visible_channel_dynamics,
)


@dataclass
class BranchResult:
    metrics: dict[str, np.ndarray]
    in_regime: np.ndarray
    summary: dict[str, object]
    channel_metrics: dict[str, np.ndarray]
    channel_steps: list[int]
    visible_channel_metrics: dict[str, np.ndarray]
    visible_channel_steps: list[int]
    snapshots: dict[int, torch.Tensor | None]


@torch.no_grad()
def _calibrate_regime(
    model: GrowingNCA,
    target: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[TargetRegime, dict[str, np.ndarray]]:
    collected = {name: [] for name in MORPHOLOGY_METRIC_NAMES}
    evaluator = MorphologyEvaluator(target, config.model.alpha_threshold)
    height, width = target.shape[-2:]
    for replicate in range(config.calibration.replicates):
        seed_everything(config.calibration.seed_start + replicate)
        state = model.seed(1, height, width, device)
        for step in range(config.calibration.mature_stop + 1):
            if step >= config.calibration.mature_start:
                metrics = evaluator.evaluate(state)
                for name, values in collected.items():
                    values.append(metrics[name])
            if step < config.calibration.mature_stop:
                state = model(state, update_probability=config.model.update_probability)
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in collected.items()}
    regime = TargetRegime.calibrate(arrays, config.calibration.acceptance_quantile)
    return regime, arrays


@torch.no_grad()
def _develop_to_target(
    model: GrowingNCA,
    target: torch.Tensor,
    regime: TargetRegime,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> tuple[
    torch.Tensor | None,
    int | None,
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
]:
    seed_everything(seed)
    height, width = target.shape[-2:]
    state = model.seed(1, height, width, device)
    evaluator = MorphologyEvaluator(target, config.model.alpha_threshold)
    consecutive = 0
    snapshots: dict[int, torch.Tensor] = {0: state[0].detach().cpu()}
    reference_steps = set(config.interventions.development_reference_steps)
    development_steps = reference_steps | {
        config.calibration.mature_start // 2,
        config.calibration.mature_start,
        config.calibration.mature_stop,
    }
    development_states: dict[int, torch.Tensor] = {}
    for step in range(config.calibration.mature_stop + 1):
        if step in development_steps:
            snapshots[step] = state[0].detach().cpu()
        if step in reference_steps:
            development_states[step] = state.detach().clone()
        if step >= config.calibration.mature_start:
            metrics = evaluator.evaluate(state)
            consecutive = consecutive + 1 if regime.contains(metrics) else 0
            if consecutive >= config.evaluation.attainment_patience:
                snapshots[step] = state[0].detach().cpu()
                return state.detach().clone(), step, snapshots, development_states
        if step < config.calibration.mature_stop:
            state = model(state, update_probability=config.model.update_probability)
    return None, None, snapshots, development_states


@torch.no_grad()
def _run_branch(
    model: GrowingNCA,
    target: torch.Tensor,
    mature_state: torch.Tensor,
    intervention_reference_state: torch.Tensor,
    regime: TargetRegime,
    condition: ConditionSpec,
    config: ExperimentConfig,
    replicate: int,
    continuation_seed: int,
    collect_snapshots: bool,
) -> BranchResult:
    state = mature_state.clone()
    seed_everything(continuation_seed)
    evaluator = MorphologyEvaluator(
        target,
        config.model.alpha_threshold,
        mature_state=mature_state,
        hidden_reference_state=intervention_reference_state,
    )
    records: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    regime_flags: list[bool] = []
    channel_records: dict[str, list[np.ndarray]] = {
        name: [] for name in HIDDEN_CHANNEL_METRIC_NAMES
    }
    channel_steps: list[int] = []
    visible_channel_records: dict[str, list[np.ndarray]] = {
        name: [] for name in VISIBLE_CHANNEL_METRIC_NAMES
    }
    visible_channel_steps: list[int] = []
    snapshots: dict[int, torch.Tensor | None] = {}
    snapshot_steps = set(config.analysis.snapshot_steps)
    previous_state: torch.Tensor | None = None
    divergence_step: int | None = None

    def pad_after_divergence(step: int) -> None:
        remaining_steps = config.evaluation.horizon - step + 1
        for record in records.values():
            record.extend([float("nan")] * remaining_steps)
        regime_flags.extend([False] * remaining_steps)
        hidden_channels = mature_state.shape[1] - 4
        for sampled_step in range(step, config.evaluation.horizon + 1):
            if sampled_step % config.analysis.hidden_channel_record_every:
                continue
            missing = np.full(hidden_channels, np.nan, dtype=np.float64)
            for name in HIDDEN_CHANNEL_METRIC_NAMES:
                channel_records[name].append(missing.copy())
            channel_steps.append(sampled_step)
        for sampled_step in range(step, config.evaluation.horizon + 1):
            if sampled_step % config.analysis.visible_channel_record_every:
                continue
            missing = np.full(4, np.nan, dtype=np.float64)
            for name in VISIBLE_CHANNEL_METRIC_NAMES:
                visible_channel_records[name].append(missing.copy())
            visible_channel_steps.append(sampled_step)
        if collect_snapshots:
            for snapshot_step in snapshot_steps:
                if snapshot_step >= step:
                    snapshots[snapshot_step] = None

    for step in range(config.evaluation.horizon + 1):
        values = evaluator.evaluate(state, previous_state)
        if not all(math.isfinite(value) for value in values.values()):
            divergence_step = step
            pad_after_divergence(step)
            break
        for name in METRIC_NAMES:
            records[name].append(values[name])
        regime_flags.append(regime.contains(values))
        if step % config.analysis.hidden_channel_record_every == 0:
            channel_values = hidden_channel_metrics(state, mature_state, previous_state)
            for name, values in channel_values.items():
                channel_records[name].append(values.detach().cpu().numpy().astype(np.float64))
            channel_steps.append(step)
        if step % config.analysis.visible_channel_record_every == 0:
            channel_values = visible_channel_metrics(state, mature_state, previous_state)
            for name, channel_data in channel_values.items():
                visible_channel_records[name].append(
                    channel_data.detach().cpu().numpy().astype(np.float64)
                )
            visible_channel_steps.append(step)
        if collect_snapshots and step in snapshot_steps:
            snapshots[step] = state[0].detach().cpu()
        if step == config.evaluation.horizon:
            break
        previous_state = state
        state = model(state, update_probability=config.model.update_probability)
        state = apply_post_step_intervention(
            state,
            intervention_reference_state,
            condition,
            step + 1,
        )

    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in records.items()}
    flags = np.asarray(regime_flags, dtype=bool)
    outcome = survival_outcome(
        flags,
        config.evaluation.failure_patience,
        terminal_failure_step=divergence_step,
    )
    stable_stop = outcome.first_failure_step if outcome.event_observed else len(flags)
    stable_stop = max(int(stable_stop), 1)
    summary = {
        "replicate": replicate,
        "condition": condition.name,
        "label": condition.label,
        "family": condition.family,
        "parameter_name": condition.parameter_name or "",
        "parameter": "" if condition.parameter is None else condition.parameter,
        "reference_kind": "development" if condition.reference_step is not None else "mature",
        "reference_step": "" if condition.reference_step is None else condition.reference_step,
        "survival_time": outcome.survival_time,
        "event_observed": outcome.event_observed,
        "first_failure_step": (
            "" if outcome.first_failure_step is None else outcome.first_failure_step
        ),
        "numerical_divergence_step": "" if divergence_step is None else divergence_step,
        "max_stable_hidden_relative_l2": float(arrays["hidden_relative_l2"][:stable_stop].max()),
        "mean_stable_hidden_relative_l2": float(arrays["hidden_relative_l2"][:stable_stop].mean()),
        "initial_hidden_reference_relative_l2": float(arrays["hidden_reference_relative_l2"][0]),
        "mean_stable_hidden_reference_relative_l2": float(
            arrays["hidden_reference_relative_l2"][:stable_stop].mean()
        ),
        "final_hidden_reference_relative_l2": float(arrays["hidden_reference_relative_l2"][-1]),
        "mean_stable_hidden_step_rms": float(arrays["hidden_step_rms"][:stable_stop].mean()),
        "final_target_mse": float(arrays["target_mse"][-1]),
        "final_target_region_mse": float(arrays["target_region_mse"][-1]),
        "final_alpha_iou": float(arrays["alpha_iou"][-1]),
        "final_area_ratio": float(arrays["area_ratio"][-1]),
    }
    channel_arrays = {name: np.stack(values) for name, values in channel_records.items()}
    visible_channel_arrays = {
        name: np.stack(values) for name, values in visible_channel_records.items()
    }
    return BranchResult(
        arrays,
        flags,
        summary,
        channel_arrays,
        channel_steps,
        visible_channel_arrays,
        visible_channel_steps,
        snapshots,
    )


def _save_trajectory_metrics(
    path: Path,
    metrics: dict[str, dict[str, np.ndarray]],
    regime_flags: dict[str, np.ndarray],
    replicate_ids: list[int],
) -> None:
    payload: dict[str, Any] = {
        "steps": np.arange(next(iter(next(iter(metrics.values())).values())).shape[1]),
        "replicate_ids": np.asarray(replicate_ids, dtype=np.int64),
        "condition_names": np.asarray(list(metrics)),
        "metric_names": np.asarray(METRIC_NAMES),
    }
    for condition, condition_metrics in metrics.items():
        for name, values in condition_metrics.items():
            payload[f"{condition}__{name}"] = values
        payload[f"{condition}__in_regime"] = regime_flags[condition]
    np.savez_compressed(path, **payload)


def _metadata(config: ExperimentConfig, device: torch.device) -> dict[str, Any]:
    return {
        "experiment": config.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(config.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(config.checkpoint),
        "target": str(Path(config.target).resolve()),
        "target_sha256": file_sha256(config.target),
        "device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "measurement_precision": "float64 storage; CPU float64 reductions when float32 squares or sums risk overflow",
    }


def run_experiment(config: ExperimentConfig) -> Path:
    """Run the complete paired Growing NCA hidden-state intervention experiment."""
    config.validate()
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a new output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(config.device)
    target = load_distill_target(
        config.target,
        size=config.model.target_size,
        padding=config.model.target_padding,
        device=device,
    )
    model = load_model(config, device)
    conditions = build_conditions(config.interventions)

    resolved = config.to_dict()
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    metadata = _metadata(config, device)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_csv(
        output_dir / "conditions.csv",
        [
            {
                "condition": condition.name,
                "label": condition.label,
                "family": condition.family,
                "parameter_name": condition.parameter_name or "",
                "parameter": "" if condition.parameter is None else condition.parameter,
                "reference_kind": (
                    "development" if condition.reference_step is not None else "mature"
                ),
                "reference_step": (
                    "" if condition.reference_step is None else condition.reference_step
                ),
                "hidden_indices": (
                    ""
                    if condition.hidden_indices is None
                    else ",".join(str(index) for index in condition.hidden_indices)
                ),
            }
            for condition in conditions
        ],
    )

    print(
        f"Calibrating target regime with {config.calibration.replicates} rollouts...",
        flush=True,
    )
    regime, calibration_metrics = _calibrate_regime(model, target, config, device)
    np.savez_compressed(output_dir / "calibration_metrics.npz", **calibration_metrics)
    (output_dir / "target_regime.json").write_text(json.dumps(regime.to_dict(), indent=2) + "\n")

    metric_lists: dict[str, dict[str, list[np.ndarray]]] = {
        condition.name: {name: [] for name in METRIC_NAMES} for condition in conditions
    }
    channel_metric_lists: dict[str, dict[str, list[np.ndarray]]] = {
        condition.name: {name: [] for name in HIDDEN_CHANNEL_METRIC_NAMES}
        for condition in conditions
    }
    channel_steps: list[int] = []
    visible_channel_metric_lists: dict[str, dict[str, list[np.ndarray]]] = {
        condition.name: {name: [] for name in VISIBLE_CHANNEL_METRIC_NAMES}
        for condition in conditions
    }
    visible_channel_steps: list[int] = []
    regime_lists: dict[str, list[np.ndarray]] = {condition.name: [] for condition in conditions}
    run_rows: list[dict[str, object]] = []
    excluded: list[dict[str, int]] = []
    included_replicates: list[int] = []
    snapshots: dict[str, dict[int, torch.Tensor | None]] = {}
    development_snapshots: dict[int, torch.Tensor] = {}
    mature_states: dict[int, dict[str, Any]] = {}

    for replicate in range(config.evaluation.replicates):
        development_seed = config.evaluation.seed_start + replicate
        mature_state, mature_step, candidate_snapshots, development_states = _develop_to_target(
            model, target, regime, config, device, development_seed
        )
        if mature_state is None or mature_step is None:
            excluded.append({"replicate": replicate, "development_seed": development_seed})
            print(
                f"Evaluation replicate {replicate + 1}/{config.evaluation.replicates}: "
                "target not attained; excluded.",
                flush=True,
            )
            continue
        included_replicates.append(replicate)
        mature_states[replicate] = {
            "mature_step": mature_step,
            "development_seed": development_seed,
            "state": mature_state[0].detach().cpu(),
            "development_states": {
                step: state[0].detach().cpu() for step, state in development_states.items()
            },
        }
        if not development_snapshots:
            development_snapshots = candidate_snapshots
        continuation_seed = (
            config.evaluation.seed_start + config.evaluation.continuation_seed_offset + replicate
        )
        detailed_index = len(included_replicates) - 1
        for condition in conditions:
            intervention_reference_state = resolve_intervention_reference(
                mature_state,
                condition,
                development_states,
            )
            print(
                f"Evaluation replicate {replicate + 1}/{config.evaluation.replicates}, "
                f"t*={mature_step}: {condition.label}",
                flush=True,
            )
            result = _run_branch(
                model,
                target,
                mature_state,
                intervention_reference_state,
                regime,
                condition,
                config,
                replicate,
                continuation_seed,
                collect_snapshots=detailed_index == 0,
            )
            result.summary["development_seed"] = development_seed
            result.summary["continuation_seed"] = continuation_seed
            result.summary["mature_step"] = mature_step
            run_rows.append(result.summary)
            for name in METRIC_NAMES:
                metric_lists[condition.name][name].append(result.metrics[name])
            for name in HIDDEN_CHANNEL_METRIC_NAMES:
                channel_metric_lists[condition.name][name].append(result.channel_metrics[name])
            if not channel_steps:
                channel_steps = result.channel_steps
            elif channel_steps != result.channel_steps:
                raise RuntimeError("hidden-channel sampling steps differ between branches")
            for name in VISIBLE_CHANNEL_METRIC_NAMES:
                visible_channel_metric_lists[condition.name][name].append(
                    result.visible_channel_metrics[name]
                )
            if not visible_channel_steps:
                visible_channel_steps = result.visible_channel_steps
            elif visible_channel_steps != result.visible_channel_steps:
                raise RuntimeError("visible-channel sampling steps differ between branches")
            regime_lists[condition.name].append(result.in_regime)
            if result.snapshots:
                snapshots[condition.name] = result.snapshots

    if not included_replicates:
        (output_dir / "excluded_replicates.json").write_text(json.dumps(excluded, indent=2) + "\n")
        raise RuntimeError("no evaluation rollout reached the calibrated target regime")

    metrics = {
        condition: {name: np.stack(values) for name, values in condition_metrics.items()}
        for condition, condition_metrics in metric_lists.items()
    }
    regime_arrays = {condition: np.stack(values) for condition, values in regime_lists.items()}
    _save_trajectory_metrics(
        output_dir / "trajectory_metrics.npz", metrics, regime_arrays, included_replicates
    )
    channel_metrics = {
        condition: {name: np.stack(values) for name, values in condition_metrics.items()}
        for condition, condition_metrics in channel_metric_lists.items()
    }
    save_hidden_channel_metrics(
        output_dir / "hidden_channel_metrics.npz",
        channel_metrics,
        channel_steps,
        included_replicates,
    )
    write_csv(
        output_dir / "hidden_channel_summary.csv",
        summarize_hidden_channels(channel_metrics, conditions, channel_steps),
    )
    visible_channel_metrics_by_condition = {
        condition: {name: np.stack(values) for name, values in condition_metrics.items()}
        for condition, condition_metrics in visible_channel_metric_lists.items()
    }
    save_visible_channel_metrics(
        output_dir / "visible_channel_metrics.npz",
        visible_channel_metrics_by_condition,
        visible_channel_steps,
        included_replicates,
    )
    write_csv(
        output_dir / "visible_channel_summary.csv",
        summarize_visible_channels(
            visible_channel_metrics_by_condition,
            conditions,
            visible_channel_steps,
        ),
    )
    write_csv(output_dir / "per_run_summary.csv", run_rows)
    (output_dir / "excluded_replicates.json").write_text(json.dumps(excluded, indent=2) + "\n")
    torch.save(
        {"intervention": snapshots, "development": development_snapshots},
        output_dir / "snapshots.pt",
    )
    torch.save(mature_states, output_dir / "mature_states.pt")

    summaries = summarize_conditions(run_rows)
    write_csv(output_dir / "condition_summary.csv", summaries)

    plot_core_morphology_metrics(
        output_dir / "core_morphology_metrics.png", metrics, conditions, regime
    )
    plot_hidden_channel_dynamics(
        output_dir / "hidden_channel_dynamics.png",
        channel_metrics,
        conditions,
        channel_steps,
    )
    plot_visible_channel_dynamics(
        output_dir / "visible_channel_dynamics.png",
        visible_channel_metrics_by_condition,
        conditions,
        visible_channel_steps,
    )
    plot_snapshots(
        output_dir / "morphology_snapshots.png",
        snapshots,
        conditions,
        config.analysis.snapshot_steps,
        include_all=config.analysis.snapshot_all_conditions,
    )
    plot_development(
        output_dir / "development_to_target.png", development_snapshots, target.detach().cpu()
    )
    return output_dir
