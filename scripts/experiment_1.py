from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import platform
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from threadpoolctl import threadpool_limits

from latent_nca.artifacts import file_sha256, write_csv
from latent_nca.data.device import get_device, seed_everything
from latent_nca.data.images import load_distill_target
from latent_nca.experiment_1.metrics import (
    HIDDEN_CHANNEL_METRIC_NAMES,
    MEASUREMENT_REVISION,
    METRIC_NAMES,
    MORPHOLOGY_METRIC_NAMES,
    VISIBLE_CHANNEL_METRIC_NAMES,
    MorphologyEvaluator,
    TargetRegime,
    hidden_channel_metrics,
    survival_outcome,
    visible_channel_metrics,
)
from latent_nca.models import GrowingNCA, load_growing_checkpoint

plt.switch_backend("Agg")
MORPHOLOGIES = ("gecko", "butterfly", "planaria", "spider")
PROTOCOL = "experiment_1_v3"


# Configuration and hidden-state interventions.
@dataclass(frozen=True)
class ModelConfig:
    channels: int = 16
    hidden_size: int = 128
    target_size: int = 40
    target_padding: int = 16
    update_probability: float = 0.5
    alpha_threshold: float = 0.1


@dataclass(frozen=True)
class CalibrationConfig:
    replicates: int = 32
    seed_start: int = 10_000
    mature_start: int = 64
    mature_stop: int = 96
    acceptance_quantile: float = 0.95


@dataclass(frozen=True)
class EvaluationConfig:
    replicates: int = 30
    seed_start: int = 20_000
    continuation_seed_offset: int = 1_000_000
    horizon: int = 10_000
    attainment_patience: int = 5
    failure_patience: int = 10


@dataclass(frozen=True)
class InterventionConfig:
    include_standard_conditions: bool = True
    include_morphology_freeze: bool = True
    partial_freeze_counts: tuple[int, ...] = (6,)
    restore_strengths: tuple[float, ...] = (0.1,)
    periodic_reset_intervals: tuple[int, ...] = (150,)
    development_reference_steps: tuple[int, ...] = (16,)
    development_restore_strengths: tuple[float, ...] = (0.1,)
    development_reset_intervals: tuple[int, ...] = (150,)


@dataclass(frozen=True)
class AnalysisConfig:
    hidden_channel_record_every: int = 10
    visible_channel_record_every: int = 10
    snapshot_steps: tuple[int, ...] = (0, 100, 500, 1_000, 2_000, 5_000, 10_000)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment_1"
    checkpoint: str = "checkpoints/growing/gecko/model.pt"
    target: str = "data/gecko.png"
    output_dir: str = "reports/experiment_1/gecko"
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    interventions: InterventionConfig = field(default_factory=InterventionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.name.startswith("experiment_1"):
            raise ValueError("name must start with 'experiment_1'")
        if self.model.channels <= 4:
            raise ValueError("the model must have at least one hidden channel")
        if not 0.0 < self.model.update_probability <= 1.0:
            raise ValueError("update_probability must be in (0, 1]")
        if not 0.0 < self.model.alpha_threshold < 1.0:
            raise ValueError("alpha_threshold must be in (0, 1)")
        if self.calibration.replicates <= 0 or self.evaluation.replicates <= 0:
            raise ValueError("replicate counts must be positive")
        if self.calibration.mature_start < 0:
            raise ValueError("mature_start must be non-negative")
        if self.calibration.mature_stop < self.calibration.mature_start:
            raise ValueError("mature_stop must be at least mature_start")
        if not 0.5 < self.calibration.acceptance_quantile < 1.0:
            raise ValueError("acceptance_quantile must be in (0.5, 1)")
        if self.evaluation.horizon <= 0:
            raise ValueError("evaluation horizon must be positive")
        if self.evaluation.continuation_seed_offset <= 0:
            raise ValueError("continuation_seed_offset must be positive")
        if self.evaluation.attainment_patience <= 0 or self.evaluation.failure_patience <= 0:
            raise ValueError("patience values must be positive")
        hidden_channels = self.model.channels - 4
        if any(
            value <= 0 or value >= hidden_channels
            for value in self.interventions.partial_freeze_counts
        ):
            raise ValueError(
                "partial freeze counts must be positive and smaller than the hidden-channel count"
            )
        if any(not 0.0 < value < 1.0 for value in self.interventions.restore_strengths):
            raise ValueError("restore strengths must be in (0, 1); freeze supplies beta=1")
        if any(value <= 0 for value in self.interventions.periodic_reset_intervals):
            raise ValueError("periodic reset intervals must be positive")
        reference_steps = self.interventions.development_reference_steps
        if len(reference_steps) != len(set(reference_steps)):
            raise ValueError("development reference steps must be unique")
        if any(step <= 0 or step >= self.calibration.mature_start for step in reference_steps):
            raise ValueError(
                "development reference steps must be positive and precede mature_start"
            )
        if any(not 0.0 < value < 1.0 for value in self.interventions.development_restore_strengths):
            raise ValueError("development restore strengths must be in (0, 1)")
        if any(value <= 0 for value in self.interventions.development_reset_intervals):
            raise ValueError("development reset intervals must be positive")
        if (
            self.interventions.development_restore_strengths
            or self.interventions.development_reset_intervals
        ) and not reference_steps:
            raise ValueError(
                "development restoration requires at least one development reference step"
            )
        if (
            self.analysis.hidden_channel_record_every <= 0
            or self.analysis.visible_channel_record_every <= 0
        ):
            raise ValueError("hidden- and visible-channel sampling intervals must be positive")
        if any(step < 0 or step > self.evaluation.horizon for step in self.analysis.snapshot_steps):
            raise ValueError("snapshot steps must fall within the evaluation horizon")


def _section(cls, values: dict[str, Any] | None):
    values = dict(values or {})
    for key in (
        "partial_freeze_counts",
        "restore_strengths",
        "periodic_reset_intervals",
        "development_reference_steps",
        "development_restore_strengths",
        "development_reset_intervals",
        "snapshot_steps",
    ):
        if key in values:
            values[key] = tuple(values[key])
    return cls(**values)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    try:
        contents = path.read_text()
    except OSError as error:
        if error.errno in (errno.ECANCELED, errno.ETIMEDOUT):
            error.add_note(f"Cannot read Experiment 1 configuration: {path.resolve()}")
            error.add_note(
                "If this file is stored on iCloud, make the required reports available locally "
                "in Finder (Download Now / Keep Downloaded), wait for the download to finish, "
                "then retry the same command. The saved configuration must not be replaced "
                "with default settings."
            )
        raise
    raw = yaml.safe_load(contents) or {}
    config = ExperimentConfig(
        name=raw.get("name", "experiment_1"),
        checkpoint=raw.get("checkpoint", ExperimentConfig.checkpoint),
        target=raw.get("target", ExperimentConfig.target),
        output_dir=raw.get("output_dir", ExperimentConfig.output_dir),
        device=raw.get("device", ExperimentConfig.device),
        model=_section(ModelConfig, raw.get("model")),
        calibration=_section(CalibrationConfig, raw.get("calibration")),
        evaluation=_section(EvaluationConfig, raw.get("evaluation")),
        interventions=_section(InterventionConfig, raw.get("interventions")),
        analysis=_section(AnalysisConfig, raw.get("analysis")),
    )
    config.validate()
    return config


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    label: str
    family: str
    parameter_name: str | None = None
    parameter: float | int | None = None
    hidden_indices: tuple[int, ...] | None = None
    reference_step: int | None = None


def _number_token(value: float) -> str:
    return str(value).replace(".", "p")


def build_conditions(config: InterventionConfig) -> tuple[ConditionSpec, ...]:
    conditions = [ConditionSpec("baseline", "Control", "baseline")]
    if config.include_standard_conditions:
        conditions.append(
            ConditionSpec("hidden_freeze", "Freeze hidden (full grid)", "freeze", "beta", 1.0)
        )
    if config.include_morphology_freeze:
        conditions.append(
            ConditionSpec(
                "hidden_freeze_morphology", "Freeze hidden (target only)", "morphology_freeze"
            )
        )
    if config.include_standard_conditions:
        conditions.extend(
            ConditionSpec(
                f"hidden_freeze_first_{count}",
                f"Freeze first {count} hidden",
                "partial_freeze",
                "frozen_channels",
                count,
                tuple(range(count)),
            )
            for count in config.partial_freeze_counts
        )
        conditions.extend(
            ConditionSpec(
                f"restore_beta_{_number_token(beta)}",
                f"Restore beta={beta:g}",
                "restore",
                "beta",
                beta,
            )
            for beta in config.restore_strengths
        )
    conditions.extend(
        ConditionSpec(
            f"periodic_reset_{interval}",
            f"Periodic reset P={interval}",
            "periodic_reset",
            "interval",
            interval,
        )
        for interval in config.periodic_reset_intervals
    )
    conditions.extend(
        ConditionSpec(
            (f"restore_development_step_{reference_step}_beta_{_number_token(beta)}"),
            f"Restore dev t={reference_step} beta={beta:g}",
            "development_restore",
            "beta",
            beta,
            reference_step=reference_step,
        )
        for reference_step in config.development_reference_steps
        for beta in config.development_restore_strengths
    )
    conditions.extend(
        ConditionSpec(
            f"restore_development_step_{reference_step}_period_{interval}",
            f"Dev reset t={reference_step} P={interval}",
            "development_periodic_reset",
            "interval",
            interval,
            reference_step=reference_step,
        )
        for reference_step in config.development_reference_steps
        for interval in config.development_reset_intervals
    )
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("intervention parameters produce duplicate condition names")
    return tuple(conditions)


def resolve_intervention_reference(
    mature_state: torch.Tensor,
    condition: ConditionSpec,
    development_states: Mapping[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return H* or the same rollout's configured developmental reference state."""
    if condition.reference_step is None:
        return mature_state
    if development_states is None or condition.reference_step not in development_states:
        raise ValueError(
            f"missing developmental state at step {condition.reference_step} "
            f"for condition {condition.name}"
        )
    reference = development_states[condition.reference_step]
    if reference.ndim == mature_state.ndim - 1:
        reference = reference.unsqueeze(0)
    reference = reference.to(device=mature_state.device, dtype=mature_state.dtype)
    if reference.shape != mature_state.shape:
        raise ValueError(
            f"developmental reference shape {tuple(reference.shape)} does not match "
            f"mature state shape {tuple(mature_state.shape)}"
        )
    return reference


def apply_post_step_intervention(
    next_state: torch.Tensor,
    reference_state: torch.Tensor,
    condition: ConditionSpec,
    relative_step: int,
    *,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if condition.family == "baseline":
        return next_state
    if condition.family == "morphology_freeze":
        if target_mask is None:
            raise ValueError("morphology freeze requires the fixed target alpha mask")
        if target_mask.shape != next_state[:, :1].shape or target_mask.dtype != torch.bool:
            raise ValueError("target mask must be boolean with shape [B, 1, H, W]")
        result = next_state.clone()
        result[:, 4:] = torch.where(target_mask, reference_state[:, 4:], next_state[:, 4:])
        return result
    if condition.family == "partial_freeze":
        if not condition.hidden_indices:
            raise ValueError("partial freeze requires hidden-channel indices")
        indices = [4 + index for index in condition.hidden_indices]
        if indices[-1] >= next_state.shape[1]:
            raise ValueError("partial-freeze channel index exceeds the NCA state size")
        result = next_state.clone()
        result[:, indices] = reference_state[:, indices]
        return result
    if condition.family == "freeze":
        beta = 1.0
    elif condition.family in {"restore", "development_restore"}:
        beta = float(condition.parameter)
    elif condition.family in {"periodic_reset", "development_periodic_reset"}:
        interval = int(condition.parameter)
        if relative_step % interval:
            return next_state
        beta = 1.0
    else:
        raise ValueError(f"unknown intervention family: {condition.family}")

    result = next_state.clone()
    result[:, 4:] = torch.lerp(result[:, 4:], reference_state[:, 4:], beta)
    return result


def load_model(config: ExperimentConfig, device: torch.device) -> GrowingNCA:
    return load_growing_checkpoint(
        config.checkpoint,
        channels=config.model.channels,
        hidden_size=config.model.hidden_size,
        update_probability=config.model.update_probability,
        device=device,
    )


# Paired simulation, calibration and measurements.
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
            if not bool(torch.isfinite(state).all()):
                raise ValueError(f"non-finite calibration state: replica {replicate}, step {step}")
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
    str | None,
]:
    seed_everything(seed)
    height, width = target.shape[-2:]
    state = model.seed(1, height, width, device)
    evaluator = MorphologyEvaluator(target, config.model.alpha_threshold)
    consecutive = 0
    snapshots: dict[int, torch.Tensor] = {0: state[0, :4].detach().cpu().clone()}
    reference_steps = set(config.interventions.development_reference_steps)
    development_steps = reference_steps | {
        config.calibration.mature_start // 2,
        config.calibration.mature_start,
        config.calibration.mature_stop,
    }
    development_states: dict[int, torch.Tensor] = {}
    for step in range(config.calibration.mature_stop + 1):
        if not bool(torch.isfinite(state).all()):
            return None, None, snapshots, development_states, "nonfinite_during_development"
        if step in development_steps:
            snapshots[step] = state[0, :4].detach().cpu().clone()
        if step in reference_steps:
            development_states[step] = state.detach().clone()
        if step >= config.calibration.mature_start:
            metrics = evaluator.evaluate(state)
            consecutive = consecutive + 1 if regime.contains(metrics) else 0
            if consecutive >= config.evaluation.attainment_patience:
                snapshots[step] = state[0, :4].detach().cpu().clone()
                return state.detach().clone(), step, snapshots, development_states, None
        if step < config.calibration.mature_stop:
            state = model(state, update_probability=config.model.update_probability)
    return None, None, snapshots, development_states, "target_not_attained"


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
    snapshot_steps = set(config.analysis.snapshot_steps) | {0, config.evaluation.horizon}
    previous_state: torch.Tensor | None = None
    divergence_step: int | None = None

    def pad_after_divergence(step: int) -> None:
        remaining_steps = config.evaluation.horizon - step + 1
        for record in records.values():
            record.extend([float("nan")] * remaining_steps)
        regime_flags.extend([False] * remaining_steps)
        hidden_channels = mature_state.shape[1] - 4
        for sampled_step in range(step, config.evaluation.horizon + 1):
            if (
                sampled_step % config.analysis.hidden_channel_record_every
                and sampled_step != config.evaluation.horizon
            ):
                continue
            missing = np.full(hidden_channels, np.nan, dtype=np.float64)
            for name in HIDDEN_CHANNEL_METRIC_NAMES:
                channel_records[name].append(missing.copy())
            channel_steps.append(sampled_step)
        for sampled_step in range(step, config.evaluation.horizon + 1):
            if (
                sampled_step % config.analysis.visible_channel_record_every
                and sampled_step != config.evaluation.horizon
            ):
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
        values = (
            evaluator.evaluate(state, previous_state) if bool(torch.isfinite(state).all()) else None
        )
        hidden_values, visible_values = {}, {}
        record_hidden = (
            step % config.analysis.hidden_channel_record_every == 0
            or step == config.evaluation.horizon
        )
        record_visible = (
            step % config.analysis.visible_channel_record_every == 0
            or step == config.evaluation.horizon
        )
        if values is not None and all(math.isfinite(value) for value in values.values()):
            if record_hidden:
                hidden_values = hidden_channel_metrics(state, mature_state, previous_state)
            if record_visible:
                visible_values = visible_channel_metrics(state, mature_state, previous_state)
        if (
            values is None
            or not all(math.isfinite(value) for value in values.values())
            or not all(
                bool(torch.isfinite(value).all())
                for value in (*hidden_values.values(), *visible_values.values())
            )
        ):
            divergence_step = step
            pad_after_divergence(step)
            break
        for name in METRIC_NAMES:
            records[name].append(values[name])
        regime_flags.append(regime.contains(values))
        if record_hidden:
            for name, channel_data in hidden_values.items():
                channel_records[name].append(channel_data.detach().cpu().numpy().astype(np.float64))
            channel_steps.append(step)
        if record_visible:
            for name, channel_data in visible_values.items():
                visible_channel_records[name].append(
                    channel_data.detach().cpu().numpy().astype(np.float64)
                )
            visible_channel_steps.append(step)
        if collect_snapshots and step in snapshot_steps:
            snapshots[step] = state[0, :4].detach().cpu().clone()
        if step == config.evaluation.horizon:
            break
        previous_state = state
        state = model(state, update_probability=config.model.update_probability)
        state = apply_post_step_intervention(
            state,
            intervention_reference_state,
            condition,
            step + 1,
            target_mask=evaluator.target_mask,
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
        "measurement_revision": MEASUREMENT_REVISION,
        "measurement_precision": "float64 storage; CPU float64 reductions when float32 squares or sums risk overflow",
    }


# Population figures.
def _condition_map(conditions: tuple[ConditionSpec, ...]) -> dict[str, ConditionSpec]:
    return {condition.name: condition for condition in conditions}


def _mask_invalid_replicates(
    metrics: Mapping[str, np.ndarray],
    valid: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """A non-finite value retires its trajectory from that step onward."""
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in metrics.items()}
    shape = next(iter(arrays.values())).shape[:2]
    valid = np.ones(shape, dtype=bool) if valid is None else np.array(valid, dtype=bool, copy=True)
    if len(shape) != 2 or valid.shape != shape:
        raise ValueError("Expected validity indexed by replica and step")
    for values in arrays.values():
        if values.shape[:2] != shape:
            raise ValueError("Metric replica and time axes must match")
        valid &= np.isfinite(values).all(axis=tuple(range(2, values.ndim)))
    valid = np.logical_and.accumulate(valid, axis=1)
    return {
        name: np.where(valid.reshape(shape + (1,) * (values.ndim - 2)), values, np.nan)
        for name, values in arrays.items()
    }, valid


def _finite_quantile(values: np.ndarray, quantile: float | tuple[float, ...]) -> np.ndarray:
    finite_values = np.where(np.isfinite(values), values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanquantile(finite_values, quantile, axis=0)


def _finite_mean_std_count(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked, valid = _mask_invalid_replicates({"values": values})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(masked["values"], axis=0)
        standard_deviations = np.nanstd(masked["values"], axis=0, ddof=1)
    counts = valid.sum(axis=0)
    return means, standard_deviations, counts


def _finite_median_quantiles_count(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masked, valid = _mask_invalid_replicates({"values": values})
    counts = valid.sum(axis=0)
    q05, q25, median, q75, q95 = _finite_quantile(
        masked["values"],
        (0.05, 0.25, 0.5, 0.75, 0.95),
    )
    return median, q25, q75, q05, q95, counts


def plot_core_morphology_metrics(
    path: Path,
    metrics: dict[str, dict[str, np.ndarray]],
    conditions: tuple[ConditionSpec, ...],
    regime: TargetRegime,
) -> None:
    _plot_core_morphology_metrics(path, metrics, conditions, regime, robust=False)


def plot_core_morphology_metrics_robust(
    path: Path,
    metrics: dict[str, dict[str, np.ndarray]],
    conditions: tuple[ConditionSpec, ...],
    regime: TargetRegime,
) -> None:
    _plot_core_morphology_metrics(path, metrics, conditions, regime, robust=True)


def _set_mse_log_scale(axis) -> None:
    minimum = float(axis.dataLim.minposy)
    maximum = float(axis.dataLim.ymax)
    if not np.isfinite(minimum) or minimum <= 0:
        minimum = 1e-6
    if not np.isfinite(maximum) or maximum < minimum:
        maximum = minimum
    log_minimum, log_maximum = math.log10(minimum), math.log10(maximum)
    padding = max(0.2, 0.05 * (log_maximum - log_minimum))
    smallest = float(np.nextafter(0.0, 1.0))
    largest = np.finfo(np.float64).max
    lower = (
        smallest
        if log_minimum - padding <= math.log10(smallest)
        else 10.0 ** (log_minimum - padding)
    )
    upper = (
        largest if log_maximum + padding >= math.log10(largest) else 10.0 ** (log_maximum + padding)
    )
    if lower == upper:
        if upper < largest:
            upper = float(np.nextafter(upper, np.inf))
        else:
            lower = float(np.nextafter(lower, 0.0))
    axis.set_ylim(lower, upper)
    axis.set_yscale("log", nonpositive="mask")
    # LogLocator adds ticks outside the view; on wide ranges those can become inf.
    first, last = math.ceil(math.log10(lower)), math.floor(math.log10(upper))
    stride = max(1, math.ceil((last - first) / 7))
    ticks = np.power(10.0, np.arange(first, last + 1, stride, dtype=np.float64))
    axis.yaxis.set_major_locator(FixedLocator(ticks if ticks.size else [minimum]))
    axis.yaxis.set_minor_locator(NullLocator())


def _plot_core_morphology_metrics(
    path: Path,
    metrics: dict[str, dict[str, np.ndarray]],
    conditions: tuple[ConditionSpec, ...],
    regime: TargetRegime,
    *,
    robust: bool,
) -> None:
    condition_by_name = _condition_map(conditions)
    names = [condition.name for condition in conditions]
    masked = {name: _mask_invalid_replicates(metrics[name]) for name in names}
    metrics = {name: values for name, (values, _) in masked.items()}
    panels = [
        ("target_mse", "RGBA target MSE"),
        ("alpha_iou", "Alpha-mask IoU"),
        ("area_ratio", "Alive-area ratio"),
    ]
    if all("target_region_mse" in metrics[name] for name in names):
        panels.insert(1, ("target_region_mse", "RGBA MSE in target region"))
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.5, 4.6 * rows),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(rows, columns)
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    original_colors = {
        "baseline": 0,
        "hidden_freeze": 1,
        "hidden_freeze_first_6": 2,
        "restore_beta_0p1": 3,
        "periodic_reset_150": 4,
        "restore_development_step_16_beta_0p1": 5,
        "restore_development_step_48_beta_0p1": 6,
        "restore_development_step_16_period_150": 7,
        "restore_development_step_48_period_150": 8,
        "hidden_freeze_morphology": 9,
    }
    colors = {
        name: cmap(original_colors.get(name, index) % cmap.N) for index, name in enumerate(names)
    }
    summarize = _finite_median_quantiles_count if robust else _finite_mean_std_count
    statistics = {
        name: {metric: summarize(metrics[name][metric]) for metric, _ in panels} for name in names
    }
    legend_labels: dict[str, str] = {}
    for name in names:
        counts = statistics[name][panels[0][0]][-1]
        initial_count = int(counts[0])
        final_count = int(counts[-1])
        sample_text = (
            f"n={initial_count}"
            if initial_count == final_count
            else f"n={initial_count}->{final_count}"
        )
        legend_labels[name] = f"{condition_by_name[name].label} ({sample_text})"

    for axis, (metric, title) in zip(axes.flat[: len(panels)], panels, strict=True):
        for name in names:
            if robust:
                center, q25, q75, q05, q95, counts = statistics[name][metric]
                bands = ((q05, q95, 0.06), (q25, q75, 0.17))
            else:
                center, standard_deviation, counts = statistics[name][metric]
                bands = ((center - standard_deviation, center + standard_deviation, 0.13),)
            remaining = np.flatnonzero(counts)
            stop = int(remaining[-1]) + 1 if remaining.size else 0
            steps = np.arange(stop)
            center = center[:stop]
            counts = counts[:stop]
            color = colors[name]
            for lower, upper, alpha in bands:
                lower, upper = lower[:stop], upper[:stop]
                if metric in {"target_mse", "target_region_mse"}:
                    lower = np.ma.masked_less_equal(lower, 0.0)
                    upper = np.ma.masked_less_equal(upper, 0.0)
                elif metric == "alpha_iou":
                    lower = np.clip(lower, 0.0, 1.0)
                    upper = np.clip(upper, 0.0, 1.0)
                elif metric == "area_ratio":
                    lower = np.maximum(lower, 0.0)
                axis.fill_between(
                    steps,
                    lower,
                    upper,
                    where=counts >= 2,
                    color=color,
                    alpha=alpha,
                    linewidth=0,
                )
            if metric in {"target_mse", "target_region_mse"}:
                center = np.ma.masked_less_equal(center, 0.0)
            visible = ~np.ma.getmaskarray(center) & np.isfinite(np.ma.getdata(center))
            isolated = visible & ~np.pad(visible, (1, 1))[:-2] & ~np.pad(visible, (1, 1))[2:]
            axis.plot(
                steps,
                center,
                color=color,
                linewidth=1.45,
                label=legend_labels[name],
                marker="." if isolated.any() else None,
                markersize=3,
                markevery=np.flatnonzero(isolated).tolist(),
            )
        if metric in {"target_mse", "target_region_mse"}:
            if metric == "target_mse":
                axis.axhline(regime.target_mse_max, color="#666666", linestyle="--")
            _set_mse_log_scale(axis)
        elif metric == "alpha_iou":
            axis.axhline(regime.alpha_iou_min, color="#666666", linestyle="--")
            axis.set_ylim(-0.02, 1.02)
        elif metric == "area_ratio":
            axis.axhspan(
                regime.area_ratio_min,
                regime.area_ratio_max,
                color="#888888",
                alpha=0.12,
            )
            axis.set_yscale("symlog", linthresh=0.1)
        axis.set_title(title)
        axis.set_xlabel("Steps after target attainment")
        axis.grid(alpha=0.2, which="both")
    for axis in axes.flat[len(panels) :]:
        axis.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_columns = min(3, len(names))
    legend_rows = int(np.ceil(len(names) / legend_columns))
    bottom_margin = 0.055 + 0.035 * legend_rows
    title = (
        "Core morphology trajectories: median and robust spread across paired replicates"
        if robust
        else "Core morphology trajectories: mean and variability across paired replicates"
    )
    figure.suptitle(title, fontsize=14)
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        fontsize=8,
        ncol=legend_columns,
        frameon=False,
    )
    note = (
        "Solid line: valid-replica median. Dark band: 25th-75th percentiles (IQR). "
        "Light band: 5th-95th percentiles."
        if robust
        else "Solid line: valid-replica mean. Shaded band: +/- 1 sample SD."
    )
    figure.text(
        0.5,
        0.008,
        f"{note} n: valid replicas (first -> final). Zero MSE omitted on log axes.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, bottom_margin, 1.0, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)
    write_csv(
        path.with_name(f"{path.stem}_n_valid.csv"),
        [
            {"condition": name, "step": step, "n_valid": int(count), "n_total": valid.shape[0]}
            for name, (_, valid) in masked.items()
            for step, count in enumerate(valid.sum(axis=0))
        ],
    )


def _plot_channel_dynamics(
    path: Path,
    metrics: dict[str, dict[str, np.ndarray]],
    conditions: tuple[ConditionSpec, ...],
    steps: list[int],
    channel_name: str,
    channel_labels: tuple[str, ...],
    max_step: int | None = None,
) -> None:
    if not metrics or not steps:
        return
    metrics = {
        condition.name: _mask_invalid_replicates(metrics[condition.name])[0]
        for condition in conditions
    }
    step_values = np.asarray(steps, dtype=np.int64)
    selected_steps = np.ones(step_values.shape, dtype=bool)
    if max_step is not None:
        selected_steps = step_values <= max_step
        if not selected_steps.any():
            raise ValueError("max_step precedes the first channel-dynamics sample")
    plotted_steps = step_values[selected_steps]

    # Fixed, clipped log ranges preserve comparability while preventing a
    # numerically divergent branch from flattening every other condition.
    relative_log_limits = (-4.0, 1.0)
    update_log_limits = (-6.0, -1.0)

    def transformed_median(
        condition: ConditionSpec,
        metric: str,
        limits: tuple[float, float],
    ) -> np.ndarray:
        median = _finite_quantile(metrics[condition.name][metric], 0.5)
        median = median[selected_steps].T
        lower, upper = (10.0 ** limits[0], 10.0 ** limits[1])
        return np.log10(np.clip(median, lower, upper))

    relative = {
        condition.name: transformed_median(
            condition,
            "relative_l2",
            relative_log_limits,
        )
        for condition in conditions
    }
    updates = {
        condition.name: transformed_median(
            condition,
            "step_rms",
            update_log_limits,
        )
        for condition in conditions
    }
    relative_cmap = plt.get_cmap("magma").with_extremes(bad="#BDBDBD")
    update_cmap = plt.get_cmap("viridis").with_extremes(bad="#BDBDBD")

    figure, axes = plt.subplots(
        len(conditions),
        2,
        figsize=(14.0, 2.25 * len(conditions)),
        squeeze=False,
        layout="constrained",
    )
    extent = (
        plotted_steps[0] - (0.5 if len(plotted_steps) == 1 else 0),
        plotted_steps[-1] + (0.5 if len(plotted_steps) == 1 else 0),
        -0.5,
        relative[next(iter(relative))].shape[0] - 0.5,
    )
    relative_image = None
    update_image = None
    for row, condition in enumerate(conditions):
        relative_image = axes[row, 0].imshow(
            relative[condition.name],
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=relative_cmap,
            vmin=relative_log_limits[0],
            vmax=relative_log_limits[1],
        )
        update_image = axes[row, 1].imshow(
            updates[condition.name],
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=update_cmap,
            vmin=update_log_limits[0],
            vmax=update_log_limits[1],
        )
        for axis in axes[row]:
            axis.set_yticks(range(relative[condition.name].shape[0]))
            axis.set_yticklabels(channel_labels)
            axis.set_ylabel(condition.label)
            if (
                channel_name == "hidden"
                and condition.family == "partial_freeze"
                and condition.hidden_indices
            ):
                axis.axhline(max(condition.hidden_indices) + 0.5, color="white", linewidth=0.8)
        if row == len(conditions) - 1:
            axes[row, 0].set_xlabel("Steps after target attainment")
            axes[row, 1].set_xlabel("Steps after target attainment")
    axes[0, 0].set_title(f"Median deviation from mature {channel_name} state H*")
    axes[0, 1].set_title(f"Median step-to-step {channel_name} update")
    if relative_image is not None and update_image is not None:
        relative_ticks = np.arange(int(relative_log_limits[0]), int(relative_log_limits[1]) + 1)
        update_ticks = np.arange(int(update_log_limits[0]), int(update_log_limits[1]) + 1)
        figure.colorbar(
            relative_image,
            ax=axes[:, 0],
            orientation="horizontal",
            fraction=0.025,
            pad=0.04,
            ticks=relative_ticks,
            format=FuncFormatter(lambda value, _: rf"$10^{{{int(value)}}}$"),
            label="Per-channel relative L2 (clipped at 1e-4 and 10)",
        )
        figure.colorbar(
            update_image,
            ax=axes[:, 1],
            orientation="horizontal",
            fraction=0.025,
            pad=0.04,
            ticks=update_ticks,
            format=FuncFormatter(lambda value, _: rf"$10^{{{int(value)}}}$"),
            label="Per-channel step RMS (clipped at 1e-6 and 1e-1)",
        )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_hidden_channel_dynamics(
    path: Path,
    metrics: dict[str, dict[str, np.ndarray]],
    conditions: tuple[ConditionSpec, ...],
    steps: list[int],
    max_step: int | None = None,
) -> None:
    channel_count = next(iter(metrics.values()))["relative_l2"].shape[-1] if metrics else 0
    _plot_channel_dynamics(
        path,
        metrics,
        conditions,
        steps,
        "hidden",
        tuple(str(index) for index in range(channel_count)),
        max_step,
    )


def plot_visible_channel_dynamics(
    path: Path,
    metrics: dict[str, dict[str, np.ndarray]],
    conditions: tuple[ConditionSpec, ...],
    steps: list[int],
    max_step: int | None = None,
) -> None:
    _plot_channel_dynamics(
        path,
        metrics,
        conditions,
        steps,
        "visible",
        ("R", "G", "B", "A"),
        max_step,
    )


def _rgba_on_white(state: torch.Tensor) -> np.ndarray:
    rgba = state[:4].detach().cpu().permute(1, 2, 0).numpy()
    alpha = np.clip(rgba[..., 3:4], 0.0, 1.0)
    return np.clip(rgba[..., :3] + 1.0 - alpha, 0.0, 1.0)


def plot_snapshots(
    path: Path,
    snapshots: dict[str, dict[int, torch.Tensor | None]],
    conditions: tuple[ConditionSpec, ...],
    snapshot_steps: tuple[int, ...],
    include_all: bool = False,
    title: str | None = None,
) -> None:
    if not snapshots:
        return
    condition_by_name = _condition_map(conditions)
    preferred = [condition.name for condition in conditions] if include_all else ["baseline"]
    if any(condition.name == "hidden_freeze" for condition in conditions):
        preferred.append("hidden_freeze")
    selectors = {
        "restore": reversed(conditions),
        "periodic_reset": iter(conditions),
    }
    for family, candidates in selectors.items():
        selected = next(
            (condition.name for condition in candidates if condition.family == family), None
        )
        if selected is not None:
            preferred.append(selected)
    names = list(dict.fromkeys(name for name in preferred if name in snapshots))
    figure, axes = plt.subplots(
        len(names),
        len(snapshot_steps),
        figsize=(2.2 * len(snapshot_steps), 2.0 * len(names)),
        squeeze=False,
    )
    for row, name in enumerate(names):
        for column, step in enumerate(snapshot_steps):
            axis = axes[row, column]
            state = snapshots[name].get(step)
            if state is not None:
                axis.imshow(_rgba_on_white(state))
            elif step in snapshots[name]:
                axis.text(
                    0.5,
                    0.5,
                    "Numerically\ndiverged",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"t={step}")
            if column == 0:
                axis.set_ylabel(condition_by_name[name].label)
    if title:
        figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_development(
    path: Path,
    snapshots: dict[int, torch.Tensor],
    target: torch.Tensor,
) -> None:
    if not snapshots:
        return
    steps = sorted(snapshots)
    figure, axes = plt.subplots(1, len(steps) + 1, figsize=(2.2 * (len(steps) + 1), 2.4))
    for axis, step in zip(axes[:-1], steps, strict=True):
        axis.imshow(_rgba_on_white(snapshots[step]))
        axis.set_title(f"t={step}")
        axis.set_xticks([])
        axis.set_yticks([])
    axes[-1].imshow(_rgba_on_white(target[0]))
    axes[-1].set_title("Target")
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


# Full-state temporal PCA.
@dataclass(frozen=True)
class TemporalAttractorSettings:
    horizon: int = 10_000
    dense_until: int = 250
    record_every: int = 10
    crop_size: int = 0
    pca_components: int = 12
    pca_seed: int = 40_000

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0 <= self.dense_until <= self.horizon:
            raise ValueError("dense_until must be between zero and horizon")
        if self.record_every <= 0 or self.crop_size < 0:
            raise ValueError("record_every must be positive; crop_size must be nonnegative")
        if self.pca_components < 3:
            raise ValueError("pca_components must be at least three")


@dataclass(frozen=True)
class StateTrajectory:
    steps: np.ndarray
    features: np.ndarray
    divergence_step: int | None
    in_regime: np.ndarray | None = None
    first_failure_step: int | None = None


@dataclass(frozen=True)
class Projection:
    scaler: Any
    pca: Any
    coordinates: np.ndarray


def _record_step(step: int, settings: TemporalAttractorSettings) -> bool:
    return (
        step <= settings.dense_until
        or step % settings.record_every == 0
        or step == settings.horizon
    )


def _state_feature(
    state: torch.Tensor,
    crop_size: int,
) -> np.ndarray:
    if state.ndim == 3:
        state = state.unsqueeze(0)
    if state.ndim != 4 or state.shape[0] != 1:
        raise ValueError("state must have shape [1, C, H, W] or [C, H, W]")
    height, width = state.shape[-2:]
    if crop_size == 0:
        return state[0].detach().cpu().numpy().reshape(-1).astype(np.float32)
    if crop_size < 0:
        raise ValueError("crop_size must be nonnegative")
    if crop_size > height or crop_size > width:
        raise ValueError("crop_size exceeds the NCA substrate")
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    crop = state[0, :, top : top + crop_size, left : left + crop_size]
    return crop.detach().cpu().numpy().reshape(-1).astype(np.float32)


@torch.no_grad()
def _record_condition(
    model,
    mature_state: torch.Tensor,
    condition: ConditionSpec,
    config: ExperimentConfig,
    settings: TemporalAttractorSettings,
    continuation_seed: int,
    intervention_reference_state: torch.Tensor | None = None,
    evaluator: MorphologyEvaluator | None = None,
    regime: TargetRegime | None = None,
    feature_path: Path | None = None,
) -> StateTrajectory:
    state = mature_state.clone()
    intervention_reference_state = (
        mature_state if intervention_reference_state is None else intervention_reference_state
    )
    seed_everything(continuation_seed)
    steps: list[int] = []
    sample_count = sum(_record_step(step, settings) for step in range(settings.horizon + 1))
    shape = (sample_count, _state_feature(state, settings.crop_size).size)
    features = (
        np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32, shape=shape)
        if feature_path is not None
        else np.empty(shape, dtype=np.float32)
    )
    flags: list[bool] = []
    previous_state = None
    divergence_step: int | None = None
    if condition.family == "morphology_freeze" and evaluator is None:
        raise ValueError("morphology freeze PCA recording requires a target evaluator")
    for step in range(settings.horizon + 1):
        values = evaluator.evaluate(state, previous_state) if evaluator is not None else None
        if not bool(torch.isfinite(state).all().item()) or (
            values is not None and not all(math.isfinite(value) for value in values.values())
        ):
            divergence_step = step
            if regime is not None:
                flags.append(False)
            break
        if regime is not None:
            if values is None:
                raise ValueError("regime classification requires a morphology evaluator")
            flags.append(regime.contains(values))
        if _record_step(step, settings):
            features[len(steps)] = _state_feature(state, settings.crop_size)
            steps.append(step)
        if step == settings.horizon:
            break
        previous_state = state
        state = model(state, update_probability=config.model.update_probability)
        state = apply_post_step_intervention(
            state,
            intervention_reference_state,
            condition,
            step + 1,
            target_mask=None if evaluator is None else evaluator.target_mask,
        )
    if not steps:
        raise ValueError(f"{condition.name}: the initial mature state has non-finite measurements")
    outcome = (
        survival_outcome(flags, config.evaluation.failure_patience, divergence_step)
        if flags
        else None
    )
    return StateTrajectory(
        np.asarray(steps, dtype=np.int64),
        features[: len(steps)],
        divergence_step,
        np.asarray(flags, dtype=bool) if flags else None,
        None if outcome is None else outcome.first_failure_step,
    )


def _fit_projection(
    matrix: np.ndarray | tuple[np.ndarray, ...],
    components: int,
    random_state: int,
    workspace: Path | None = None,
) -> Projection:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    matrices = (matrix,) if isinstance(matrix, np.ndarray) else matrix
    samples = sum(part.shape[0] for part in matrices)
    if (
        not matrices
        or samples < 3
        or any(part.ndim != 2 or part.shape[1] != matrices[0].shape[1] for part in matrices)
    ):
        raise ValueError("PCA needs at least three samples with matching feature dimensions")
    if matrices[0].shape[1] < 3:
        raise ValueError("PCA visualization requires at least three features")
    scaler = StandardScaler()
    for part in matrices:
        for start in range(0, len(part), 64):
            block = np.asarray(part[start : start + 64], dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("PCA matrix contains non-finite values")
            scaler.partial_fit(block)
    shape = (samples, matrices[0].shape[1])
    scaled = (
        np.lib.format.open_memmap(
            workspace / "scaled_pca.npy", mode="w+", dtype=np.float64, shape=shape
        )
        if workspace is not None
        else np.empty(shape, dtype=np.float64)
    )
    offset = 0
    for part in matrices:
        for start in range(0, len(part), 64):
            block = part[start : start + 64]
            scaled[offset : offset + len(block)] = scaler.transform(
                np.asarray(block, dtype=np.float64)
            )
            offset += len(block)
    component_count = min(components, samples, shape[1])
    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=random_state,
        copy=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in divide", category=RuntimeWarning
        )
        coordinates = pca.fit_transform(scaled)
    pca.explained_variance_ratio_ = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0)
    return Projection(scaler, pca, coordinates)


def _transform_with_residual(
    features: np.ndarray,
    projection: Projection,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.empty((len(features), projection.pca.n_components_))
    residual = np.empty(len(features))
    for start in range(0, len(features), 64):
        block = slice(start, start + 64)
        scaled = projection.scaler.transform(np.asarray(features[block], dtype=np.float64))
        coordinates[block] = projection.pca.transform(scaled)
        reconstructed = projection.pca.inverse_transform(coordinates[block])
        residual[block] = np.sqrt(np.mean(np.square(scaled - reconstructed), axis=1))
    return coordinates, residual


def _save_projection(path: Path, projection: Projection) -> None:
    save_npz(
        path,
        scaler_mean=projection.scaler.mean_,
        scaler_scale=projection.scaler.scale_,
        scaler_var=projection.scaler.var_,
        components=projection.pca.components_,
        pca_mean=projection.pca.mean_,
        explained_variance=projection.pca.explained_variance_,
        explained_variance_ratio=projection.pca.explained_variance_ratio_,
    )


def _nearest_step_index(steps: np.ndarray, requested: int) -> int:
    return int(np.abs(steps - requested).argmin())


def _trajectory_legend_handles(
    reference_label: str,
    reference_as_points: bool,
) -> tuple[Line2D, ...]:
    reference = (
        Line2D(
            [0],
            [0],
            color="#8A8A8A",
            marker="o",
            linestyle="None",
            markersize=5,
            alpha=0.55,
            label=reference_label,
        )
        if reference_as_points
        else Line2D(
            [0],
            [0],
            color="#777777",
            linestyle="--",
            linewidth=1.2,
            label=reference_label,
        )
    )
    return (
        reference,
        Line2D(
            [0],
            [0],
            color="#B73779",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Trajectory state (color encodes step)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle="None",
            markersize=6,
            label="H* (intervention start)",
        ),
        Line2D(
            [0],
            [0],
            color="#0072B2",
            marker="x",
            linestyle="None",
            markersize=7,
            label="Last recorded state",
        ),
        Line2D(
            [0],
            [0],
            color="#D55E00",
            marker="D",
            markerfacecolor="white",
            linestyle="None",
            markersize=6,
            label="First sustained morphology failure",
        ),
    )


def _add_trajectory_legend(
    figure,
    reference_label: str,
    reference_as_points: bool,
) -> None:
    figure.legend(
        handles=_trajectory_legend_handles(reference_label, reference_as_points),
        loc="outside lower center",
        ncols=3,
        fontsize=8,
        frameon=True,
    )


def _draw_trajectory(
    axis,
    steps: np.ndarray,
    coordinates: np.ndarray,
    x_index: int,
    y_index: int,
    normalization: Normalize,
    failure_step: int | None,
):
    axis.plot(
        coordinates[:, x_index],
        coordinates[:, y_index],
        color="#555555",
        linewidth=0.7,
        alpha=0.55,
    )
    points = axis.scatter(
        coordinates[:, x_index],
        coordinates[:, y_index],
        c=steps,
        cmap="plasma",
        norm=normalization,
        s=13,
        linewidths=0,
        zorder=2,
    )
    axis.scatter(
        coordinates[0, x_index],
        coordinates[0, y_index],
        color="black",
        s=38,
        zorder=4,
    )
    axis.scatter(
        coordinates[-1, x_index],
        coordinates[-1, y_index],
        color="#0072B2",
        marker="x",
        s=42,
        zorder=4,
    )
    if failure_step is not None and steps[0] <= failure_step <= steps[-1]:
        failure_index = _nearest_step_index(steps, failure_step)
        axis.scatter(
            coordinates[failure_index, x_index],
            coordinates[failure_index, y_index],
            facecolor="white",
            edgecolor="#D55E00",
            marker="D",
            s=42,
            linewidth=1.2,
            zorder=5,
        )
    return points


def _projection_axes(condition_count: int):
    columns = min(condition_count, 3)
    blocks = (condition_count + columns - 1) // columns
    figure, axes = plt.subplots(
        2 * blocks,
        columns,
        figsize=(5.0 * columns, 6.2 * blocks + 1),
        squeeze=False,
        layout="constrained",
    )
    for index in range(condition_count, blocks * columns):
        block, column = divmod(index, columns)
        for row in (0, 1):
            axes[2 * block + row, column].set_visible(False)
    return figure, axes, columns


def _plot_reference_projection(
    path: Path,
    conditions: tuple[ConditionSpec, ...],
    reference_coordinates: np.ndarray,
    trajectories: dict[str, StateTrajectory],
    coordinates: dict[str, np.ndarray],
    projection: Projection,
    failures: dict[str, int | None],
    horizon: int,
) -> None:
    figure, axes, columns = _projection_axes(len(conditions))
    normalization = Normalize(0, horizon)
    points = None
    pairs = ((0, 1), (0, 2))
    ratio = projection.pca.explained_variance_ratio_ * 100.0
    for index, condition in enumerate(conditions):
        block, column = divmod(index, columns)
        for row, (x_index, y_index) in enumerate(pairs):
            axis = axes[2 * block + row, column]
            axis.scatter(
                reference_coordinates[:, x_index],
                reference_coordinates[:, y_index],
                color="#8A8A8A",
                s=10,
                alpha=0.28,
                label="Target-regime reference" if row == 0 and column == 0 else None,
            )
            points = _draw_trajectory(
                axis,
                trajectories[condition.name].steps,
                coordinates[condition.name],
                x_index,
                y_index,
                normalization,
                failures.get(condition.name),
            )
            axis.set_xlabel(f"PC{x_index + 1} ({ratio[x_index]:.1f}%)")
            axis.set_ylabel(f"PC{y_index + 1} ({ratio[y_index]:.1f}%)")
            axis.grid(alpha=0.18)
            if row == 0:
                axis.set_title(condition.label, fontsize=11)
    if points is not None:
        figure.colorbar(points, ax=axes, label="Steps after H*", shrink=0.82)
    figure.suptitle("Trajectories in the target-regime reference PCA", fontsize=15)
    _add_trajectory_legend(figure, "Target-regime PCA reference", reference_as_points=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _plot_pairwise_projections(
    path: Path,
    conditions: tuple[ConditionSpec, ...],
    condition_coordinates: dict[str, np.ndarray],
    baseline_coordinates: dict[str, np.ndarray],
    projections: dict[str, Projection],
    trajectories: dict[str, StateTrajectory],
    failures: dict[str, int | None],
    horizon: int,
) -> None:
    figure, axes, columns = _projection_axes(len(conditions))
    normalization = Normalize(0, horizon)
    points = None
    for index, condition in enumerate(conditions):
        block, column = divmod(index, columns)
        ratio = projections[condition.name].pca.explained_variance_ratio_ * 100.0
        for row, (x_index, y_index) in enumerate(((0, 1), (0, 2))):
            axis = axes[2 * block + row, column]
            if condition.name != "baseline":
                axis.plot(
                    baseline_coordinates[condition.name][:, x_index],
                    baseline_coordinates[condition.name][:, y_index],
                    color="#777777",
                    linewidth=1.0,
                    alpha=0.75,
                    linestyle="--",
                )
                axis.scatter(
                    baseline_coordinates[condition.name][0, x_index],
                    baseline_coordinates[condition.name][0, y_index],
                    color="black",
                    s=28,
                    zorder=4,
                )
            points = _draw_trajectory(
                axis,
                trajectories[condition.name].steps,
                condition_coordinates[condition.name],
                x_index,
                y_index,
                normalization,
                failures.get(condition.name),
            )
            axis.set_xlabel(f"PC{x_index + 1} ({ratio[x_index]:.1f}%)")
            axis.set_ylabel(f"PC{y_index + 1} ({ratio[y_index]:.1f}%)")
            axis.grid(alpha=0.18)
            if row == 0:
                axis.set_title(condition.label, fontsize=11)
    if points is not None:
        figure.colorbar(points, ax=axes, label="Steps after H*", shrink=0.82)
    figure.suptitle(
        "Pairwise PCA: control reference and intervention trajectory",
        fontsize=15,
    )
    _add_trajectory_legend(figure, "Control trajectory", reference_as_points=False)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _plot_pairwise_3d(
    path: Path,
    conditions: tuple[ConditionSpec, ...],
    condition_coordinates: dict[str, np.ndarray],
    baseline_coordinates: dict[str, np.ndarray],
    projections: dict[str, Projection],
    trajectories: dict[str, StateTrajectory],
    failures: dict[str, int | None],
    horizon: int,
) -> None:
    columns = min(len(conditions), 3)
    rows = (len(conditions) + columns - 1) // columns
    figure = plt.figure(figsize=(5.5 * columns, 4.8 * rows + 1), layout="constrained")
    normalization = Normalize(0, horizon)
    points = None
    for index, condition in enumerate(conditions, start=1):
        axis = figure.add_subplot(rows, columns, index, projection="3d")
        if condition.name != "baseline":
            reference = baseline_coordinates[condition.name]
            axis.plot(
                reference[:, 0],
                reference[:, 1],
                reference[:, 2],
                color="#777777",
                linewidth=0.9,
                alpha=0.7,
                linestyle="--",
            )
        coordinates = condition_coordinates[condition.name]
        steps = trajectories[condition.name].steps
        axis.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            color="#555555",
            linewidth=0.6,
            alpha=0.5,
        )
        points = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=steps,
            cmap="plasma",
            norm=normalization,
            s=10,
            depthshade=False,
        )
        axis.scatter(*coordinates[0, :3], color="black", s=34)
        axis.scatter(*coordinates[-1, :3], color="#0072B2", marker="x", s=38)
        failure_step = failures.get(condition.name)
        if failure_step is not None and steps[0] <= failure_step <= steps[-1]:
            failure_index = _nearest_step_index(steps, failure_step)
            axis.scatter(
                *coordinates[failure_index, :3],
                facecolor="white",
                edgecolor="#D55E00",
                marker="D",
                s=40,
                linewidth=1.2,
            )
        axis.set_title(condition.label, fontsize=11)
        ratio = projections[condition.name].pca.explained_variance_ratio_ * 100
        axis.set_xlabel(f"PC1 ({ratio[0]:.1f}%)")
        axis.set_ylabel(f"PC2 ({ratio[1]:.1f}%)")
        axis.set_zlabel(f"PC3 ({ratio[2]:.1f}%)")
        axis.view_init(elev=24, azim=42)
    if points is not None:
        figure.colorbar(points, ax=figure.axes, label="Steps after H*", shrink=0.72)
    figure.suptitle("Pairwise temporal attractor geometry", fontsize=15)
    _add_trajectory_legend(figure, "Control trajectory", reference_as_points=False)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _resolved_conditions(
    config: ExperimentConfig,
    requested: tuple[str, ...] | None,
) -> tuple[ConditionSpec, ...]:
    available = {condition.name: condition for condition in build_conditions(config.interventions)}
    if requested is None:
        return tuple(available.values())
    if len(requested) != len(set(requested)):
        raise ValueError("requested conditions must be unique")
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"conditions are absent from the resolved experiment: {missing}")
    if "baseline" not in requested:
        raise ValueError("baseline must be included as the shared reference")
    return tuple(available[name] for name in requested)


def run_temporal_attractor_analysis(
    results_dir: str | Path,
    output_dir: str | Path,
    replicate: int,
    condition_names: tuple[str, ...] | None,
    settings: TemporalAttractorSettings,
    device_name: str | None = None,
) -> Path:
    with (
        tempfile.TemporaryDirectory(prefix="nca-temporal-pca-") as temporary,
        threadpool_limits(limits=1),
    ):
        return _run_temporal_attractor_analysis(
            results_dir,
            output_dir,
            replicate,
            condition_names,
            settings,
            device_name,
            Path(temporary),
        )


def _run_temporal_attractor_analysis(
    results_dir: str | Path,
    output_dir: str | Path,
    replicate: int,
    condition_names: tuple[str, ...] | None,
    settings: TemporalAttractorSettings,
    device_name: str | None,
    workspace: Path,
) -> Path:
    settings.validate()
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    config, _, recorded, _ = _load_population(results_dir)
    conditions = _resolved_conditions(config, condition_names)
    source_metadata = read_json(results_dir / "metadata.json")
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
    }
    if any(str(source_metadata[name]) != value for name, value in runtime.items()):
        raise ValueError("PCA replay requires the original numerical environment")
    hashes = {name: file_sha256(getattr(config, name)) for name in ("checkpoint", "target")}
    for name, digest in hashes.items():
        expected = source_metadata.get(f"{name}_sha256")
        if expected is not None and expected != digest:
            raise ValueError(f"{name} differs from the file used in {results_dir}")
    device = get_device(device_name or config.device)
    if str(device) != source_metadata["device"]:
        raise ValueError("PCA replay must use the original experiment device")
    model = load_model(config, device)
    mature_states = {
        payload["metadata"]["replicate"]: {
            "state": torch.from_numpy(payload["mature_state"]),
            "mature_step": payload["metadata"]["mature_step"],
            "development_seed": payload["metadata"]["development_seed"],
            "development_states": {
                int(step): torch.from_numpy(state)
                for step, state in zip(
                    payload["development_reference_steps"],
                    payload["development_states"],
                    strict=True,
                )
            },
        }
        for payload in recorded
    }
    if replicate not in mature_states:
        raise ValueError(f"replicate {replicate} did not attain the target")

    selected = mature_states[replicate]
    mature_state = selected["state"].unsqueeze(0).to(device)
    development_states = selected.get("development_states", {})
    continuation_seed = (
        config.evaluation.seed_start + config.evaluation.continuation_seed_offset + replicate
    )
    source_failures = _condition_failures(results_dir, replicate)
    regime = TargetRegime(**read_json(results_dir / "target_regime.json"))
    target = load_distill_target(
        config.target, config.model.target_size, config.model.target_padding, device
    )
    _state_feature(mature_state, settings.crop_size)
    references = {
        condition.name: resolve_intervention_reference(mature_state, condition, development_states)
        for condition in conditions
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories: dict[str, StateTrajectory] = {}
    for condition in conditions:
        intervention_reference_state = references[condition.name]
        print(f"Recording {condition.label}...", flush=True)
        trajectory = _record_condition(
            model,
            mature_state,
            condition,
            config,
            settings,
            continuation_seed,
            intervention_reference_state=intervention_reference_state,
            evaluator=MorphologyEvaluator(
                target, config.model.alpha_threshold, mature_state, intervention_reference_state
            ),
            regime=regime,
            feature_path=workspace / f"{condition.name}.npy",
        )
        trajectories[condition.name] = trajectory

    baseline = trajectories["baseline"]
    failures = {name: trajectory.first_failure_step for name, trajectory in trajectories.items()}
    mature_features = np.stack(
        [
            _state_feature(record["state"], settings.crop_size)
            for _, record in sorted(mature_states.items())
        ]
    )
    baseline_reference_mask = baseline.in_regime[baseline.steps]
    reference_matrix = (mature_features, baseline.features[baseline_reference_mask])
    reference_projection = _fit_projection(
        reference_matrix,
        settings.pca_components,
        settings.pca_seed,
        workspace,
    )
    _save_projection(output_dir / "reference_pca_model.npz", reference_projection)
    reference_coordinates = reference_projection.coordinates
    reference_coordinates_by_condition: dict[str, np.ndarray] = {}
    reference_residuals: dict[str, np.ndarray] = {}
    for condition in conditions:
        coordinates, residual = _transform_with_residual(
            trajectories[condition.name].features,
            reference_projection,
        )
        reference_coordinates_by_condition[condition.name] = coordinates
        reference_residuals[condition.name] = residual
    _save_coordinates(
        output_dir / "reference_pca_coordinates.npz",
        trajectories,
        reference_coordinates_by_condition,
        reference_residuals,
    )
    _plot_reference_projection(
        output_dir / "reference_pca_trajectories.png",
        conditions,
        reference_coordinates,
        trajectories,
        reference_coordinates_by_condition,
        reference_projection,
        failures,
        settings.horizon,
    )

    pairwise_condition_coordinates: dict[str, np.ndarray] = {}
    pairwise_baseline_coordinates: dict[str, np.ndarray] = {}
    pairwise_projections: dict[str, Projection] = {}
    for offset, condition in enumerate(conditions):
        condition_trajectory = trajectories[condition.name]
        fit_matrix = (
            baseline.features
            if condition.name == "baseline"
            else (baseline.features, condition_trajectory.features)
        )
        projection = _fit_projection(
            fit_matrix,
            settings.pca_components,
            settings.pca_seed + offset + 1,
            workspace,
        )
        pairwise_projections[condition.name] = projection
        pairwise_baseline_coordinates[condition.name], _ = _transform_with_residual(
            baseline.features, projection
        )
        pairwise_condition_coordinates[condition.name], _ = _transform_with_residual(
            condition_trajectory.features, projection
        )
        _save_projection(
            output_dir / f"pairwise_pca_model__{condition.name}.npz",
            projection,
        )

    _save_coordinates(
        output_dir / "pairwise_pca_coordinates.npz",
        trajectories,
        pairwise_condition_coordinates,
        baseline_coordinates=pairwise_baseline_coordinates,
    )
    _plot_pairwise_projections(
        output_dir / "pairwise_pca_trajectories.png",
        conditions,
        pairwise_condition_coordinates,
        pairwise_baseline_coordinates,
        pairwise_projections,
        trajectories,
        failures,
        settings.horizon,
    )
    _plot_pairwise_3d(
        output_dir / "pairwise_pca_3d.png",
        conditions,
        pairwise_condition_coordinates,
        pairwise_baseline_coordinates,
        pairwise_projections,
        trajectories,
        failures,
        settings.horizon,
    )

    metadata = {
        "source_results": str(results_dir.resolve()),
        "source_run_signature": source_metadata["run_signature"],
        "replicate": replicate,
        "mature_step": int(selected["mature_step"]),
        "development_seed": int(selected["development_seed"]),
        "continuation_seed": continuation_seed,
        "conditions": [condition.name for condition in conditions],
        "settings": settings.__dict__,
        "effective_update_probability": config.model.update_probability,
        "device": str(device),
        "source_device": source_metadata.get("device"),
        "checkpoint_sha256": hashes["checkpoint"],
        "target_sha256": hashes["target"],
        "source_first_failure_steps": source_failures,
        "replayed_first_failure_steps": failures,
        "regime_flags_and_markers": "Recomputed at every step of the replay, using the saved target thresholds.",
        "representation": {
            "channels": f"all {config.model.channels} NCA channels",
            "spatial_region": (
                f"full {mature_state.shape[-2]}x{mature_state.shape[-1]} substrate, including background"
                if settings.crop_size == 0
                else f"center {settings.crop_size}x{settings.crop_size} crop"
            ),
            "features": baseline.features.shape[1],
            "pooling": False,
        },
        "reference_fit": {
            "mature_states": len(mature_features),
            "baseline_in_regime_states": int(baseline_reference_mask.sum()),
            "standardized": True,
        },
        "interpretation": (
            "Reference PCA uses the mature-state ensemble and the selected control trajectory "
            "while it remains in the target regime. Pairwise PCA models are fitted separately "
            "and their axes are not directly comparable between conditions. A compact PCA "
            "trajectory alone is not evidence of attraction. Time sampling is dense initially "
            "and sparser later; this affects the weight of each phase in the pairwise fit."
        ),
        "trajectory_source": "Independent paired replay from saved H*, not the stored RGBA snapshots; failure markers are recomputed.",
        "raw_features_saved": False,
    }
    save_json(output_dir / "metadata.json", metadata)
    return output_dir


def _snapshot_replica_scores(metrics: dict) -> np.ndarray:
    """Equal-weight, IQR-scaled distance to temporal medians across all conditions."""
    scores = np.zeros(next(iter(metrics.values()))["alpha_iou"].shape[0])
    for condition in metrics.values():
        condition, _ = _mask_invalid_replicates(condition)
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


def _plot_final_gallery(
    path: Path,
    frames: dict,
    title: str,
    step: int,
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
                "Numerically\ndiverged",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=8,
            )
        else:
            axis.imshow(_rgba_on_white(state))
        axis.set_title(f"Replica {replicate:03d}", fontsize=9)
    figure.suptitle(f"{title} | t={step} | n={len(frames)}")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=160)
    plt.close(figure)


# Saved outputs and CLI workflow.
def _save_coordinates(path, trajectories, coordinates, residuals=None, baseline_coordinates=None):
    payload = {"condition_names": np.asarray(list(trajectories))}
    for name, trajectory in trajectories.items():
        payload[f"{name}__steps"] = trajectory.steps
        payload[f"{name}__coordinates"] = coordinates[name]
        if residuals is not None:
            payload[f"{name}__residual"] = residuals[name]
        if baseline_coordinates is not None:
            payload[f"{name}__control_coordinates"] = baseline_coordinates[name]
    save_npz(path, **payload)


def save_npz(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)
    except OSError as error:
        if error.errno == errno.ENOSPC:
            error.add_note("Free disk space and retry: completed replicas are preserved.")
        raise
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError as error:
        if error.errno in (errno.ECANCELED, errno.ETIMEDOUT):
            error.add_note(f"Download this file locally before retrying: {path.resolve()}")
        raise


def load_payload(folder: Path) -> dict[str, Any]:
    with np.load(folder / "trajectories.npz", allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["metadata"] = json.loads(str(payload["metadata"]))
    return payload


def _pack_replica(results, mature, development, development_snapshots, config, metadata):
    steps = sorted(results[0].snapshots)
    shape = (4, *mature.shape[-2:])
    frames = np.stack(
        [
            np.stack(
                [
                    np.full(shape, np.nan, dtype=np.float32)
                    if result.snapshots[step] is None
                    else result.snapshots[step].numpy()
                    for step in steps
                ]
            )
            for result in results
        ]
    )
    reference_steps = sorted(development)
    development_steps = sorted(development_snapshots)
    return {
        "metadata": np.asarray(json.dumps(metadata)),
        "condition_names": np.asarray([result.summary["condition"] for result in results]),
        "metric_names": np.asarray(METRIC_NAMES),
        "channel_metric_names": np.asarray(HIDDEN_CHANNEL_METRIC_NAMES),
        "steps": np.arange(config.evaluation.horizon + 1),
        "metrics": np.stack(
            [
                np.stack([result.metrics[name] for name in METRIC_NAMES], axis=-1)
                for result in results
            ]
        ),
        "in_regime": np.stack([result.in_regime for result in results]),
        "hidden_steps": np.asarray(results[0].channel_steps),
        "visible_steps": np.asarray(results[0].visible_channel_steps),
        "hidden_channels": np.stack(
            [
                np.stack(
                    [result.channel_metrics[name] for name in HIDDEN_CHANNEL_METRIC_NAMES], axis=-1
                )
                for result in results
            ]
        ),
        "visible_channels": np.stack(
            [
                np.stack(
                    [result.visible_channel_metrics[name] for name in VISIBLE_CHANNEL_METRIC_NAMES],
                    axis=-1,
                )
                for result in results
            ]
        ),
        "snapshot_steps": np.asarray(steps),
        "snapshots": frames,
        "mature_state": mature[0].detach().cpu().numpy(),
        "development_reference_steps": np.asarray(reference_steps, dtype=np.int64),
        "development_states": np.stack(
            [development[step][0].detach().cpu().numpy() for step in reference_steps]
        )
        if reference_steps
        else np.empty((0, *mature.shape[1:]), dtype=np.float32),
        "development_snapshot_steps": np.asarray(development_steps),
        "development_snapshots": np.stack(
            [development_snapshots[step].numpy() for step in development_steps]
        ),
    }


def _mask_invalid_payload(payload: dict[str, Any]) -> None:
    """Apply saved failure signals in memory; never rewrite the raw archives."""
    valid = np.ones(payload["metrics"].shape[:2], dtype=bool)
    names = payload["condition_names"].tolist()
    for summary in payload["metadata"].get("summaries", []):
        failure = summary.get("numerical_divergence_step")
        if failure not in (None, ""):
            valid[names.index(summary["condition"]), int(failure) :] = False
    if not np.isfinite(payload["mature_state"]).all():
        valid[:] = False
    sampled_fields = (
        ("hidden_channels", "hidden_steps"),
        ("visible_channels", "visible_steps"),
        ("snapshots", "snapshot_steps"),
    )
    for name, steps in sampled_fields:
        values = payload[name]
        finite = np.isfinite(values).all(axis=tuple(range(2, values.ndim)))
        for index in range(len(names)):
            failures = payload[steps][~finite[index]]
            if failures.size:
                valid[index, int(failures.min()) :] = False
    masked, valid = _mask_invalid_replicates({"metrics": payload["metrics"]}, valid)
    payload["metrics"] = masked["metrics"]
    for name, steps in sampled_fields:
        sampled = valid[:, payload[steps]]
        shape = sampled.shape + (1,) * (payload[name].ndim - 2)
        payload[name] = np.where(sampled.reshape(shape), payload[name], np.nan)


def _load_population(folder: Path):
    metadata = read_json(folder / "metadata.json")
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError(
            f"Old Experiment 1 format in {folder}; start a fresh run in an empty output root"
        )
    if (
        str(metadata.get("device", "")).startswith("mps")
        and metadata.get("measurement_revision") != MEASUREMENT_REVISION
    ):
        raise ValueError(
            f"Unsafe saved MPS measurements in {folder}: the old CPU/float64 transfer can corrupt "
            "finite metrics. Plotting cannot reconstruct the missing per-step states. "
            "Preserve these archives and rerun with a new --output-root."
        )
    config = load_experiment_config(folder / "resolved_config.json")
    conditions = build_conditions(config.interventions)
    names = [item.name for item in conditions]
    included, excluded = [], []
    for replicate in range(config.evaluation.replicates):
        path = folder / f"replicate_{replicate:03d}"
        if not (path / "trajectories.npz").is_file():
            raise ValueError(
                f"Replica {replicate} is incomplete; resume the run before analyzing {folder}"
            )
        payload = load_payload(path)
        record = payload["metadata"]
        if record["replicate"] != replicate or record["run_signature"] != metadata["run_signature"]:
            raise ValueError(f"Replica metadata does not match the experiment: {path}")
        if record["status"] == "excluded":
            excluded.append(record)
        else:
            if payload["condition_names"].tolist() != names:
                raise ValueError(f"Condition order differs: {path}")
            if payload["metric_names"].tolist() != list(METRIC_NAMES):
                raise ValueError(f"Metric columns differ: {path}")
            _mask_invalid_payload(payload)
            included.append(payload)
    if not included:
        raise ValueError(f"No replica attained the target in {folder}")
    return config, conditions, included, excluded


def _population_metrics(payloads, conditions):
    return {
        condition.name: {
            name: np.stack([payload["metrics"][index, :, column] for payload in payloads])
            for column, name in enumerate(METRIC_NAMES)
        }
        for index, condition in enumerate(conditions)
    }


def analyze_folder(folder: Path) -> None:
    _, _, payloads, excluded = _load_population(folder)
    write_csv(
        folder / "per_run_summary.csv",
        [
            {
                **summary,
                "development_seed": payload["metadata"]["development_seed"],
                "continuation_seed": payload["metadata"]["continuation_seed"],
                "mature_step": payload["metadata"]["mature_step"],
            }
            for payload in payloads
            for summary in payload["metadata"]["summaries"]
        ],
    )
    save_json(folder / "excluded_replicates.json", {"replicas": excluded})


def run_experiment(config: ExperimentConfig) -> Path:
    config.validate()
    folder = Path(config.output_dir)
    device = get_device(config.device)
    config = replace(config, device=str(device))
    inputs = _metadata(config, device)
    settings = config.to_dict()
    settings.pop("output_dir")
    signature = {
        "protocol": PROTOCOL,
        "measurement_revision": MEASUREMENT_REVISION,
        "config": settings,
        **{
            name: inputs[name]
            for name in ("checkpoint_sha256", "target_sha256", "device", "python", "torch", "numpy")
        },
    }
    run_signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if folder.exists() and any(folder.iterdir()):
        if not (folder / "metadata.json").is_file():
            raise FileExistsError(
                f"Output is not an Experiment 1 run: {folder}; choose an empty output root"
            )
        previous = read_json(folder / "metadata.json")
        if previous.get("run_signature") != run_signature:
            raise ValueError(f"Protocol or inputs differ from {folder}; choose a new output root")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        save_json(folder / "resolved_config.json", config.to_dict())
        save_json(
            folder / "metadata.json",
            {
                **inputs,
                "protocol": PROTOCOL,
                "run_signature": run_signature,
                "mask": "Fixed target alpha mask; morphology freeze affects hidden channels only.",
            },
        )
    conditions = build_conditions(config.interventions)
    write_csv(
        folder / "conditions.csv",
        [
            {
                **asdict(condition),
                "spatial_scope": "target_mask"
                if condition.family == "morphology_freeze"
                else "full_grid",
            }
            for condition in conditions
        ],
    )
    model = load_model(config, device)
    target = load_distill_target(
        config.target, config.model.target_size, config.model.target_padding, device
    )
    if (folder / "target_regime.json").is_file():
        regime = TargetRegime(**read_json(folder / "target_regime.json"))
    else:
        print(
            f"{folder.name}: calibrating {config.calibration.replicates} independent rollouts",
            flush=True,
        )
        regime, observations = _calibrate_regime(model, target, config, device)
        save_npz(folder / "calibration_metrics.npz", **observations)
        save_json(folder / "target_regime.json", regime.to_dict())
    for replicate in range(config.evaluation.replicates):
        result_dir = folder / f"replicate_{replicate:03d}"
        path = result_dir / "trajectories.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                recorded = json.loads(str(archive["metadata"]))
            if recorded["replicate"] != replicate or recorded["run_signature"] != run_signature:
                raise ValueError(f"Saved replica does not match this run: {path}")
            print(f"Already recorded: {folder.name}, replica {replicate:03d}", flush=True)
            continue
        development_seed = config.evaluation.seed_start + replicate
        continuation_seed = development_seed + config.evaluation.continuation_seed_offset
        mature, mature_step, development_snapshots, development, reason = _develop_to_target(
            model, target, regime, config, device, development_seed
        )
        metadata = {
            "replicate": replicate,
            "run_signature": run_signature,
            "development_seed": development_seed,
            "continuation_seed": continuation_seed,
        }
        if mature is None:
            metadata.update(status="excluded", reason=reason)
            save_npz(path, metadata=np.asarray(json.dumps(metadata)))
            print(
                f"{folder.name}: replica {replicate:03d} did not attain target; excluded",
                flush=True,
            )
            continue
        results = []
        for condition in conditions:
            reference = resolve_intervention_reference(mature, condition, development)
            print(
                f"{folder.name}, replica {replicate + 1}/{config.evaluation.replicates}: {condition.label}",
                flush=True,
            )
            results.append(
                _run_branch(
                    model,
                    target,
                    mature,
                    reference,
                    regime,
                    condition,
                    config,
                    replicate,
                    continuation_seed,
                    collect_snapshots=True,
                )
            )
        metadata.update(
            status="complete",
            mature_step=mature_step,
            summaries=[result.summary for result in results],
        )
        save_npz(
            path,
            **_pack_replica(results, mature, development, development_snapshots, config, metadata),
        )
    analyze_folder(folder)
    return folder


def select_representative_replicate(results_dir: str | Path) -> int:
    _, conditions, payloads, _ = _load_population(Path(results_dir))
    scores = _snapshot_replica_scores(_population_metrics(payloads, conditions))
    return int(payloads[int(np.argmin(scores))]["metadata"]["replicate"])


def _condition_failures(results_dir: Path, replicate: int) -> dict:
    payload = load_payload(results_dir / f"replicate_{replicate:03d}")
    return {
        row["condition"]: None
        if row["first_failure_step"] == ""
        else int(row["first_failure_step"])
        for row in payload["metadata"]["summaries"]
    }


def _saved_snapshot_frames(payload, condition_index, steps):
    indices = {int(step): index for index, step in enumerate(payload["snapshot_steps"])}
    missing = set(steps) - indices.keys()
    if missing:
        raise ValueError(f"Requested snapshot steps were not recorded: {sorted(missing)}")
    return {
        step: torch.from_numpy(frame) if np.isfinite(frame).all() else None
        for step in steps
        for frame in [payload["snapshots"][condition_index, indices[step]]]
    }


def plot_folder(folder: Path, *, robust: bool = True) -> None:
    config, conditions, payloads, _ = _load_population(folder)
    if file_sha256(config.target) != read_json(folder / "metadata.json")["target_sha256"]:
        raise ValueError(
            "Target image changed since this run; restore the original target before plotting"
        )
    output = folder / "analysis"
    output.mkdir(exist_ok=True)
    metrics = _population_metrics(payloads, conditions)
    _plot_core_morphology_metrics(
        output / "core_morphology_metrics.png",
        metrics,
        conditions,
        TargetRegime(**read_json(folder / "target_regime.json")),
        robust=robust,
    )
    for scope, plotter in (
        ("hidden", plot_hidden_channel_dynamics),
        ("visible", plot_visible_channel_dynamics),
    ):
        steps = payloads[0][f"{scope}_steps"].astype(int).tolist()
        if any(payload[f"{scope}_steps"].tolist() != steps for payload in payloads):
            raise ValueError(f"Sampling steps differ between {scope} archives")
        channels = {
            condition.name: {
                name: np.stack(
                    [payload[f"{scope}_channels"][index, :, :, column] for payload in payloads]
                )
                for column, name in enumerate(HIDDEN_CHANNEL_METRIC_NAMES)
            }
            for index, condition in enumerate(conditions)
        }
        plotter(output / f"{scope}_channel_dynamics.png", channels, conditions, steps)
    scores = _snapshot_replica_scores(metrics)
    selected = payloads[int(np.argmin(scores))]
    selected_id = selected["metadata"]["replicate"]
    steps = tuple(selected["snapshot_steps"].astype(int))
    plot_snapshots(
        output / "morphology_snapshots.png",
        {
            condition.name: _saved_snapshot_frames(selected, index, steps)
            for index, condition in enumerate(conditions)
        },
        conditions,
        steps,
        include_all=True,
        title=f"{folder.name}: paired replica {selected_id:03d}, nearest median trajectories",
    )
    plot_development(
        output / "development_to_target.png",
        {
            int(step): torch.from_numpy(frame)
            for step, frame in zip(
                selected["development_snapshot_steps"],
                selected["development_snapshots"],
                strict=True,
            )
        },
        load_distill_target(
            config.target,
            config.model.target_size,
            config.model.target_padding,
            torch.device("cpu"),
        ),
    )
    write_csv(
        output / "representative_selection.csv",
        [
            {
                "replicate": payload["metadata"]["replicate"],
                "distance_score": float(score),
                "selected": payload is selected,
            }
            for payload, score in zip(payloads, scores, strict=True)
        ],
    )
    for index, condition in enumerate(conditions):
        frames = {
            payload["metadata"]["replicate"]: _saved_snapshot_frames(
                payload, index, (config.evaluation.horizon,)
            )[config.evaluation.horizon]
            for payload in payloads
        }
        _plot_final_gallery(
            output / f"final_t{config.evaluation.horizon}__{condition.name}.png",
            frames,
            f"{folder.name}: {condition.label}",
            config.evaluation.horizon,
        )
    print(f"Saved all core, channel and snapshot figures: {output}", flush=True)


def run(args: argparse.Namespace) -> None:
    base = load_experiment_config(args.config)
    for morphology in args.morphologies:
        config = replace(
            base,
            checkpoint=str(Path(args.checkpoint_root) / morphology / "model.pt"),
            target=str(Path(args.data_dir) / f"{morphology}.png"),
            output_dir=str(Path(args.output_root) / morphology),
            device=args.device or base.device,
            evaluation=replace(base.evaluation, replicates=args.replicates),
        )
        with threadpool_limits(limits=1):
            run_experiment(config)


def analyze(args: argparse.Namespace) -> None:
    for morphology in args.morphologies:
        analyze_folder(Path(args.results_root) / morphology)


def plot(args: argparse.Namespace) -> None:
    for morphology in args.morphologies:
        plot_folder(Path(args.results_root) / morphology, robust=args.summary == "median-quantiles")


def analyze_attractor(args: argparse.Namespace) -> None:
    if Path(args.analysis_name).name != args.analysis_name or args.analysis_name in {".", ".."}:
        raise ValueError("analysis-name must be a single folder name")
    for morphology in args.morphologies:
        folder = Path(args.results_root) / morphology
        config, _, payloads, _ = _load_population(folder)
        replicas = (
            [payload["metadata"]["replicate"] for payload in payloads]
            if args.all_replicates
            else [
                args.replicate
                if args.replicate is not None
                else select_representative_replicate(folder)
            ]
        )
        settings = TemporalAttractorSettings(
            horizon=config.evaluation.horizon,
            dense_until=min(250, config.evaluation.horizon),
            record_every=args.record_every,
            crop_size=args.crop_size,
            pca_components=args.pca_components,
        )
        settings.validate()
        for replicate in replicas:
            output = (
                folder
                / "analysis"
                / "attractor"
                / args.analysis_name
                / f"replicate_{replicate:03d}"
            )
            print(f"PCA: {morphology}, replica {replicate:03d}, every saved condition", flush=True)
            conditions = tuple(item.name for item in build_conditions(config.interventions))
            marker = output / "metadata.json"
            if marker.is_file():
                completed = read_json(marker)
                signature = read_json(folder / "metadata.json")["run_signature"]
                figures = (
                    "reference_pca_trajectories.png",
                    "pairwise_pca_trajectories.png",
                    "pairwise_pca_3d.png",
                )
                if (
                    completed.get("source_run_signature") == signature
                    and completed.get("settings") == asdict(settings)
                    and completed.get("conditions") == list(conditions)
                    and completed.get("replicate") == replicate
                    and all((output / name).is_file() for name in figures)
                ):
                    print(f"Already completed: {output}", flush=True)
                    continue
            run_temporal_attractor_analysis(
                folder, output, replicate, conditions, settings, args.device
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 1: paired hidden-state interventions in Growing NCA."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="Run all eight conditions; resume completed replicas."
    )
    run_parser.add_argument("--config", default="configs/experiment_1.yaml")
    run_parser.add_argument("--data-dir", default="data")
    run_parser.add_argument("--checkpoint-root", default="checkpoints/growing")
    run_parser.add_argument("--output-root", default="reports/experiment_1")
    run_parser.add_argument("--replicates", type=int, default=30)
    run_parser.add_argument("--device")
    run_parser.set_defaults(handler=run)
    for command, handler in (
        ("analyze", analyze),
        ("plot", plot),
        ("attractor", analyze_attractor),
    ):
        help_text = {
            "analyze": "Write per-run results and exclusions from completed replicas.",
            "plot": "Generate core metrics, channel activity, representative snapshots and final galleries.",
            "attractor": "Generate full-state reference and pairwise PCA for every condition.",
        }
        subparser = commands.add_parser(command, help=help_text[command])
        subparser.add_argument("--results-root", default="reports/experiment_1")
        subparser.set_defaults(handler=handler)
        if command == "plot":
            subparser.add_argument(
                "--summary", choices=("median-quantiles", "mean-sd"), default="median-quantiles"
            )
        if command == "attractor":
            replicas = subparser.add_mutually_exclusive_group()
            replicas.add_argument("--replicate", type=int)
            replicas.add_argument("--all-replicates", action="store_true")
            subparser.add_argument(
                "--crop-size",
                type=int,
                default=0,
                help="0 includes the entire substrate and background.",
            )
            subparser.add_argument("--record-every", type=int, default=10)
            subparser.add_argument("--pca-components", type=int, default=12)
            subparser.add_argument("--analysis-name", default="temporal_full_state")
            subparser.add_argument("--device")
    for subparser in commands.choices.values():
        subparser.add_argument(
            "--morphologies", nargs="+", choices=MORPHOLOGIES, default=MORPHOLOGIES
        )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
