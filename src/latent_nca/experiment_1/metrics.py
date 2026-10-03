from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import torch

MORPHOLOGY_METRIC_NAMES = (
    "target_mse",
    "alpha_iou",
    "area_ratio",
    "target_region_mse",
)

METRIC_NAMES = MORPHOLOGY_METRIC_NAMES + (
    "hidden_relative_l2",
    "hidden_reference_relative_l2",
    "hidden_rms_delta",
    "hidden_step_rms",
)

CHANNEL_METRIC_NAMES = (
    "relative_l2",
    "rms_delta",
    "step_rms",
)
HIDDEN_CHANNEL_METRIC_NAMES = CHANNEL_METRIC_NAMES
VISIBLE_CHANNEL_METRIC_NAMES = CHANNEL_METRIC_NAMES


@dataclass(frozen=True)
class TargetRegime:
    target_mse_max: float
    alpha_iou_min: float
    area_ratio_min: float
    area_ratio_max: float
    acceptance_quantile: float

    @classmethod
    def calibrate(
        cls,
        records: Mapping[str, np.ndarray],
        acceptance_quantile: float,
    ) -> TargetRegime:
        lower_quantile = 1.0 - acceptance_quantile
        return cls(
            target_mse_max=float(np.quantile(records["target_mse"], acceptance_quantile)),
            alpha_iou_min=float(np.quantile(records["alpha_iou"], lower_quantile)),
            area_ratio_min=float(np.quantile(records["area_ratio"], lower_quantile)),
            area_ratio_max=float(np.quantile(records["area_ratio"], acceptance_quantile)),
            acceptance_quantile=acceptance_quantile,
        )

    def contains(self, metrics: Mapping[str, float]) -> bool:
        return (
            metrics["target_mse"] <= self.target_mse_max
            and metrics["alpha_iou"] >= self.alpha_iou_min
            and self.area_ratio_min <= metrics["area_ratio"] <= self.area_ratio_max
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SurvivalOutcome:
    survival_time: int
    event_observed: bool
    first_failure_step: int | None


class MorphologyEvaluator:
    def __init__(
        self,
        target: torch.Tensor,
        alpha_threshold: float = 0.1,
        mature_state: torch.Tensor | None = None,
        hidden_reference_state: torch.Tensor | None = None,
    ) -> None:
        self.target = target[:, :4]
        self.alpha_threshold = alpha_threshold
        self.target_mask = self.target[:, 3:4] > alpha_threshold
        self.target_area = self.target_mask.sum().clamp_min(1)
        self.mature_state = mature_state
        self.mature_hidden_norm = None
        if mature_state is not None:
            self.mature_hidden_norm = mature_state[:, 4:].norm().clamp_min(1e-8)
        self.hidden_reference_state = (
            mature_state if hidden_reference_state is None else hidden_reference_state
        )
        self.hidden_reference_norm = None
        if self.hidden_reference_state is not None:
            self.hidden_reference_norm = self.hidden_reference_state[:, 4:].norm().clamp_min(1e-8)

    @torch.no_grad()
    def evaluate(
        self,
        state: torch.Tensor,
        previous_state: torch.Tensor | None = None,
    ) -> dict[str, float]:
        visible = state[:, :4]
        state_mask = visible[:, 3:4] > self.alpha_threshold
        intersection = (state_mask & self.target_mask).sum().float()
        union = (state_mask | self.target_mask).sum().float().clamp_min(1.0)
        target_region = self.target_mask.expand_as(visible)
        target_region_mse = (visible - self.target).square().mul(
            target_region
        ).sum() / target_region.sum().clamp_min(1)
        values = torch.stack(
            (
                (visible - self.target).square().mean(),
                intersection / union,
                state_mask.sum().float() / self.target_area,
                target_region_mse,
            )
        )
        result = {
            "target_mse": float(values[0].cpu()),
            "alpha_iou": float(values[1].cpu()),
            "area_ratio": float(values[2].cpu()),
            "target_region_mse": float(values[3].cpu()),
            "hidden_relative_l2": 0.0,
            "hidden_reference_relative_l2": 0.0,
            "hidden_rms_delta": 0.0,
            "hidden_step_rms": 0.0,
        }
        if self.mature_state is not None and self.mature_hidden_norm is not None:
            hidden_delta = state[:, 4:] - self.mature_state[:, 4:]
            result["hidden_relative_l2"] = float(
                (hidden_delta.norm() / self.mature_hidden_norm).cpu()
            )
            result["hidden_rms_delta"] = float(hidden_delta.square().mean().sqrt().cpu())
        if self.hidden_reference_state is not None and self.hidden_reference_norm is not None:
            reference_delta = state[:, 4:] - self.hidden_reference_state[:, 4:]
            result["hidden_reference_relative_l2"] = float(
                (reference_delta.norm() / self.hidden_reference_norm).cpu()
            )
        if previous_state is not None:
            step_delta = state[:, 4:] - previous_state[:, 4:]
            result["hidden_step_rms"] = float(step_delta.square().mean().sqrt().cpu())
        return result


def _per_channel_metrics(
    current: torch.Tensor,
    mature: torch.Tensor,
    previous: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    delta = current - mature
    delta_norm = delta.flatten(start_dim=2).norm(dim=2)
    reference_norm = mature.flatten(start_dim=2).norm(dim=2).clamp_min(1e-8)
    rms_delta = delta.square().mean(dim=(0, 2, 3)).sqrt()
    step_rms = torch.zeros_like(rms_delta)
    if previous is not None:
        step_delta = current - previous
        step_rms = step_delta.square().mean(dim=(0, 2, 3)).sqrt()
    return {
        "relative_l2": (delta_norm / reference_norm).mean(dim=0),
        "rms_delta": rms_delta,
        "step_rms": step_rms,
    }


@torch.no_grad()
def hidden_channel_metrics(
    state: torch.Tensor,
    mature_state: torch.Tensor,
    previous_state: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Measure each hidden channel against H* and against the preceding state."""
    previous = None if previous_state is None else previous_state[:, 4:]
    return _per_channel_metrics(state[:, 4:], mature_state[:, 4:], previous)


@torch.no_grad()
def visible_channel_metrics(
    state: torch.Tensor,
    mature_state: torch.Tensor,
    previous_state: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Measure R, G, B, and alpha against H* and against the preceding state."""
    previous = None if previous_state is None else previous_state[:, :4]
    return _per_channel_metrics(state[:, :4], mature_state[:, :4], previous)


def first_sustained_true(flags: list[bool] | np.ndarray, patience: int) -> int | None:
    run = 0
    for index, value in enumerate(flags):
        run = run + 1 if bool(value) else 0
        if run >= patience:
            return index
    return None


def survival_outcome(
    in_regime: list[bool] | np.ndarray,
    patience: int,
    terminal_failure_step: int | None = None,
) -> SurvivalOutcome:
    flags = np.asarray(in_regime, dtype=bool)
    if flags.ndim != 1 or flags.size == 0:
        raise ValueError("in_regime must be a non-empty 1D sequence")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if terminal_failure_step is not None and not 0 <= terminal_failure_step < flags.size:
        raise ValueError("terminal_failure_step must index in_regime")

    sustained_failure: int | None = None
    run = 0
    for index in range(1, flags.size):
        run = run + 1 if not flags[index] else 0
        if run >= patience:
            sustained_failure = index - patience + 1
            break

    observed_failures = [
        step for step in (sustained_failure, terminal_failure_step) if step is not None
    ]
    if observed_failures:
        first_failure = min(observed_failures)
        return SurvivalOutcome(first_failure, True, first_failure)
    return SurvivalOutcome(flags.size - 1, False, None)
