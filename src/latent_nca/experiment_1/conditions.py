from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .config import InterventionConfig


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
        conditions.append(ConditionSpec("hidden_freeze", "Freeze", "freeze", "beta", 1.0))
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
) -> torch.Tensor:
    if condition.family == "baseline":
        return next_state
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
