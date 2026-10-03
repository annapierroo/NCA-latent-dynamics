from __future__ import annotations

from pathlib import Path

import numpy as np

from .conditions import ConditionSpec
from .metrics import HIDDEN_CHANNEL_METRIC_NAMES, VISIBLE_CHANNEL_METRIC_NAMES


def _save_channel_metrics(
    path: Path,
    metrics: dict[str, dict[str, np.ndarray]],
    metric_names: tuple[str, ...],
    steps: list[int],
    replicate_ids: list[int],
    channel_labels: tuple[str, ...] | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "steps": np.asarray(steps, dtype=np.int64),
        "replicate_ids": np.asarray(replicate_ids, dtype=np.int64),
        "condition_names": np.asarray(list(metrics)),
        "metric_names": np.asarray(metric_names),
    }
    if channel_labels is not None:
        payload["channel_labels"] = np.asarray(channel_labels)
    for condition, condition_metrics in metrics.items():
        for name, values in condition_metrics.items():
            payload[f"{condition}__{name}"] = values
    np.savez_compressed(path, **payload)


def save_hidden_channel_metrics(
    path: Path,
    metrics: dict[str, dict[str, np.ndarray]],
    steps: list[int],
    replicate_ids: list[int],
) -> None:
    _save_channel_metrics(
        path,
        metrics,
        HIDDEN_CHANNEL_METRIC_NAMES,
        steps,
        replicate_ids,
    )


def save_visible_channel_metrics(
    path: Path,
    metrics: dict[str, dict[str, np.ndarray]],
    steps: list[int],
    replicate_ids: list[int],
) -> None:
    _save_channel_metrics(
        path,
        metrics,
        VISIBLE_CHANNEL_METRIC_NAMES,
        steps,
        replicate_ids,
        channel_labels=("R", "G", "B", "A"),
    )


def _mean_over_replicates(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0, dtype=np.float64)
    return np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else float("nan")


def summarize_hidden_channels(
    metrics: dict[str, dict[str, np.ndarray]],
    conditions: tuple[ConditionSpec, ...],
    steps: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in conditions:
        relative = _mean_over_replicates(metrics[condition.name]["relative_l2"])
        rms_delta = _mean_over_replicates(metrics[condition.name]["rms_delta"])
        step_rms = _mean_over_replicates(metrics[condition.name]["step_rms"])
        for channel in range(relative.shape[1]):
            frozen = condition.family == "freeze" or (
                condition.family == "partial_freeze"
                and condition.hidden_indices is not None
                and channel in condition.hidden_indices
            )
            series = relative[:, channel]
            finite_indices = np.flatnonzero(np.isfinite(series))
            last_index = int(finite_indices[-1]) if finite_indices.size else None
            rows.append(
                {
                    "condition": condition.name,
                    "label": condition.label,
                    "family": condition.family,
                    "hidden_channel": channel,
                    "tensor_channel": channel + 4,
                    "frozen_by_design": frozen,
                    "mean_relative_l2": _finite_mean(series),
                    "max_mean_relative_l2": _finite_max(series),
                    "final_relative_l2": float(relative[-1, channel]),
                    "last_finite_relative_l2": (
                        "" if last_index is None else float(series[last_index])
                    ),
                    "last_finite_step": "" if last_index is None else steps[last_index],
                    "mean_rms_delta": _finite_mean(rms_delta[:, channel]),
                    "max_mean_rms_delta": _finite_max(rms_delta[:, channel]),
                    "mean_step_rms": _finite_mean(step_rms[:, channel]),
                }
            )
    return rows


def summarize_visible_channels(
    metrics: dict[str, dict[str, np.ndarray]],
    conditions: tuple[ConditionSpec, ...],
    steps: list[int],
) -> list[dict[str, object]]:
    labels = ("R", "G", "B", "A")
    rows: list[dict[str, object]] = []
    for condition in conditions:
        relative = _mean_over_replicates(metrics[condition.name]["relative_l2"])
        rms_delta = _mean_over_replicates(metrics[condition.name]["rms_delta"])
        step_rms = _mean_over_replicates(metrics[condition.name]["step_rms"])
        for channel, label in enumerate(labels):
            series = relative[:, channel]
            finite_indices = np.flatnonzero(np.isfinite(series))
            last_index = int(finite_indices[-1]) if finite_indices.size else None
            rows.append(
                {
                    "condition": condition.name,
                    "label": condition.label,
                    "family": condition.family,
                    "visible_channel": label,
                    "tensor_channel": channel,
                    "mean_relative_l2": _finite_mean(series),
                    "max_mean_relative_l2": _finite_max(series),
                    "final_relative_l2": float(relative[-1, channel]),
                    "last_finite_relative_l2": (
                        "" if last_index is None else float(series[last_index])
                    ),
                    "last_finite_step": "" if last_index is None else steps[last_index],
                    "mean_rms_delta": _finite_mean(rms_delta[:, channel]),
                    "max_mean_rms_delta": _finite_max(rms_delta[:, channel]),
                    "mean_step_rms": _finite_mean(step_rms[:, channel]),
                }
            )
    return rows


__all__ = [
    "save_hidden_channel_metrics",
    "save_visible_channel_metrics",
    "summarize_hidden_channels",
    "summarize_visible_channels",
]
