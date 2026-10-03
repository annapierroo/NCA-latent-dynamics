from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from .conditions import ConditionSpec
from .metrics import TargetRegime

plt.switch_backend("Agg")


def _condition_map(conditions: tuple[ConditionSpec, ...]) -> dict[str, ConditionSpec]:
    return {condition.name: condition for condition in conditions}


def _finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=axis)
    sums = np.where(finite, values, 0.0).sum(axis=axis, dtype=np.float64)
    return np.divide(
        sums,
        counts,
        out=np.full(np.shape(sums), np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _finite_quantile(values: np.ndarray, quantile: float | tuple[float, ...]) -> np.ndarray:
    finite_values = np.where(np.isfinite(values), values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanquantile(finite_values, quantile, axis=0)


def _finite_mean_std_count(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    means = np.divide(
        np.where(finite, values, 0.0).sum(axis=0, dtype=np.float64),
        counts,
        out=np.full(values.shape[1:], np.nan, dtype=np.float64),
        where=counts > 0,
    )
    squared_deviations = np.where(finite, np.square(values - means), 0.0).sum(
        axis=0,
        dtype=np.float64,
    )
    standard_deviations = np.sqrt(
        np.divide(
            squared_deviations,
            counts - 1,
            out=np.full(values.shape[1:], np.nan, dtype=np.float64),
            where=counts > 1,
        )
    )
    return means, standard_deviations, counts


def _finite_median_quantiles_count(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    counts = np.isfinite(values).sum(axis=0)
    q05, q25, median, q75, q95 = _finite_quantile(
        values,
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
    colors = {name: cmap(index % cmap.N) for index, name in enumerate(names)}
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
        positive_centers = np.concatenate([statistics[name][metric][0] for name in names])
        positive_centers = positive_centers[
            np.isfinite(positive_centers) & (positive_centers > 0.0)
        ]
        positive_floor = (
            max(float(positive_centers.min()) * 0.5, np.finfo(np.float64).tiny)
            if positive_centers.size
            else np.finfo(np.float64).tiny
        )
        for name in names:
            if robust:
                center, q25, q75, q05, q95, counts = statistics[name][metric]
                bands = ((q05, q95, 0.06), (q25, q75, 0.17))
            else:
                center, standard_deviation, counts = statistics[name][metric]
                bands = ((center - standard_deviation, center + standard_deviation, 0.13),)
            steps = np.arange(center.size)
            color = colors[name]
            for lower, upper, alpha in bands:
                if metric in {"target_mse", "target_region_mse"}:
                    lower = np.maximum(lower, positive_floor)
                    upper = np.maximum(upper, positive_floor)
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
            axis.plot(
                steps,
                center,
                color=color,
                linewidth=1.45,
                label=legend_labels[name],
            )
        if metric in {"target_mse", "target_region_mse"}:
            axis.set_yscale("log")
            if metric == "target_mse":
                axis.axhline(regime.target_mse_max, color="#666666", linestyle="--")
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
        "Solid line: finite median. Dark band: 25th-75th percentiles (IQR). "
        "Light band: 5th-95th percentiles."
        if robust
        else "Solid line: finite mean. Shaded band: +/- 1 sample SD."
    )
    figure.text(
        0.5,
        0.008,
        f"{note} n labels show finite replicates at the first and final step.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, bottom_margin, 1.0, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


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
        plotted_steps[0],
        plotted_steps[-1],
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
    figure.tight_layout()
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
