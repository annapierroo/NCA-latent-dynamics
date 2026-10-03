from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from latent_nca.data.device import get_device, seed_everything

from .conditions import (
    ConditionSpec,
    apply_post_step_intervention,
    build_conditions,
    resolve_intervention_reference,
)
from .config import ExperimentConfig, load_experiment_config
from .model import load_model

plt.switch_backend("Agg")


@dataclass(frozen=True)
class TemporalAttractorSettings:
    horizon: int = 10_000
    dense_until: int = 250
    record_every: int = 10
    crop_size: int = 40
    pca_components: int = 12
    pca_seed: int = 40_000
    update_probability: float | None = None

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0 <= self.dense_until <= self.horizon:
            raise ValueError("dense_until must be between zero and horizon")
        if self.record_every <= 0 or self.crop_size <= 0:
            raise ValueError("record_every and crop_size must be positive")
        if self.pca_components < 3:
            raise ValueError("pca_components must be at least three")
        if self.update_probability is not None and not 0.0 < self.update_probability <= 1.0:
            raise ValueError("update_probability must be in (0, 1]")


@dataclass(frozen=True)
class StateTrajectory:
    steps: np.ndarray
    features: np.ndarray
    divergence_step: int | None


@dataclass(frozen=True)
class Projection:
    scaler: StandardScaler
    pca: PCA
    coordinates: np.ndarray


def select_representative_replicate(results_dir: str | Path) -> int:
    rows: list[tuple[int, int]] = []
    with (Path(results_dir) / "per_run_summary.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == "baseline":
                rows.append((int(row["replicate"]), int(row["survival_time"])))
    if not rows:
        raise ValueError("per_run_summary.csv contains no baseline rows")
    median = float(np.median([survival for _, survival in rows]))
    return min(rows, key=lambda item: (abs(item[1] - median), item[0]))[0]


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
) -> StateTrajectory:
    state = mature_state.clone()
    intervention_reference_state = (
        mature_state if intervention_reference_state is None else intervention_reference_state
    )
    seed_everything(continuation_seed)
    steps: list[int] = []
    features: list[np.ndarray] = []
    divergence_step: int | None = None
    update_probability = (
        config.model.update_probability
        if settings.update_probability is None
        else settings.update_probability
    )
    for step in range(settings.horizon + 1):
        if _record_step(step, settings):
            if not bool(torch.isfinite(state).all().item()):
                divergence_step = step
                break
            steps.append(step)
            features.append(_state_feature(state, settings.crop_size))
        if step == settings.horizon:
            break
        state = model(state, update_probability=update_probability)
        state = apply_post_step_intervention(
            state,
            intervention_reference_state,
            condition,
            step + 1,
        )
    return StateTrajectory(
        np.asarray(steps, dtype=np.int64),
        np.stack(features),
        divergence_step,
    )


def _fit_projection(
    matrix: np.ndarray,
    components: int,
    random_state: int,
) -> Projection:
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        raise ValueError("PCA matrix must contain at least three samples")
    if not np.isfinite(matrix).all():
        raise ValueError("PCA matrix contains non-finite values")
    matrix = np.asarray(matrix, dtype=np.float64)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    component_count = min(components, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=random_state,
    )
    coordinates = pca.fit_transform(scaled)
    return Projection(scaler, pca, coordinates)


def _transform_with_residual(
    features: np.ndarray,
    projection: Projection,
) -> tuple[np.ndarray, np.ndarray]:
    scaled = projection.scaler.transform(np.asarray(features, dtype=np.float64))
    coordinates = projection.pca.transform(scaled)
    reconstructed = projection.pca.inverse_transform(coordinates)
    residual = np.sqrt(np.mean(np.square(scaled - reconstructed), axis=1))
    return coordinates, residual


def _save_projection(path: Path, projection: Projection) -> None:
    np.savez_compressed(
        path,
        scaler_mean=projection.scaler.mean_,
        scaler_scale=projection.scaler.scale_,
        scaler_var=projection.scaler.var_,
        components=projection.pca.components_,
        pca_mean=projection.pca.mean_,
        explained_variance=projection.pca.explained_variance_,
        explained_variance_ratio=projection.pca.explained_variance_ratio_,
    )


def _write_coordinates(
    path: Path,
    trajectories: dict[str, StateTrajectory],
    coordinates: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray] | None = None,
) -> None:
    component_count = next(iter(coordinates.values())).shape[1]
    fields = ["condition", "step"] + [f"pc{i + 1}" for i in range(component_count)]
    if residuals is not None:
        fields.append("reconstruction_rmse")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, trajectory in trajectories.items():
            for index, step in enumerate(trajectory.steps):
                row: dict[str, object] = {
                    "condition": name,
                    "step": int(step),
                    **{
                        f"pc{component + 1}": float(coordinates[name][index, component])
                        for component in range(component_count)
                    },
                }
                if residuals is not None:
                    row["reconstruction_rmse"] = float(residuals[name][index])
                writer.writerow(row)


def _condition_failures(results_dir: Path, replicate: int) -> dict[str, int | None]:
    failures: dict[str, int | None] = {}
    with (results_dir / "per_run_summary.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["replicate"]) != replicate:
                continue
            value = row["first_failure_step"]
            failures[row["condition"]] = None if value == "" else int(value)
    return failures


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
    if failure_step is not None:
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
    figure, axes = plt.subplots(
        2,
        len(conditions),
        figsize=(4.0 * len(conditions), 7.8),
        squeeze=False,
        layout="constrained",
    )
    normalization = Normalize(0, horizon)
    points = None
    pairs = ((0, 1), (0, 2))
    ratio = projection.pca.explained_variance_ratio_ * 100.0
    for column, condition in enumerate(conditions):
        for row, (x_index, y_index) in enumerate(pairs):
            axis = axes[row, column]
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
                axis.set_title(condition.label)
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
    figure, axes = plt.subplots(
        2,
        len(conditions),
        figsize=(4.0 * len(conditions), 7.8),
        squeeze=False,
        layout="constrained",
    )
    normalization = Normalize(0, horizon)
    points = None
    for column, condition in enumerate(conditions):
        ratio = projections[condition.name].pca.explained_variance_ratio_ * 100.0
        for row, (x_index, y_index) in enumerate(((0, 1), (0, 2))):
            axis = axes[row, column]
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
                axis.set_title(condition.label)
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
    trajectories: dict[str, StateTrajectory],
    failures: dict[str, int | None],
    horizon: int,
) -> None:
    figure = plt.figure(figsize=(4.2 * len(conditions), 4.5), layout="constrained")
    normalization = Normalize(0, horizon)
    points = None
    for column, condition in enumerate(conditions, start=1):
        axis = figure.add_subplot(1, len(conditions), column, projection="3d")
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
        if failure_step is not None:
            failure_index = _nearest_step_index(steps, failure_step)
            axis.scatter(
                *coordinates[failure_index, :3],
                facecolor="white",
                edgecolor="#D55E00",
                marker="D",
                s=40,
                linewidth=1.2,
            )
        axis.set_title(condition.label)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.set_zlabel("PC3")
        axis.view_init(elev=24, azim=42)
    if points is not None:
        figure.colorbar(points, ax=figure.axes, label="Steps after H*", shrink=0.72)
    figure.suptitle("Pairwise temporal attractor geometry", fontsize=15)
    _add_trajectory_legend(figure, "Control trajectory", reference_as_points=False)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _load_reference_flags(results_dir: Path, replicate: int) -> np.ndarray:
    with np.load(results_dir / "trajectory_metrics.npz") as archive:
        replicate_ids = archive["replicate_ids"].astype(int)
        matches = np.flatnonzero(replicate_ids == replicate)
        if matches.size != 1:
            raise ValueError(f"replicate {replicate} is absent or duplicated in trajectory metrics")
        return archive["baseline__in_regime"][int(matches[0])]


def _resolved_conditions(
    config: ExperimentConfig,
    requested: tuple[str, ...],
) -> tuple[ConditionSpec, ...]:
    available = {condition.name: condition for condition in build_conditions(config.interventions)}
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
    condition_names: tuple[str, ...],
    settings: TemporalAttractorSettings,
    device_name: str | None = None,
) -> Path:
    settings.validate()
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_experiment_config(results_dir / "resolved_config.json")
    conditions = _resolved_conditions(config, condition_names)
    device = get_device(device_name or config.device)
    model = load_model(config, device)
    mature_states: dict[int, dict[str, Any]] = torch.load(
        results_dir / "mature_states.pt",
        map_location="cpu",
    )
    if replicate not in mature_states:
        raise ValueError(f"replicate {replicate} is absent from mature_states.pt")

    selected = mature_states[replicate]
    mature_state = selected["state"].unsqueeze(0).to(device)
    development_states = selected.get("development_states", {})
    continuation_seed = (
        config.evaluation.seed_start + config.evaluation.continuation_seed_offset + replicate
    )
    failures = _condition_failures(results_dir, replicate)
    reference_flags = _load_reference_flags(results_dir, replicate)

    trajectories: dict[str, StateTrajectory] = {}
    for condition in conditions:
        intervention_reference_state = resolve_intervention_reference(
            mature_state,
            condition,
            development_states,
        )
        print(f"Recording {condition.label}...", flush=True)
        trajectory = _record_condition(
            model,
            mature_state,
            condition,
            config,
            settings,
            continuation_seed,
            intervention_reference_state=intervention_reference_state,
        )
        trajectories[condition.name] = trajectory
        np.savez_compressed(
            output_dir / f"full_state_features__{condition.name}.npz",
            steps=trajectory.steps,
            features=trajectory.features,
            divergence_step=np.asarray(
                -1 if trajectory.divergence_step is None else trajectory.divergence_step
            ),
        )

    baseline = trajectories["baseline"]
    mature_features = np.stack(
        [
            _state_feature(record["state"], settings.crop_size)
            for _, record in sorted(mature_states.items())
        ]
    )
    baseline_reference_mask = np.asarray(
        [step < reference_flags.size and bool(reference_flags[step]) for step in baseline.steps],
        dtype=bool,
    )
    reference_matrix = np.concatenate(
        (mature_features, baseline.features[baseline_reference_mask]),
        axis=0,
    )
    reference_projection = _fit_projection(
        reference_matrix,
        settings.pca_components,
        settings.pca_seed,
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
    _write_coordinates(
        output_dir / "reference_pca_coordinates.csv",
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
            else np.concatenate((baseline.features, condition_trajectory.features), axis=0)
        )
        projection = _fit_projection(
            fit_matrix,
            settings.pca_components,
            settings.pca_seed + offset + 1,
        )
        pairwise_projections[condition.name] = projection
        pairwise_baseline_coordinates[condition.name] = projection.pca.transform(
            projection.scaler.transform(np.asarray(baseline.features, dtype=np.float64))
        )
        pairwise_condition_coordinates[condition.name] = projection.pca.transform(
            projection.scaler.transform(np.asarray(condition_trajectory.features, dtype=np.float64))
        )
        _save_projection(
            output_dir / f"pairwise_pca_model__{condition.name}.npz",
            projection,
        )

    _write_coordinates(
        output_dir / "pairwise_pca_coordinates.csv",
        trajectories,
        pairwise_condition_coordinates,
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
        trajectories,
        failures,
        settings.horizon,
    )

    metadata = {
        "source_results": str(results_dir.resolve()),
        "replicate": replicate,
        "mature_step": int(selected["mature_step"]),
        "development_seed": int(selected["development_seed"]),
        "continuation_seed": continuation_seed,
        "conditions": [condition.name for condition in conditions],
        "settings": settings.__dict__,
        "effective_update_probability": (
            config.model.update_probability
            if settings.update_probability is None
            else settings.update_probability
        ),
        "device": str(device),
        "representation": {
            "channels": f"all {config.model.channels} NCA channels",
            "spatial_region": f"center {settings.crop_size}x{settings.crop_size} crop",
            "features": config.model.channels * settings.crop_size**2,
        },
        "reference_fit": {
            "mature_states": len(mature_features),
            "baseline_in_regime_states": int(baseline_reference_mask.sum()),
            "standardized": True,
        },
        "interpretation": (
            "Reference PCA uses the mature-state ensemble and the selected control trajectory "
            "while it remains in the target regime. Pairwise PCA models are fitted separately "
            "and their axes are not directly comparable between conditions."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output_dir


__all__ = [
    "StateTrajectory",
    "TemporalAttractorSettings",
    "run_temporal_attractor_analysis",
    "select_representative_replicate",
]
