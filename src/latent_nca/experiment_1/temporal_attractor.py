from __future__ import annotations

import csv
import json
import math
import tempfile
import warnings
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
from threadpoolctl import threadpool_limits

from latent_nca.artifacts import file_sha256
from latent_nca.data.device import get_device, seed_everything
from latent_nca.data.images import load_distill_target

from .conditions import (
    ConditionSpec,
    apply_post_step_intervention,
    build_conditions,
    resolve_intervention_reference,
)
from .config import ExperimentConfig, load_experiment_config
from .metrics import MorphologyEvaluator, TargetRegime, survival_outcome
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
        if self.record_every <= 0 or self.crop_size < 0:
            raise ValueError("record_every must be positive; crop_size must be nonnegative")
        if self.pca_components < 3:
            raise ValueError("pca_components must be at least three")
        if self.update_probability is not None and not 0.0 < self.update_probability <= 1.0:
            raise ValueError("update_probability must be in (0, 1]")


@dataclass(frozen=True)
class StateTrajectory:
    steps: np.ndarray
    features: np.ndarray
    divergence_step: int | None
    in_regime: np.ndarray | None = None
    first_failure_step: int | None = None


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
    update_probability = (
        config.model.update_probability
        if settings.update_probability is None
        else settings.update_probability
    )
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
        state = model(state, update_probability=update_probability)
        state = apply_post_step_intervention(
            state,
            intervention_reference_state,
            condition,
            step + 1,
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
    config = load_experiment_config(results_dir / "resolved_config.json")
    conditions = _resolved_conditions(config, condition_names)
    source_metadata_path = results_dir / "metadata.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text()) if source_metadata_path.exists() else {}
    )
    hashes = {name: file_sha256(getattr(config, name)) for name in ("checkpoint", "target")}
    for name, digest in hashes.items():
        expected = source_metadata.get(f"{name}_sha256")
        if expected is not None and expected != digest:
            raise ValueError(f"{name} differs from the file used in {results_dir}")
    device = get_device(device_name or config.device)
    model = load_model(config, device)
    mature_states: dict[int, dict[str, Any]] = torch.load(
        results_dir / "mature_states.pt",
        map_location="cpu",
        weights_only=True,
    )
    if replicate not in mature_states:
        raise ValueError(f"replicate {replicate} is absent from mature_states.pt")

    selected = mature_states[replicate]
    mature_state = selected["state"].unsqueeze(0).to(device)
    development_states = selected.get("development_states", {})
    continuation_seed = (
        config.evaluation.seed_start + config.evaluation.continuation_seed_offset + replicate
    )
    source_failures = _condition_failures(results_dir, replicate)
    regime = TargetRegime(**json.loads((results_dir / "target_regime.json").read_text()))
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
        np.savez_compressed(
            output_dir / f"full_state_features__{condition.name}.npz",
            steps=trajectory.steps,
            features=trajectory.features,
            divergence_step=np.asarray(
                -1 if trajectory.divergence_step is None else trajectory.divergence_step
            ),
            in_regime=trajectory.in_regime,
        )

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
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output_dir


__all__ = [
    "StateTrajectory",
    "TemporalAttractorSettings",
    "run_temporal_attractor_analysis",
    "select_representative_replicate",
]
