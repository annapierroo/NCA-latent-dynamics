"""Persistent NCA versus Growing NCA with mature hidden-state restoration.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from latent_nca.artifacts import file_sha256, write_csv
from latent_nca.data.device import get_device, seed_everything
from latent_nca.data.images import load_distill_target
from latent_nca.models import GrowingNCA, load_growing_checkpoint
from latent_nca.training.growing import TrainingConfig, train_growing, train_persistent

MORPHOLOGIES = ("gecko", "butterfly")
SYSTEMS = ("persistent", "growing_restore")
BRANCHES = (
    "control",
    "independent_control",
    "noise_0p01",
    "noise_0p05",
    "noise_0p1",
    "erase_0p05",
    "erase_0p1",
)
LABELS = (
    "Control",
    "Independent update masks",
    "Hidden noise 1%",
    "Hidden noise 5%",
    "Hidden noise 10%",
    "Erase 5%",
    "Erase 10%",
)
COLORS = ("#343a40", "#889096", "#2a6fbb", "#12a89d", "#7761b2", "#d48c18", "#c44949")
GROUPS = {"hidden": slice(4, 16), "rgba": slice(0, 4)}
METRICS = (
    "target_mse",
    "target_region_mse",
    "alpha_iou",
    "area_ratio",
    "hidden_step_rms",
    "rgba_step_rms",
    "hidden_mature_rms",
    "rgba_mature_rms",
    "hidden_pair_rms",
    "rgba_pair_rms",
    "hidden_pair_relative",
    "rgba_pair_relative",
    "hidden_cloud_distance",
    "hidden_pca_residual",
    "rgba_cloud_distance",
    "rgba_pca_residual",
    "hidden_radius",
    "rgba_radius",
)
EPS = 1e-12


@dataclass(frozen=True)
class Settings:
    perturb_step: int = 500
    horizon: int = 10_000
    reference_start: int = 250
    mature_start: int = 64
    mature_stop: int = 96
    development_limit: int = 256
    attainment_patience: int = 5
    recovery_patience: int = 20
    calibration_replicates: int = 32
    acceptance_quantile: float = 0.95
    channels: int = 16
    hidden_size: int = 128
    target_size: int = 40
    padding: int = 16
    fire_rate: float = 0.5
    beta: float = 0.1
    alpha_threshold: float = 0.1
    pca_components: int = 12
    seed: int = 50_000
    snapshot_replicates: int = 3

    def validate(self) -> None:
        if not 0 <= self.reference_start < self.perturb_step < self.horizon:
            raise ValueError("Require 0 <= reference_start < perturb_step < horizon")
        if self.perturb_step - self.reference_start < 20:
            raise ValueError("The pre-pulse reference must contain at least 20 states")
        if not 0 <= self.mature_start <= self.mature_stop <= self.development_limit:
            raise ValueError("Invalid calibration/development interval")
        if min(self.calibration_replicates, self.attainment_patience, self.recovery_patience) < 1:
            raise ValueError("Replicate counts and patience must be positive")
        if not 0.5 < self.acceptance_quantile < 1 or not 0 < self.fire_rate <= 1:
            raise ValueError("Invalid acceptance quantile or update probability")
        if not 0 < self.beta < 1 or self.pca_components < 3:
            raise ValueError("Require 0 < beta < 1 and at least three PCA components")
        if self.channels != 16 or self.snapshot_replicates < 0:
            raise ValueError("This protocol uses 16 channels and nonnegative snapshot counts")


def save_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path: Path, **payload) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def checkpoint_path(root: Path, morphology: str, system: str, tag: str) -> Path:
    variant = "growing" if system == "growing_restore" else "persistent"
    folder = root / variant / morphology
    return (folder / tag if tag != "default" else folder) / "model.pt"


def seed_nca(seed: int, device: torch.device) -> None:
    seed_everything(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)


def capture_rng(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    accelerator = None
    if device.type == "mps":
        accelerator = torch.mps.get_rng_state()
    elif device.type == "cuda":
        accelerator = torch.cuda.get_rng_state(device)
    return torch.random.get_rng_state(), accelerator


def restore_rng(saved, device: torch.device) -> None:
    cpu, accelerator = saved
    torch.random.set_rng_state(cpu)
    if device.type == "mps":
        torch.mps.set_rng_state(accelerator)
    elif device.type == "cuda":
        torch.cuda.set_rng_state(accelerator, device)


def update(model, state: torch.Tensor, reference: torch.Tensor, system: str, cfg: Settings):
    result = model(state, update_probability=cfg.fire_rate)
    if system == "growing_restore":
        result[:, 4:] = torch.lerp(result[:, 4:], reference[:, 4:], cfg.beta)
    return result


def as_array(state: torch.Tensor) -> np.ndarray:
    return state[0].detach().cpu().numpy().astype(np.float64)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def morphology_metrics(state: np.ndarray, target: np.ndarray, threshold: float) -> np.ndarray:
    mask = target[3] > threshold
    alive = state[3] > threshold
    error = np.square(state[:4] - target[:4])
    return np.asarray(
        [
            error.mean(),
            error[:, mask].mean(),
            (alive & mask).sum() / max(int((alive | mask).sum()), 1),
            alive.sum() / max(int(mask.sum()), 1),
        ],
        dtype=np.float64,
    )


def inside_regime(values: np.ndarray, regime: dict) -> np.ndarray:
    return (
        np.isfinite(values[..., :4]).all(axis=-1)
        & (values[..., 0] <= regime["target_mse_max"])
        & (values[..., 2] >= regime["alpha_iou_min"])
        & (values[..., 3] >= regime["area_ratio_min"])
        & (values[..., 3] <= regime["area_ratio_max"])
    )


def first_sustained(flags: np.ndarray, patience: int) -> int | None:
    run = 0
    for index, value in enumerate(flags):
        run = run + 1 if value else 0
        if run >= patience:
            return index - patience + 1
    return None


def shared_regime(records: np.ndarray, quantile: float) -> dict:
    if not np.isfinite(records).all():
        raise ValueError("Non-finite calibration states; inspect the checkpoint before evaluation")
    return {
        "target_mse_max": float(np.quantile(records[..., 0], quantile)),
        "alpha_iou_min": float(np.quantile(records[..., 2], 1 - quantile)),
        "area_ratio_min": float(np.quantile(records[..., 3], 1 - quantile)),
        "area_ratio_max": float(np.quantile(records[..., 3], quantile)),
        "acceptance_quantile": quantile,
        "definition": "Equal numbers of independent development rollouts per system; "
        "within-rollout snapshots are temporally correlated. Shared per morphology/checkpoint tag. "
        "Quantiles are empirical, not accuracy guarantees.",
    }


@torch.no_grad()
def calibrate(models: dict, target: torch.Tensor, cfg: Settings) -> tuple[dict, np.ndarray]:
    records = []
    target_array = as_array(target)
    for system_index, system in enumerate(SYSTEMS):
        model_records = []
        model = models[system]
        for replicate in range(cfg.calibration_replicates):
            seed_nca(cfg.seed + 100_000 + replicate, target.device)
            state = model.seed(1, *target.shape[-2:], target.device)
            rows = []
            for step in range(cfg.mature_stop + 1):
                if step >= cfg.mature_start:
                    rows.append(
                        morphology_metrics(as_array(state), target_array, cfg.alpha_threshold)
                    )
                if step < cfg.mature_stop:
                    state = model(state, update_probability=cfg.fire_rate)
            model_records.append(rows)
        records.append(model_records)
        print(f"Calibration complete: {system_index + 1}/{len(SYSTEMS)} {system}", flush=True)
    observations = np.asarray(records)
    regime = shared_regime(observations, cfg.acceptance_quantile)
    regime["calibration_steps"] = [cfg.mature_start, cfg.mature_stop]
    return regime, observations


@torch.no_grad()
def develop(model, target: torch.Tensor, regime: dict, cfg: Settings, seed: int):
    seed_nca(seed, target.device)
    state = model.seed(1, *target.shape[-2:], target.device)
    target_array = as_array(target)
    consecutive = 0
    for step in range(cfg.development_limit + 1):
        if not torch.isfinite(state).all():
            return None, None, "nonfinite_during_development"
        if step >= cfg.mature_start:
            values = morphology_metrics(as_array(state), target_array, cfg.alpha_threshold)
            consecutive = consecutive + 1 if inside_regime(values, regime) else 0
            if consecutive >= cfg.attainment_patience:
                return state.clone(), step, None
        if step < cfg.development_limit:
            state = model(state, update_probability=cfg.fire_rate)
    return None, None, "target_not_attained"


@dataclass
class LocalPCA:
    mean: np.ndarray
    components: np.ndarray
    variance_ratio: np.ndarray
    scale: np.ndarray
    reference_scores: np.ndarray
    cloud_limit: float
    residual_limit: float
    effective_components: int
    mean_shift_in_radii: float
    speed_ratio: float

    def project(self, state: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        centered = (state / self.scale[:, None, None]).reshape(-1) - self.mean
        scores = centered @ self.components.T
        residual = float(
            np.sqrt(max(float(centered @ centered - scores @ scores), 0) / centered.size)
        )
        cloud = float(np.sqrt(np.min(np.mean((self.reference_scores - scores) ** 2, axis=1))))
        return scores, residual, cloud, rms(centered)


def fit_local_pca(reference: np.ndarray, n_components: int, seed: int) -> LocalPCA:
    from sklearn.decomposition import PCA

    split = len(reference) // 2
    training, validation = reference[:split], reference[split:]
    scale = np.sqrt(np.mean(training.astype(np.float64) ** 2, axis=(0, 2, 3)))
    scale = np.maximum(scale, max(float(scale.max()) * 1e-6, 1e-8))
    matrix = (training / scale[None, :, None, None]).reshape(split, -1)
    count = min(n_components, split - 1, matrix.shape[1])
    mean = matrix.mean(axis=0)
    if np.max(np.abs(matrix - mean)) < 1e-12:
        components = np.zeros((count, matrix.shape[1]))
        ratio = np.zeros(count)
    else:
        pca = PCA(n_components=count, svd_solver="randomized", random_state=seed)
        pca.fit(matrix)
        mean, components, ratio = pca.mean_, pca.components_, pca.explained_variance_ratio_
    fit_scores = (matrix - mean) @ components.T
    local = LocalPCA(mean, components, ratio, scale, fit_scores, 0, 0, 0, 0, 0)
    heldout = [local.project(frame) for frame in validation]
    local.cloud_limit = max(float(np.quantile([x[2] for x in heldout], 0.99)), 1e-8)
    local.residual_limit = max(float(np.quantile([x[1] for x in heldout], 0.99)), 1e-8)
    local.effective_components = int(np.searchsorted(np.cumsum(ratio), 0.95) + 1)
    local.effective_components = min(local.effective_components, count)
    validation_matrix = (validation / scale[None, :, None, None]).reshape(len(validation), -1)
    radius = rms(matrix - mean)
    local.mean_shift_in_radii = rms(validation_matrix.mean(axis=0) - mean) / max(radius, EPS)
    speed_train = rms(np.diff(matrix, axis=0))
    speed_validation = rms(np.diff(validation_matrix, axis=0))
    local.speed_ratio = (
        speed_validation / speed_train
        if speed_train > EPS
        else (1.0 if speed_validation <= EPS else float("inf"))
    )
    return local


def noise_and_disks(
    target: np.ndarray, seed: int, threshold: float
) -> tuple[np.ndarray, list, list]:
    mask = target[3] > threshold
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("Target alpha mask is empty")
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(12, *mask.shape)) * mask
    center = points[int(rng.integers(len(points)))]
    yy, xx = np.indices(mask.shape)
    distance = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    radii, counts = np.unique(distance[mask], return_counts=True)
    area_fractions = np.cumsum(counts) / mask.sum()
    disks, metadata = [], []
    for fraction in (0.05, 0.10):
        index = int(np.argmin(np.abs(area_fractions - fraction)))
        disk = distance <= radii[index]
        disks.append(disk)
        metadata.append(
            {
                "requested_target_fraction": fraction,
                "actual_target_fraction": float((disk & mask).sum() / mask.sum()),
                "center_yx": center.tolist(),
                "radius": float(np.sqrt(radii[index])),
            }
        )
    return direction, disks, metadata


def perturbations(
    anchor: torch.Tensor, target: np.ndarray, reference: np.ndarray, seed: int, threshold: float
):
    mask = target[3] > threshold
    direction, disks, details = noise_and_disks(target, seed, threshold)
    hidden_scale = np.sqrt(
        np.mean(reference[:, 4:, :, :][:, :, mask].astype(np.float64) ** 2, axis=(0, 2))
    )
    states = [anchor.clone(), anchor.clone()]
    metadata: dict[str, object] = {}
    for name, amplitude in zip(BRANCHES[2:5], (0.01, 0.05, 0.10), strict=True):
        delta = amplitude * hidden_scale[:, None, None] * direction
        state = anchor.clone()
        state[:, 4:] += torch.as_tensor(delta, dtype=state.dtype, device=state.device)[None]
        states.append(state)
        metadata[name] = {
            "relative_sigma": amplitude,
            "channel_sigma": (amplitude * hidden_scale).tolist(),
            "actual_rms_in_target": rms(delta[:, mask]),
        }
    alive = as_array(anchor)[3] > threshold
    for name, disk, detail in zip(BRANCHES[5:], disks, details, strict=True):
        state = anchor.clone()
        state[:, :, torch.as_tensor(disk, device=state.device)] = 0
        states.append(state)
        metadata[name] = {
            **detail,
            "removed_alive_fraction": float((disk & alive).sum() / max(int(alive.sum()), 1)),
        }
    return states, metadata, np.stack(disks)


def paired_updates(
    model,
    states: list,
    reference: torch.Tensor,
    system: str,
    cfg: Settings,
    shared_rng,
    independent_rng,
):
    next_states = []
    next_shared, next_independent = shared_rng, independent_rng
    for index, state in enumerate(states):
        if state is None:
            next_states.append(None)
            continue
        restore_rng(independent_rng if index == 1 else shared_rng, state.device)
        result = update(model, state, reference, system, cfg)
        if index == 1:
            next_independent = capture_rng(state.device)
        else:
            next_shared = capture_rng(state.device)
        next_states.append(result if bool(torch.isfinite(result).all()) else None)
    return next_states, next_shared, next_independent


def measure(
    state: np.ndarray,
    previous: np.ndarray | None,
    mature: np.ndarray,
    control: np.ndarray | None,
    target: np.ndarray,
    cfg: Settings,
    pcas: dict,
):
    values = np.full(len(METRICS), np.nan)
    values[:4] = morphology_metrics(state, target, cfg.alpha_threshold)
    scores, channel_rows = {}, []
    for offset, (group, selection) in enumerate(GROUPS.items()):
        selected = state[selection]
        delta = selected - mature[selection]
        speed = np.zeros_like(selected) if previous is None else selected - previous[selection]
        values[4 + offset] = rms(speed)
        values[6 + offset] = rms(delta)
        if control is not None:
            values[8 + offset] = rms(selected - control[selection])
            values[10 + offset] = values[8 + offset] / max(rms(mature[selection]), EPS)
        local_scores, residual, cloud, radius = pcas[group].project(selected)
        values[12 + offset * 2] = cloud
        values[13 + offset * 2] = residual
        values[16 + offset] = radius
        scores[group] = local_scores
        channel_rows.append(
            np.stack(
                (np.sqrt(np.mean(delta**2, axis=(1, 2))), np.sqrt(np.mean(speed**2, axis=(1, 2))))
            )
        )
    return values, scores, np.concatenate(channel_rows, axis=1)


def snapshot_schedule(cfg: Settings) -> list[int]:
    return sorted(
        {
            step
            for step in (
                0,
                cfg.perturb_step,
                cfg.perturb_step + 1,
                cfg.perturb_step + 10,
                cfg.perturb_step + 100,
                1000,
                2000,
                4000,
                6000,
                8000,
                cfg.horizon,
            )
            if 0 <= step <= cfg.horizon
        }
    )


@torch.no_grad()
def simulate(
    model,
    target: torch.Tensor,
    mature: torch.Tensor,
    system: str,
    replicate: int,
    regime: dict,
    cfg: Settings,
) -> tuple[dict, dict]:
    """Keep seven branches in memory; persist projections rather than dense full-state movies."""
    target_array, mature_array = as_array(target), as_array(mature)
    seed_nca(cfg.seed + 1_000_000 + replicate, target.device)
    state = mature.clone()
    pre_states = []
    for step in range(cfg.perturb_step + 1):
        if not torch.isfinite(state).all():
            return {}, {
                "status": "excluded",
                "reason": "nonfinite_before_perturbation",
                "step": step,
            }
        pre_states.append(as_array(state).astype(np.float32))
        if step < cfg.perturb_step:
            state = update(model, state, mature, system, cfg)
    shared_rng = capture_rng(target.device)
    seed_nca(cfg.seed + 2_000_000 + replicate, target.device)
    independent_rng = capture_rng(target.device)
    pre_states = np.stack(pre_states)
    reference = pre_states[cfg.reference_start : cfg.perturb_step]
    pcas = {
        group: fit_local_pca(reference[:, selection], cfg.pca_components, cfg.seed + replicate)
        for group, selection in GROUPS.items()
    }
    states, perturbation_metadata, disks = perturbations(
        state, target_array, reference, cfg.seed + 3_000_000 + replicate, cfg.alpha_threshold
    )
    shape = (len(BRANCHES), cfg.horizon + 1)
    curves = np.full((*shape, len(METRICS)), np.nan)
    in_target = np.zeros(shape, dtype=bool)
    in_reference = np.zeros(shape, dtype=bool)
    channel_curves = np.full((*shape, 2, 16), np.nan, dtype=np.float32)
    coordinates = {
        group: np.full((*shape, local.components.shape[0]), np.nan, dtype=np.float32)
        for group, local in pcas.items()
    }
    snap_steps = snapshot_schedule(cfg) if replicate < cfg.snapshot_replicates else []
    snapshots = np.full(
        (len(BRANCHES), len(snap_steps), 4, *target.shape[-2:]), np.nan, dtype=np.float32
    )
    snap_indices = {step: index for index, step in enumerate(snap_steps)}
    divergence = np.full(len(BRANCHES), -1, dtype=np.int64)
    windows = analysis_windows(cfg)
    temporal_variance = np.full((len(BRANCHES), len(windows), 16), np.nan)
    temporal_variance[:, 0] = reference.astype(np.float64).var(axis=0).mean(axis=(1, 2))
    window_index = 1
    running_mean = np.zeros((len(BRANCHES), *mature_array.shape))
    running_m2 = np.zeros_like(running_mean)
    sample_counts = np.zeros(len(BRANCHES), dtype=int)

    def record(branch, step, array, previous, control):
        values, scores, channels = measure(
            array, previous, mature_array, control, target_array, cfg, pcas
        )
        curves[branch, step] = values
        channel_curves[branch, step] = channels
        in_target[branch, step] = inside_regime(values, regime)
        in_reference[branch, step] = all(
            values[12 + i * 2] <= local.cloud_limit and values[13 + i * 2] <= local.residual_limit
            for i, local in enumerate(pcas.values())
        )
        for group in GROUPS:
            coordinates[group][branch, step] = scores[group]
        if step in snap_indices:
            snapshots[branch, snap_indices[step]] = array[:4]

    for step in range(cfg.perturb_step):
        record(
            0, step, pre_states[step], None if step == 0 else pre_states[step - 1], pre_states[step]
        )
    for branch in range(1, len(BRANCHES)):
        curves[branch, : cfg.perturb_step] = curves[0, : cfg.perturb_step]
        channel_curves[branch, : cfg.perturb_step] = channel_curves[0, : cfg.perturb_step]
        in_target[branch, : cfg.perturb_step] = in_target[0, : cfg.perturb_step]
        in_reference[branch, : cfg.perturb_step] = in_reference[0, : cfg.perturb_step]
        for group in GROUPS:
            coordinates[group][branch, : cfg.perturb_step] = coordinates[group][
                0, : cfg.perturb_step
            ]
        for step, index in snap_indices.items():
            if step < cfg.perturb_step:
                snapshots[branch, index] = snapshots[0, index]

    previous = [pre_states[-1]] * len(BRANCHES)
    initial_hidden_distance = []
    for step in range(cfg.perturb_step, cfg.horizon + 1):
        arrays = [None if state is None else as_array(state) for state in states]
        for branch, array in enumerate(arrays):
            if array is None:
                if divergence[branch] == -1:
                    divergence[branch] = step
                continue
            record(branch, step, array, previous[branch], arrays[0])
            if step > cfg.perturb_step:
                sample_counts[branch] += 1
                delta = array - running_mean[branch]
                running_mean[branch] += delta / sample_counts[branch]
                running_m2[branch] += delta * (array - running_mean[branch])
        if step == cfg.perturb_step:
            initial_hidden_distance = curves[:, step, 8].tolist()
        if window_index < len(windows) and step == windows[window_index][2]:
            _, start, stop = windows[window_index]
            complete = sample_counts == stop - start + 1
            temporal_variance[complete, window_index] = (
                running_m2[complete] / (stop - start + 1)
            ).mean(axis=(2, 3))
            running_mean.fill(0)
            running_m2.fill(0)
            sample_counts.fill(0)
            window_index += 1
        if step == cfg.horizon or all(state is None for state in states):
            break
        previous = arrays
        states, shared_rng, independent_rng = paired_updates(
            model, states, mature, system, cfg, shared_rng, independent_rng
        )
    diagnostics = {}
    payload = {
        "steps": np.arange(cfg.horizon + 1),
        "branch_names": np.asarray(BRANCHES),
        "metric_names": np.asarray(METRICS),
        "metrics": curves,
        "channel_metrics": channel_curves,
        "channel_order": np.asarray([f"H{i}" for i in range(12)] + ["R", "G", "B", "A"]),
        "channel_temporal_variance": temporal_variance,
        "variance_channel_order": np.asarray(["R", "G", "B", "A"] + [f"H{i}" for i in range(12)]),
        "variance_window_names": np.asarray([window[0] for window in windows]),
        "in_target": in_target,
        "in_reference": in_reference,
        "divergence_step": divergence,
        "snapshot_steps": np.asarray(snap_steps, dtype=np.int64),
        "snapshots": snapshots,
        "mature_state": mature_array.astype(np.float32),
        "pre_pulse_state": pre_states[-1],
        "erase_masks": disks,
        "initial_hidden_distance": np.asarray(initial_hidden_distance),
    }
    for group, local in pcas.items():
        payload.update(
            {
                f"{group}_scores": coordinates[group],
                f"{group}_pca_mean": local.mean,
                f"{group}_pca_components": local.components,
                f"{group}_scale": local.scale,
                f"{group}_variance_ratio": local.variance_ratio,
                f"{group}_reference_scores": local.reference_scores,
            }
        )
        diagnostics[group] = {
            "cloud_limit": local.cloud_limit,
            "residual_limit": local.residual_limit,
            "explained_variance": float(local.variance_ratio.sum()),
            "components_to_95_or_cap": local.effective_components,
            "reached_95_percent": bool(local.variance_ratio.sum() >= 0.95),
            "pre_mean_shift_in_radii": local.mean_shift_in_radii,
            "pre_speed_ratio": local.speed_ratio if np.isfinite(local.speed_ratio) else None,
            "approximately_stationary_heuristic": bool(
                local.mean_shift_in_radii <= 1 and 0.5 <= local.speed_ratio <= 2
            ),
        }
    split_step = cfg.reference_start + len(reference) // 2
    return payload, {
        "status": "complete",
        "system": system,
        "replicate": replicate,
        "perturbations": perturbation_metadata,
        "local_pca": diagnostics,
        "fit_steps_inclusive": [cfg.reference_start, split_step - 1],
        "validation_steps_inclusive": [split_step, cfg.perturb_step - 1],
        "in_target_before_pulse": bool(
            inside_regime(
                morphology_metrics(pre_states[-1], target_array, cfg.alpha_threshold), regime
            )
        ),
        "timing": "t=0 is target attainment; t=pulse contains post-pulse state; "
        "step RMS at the pulse includes the instantaneous perturbation.",
        "limitations": "Reference stationarity is a heuristic on a short pre-pulse window. "
        "PCA/FFT and finite return tests do not establish a mathematical attractor.",
    }


def train(args) -> None:
    device = get_device(args.device)
    for morphology in args.morphologies:
        for system in args.systems:
            output = checkpoint_path(
                Path(args.checkpoint_root), morphology, system, args.checkpoint_tag
            )
            if output.exists():
                print(f"Preserving existing checkpoint: {output}", flush=True)
                continue
            seed_nca(args.training_seed, device)
            target = load_distill_target(Path(args.data_dir) / f"{morphology}.png", device=device)
            model = GrowingNCA().to(device)
            variant = "persistent" if system == "persistent" else "growing"
            cfg = TrainingConfig(mode=variant)
            trainer = train_persistent if variant == "persistent" else train_growing
            print(
                f"Training {morphology} / {variant}: {cfg.training_iterations} iterations",
                flush=True,
            )
            trainer(model, target, cfg, output, output.with_name("loss.txt"))
            save_json(
                output.with_name("training_metadata.json"),
                {
                    "training": asdict(cfg),
                    "seed": args.training_seed,
                    "device": str(device),
                    "target_sha256": file_sha256(Path(args.data_dir) / f"{morphology}.png"),
                    "checkpoint_sha256": file_sha256(output),
                    "torch": str(torch.__version__),
                    "channels": 16,
                    "hidden_size": 128,
                    "target_size": 40,
                    "padding": 16,
                },
            )
            print(f"Saved {output}", flush=True)


def run(args) -> None:
    from threadpoolctl import threadpool_limits

    cfg = Settings(
        **{
            key: getattr(args, key)
            for key in (
                "perturb_step",
                "horizon",
                "reference_start",
                "seed",
                "calibration_replicates",
                "snapshot_replicates",
                "pca_components",
            )
        }
    )
    cfg.validate()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    device = get_device(args.device)
    # Check all inputs before creating an output folder or starting calibration.
    for morphology in args.morphologies:
        for system in SYSTEMS:
            path = checkpoint_path(
                Path(args.checkpoint_root), morphology, system, args.checkpoint_tag
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. Train the corresponding system first.")
    for morphology in args.morphologies:
        folder = Path(args.output_root) / morphology / args.checkpoint_tag
        folder.mkdir(parents=True, exist_ok=True)
        paths = {
            system: checkpoint_path(
                Path(args.checkpoint_root), morphology, system, args.checkpoint_tag
            )
            for system in SYSTEMS
        }
        target_path = Path(args.data_dir) / f"{morphology}.png"
        manifest = {
            "protocol": "persistent_vs_mature_restore_v1",
            "settings": asdict(cfg),
            "morphology": morphology,
            "checkpoint_tag": args.checkpoint_tag,
            "checkpoint_hashes": {system: file_sha256(path) for system, path in paths.items()},
            "checkpoint_paths": {system: str(path.resolve()) for system, path in paths.items()},
            "target_path": str(target_path.resolve()),
            "target_hash": file_sha256(target_path),
            "device": str(device),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "implementation_sha256": file_sha256(Path(__file__)),
        }
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(
                    f"Protocol/inputs/environment changed: choose a new output root ({folder})"
                )
        else:
            if any(folder.iterdir()):
                raise FileExistsError(f"Output contains files without a manifest: {folder}")
            save_json(manifest_path, manifest)
        models = {
            system: load_growing_checkpoint(
                path, channels=16, hidden_size=128, update_probability=cfg.fire_rate, device=device
            )
            for system, path in paths.items()
        }
        target = load_distill_target(target_path, cfg.target_size, cfg.padding, device)
        regime_path = folder / "calibration.json"
        if regime_path.exists():
            regime = json.loads(regime_path.read_text())
        else:
            regime, records = calibrate(models, target, cfg)
            save_npz(
                folder / "calibration_metrics.npz", records=records, systems=np.asarray(SYSTEMS)
            )
            save_json(regime_path, regime)
        for system, model in models.items():
            for replicate in range(args.replicates):
                result_dir = folder / system / f"replicate_{replicate:03d}"
                result_dir.mkdir(parents=True, exist_ok=True)
                metadata_path = result_dir / "metadata.json"
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    if (
                        metadata["status"] == "excluded"
                        or (result_dir / "trajectories.npz").exists()
                    ):
                        print(f"Already recorded: {morphology}/{system}/{replicate}", flush=True)
                        continue
                print(
                    f"Recording {morphology}/{system}, replicate {replicate + 1}/{args.replicates}",
                    flush=True,
                )
                mature, mature_step, reason = develop(
                    model, target, regime, cfg, cfg.seed + replicate
                )
                if mature is None:
                    save_json(
                        metadata_path,
                        {
                            "status": "excluded",
                            "reason": reason,
                            "system": system,
                            "replicate": replicate,
                        },
                    )
                    continue
                with threadpool_limits(limits=1):
                    payload, metadata = simulate(
                        model, target, mature, system, replicate, regime, cfg
                    )
                metadata.update(
                    {
                        "system": system,
                        "replicate": replicate,
                        "mature_step": mature_step,
                        "development_seed": cfg.seed + replicate,
                        "continuation_seed": cfg.seed + 1_000_000 + replicate,
                        "independent_seed": cfg.seed + 2_000_000 + replicate,
                        "perturbation_seed": cfg.seed + 3_000_000 + replicate,
                    }
                )
                if payload:
                    save_npz(result_dir / "trajectories.npz", **payload)
                save_json(metadata_path, metadata)
                print(f"Saved {result_dir}", flush=True)


def analysis_windows(cfg: Settings) -> list[tuple[str, int, int]]:
    windows = [("pre_pulse", cfg.reference_start, cfg.perturb_step - 1)]
    boundaries = sorted(
        {
            point
            for point in (1000, 2000, 4000, 6000, 8000, cfg.horizon)
            if cfg.perturb_step < point <= cfg.horizon
        }
    )
    start = cfg.perturb_step + 1
    for stop in boundaries:
        windows.append((f"t{start}_{stop}", start, stop))
        start = stop + 1
    return windows


def spectral_diagnostics(scores: np.ndarray, segment_length: int) -> tuple[dict, dict]:
    from scipy import signal

    centered = signal.detrend(scores.astype(np.float64), axis=0, type="linear")
    window = signal.windows.hann(len(centered), sym=False)
    spectrum = np.abs(np.fft.rfft(centered * window[:, None], axis=0)) ** 2
    power = spectrum.sum(axis=1) / max(float(np.sum(window**2)), EPS)
    frequency = np.fft.rfftfreq(len(centered), d=1.0)
    welch_frequency, welch_power = signal.welch(
        scores,
        fs=1.0,
        window="hann",
        nperseg=min(segment_length, len(scores)),
        noverlap=min(segment_length, len(scores)) // 2,
        detrend="linear",
        axis=0,
    )
    welch_power = welch_power.sum(axis=1)
    non_dc = power[1:]
    total = float(non_dc.sum())
    distribution = non_dc / total if total > EPS else np.zeros_like(non_dc)
    dominant = int(np.argmax(non_dc)) + 1 if total > EPS else None
    peaks = signal.find_peaks(distribution, height=0.05)[0]
    # A boundary-bin maximum need not be a resolved oscillation.
    period = 1 / frequency[dominant] if dominant is not None else float("nan")
    summary = {
        "detrended_variance": float(np.var(centered, axis=0).sum()),
        "dominant_frequency": float(frequency[dominant]) if dominant else float("nan"),
        "dominant_period": float(period),
        "dominant_power_fraction": float(distribution.max()) if distribution.size else 0.0,
        "dominant_observed_cycles": float(len(scores) / period),
        "at_least_three_cycles": bool(dominant and len(scores) / period >= 3),
        "spectral_entropy": float(
            -np.sum(distribution * np.log(distribution + EPS)) / np.log(max(len(distribution), 2))
        ),
        "top5_power_fraction": float(np.sort(distribution)[-5:].sum()),
        "peaks_above_5_percent": len(peaks),
        "fft_resolution": 1 / len(scores),
        "welch_segment_length": min(segment_length, len(scores)),
    }
    return summary, {
        "fft_frequency": frequency,
        "fft_power": power,
        "welch_frequency": welch_frequency,
        "welch_power": welch_power,
    }


def geometric_diagnostics(scores: np.ndarray, steps: np.ndarray) -> tuple[dict, dict]:
    from scipy.spatial.distance import cdist

    mean = scores.mean(axis=0)
    radius = rms(scores - mean)
    quarter = max(len(scores) // 4, 2)
    start_speed = rms(np.diff(scores[:quarter], axis=0))
    end_speed = rms(np.diff(scores[-quarter:], axis=0))
    lags = np.arange(1, min(500, len(scores) // 4) + 1)
    distances = np.asarray([rms(scores[lag:] - scores[:-lag]) for lag in lags])
    relative = distances / max(radius, EPS)
    valid_lags = np.flatnonzero(lags >= 20)
    best = int(valid_lags[np.argmin(relative[valid_lags])]) if valid_lags.size else None
    indices = np.unique(np.linspace(0, len(scores) - 1, min(240, len(scores)), dtype=int))
    recurrence = cdist(scores[indices], scores[indices]) / np.sqrt(scores.shape[1])
    off_diagonal = np.abs(steps[indices, None] - steps[None, indices]) > 20
    robust_diameter = (
        float(np.quantile(recurrence[off_diagonal], 0.95)) if off_diagonal.any() else 0.0
    )
    threshold = 0.1 * robust_diameter
    rate = (
        float(np.mean(recurrence[off_diagonal] <= threshold))
        if off_diagonal.any()
        else float("nan")
    )
    summary = {
        "pca_radius": radius,
        "pca_step_rms": rms(np.diff(scores, axis=0)),
        "pca_start_speed": start_speed,
        "pca_end_speed": end_speed,
        "pca_end_start_speed_ratio": end_speed / max(start_speed, EPS),
        "pca_mean_shift_in_radii": rms(
            scores[-quarter:].mean(axis=0) - scores[:quarter].mean(axis=0)
        )
        / max(radius, EPS),
        "best_return_lag": int(lags[best]) if best is not None else "",
        "best_relative_return_distance": float(relative[best])
        if best is not None
        else float("nan"),
        "recurrence_rate": rate,
        "recurrence_threshold": threshold,
        "recurrence_theiler_steps": 20,
        "recurrence_robust_diameter": robust_diameter,
    }
    return summary, {
        "return_lags": lags,
        "return_distances": distances,
        "relative_return_distances": relative,
        "recurrence_distance": recurrence,
        "recurrence_steps": steps[indices],
    }


def window_rows(payload: dict, cfg: Settings) -> tuple[list[dict], dict]:
    rows, details = [], {}
    metrics, steps = payload["metrics"], payload["steps"]
    for branch, name in enumerate(BRANCHES):
        for group_index, group in enumerate(GROUPS):
            for window_index, (label, start, stop) in enumerate(analysis_windows(cfg)):
                selected = (steps >= start) & (steps <= stop)
                scores = payload[f"{group}_scores"][branch, selected].astype(np.float64)
                finite = np.isfinite(scores).all(axis=1)
                complete = bool(finite.all() and len(scores) >= 8)
                row = {
                    "branch": name,
                    "group": group,
                    "window": label,
                    "start": start,
                    "stop": stop,
                    "complete": complete,
                    "n_finite": int(finite.sum()),
                }
                if complete:
                    spectral, spectrum = spectral_diagnostics(
                        scores, min(128, cfg.perturb_step - cfg.reference_start)
                    )
                    geometry, recurrence = geometric_diagnostics(scores, steps[selected])
                    row.update(spectral)
                    row.update(geometry)
                    row["full_group_step_rms"] = float(
                        metrics[branch, selected, 4 + group_index].mean()
                    )
                    row["mean_cell_temporal_variance"] = float(
                        payload["channel_temporal_variance"][
                            branch, window_index, GROUPS[group]
                        ].mean()
                    )
                    row["target_fraction"] = float(payload["in_target"][branch, selected].mean())
                    row["reference_fraction"] = float(
                        payload["in_reference"][branch, selected].mean()
                    )
                    prefix = f"{name}__{group}__{label}__"
                    details.update(
                        {prefix + key: value for key, value in {**spectrum, **recurrence}.items()}
                    )
                rows.append(row)
    return rows, details


def recovery_rows(payload: dict, metadata: dict, windows: list[dict], cfg: Settings) -> list[dict]:
    rows = []
    pulse = cfg.perturb_step
    last_start = max(pulse + 1, cfg.horizon - 199)
    reference_ok = all(
        d["approximately_stationary_heuristic"] for d in metadata["local_pca"].values()
    )
    late_label = analysis_windows(cfg)[-1][0]
    for branch, name in enumerate(BRANCHES):
        target = payload["in_target"][branch]
        flags = target[pulse:]
        exits = np.flatnonzero(~flags)
        first_exit = int(exits[0]) if exits.size else None
        if first_exit is None:
            recovery = 0
        else:
            returned = first_sustained(flags[first_exit + 1 :], cfg.recovery_patience)
            recovery = None if returned is None else first_exit + 1 + returned
        initial_distance = float(payload["metrics"][branch, pulse, 8])
        ratios = payload["metrics"][branch, pulse:, 8] / max(initial_distance, EPS)
        pair_return = first_sustained(np.isfinite(ratios) & (ratios <= 0.1), cfg.recovery_patience)
        if initial_distance <= EPS:
            pair_return = None
        divergence = int(payload["divergence_step"][branch])
        sustained = bool(
            recovery is not None and divergence < 0 and target[last_start:].mean() >= 0.95
        )
        local_fraction = float(payload["in_reference"][branch, last_start:].mean())
        baseline_local = float(payload["in_reference"][0, last_start:].mean())
        late = [row for row in windows if row["branch"] == name and row["window"] == late_label]
        bounded_candidate = len(late) == 2 and all(
            row["complete"]
            and row["pca_mean_shift_in_radii"] <= 1
            and (0.5 <= row["pca_end_start_speed_ratio"] <= 2 or row["pca_step_rms"] < EPS)
            for row in late
        )
        if divergence >= 0:
            interpretation = "numerical_divergence"
        elif name in BRANCHES[:2]:
            interpretation = "unperturbed_comparator"
        elif not sustained:
            interpretation = "no_sustained_morphological_recovery"
        elif not metadata["in_target_before_pulse"]:
            interpretation = "target_reached_after_pulse_but_pre_pulse_morphology_failed"
        elif not reference_ok or baseline_local < 0.9:
            interpretation = "morphology_recovered_reference_not_stationary"
        elif local_fraction >= 0.9:
            interpretation = "compatible_with_pre_pulse_regime"
        elif bounded_candidate:
            interpretation = "different_late_regime_candidate_requires_followup"
        else:
            interpretation = "morphology_recovered_dynamics_unresolved"
        final_values = payload["metrics"][branch, -1]
        independent_scale = payload["metrics"][1, last_start:, 8]
        finite_scale = independent_scale[np.isfinite(independent_scale)]
        rows.append(
            {
                "branch": name,
                "in_target_before_pulse": metadata["in_target_before_pulse"],
                "reference_stationary_heuristic": reference_ok,
                "first_target_exit_delay": "" if first_exit is None else first_exit,
                "first_recovery_delay": "" if recovery is None else recovery,
                "recovery_observed": recovery is not None,
                "sustained_target_at_horizon": sustained,
                "never_left_target": bool(flags.all()),
                "relapse_after_recovery": bool(recovery is not None and not flags[recovery:].all()),
                "time_to_90_percent_hidden_distance_reduction": ""
                if pair_return is None
                else pair_return,
                "recovery_followup_steps": cfg.horizon - pulse,
                "last_200_target_fraction": float(target[last_start:].mean()),
                "last_200_reference_fraction": local_fraction,
                "final_target_mse": float(final_values[0]),
                "final_target_region_mse": float(final_values[1]),
                "final_alpha_iou": float(final_values[2]),
                "final_area_ratio": float(final_values[3]),
                "final_hidden_pair_rms": float(final_values[8]),
                "final_rgba_pair_rms": float(final_values[9]),
                "independent_control_late_hidden_distance_q95": float(
                    np.quantile(finite_scale, 0.95)
                )
                if finite_scale.size
                else float("nan"),
                "numerical_divergence_step": "" if divergence < 0 else divergence,
                "interpretation": interpretation,
            }
        )
    return rows


def result_folders(args):
    root = Path(args.output_root)
    for morphology in args.morphologies:
        folders = (
            [root / morphology / args.checkpoint_tag]
            if args.checkpoint_tag
            else sorted((root / morphology).glob("*"))
        )
        for folder in folders:
            if (folder / "manifest.json").exists():
                yield morphology, folder


def wilson(successes: int, count: int) -> tuple[float, float]:
    if not count:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return float(center - half), float(center + half)


def aggregate(rows: list[dict], bootstrap_samples: int, seed: int) -> tuple[list[dict], list[dict]]:
    summaries, comparisons = [], []
    rng = np.random.default_rng(seed)
    for system in SYSTEMS:
        for branch in BRANCHES:
            selected = [row for row in rows if row["system"] == system and row["branch"] == branch]
            eligible = [row for row in selected if row["in_target_before_pulse"]]
            successes = sum(row["sustained_target_at_horizon"] for row in eligible)
            low, high = wilson(successes, len(eligible))
            delays = [
                row["first_recovery_delay"]
                for row in eligible
                if row["recovery_observed"] and not row["never_left_target"]
            ]
            summaries.append(
                {
                    "system": system,
                    "branch": branch,
                    "n_recorded": len(selected),
                    "n_pre_pulse_target": len(eligible),
                    "sustained_recoveries": successes,
                    "n_never_left_target": sum(row["never_left_target"] for row in eligible),
                    "recovery_fraction": successes / len(eligible) if eligible else float("nan"),
                    "recovery_ci95_low": low,
                    "recovery_ci95_high": high,
                    "median_delay_among_recovered_after_exit": float(np.median(delays))
                    if delays
                    else "",
                    "n_without_observed_recovery": sum(
                        not row["recovery_observed"] for row in eligible
                    ),
                }
            )
    for branch in BRANCHES:
        persistent = {
            row["replicate"]: row
            for row in rows
            if row["system"] == SYSTEMS[0]
            and row["branch"] == branch
            and row["in_target_before_pulse"]
        }
        restored = {
            row["replicate"]: row
            for row in rows
            if row["system"] == SYSTEMS[1]
            and row["branch"] == branch
            and row["in_target_before_pulse"]
        }
        pairs = sorted(persistent.keys() & restored.keys())
        if not pairs:
            continue
        delta = np.asarray(
            [
                int(restored[i]["sustained_target_at_horizon"])
                - int(persistent[i]["sustained_target_at_horizon"])
                for i in pairs
            ]
        )
        boot = np.empty(bootstrap_samples)
        for start in range(0, bootstrap_samples, 500):
            count = min(500, bootstrap_samples - start)
            indices = rng.integers(len(delta), size=(count, len(delta)))
            boot[start : start + count] = delta[indices].mean(axis=1)
        comparisons.append(
            {
                "branch": branch,
                "n_matched_evaluation_seeds": len(pairs),
                "growing_minus_persistent_recovery_fraction": float(delta.mean()),
                "paired_bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                "paired_bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                "bootstrap_samples": bootstrap_samples,
                "inference_scope": "This checkpoint pair only; rollout seeds are not independent trainings",
            }
        )
    return summaries, comparisons


def analyze(args) -> None:
    found = False
    for morphology, folder in result_folders(args):
        found = True
        cfg = Settings(**json.loads((folder / "manifest.json").read_text())["settings"])
        rows, exclusions = [], []
        for system in SYSTEMS:
            for result_dir in sorted((folder / system).glob("replicate_*")):
                metadata_path = result_dir / "metadata.json"
                if not metadata_path.exists():
                    continue
                metadata = json.loads(metadata_path.read_text())
                if metadata["status"] != "complete":
                    exclusions.append(
                        {
                            "system": system,
                            "replicate": metadata.get("replicate", result_dir.name),
                            "reason": metadata["reason"],
                        }
                    )
                    continue
                with np.load(result_dir / "trajectories.npz", allow_pickle=False) as archive:
                    payload = {key: archive[key] for key in archive.files}
                windows, details = window_rows(payload, cfg)
                recovered = recovery_rows(payload, metadata, windows, cfg)
                write_csv(result_dir / "window_metrics.csv", windows)
                write_csv(result_dir / "recovery.csv", recovered)
                save_npz(result_dir / "window_diagnostics.npz", **details)
                rows.extend(
                    {
                        "morphology": morphology,
                        "checkpoint_tag": folder.name,
                        "system": system,
                        "replicate": metadata["replicate"],
                        **row,
                    }
                    for row in recovered
                )
                print(f"Analyzed {result_dir}", flush=True)
        if not rows:
            save_json(folder / "exclusions.json", exclusions)
            print(f"No completed, eligible trajectories in {folder}", flush=True)
            continue
        summaries, comparisons = aggregate(rows, args.bootstrap_samples, cfg.seed + 4_000_000)
        write_csv(folder / "per_run_metrics.csv", rows)
        write_csv(folder / "condition_metrics.csv", summaries)
        write_csv(folder / "matched_system_comparison.csv", comparisons)
        save_json(folder / "exclusions.json", exclusions)
        save_json(
            folder / "analysis_settings.json",
            {
                "windows_inclusive": analysis_windows(cfg),
                "sample_interval": 1,
                "pca_fit": "First half of pre-pulse reference, each replica/model separately",
                "pca_validation": "Second temporal half; q99 cloud/residual bounds; autocorrelated observations",
                "pre_stationarity": "Diagnostic: mean shift <= 1 radius and speed ratio 0.5-2; not a statistical test",
                "spectra": "Linear detrending, Hann, multicomponent summed FFT power and Welch PSD",
                "welch_segment_length": min(128, cfg.perturb_step - cfg.reference_start),
                "recurrence": "PCA-space distance; threshold=0.1*q95 pair distance, exclude |lag|<=20",
                "morphology_recovery": f"After first target exit: {cfg.recovery_patience} consecutive target states; delay measured from pulse; final 200 >=95% in target. Never-exited trajectories counted separately and included in target-retention/recovery fraction.",
                "regime_compatibility": "Final 200 >=90% inside pre-reference geometric envelope, including PCA residual",
                "pair_return": f"{cfg.recovery_patience} consecutive samples with hidden distance <=10% of initial perturbation distance",
                "temporal_variance": "Population variance over time at each cell/channel, averaged over the entire spatial grid; no pooling",
                "population": "Morphologies and checkpoint tags analyzed separately; paired bootstrap over matching rollout IDs",
                "bootstrap_samples": args.bootstrap_samples,
                "classification": "Operational descriptors only; alternate regime candidates require longer observation and re-perturbation",
            },
        )
    if not found:
        raise FileNotFoundError("No Experiment 2 manifests found under the requested output root")


def plotting():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def finish_figure(plt, figure, path: Path) -> None:
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def display_rgba(rgba: np.ndarray) -> np.ndarray:
    alpha = np.clip(rgba[3:4], 0, 1)
    return np.clip(rgba[:3] + 1 - alpha, 0, 1).transpose(1, 2, 0)


def plot_channels(plt, payload: dict, result_dir: Path, cfg: Settings) -> None:
    from matplotlib.colors import SymLogNorm

    for group, selection in (("hidden", slice(0, 12)), ("rgba", slice(12, 16))):
        values = payload["channel_metrics"][:, :, :, selection]
        names = payload["channel_order"][selection]
        figure, axes = plt.subplots(7, 2, figsize=(13, 14), sharex=True, layout="constrained")
        for column, title in enumerate(("Deviation from mature state (RMS)", "Step-to-step RMS")):
            finite = values[:, :, column][np.isfinite(values[:, :, column])]
            maximum = max(float(finite.max()), 1e-8) if finite.size else 1e-8
            positive = finite[finite > 0]
            floor = max(float(np.quantile(positive, 0.05)), 1e-10) if positive.size else 1e-10
            norm = SymLogNorm(linthresh=floor, vmin=0, vmax=maximum)
            cmap = plt.get_cmap("magma" if column == 0 else "viridis").copy()
            cmap.set_bad("0.8")
            for branch, label in enumerate(LABELS):
                axis = axes[branch, column]
                image = axis.imshow(
                    values[branch, :, column].T,
                    origin="lower",
                    aspect="auto",
                    cmap=cmap,
                    norm=norm,
                    extent=(-0.5, cfg.horizon + 0.5, -0.5, len(names) - 0.5),
                )
                axis.set_yticks(np.arange(len(names)), names, fontsize=7)
                axis.axvline(cfg.perturb_step, color="white", linestyle="--", linewidth=0.7)
                if column == 0:
                    axis.set_ylabel(label, fontsize=8)
                if branch == 0:
                    axis.set_title(title)
            figure.colorbar(
                image,
                ax=axes[:, column],
                orientation="horizontal",
                fraction=0.025,
                label=f"RMS; linear below {floor:.2g}, logarithmic above",
            )
            axes[-1, column].set_xlabel("Steps after target attainment")
        figure.suptitle(f"{group}: shared scales across branches; gray = missing/non-finite")
        finish_figure(plt, figure, result_dir / f"{group}_channel_dynamics.png")


def plot_replica(result_dir: Path, cfg: Settings) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    plt = plotting()
    with np.load(result_dir / "trajectories.npz", allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads((result_dir / "metadata.json").read_text())
    steps, metrics = payload["steps"], payload["metrics"]
    plot_channels(plt, payload, result_dir, cfg)
    time_indices = np.unique(
        np.r_[np.arange(0, cfg.horizon + 1, 10), cfg.perturb_step, cfg.horizon]
    )
    for group in GROUPS:
        scores = payload[f"{group}_scores"]
        ratio = payload[f"{group}_variance_ratio"]
        reference = (steps >= cfg.reference_start) & (steps < cfg.perturb_step)
        for dimension in (2, 3):
            figure = plt.figure(figsize=(16, 8) if dimension == 3 else (14, 7))
            last_scatter = None
            for branch, label in enumerate(LABELS):
                axis = figure.add_subplot(
                    2, 4, branch + 1, projection="3d" if dimension == 3 else None
                )
                gray = scores[0, reference, :dimension]
                base = scores[0, time_indices, :dimension]
                points = scores[branch, time_indices, :dimension]
                axis.plot(*base.T, color="0.7", linewidth=0.7, linestyle="--")
                axis.scatter(*gray.T, color="0.65", s=3, alpha=0.35)
                valid = np.isfinite(points).all(axis=1)
                last_scatter = axis.scatter(
                    *points[valid].T,
                    c=steps[time_indices][valid],
                    cmap="plasma",
                    vmin=0,
                    vmax=cfg.horizon,
                    s=4,
                )
                axis.plot(*points.T, color=COLORS[branch], linewidth=0.5, alpha=0.5)
                for index, marker, color in (
                    (cfg.perturb_step - 1, "o", "black"),
                    (cfg.perturb_step, "D", "#db6d00"),
                    (cfg.horizon, "x", "#008a96"),
                ):
                    if np.isfinite(scores[branch, index, :dimension]).all():
                        axis.scatter(
                            *scores[branch, index, :dimension], marker=marker, color=color, s=24
                        )
                axis.set_title(label, fontsize=9)
                axis.set_xlabel(f"PC1 ({ratio[0]:.1%})", fontsize=8)
                axis.set_ylabel(f"PC2 ({ratio[1]:.1%})", fontsize=8)
                if dimension == 3:
                    axis.set_zlabel(f"PC3 ({ratio[2]:.1%})", fontsize=8)
                for coordinate_axis in (
                    axis.xaxis,
                    axis.yaxis,
                    *([axis.zaxis] if dimension == 3 else []),
                ):
                    coordinate_axis.set_major_locator(MaxNLocator(nbins=3))
                    coordinate_axis.set_major_formatter(
                        FuncFormatter(lambda value, _: f"{value:.2g}")
                    )
                axis.tick_params(labelsize=7)
            from matplotlib.lines import Line2D

            handles = [
                Line2D([], [], color="0.7", linestyle="--", label="Unperturbed control"),
                Line2D([], [], linestyle="", marker="o", color="0.65", label="Pre-pulse reference"),
                Line2D(
                    [], [], linestyle="", marker="o", color="black", label="Last pre-pulse sample"
                ),
                Line2D(
                    [],
                    [],
                    linestyle="",
                    marker="D",
                    color="#db6d00",
                    label="Immediately after pulse",
                ),
                Line2D([], [], linestyle="", marker="x", color="#008a96", label="State at horizon"),
            ]
            figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
            figure.suptitle(
                f"{metadata['system']} | {group} local PCA | replica {metadata['replicate']}\n"
                "Axes fitted only before perturbation; subplot limits may differ",
                fontsize=12,
            )
            figure.subplots_adjust(bottom=0.16, top=0.87, wspace=0.42, hspace=0.45, right=0.88)
            color_axis = figure.add_axes([0.95, 0.22, 0.012, 0.53])
            figure.colorbar(last_scatter, cax=color_axis, label="Steps after target attainment")
            finish_figure(plt, figure, result_dir / f"{group}_local_pca_{dimension}d.png")

    figure, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for branch, label in enumerate(LABELS):
        for column in range(2):
            distance = metrics[branch, :, 10 + column]
            axes[0, column].plot(steps, distance, color=COLORS[branch], label=label, linewidth=0.8)
            axes[1, column].plot(
                steps, metrics[branch, :, 4 + column], color=COLORS[branch], linewidth=0.8
            )
            group = tuple(GROUPS)[column]
            for row, offset, limit in ((2, 12, "cloud_limit"), (3, 13, "residual_limit")):
                axes[row, column].plot(
                    steps,
                    metrics[branch, :, offset + 2 * column] / metadata["local_pca"][group][limit],
                    color=COLORS[branch],
                    linewidth=0.8,
                )
    for column, group in enumerate(GROUPS):
        axes[0, column].set_title(f"{group}: distance to paired control / mature RMS")
        axes[1, column].set_title(f"{group}: step-to-step RMS")
        axes[2, column].set_title(f"{group}: PCA cloud distance / held-out bound")
        axes[3, column].set_title(f"{group}: PCA residual / held-out bound")
        for row in (2, 3):
            axes[row, column].axhline(1, color="0.3", linestyle=":", linewidth=0.8)
    for axis in axes.flat:
        axis.set_yscale("symlog", linthresh=1e-8)
        axis.axvline(cfg.perturb_step, color="0.4", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Steps after target attainment")
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    finish_figure(plt, figure, result_dir / "paired_recovery_and_activity.png")

    snap_steps = payload["snapshot_steps"]
    if snap_steps.size:
        snapshot_columns = [(int(step), index) for index, step in enumerate(snap_steps)]
        snapshot_columns.insert(
            int(np.searchsorted(snap_steps, cfg.perturb_step)), (cfg.perturb_step, None)
        )
        figure, axes = plt.subplots(
            len(BRANCHES),
            len(snap_steps) + 1,
            figsize=(1.7 * (len(snap_steps) + 1), 11),
            squeeze=False,
        )
        for branch, label in enumerate(LABELS):
            for column, (step, snapshot_index) in enumerate(snapshot_columns):
                axis = axes[branch, column]
                rgba = (
                    payload["pre_pulse_state"][:4]
                    if snapshot_index is None
                    else payload["snapshots"][branch, snapshot_index]
                )
                if np.isfinite(rgba).all():
                    axis.imshow(display_rgba(rgba))
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "Non-finite",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        fontsize=8,
                    )
                axis.set_xticks([])
                axis.set_yticks([])
                if branch == 0:
                    axis.set_title(
                        f"t={step}, pre" if snapshot_index is None else f"t={step}",
                        fontsize=8,
                    )
                if column == 0:
                    axis.set_ylabel(label, fontsize=8)
        figure.tight_layout()
        finish_figure(plt, figure, result_dir / "perturbation_snapshots.png")

    path = result_dir / "window_diagnostics.npz"
    if not path.exists():
        return
    with np.load(path, allow_pickle=False) as archive:
        details = {key: archive[key] for key in archive.files}
    windows = analysis_windows(cfg)
    for group in GROUPS:
        figure, axes = plt.subplots(2, 4, figsize=(14, 7))
        return_figure, return_axes = plt.subplots(2, 4, figsize=(14, 7))
        recurrence_figure, recurrence_axes = plt.subplots(2, 4, figsize=(14, 7))
        for branch, label in enumerate(LABELS):
            axis = axes.flat[branch]
            rec_axis = recurrence_axes.flat[branch]
            return_axis = return_axes.flat[branch]
            for window_label, start, stop in windows:
                prefix = f"{BRANCHES[branch]}__{group}__{window_label}__"
                if prefix + "welch_frequency" in details:
                    frequency, power = (
                        details[prefix + "welch_frequency"],
                        details[prefix + "welch_power"],
                    )
                    axis.plot(
                        frequency[1:],
                        np.maximum(power[1:], 1e-30),
                        label=f"{start}-{stop}",
                        linewidth=0.8,
                    )
                    return_axis.plot(
                        details[prefix + "return_lags"],
                        details[prefix + "relative_return_distances"],
                        label=f"{start}-{stop}",
                        linewidth=0.8,
                    )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_title(label, fontsize=9)
            axis.set_xlabel("Cycles per step", fontsize=8)
            axis.set_ylabel("Welch power density", fontsize=8)
            axis.tick_params(labelsize=7)
            return_axis.set_title(label, fontsize=9)
            return_axis.set_xlabel("Lag (NCA steps)", fontsize=8)
            return_axis.set_ylabel("Return distance / window radius", fontsize=8)
            return_axis.tick_params(labelsize=7)
            return_axis.grid(alpha=0.2)
            prefix = f"{BRANCHES[branch]}__{group}__{windows[-1][0]}__"
            if prefix + "recurrence_distance" in details:
                distance = details[prefix + "recurrence_distance"]
                rec_steps = details[prefix + "recurrence_steps"]
                valid = np.abs(rec_steps[:, None] - rec_steps[None]) > 20
                diameter = (
                    max(float(np.quantile(distance[valid], 0.95)), EPS) if valid.any() else EPS
                )
                image = rec_axis.imshow(
                    distance / diameter,
                    origin="lower",
                    cmap="magma_r",
                    vmin=0,
                    vmax=1,
                    extent=(rec_steps[0], rec_steps[-1], rec_steps[0], rec_steps[-1]),
                )
                if np.min(distance / diameter) < 0.1 < np.max(distance / diameter):
                    rec_axis.contour(
                        rec_steps,
                        rec_steps,
                        distance / diameter,
                        levels=[0.1],
                        colors="cyan",
                        linewidths=0.4,
                    )
                recurrence_figure.colorbar(image, ax=rec_axis, fraction=0.045)
            else:
                rec_axis.text(
                    0.5, 0.5, "Incomplete window", ha="center", transform=rec_axis.transAxes
                )
            rec_axis.set_title(label, fontsize=9)
            rec_axis.tick_params(labelsize=7)
        axes.flat[-1].axis("off")
        recurrence_axes.flat[-1].axis("off")
        return_axes.flat[-1].axis("off")
        axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", fontsize=9)
        return_axes.flat[-1].legend(
            *return_axes.flat[0].get_legend_handles_labels(), loc="center", fontsize=9
        )
        figure.suptitle(f"{metadata['system']} | {group} | window spectra (pulse excluded)")
        recurrence_figure.suptitle(
            f"{group}: late-window distance / robust diameter; cyan = 10% threshold"
        )
        return_figure.suptitle(f"{metadata['system']} | {group} | return distances by window")
        figure.tight_layout()
        recurrence_figure.tight_layout()
        return_figure.tight_layout()
        finish_figure(plt, figure, result_dir / f"{group}_window_spectra.png")
        finish_figure(plt, recurrence_figure, result_dir / f"{group}_late_recurrence.png")
        finish_figure(plt, return_figure, result_dir / f"{group}_return_distances.png")

    with (result_dir / "window_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    figure, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
    for group_index, group in enumerate(GROUPS):
        for column, (field, title) in enumerate(
            (
                ("full_group_step_rms", "Full-grid step RMS"),
                ("mean_cell_temporal_variance", "Mean per-cell temporal variance"),
                ("spectral_entropy", "PCA spectral entropy"),
            )
        ):
            axis = axes[group_index, column]
            for branch, label in enumerate(LABELS):
                selected = {
                    row["window"]: row
                    for row in rows
                    if row["branch"] == BRANCHES[branch] and row["group"] == group
                }
                values = [float(selected[name].get(field) or "nan") for name, _, _ in windows]
                axis.plot(
                    np.arange(len(windows)), values, color=COLORS[branch], marker=".", label=label
                )
            axis.set_title(f"{group}: {title}", fontsize=10)
            axis.set_xticks(
                np.arange(len(windows)),
                [f"{start}-{stop}" for _, start, stop in windows],
                rotation=40,
                ha="right",
                fontsize=8,
            )
            axis.grid(alpha=0.2)
            if column < 2:
                axis.set_yscale("symlog", linthresh=1e-10)
            else:
                axis.set_ylim(0, 1.05)
    figure.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8
    )
    finish_figure(plt, figure, result_dir / "window_dynamics.png")


def plot_population(folder: Path, cfg: Settings) -> None:
    plt = plotting()
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    titles = (
        "Global RGBA target MSE",
        "Target-region RGBA MSE",
        "Alpha-mask IoU",
        "Alive-area ratio",
    )
    for system_index, system in enumerate(SYSTEMS):
        curves = []
        for file in sorted((folder / system).glob("replicate_*/trajectories.npz")):
            with np.load(file, allow_pickle=False) as archive:
                curves.append(archive["metrics"][:, ::10, :4])
        if not curves:
            continue
        values = np.stack(curves)
        steps = np.arange(0, cfg.horizon + 1, 10)
        for metric in range(4):
            axis = axes[system_index, metric]
            for branch, label in enumerate(LABELS):
                sample = values[:, branch, :, metric]
                count = np.isfinite(sample).sum(axis=0)
                present = count > 0
                median, q25, q75 = np.full((3, len(steps)), np.nan)
                if present.any():
                    q25[present], median[present], q75[present] = np.nanquantile(
                        sample[:, present], [0.25, 0.5, 0.75], axis=0
                    )
                axis.plot(steps, median, color=COLORS[branch], label=label, linewidth=0.9)
                axis.fill_between(steps, q25, q75, color=COLORS[branch], alpha=0.10)
            axis.set_title(f"{system}\n{titles[metric]}", fontsize=10)
            axis.axvline(cfg.perturb_step, color="0.5", linestyle="--", linewidth=0.8)
            if metric < 2:
                axis.set_yscale("symlog", linthresh=1e-6)
            axis.set_xlabel("Steps after target attainment")
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=8)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9)
    figure.suptitle(
        "Median and IQR across recorded replicas; non-finite observations omitted (see counts)"
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    finish_figure(plt, figure, folder / "core_morphology_comparison.png")

    count_figure, count_axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    for index, system in enumerate(SYSTEMS):
        arrays = []
        for file in sorted((folder / system).glob("replicate_*/trajectories.npz")):
            with np.load(file, allow_pickle=False) as archive:
                arrays.append(np.isfinite(archive["metrics"][:, ::10, 0]))
        if arrays:
            counts = np.stack(arrays).sum(axis=0)
            for branch, label in enumerate(LABELS):
                count_axes[0, index].plot(
                    np.arange(0, cfg.horizon + 1, 10),
                    counts[branch],
                    label=label,
                    color=COLORS[branch],
                )
        count_axes[0, index].set_title(system)
        count_axes[0, index].set_ylabel("Finite replicas contributing")
        count_axes[0, index].set_xlabel("Steps after target attainment")
    count_figure.tight_layout()
    finish_figure(plt, count_figure, folder / "contributing_replicates.png")

    summary_path = folder / "condition_metrics.csv"
    if summary_path.exists():
        with summary_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        figure, axis = plt.subplots(figsize=(12, 4))
        for index, system in enumerate(SYSTEMS):
            selected = {row["branch"]: row for row in rows if row["system"] == system}
            for branch, name in enumerate(BRANCHES):
                row = selected.get(name)
                if row is None or not int(row["n_pre_pulse_target"]):
                    continue
                value = float(row["recovery_fraction"])
                errors = [
                    [max(0, value - float(row["recovery_ci95_low"]))],
                    [max(0, float(row["recovery_ci95_high"]) - value)],
                ]
                axis.errorbar(
                    branch + (index - 0.5) * 0.2,
                    value,
                    yerr=errors,
                    fmt="o",
                    capsize=3,
                    color=("#2a6fbb", "#c44949")[index],
                    label=system if branch == 0 else None,
                )
        axis.set_xticks(np.arange(len(BRANCHES)), LABELS, rotation=20, ha="right")
        axis.set_ylim(-0.04, 1.04)
        axis.set_ylabel("Retained or recovered target fraction")
        axis.set_title(
            "95% Wilson intervals; denominator: replicas in target immediately before pulse"
        )
        axis.legend()
        figure.tight_layout()
        finish_figure(plt, figure, folder / "recovery_comparison.png")


def plot(args) -> None:
    found = False
    for _, folder in result_folders(args):
        found = True
        cfg = Settings(**json.loads((folder / "manifest.json").read_text())["settings"])
        plot_population(folder, cfg)
        for system in SYSTEMS:
            directories = [
                path.parent
                for path in sorted((folder / system).glob("replicate_*/trajectories.npz"))
            ]
            selected = (
                directories[:3]
                if args.replicates is None
                else [
                    path for path in directories if int(path.name.split("_")[-1]) in args.replicates
                ]
            )
            for result_dir in selected:
                plot_replica(result_dir, cfg)
                print(f"Plotted {result_dir}", flush=True)
    if not found:
        raise FileNotFoundError("No Experiment 2 manifests found")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser(
        "train", help="Train Persistent models, optionally Growing, without overwriting checkpoints"
    )
    training.add_argument("--systems", nargs="+", choices=SYSTEMS, default=("persistent",))
    training.add_argument("--training-seed", type=int, default=0)
    training.set_defaults(handler=train)
    running = commands.add_parser(
        "run", help="Calibrate and record paired trajectories; resume completed replicas"
    )
    running.add_argument("--replicates", type=int, default=30)
    running.add_argument("--perturb-step", type=int, default=500)
    running.add_argument("--horizon", type=int, default=10_000)
    running.add_argument("--reference-start", type=int, default=250)
    running.add_argument("--calibration-replicates", type=int, default=32)
    running.add_argument("--snapshot-replicates", type=int, default=3)
    running.add_argument("--pca-components", type=int, default=12)
    running.add_argument("--seed", type=int, default=50_000)
    running.set_defaults(handler=run)
    analysis = commands.add_parser(
        "analyze", help="PCA-window, Fourier, recurrence and recovery statistics from recorded data"
    )
    analysis.add_argument("--bootstrap-samples", type=int, default=5000)
    analysis.set_defaults(handler=analyze)
    plots = commands.add_parser(
        "plot", help="Population comparisons and selected per-replica figures"
    )
    plots.add_argument(
        "--replicates",
        nargs="+",
        type=int,
        help="Replica IDs for detailed figures; default: first three recorded",
    )
    plots.set_defaults(handler=plot)
    for command in (training, running, analysis, plots):
        command.add_argument(
            "--morphologies", nargs="+", choices=MORPHOLOGIES, default=MORPHOLOGIES
        )
        command.add_argument(
            "--checkpoint-tag",
            default="default",
            help="Independent training identifier; analyzed separately",
        )
    for command in (training, running):
        command.add_argument("--checkpoint-root", default="checkpoints")
        command.add_argument("--data-dir", default="data")
        command.add_argument("--device", default="auto")
    for command in (running, analysis, plots):
        command.add_argument("--output-root", default="reports/experiment_2")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "bootstrap_samples", 1) <= 0:
        raise ValueError("bootstrap-samples must be positive")
    if (
        args.checkpoint_tag in ("", ".", "..")
        or "/" in args.checkpoint_tag
        or "\\" in args.checkpoint_tag
    ):
        raise ValueError("checkpoint-tag must be a single nonempty folder name")
    args.handler(args)


if __name__ == "__main__":
    main()
