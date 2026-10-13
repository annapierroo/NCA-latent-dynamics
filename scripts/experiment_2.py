from __future__ import annotations

import argparse
import errno
import heapq
import json
import platform
import shutil
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
NOISE_BRANCHES = ("noise_0p1", "noise_0p25")
COLORS = (
    "#343a40",
    "#2a6fbb",
    "#7761b2",
    "#d48c18",
    "#c44949",
    "#12a89d",
    "#b04c96",
    "#557b38",
    "#866448",
)
# Anatomical annotations in the original target image's (x, y) coordinates.
PRIMITIVE_REGIONS = {
    "gecko": {
        "leg": (((85, 151), (131, 146), (149, 176), (126, 205), (83, 203)), (107, 177)),
        "head": (((30, 0), (96, 0), (104, 69), (74, 88), (32, 79)), (65, 14)),
        "tail": (((151, 165), (189, 159), (239, 184), (239, 239), (142, 239)), (208, 222)),
        "torso": (((78, 84), (102, 71), (160, 127), (159, 152), (128, 159), (86, 128)), (119, 115)),
    },
    "butterfly": {
        "upper_left_wing": (
            ((20, 120), (259, 120), (282, 217), (252, 273), (139, 318), (20, 252)),
            (145, 205),
        ),
        "upper_right_wing": (
            ((359, 120), (598, 120), (598, 252), (479, 318), (366, 273), (336, 217)),
            (473, 205),
        ),
        "lower_left_wing": (
            ((151, 320), (262, 283), (284, 316), (263, 462), (214, 489), (135, 428)),
            (205, 390),
        ),
        "central_body": (
            (
                (284, 179),
                (334, 179),
                (347, 244),
                (335, 337),
                (319, 412),
                (299, 412),
                (281, 337),
                (270, 244),
            ),
            (309, 265),
        ),
    },
}
PRIMITIVE_SOURCE_SIZE = {"gecko": 240, "butterfly": 618}
METRICS = (
    "target_mse",
    "alpha_iou",
    "area_ratio",
    "hidden_pair_rms",
    "hidden_pca_residual",
)
METRIC_INDEX = {name: index for index, name in enumerate(METRICS)}
MORPHOLOGY_METRICS = ("target_mse", "alpha_iou", "area_ratio")
CONTROL_MORPHOLOGY_METRICS = ("rgba_control_mse", "alpha_control_iou")
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
    erase_fraction: float | None = None

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
        if self.erase_fraction is not None and not 0 < self.erase_fraction < 1:
            raise ValueError("erase-fraction must lie between 0 and 1")


def condition_names(erasure_plan: dict) -> tuple[str, ...]:
    return ("control", *NOISE_BRANCHES, *erasure_plan)


def retained_branch_indices(branches) -> list[int]:
    # Older archives include a control condition no longer used in this experiment.
    return [index for index, name in enumerate(branches) if name != "independent_control"]


def condition_labels(branches) -> list[str]:
    labels = []
    for name in branches:
        if name == "control":
            labels.append("Control")
        elif name.startswith("noise_"):
            labels.append(f"Hidden noise {float(name[6:].replace('p', '.')):.0%}")
        else:
            part = name.removeprefix("erase_")
            labels.append(
                f"Erase {float(part.replace('p', '.')):.0%}"
                if part.startswith("0p")
                else "Erase " + part.replace("_", " ")
            )
    return labels


def save_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_disk_space(path: Path, required_bytes: int = 256 * 1024**2) -> None:
    free = shutil.disk_usage(path.parent).free
    if free < required_bytes:
        raise OSError(
            errno.ENOSPC,
            f"Insufficient disk space: {free / 1024**2:.0f} MiB free, "
            f"at least {required_bytes / 1024**2:.0f} MiB required. "
            "Free space and rerun the same command; completed replicas will be skipped.",
            str(path),
        )


def save_npz(path: Path, **payload) -> None:
    # Allow for incompressible data and ZIP overhead without touching an existing archive.
    required_bytes = sum(np.asarray(value).nbytes for value in payload.values()) + 64 * 1024**2
    require_disk_space(path, required_bytes)
    temporary = path.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)
    except OSError as error:
        if error.errno == errno.ENOSPC:
            raise OSError(
                errno.ENOSPC,
                "Disk filled while saving. Free space and rerun the same command; "
                "completed replicas are unchanged, but the unsaved replica must be repeated.",
                str(path),
            ) from error
        raise
    finally:
        temporary.unlink(missing_ok=True)


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
            (alive & mask).sum() / max(int((alive | mask).sum()), 1),
            alive.sum() / max(int(mask.sum()), 1),
        ],
        dtype=np.float64,
    )


def control_morphology_metrics(state, control, threshold):
    if control is None:
        return np.full(2, np.nan)
    current_mask, control_mask = state[3] > threshold, control[3] > threshold
    union = int((current_mask | control_mask).sum())
    return np.asarray(
        [
            np.mean((state[:4].astype(np.float64) - control[:4]) ** 2),
            (current_mask & control_mask).sum() / union if union else 1.0,
        ]
    )


def inside_regime(values: np.ndarray, regime: dict) -> np.ndarray:
    mse, iou, area = (values[..., METRIC_INDEX[name]] for name in MORPHOLOGY_METRICS)
    return (
        np.isfinite(mse)
        & np.isfinite(iou)
        & np.isfinite(area)
        & (mse <= regime["target_mse_max"])
        & (iou >= regime["alpha_iou_min"])
        & (area >= regime["area_ratio_min"])
        & (area <= regime["area_ratio_max"])
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
        "target_mse_max": float(np.quantile(records[..., METRIC_INDEX["target_mse"]], quantile)),
        "alpha_iou_min": float(np.quantile(records[..., METRIC_INDEX["alpha_iou"]], 1 - quantile)),
        "area_ratio_min": float(
            np.quantile(records[..., METRIC_INDEX["area_ratio"]], 1 - quantile)
        ),
        "area_ratio_max": float(np.quantile(records[..., METRIC_INDEX["area_ratio"]], quantile)),
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
    residual_limit: float
    effective_components: int
    mean_shift_in_radii: float
    speed_ratio: float

    def project(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        centered = (state / self.scale[:, None, None]).reshape(-1) - self.mean
        scores = centered @ self.components.T
        residual = float(
            np.sqrt(max(float(centered @ centered - scores @ scores), 0) / centered.size)
        )
        return scores, residual


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
    local = LocalPCA(mean, components, ratio, scale, 0, 0, 0, 0)
    residuals = [local.project(frame)[1] for frame in validation]
    local.residual_limit = max(float(np.quantile(residuals, 0.99)), 1e-8)
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


def localized_patch(region: np.ndarray, center: tuple[float, float], count: int) -> np.ndarray:
    """Grow a connected patch inside one primitive, stopping at an exact pixel count."""
    points = np.argwhere(region)
    if len(points) < count:
        raise ValueError(f"Primitive contains {len(points)} pixels, but erasure needs {count}")
    center_y, center_x = center
    start = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
    queue = [(0.0, int(start[0]), int(start[1]))]
    queued = {tuple(start)}
    patch = np.zeros_like(region)
    removed = 0
    while queue and removed < count:
        _, y, x = heapq.heappop(queue)
        patch[y, x] = True
        removed += 1
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            yy, xx = y + dy, x + dx
            if (
                0 <= yy < region.shape[0]
                and 0 <= xx < region.shape[1]
                and region[yy, xx]
                and (yy, xx) not in queued
            ):
                queued.add((yy, xx))
                heapq.heappush(queue, ((yy - center_y) ** 2 + (xx - center_x) ** 2, yy, xx))
    if removed != count:
        raise ValueError("Primitive is too small or disconnected for this erasure fraction")
    return patch


def primitive_erasure_plan(target: np.ndarray, morphology: str, cfg: Settings) -> dict:
    from PIL import Image, ImageDraw
    from scipy.ndimage import label

    mask = target[3] > cfg.alpha_threshold
    target_count = int(mask.sum())
    if target_count == 0:
        raise ValueError("Target alpha mask is empty")
    scale = cfg.target_size / PRIMITIVE_SOURCE_SIZE[morphology]
    regions = {}
    for part, (polygon, center) in PRIMITIVE_REGIONS[morphology].items():
        image = Image.new("1", (mask.shape[1], mask.shape[0]))
        ImageDraw.Draw(image).polygon(
            [(x * scale + cfg.padding, y * scale + cfg.padding) for x, y in polygon], fill=1
        )
        region = np.asarray(image, dtype=bool) & mask
        center_yx = (center[1] * scale + cfg.padding, center[0] * scale + cfg.padding)
        points = np.argwhere(region)
        if not len(points):
            raise ValueError(f"{morphology}/{part}: anatomical annotation misses the target")
        closest = points[np.argmin(np.sum((points - center_yx) ** 2, axis=1))]
        components, _ = label(region, structure=np.ones((3, 3)))
        # Polygon boundaries can catch isolated pixels belonging to an adjacent body part.
        region = components == components[tuple(closest)]
        regions[part] = (region, center_yx)
    fraction = cfg.erase_fraction
    if fraction is None:
        fraction = int(regions["leg"][0].sum()) / target_count if morphology == "gecko" else 0.10
    count = max(1, round(fraction * target_count))
    plan = {}
    for part, (region, center_yx) in regions.items():
        try:
            patch = localized_patch(region, center_yx, count)
        except ValueError as error:
            raise ValueError(f"{morphology}/{part}: {error}; decrease --erase-fraction") from error
        plan[f"erase_{part}"] = {
            "mask": patch,
            "primitive": part,
            "requested_target_fraction": fraction,
            "actual_target_fraction": count / target_count,
            "removed_target_pixels": count,
            "primitive_target_pixels": int(region.sum()),
            "removed_primitive_fraction": count / int(region.sum()),
            "center_yx": list(center_yx),
        }
    return plan


def perturbations(
    anchor: torch.Tensor,
    target: np.ndarray,
    reference: np.ndarray,
    seed: int,
    threshold: float,
    erasure_plan: dict,
    branches=None,
):
    branches = condition_names(erasure_plan) if branches is None else tuple(branches)
    mask = target[3] > threshold
    direction = np.random.default_rng(seed).normal(size=(12, *mask.shape)) * mask
    hidden_scale = np.sqrt(
        np.mean(reference[:, 4:, :, :][:, :, mask].astype(np.float64) ** 2, axis=(0, 2))
    )
    states = [anchor.clone()]
    metadata: dict[str, object] = {}
    alive = as_array(anchor)[3] > threshold
    masks = []
    for name in branches[1:]:
        state = anchor.clone()
        if name.startswith("noise_"):
            amplitude = float(name.removeprefix("noise_").replace("p", "."))
            delta = amplitude * hidden_scale[:, None, None] * direction
            state[:, 4:] += torch.as_tensor(delta, dtype=state.dtype, device=state.device)[None]
            metadata[name] = {
                "relative_sigma": amplitude,
                "channel_sigma": (amplitude * hidden_scale).tolist(),
                "actual_rms_in_target": rms(delta[:, mask]),
            }
        else:
            details = erasure_plan[name]
            erase_mask = details["mask"]
            state[:, :, torch.as_tensor(erase_mask, device=state.device)] = 0
            masks.append(erase_mask)
            metadata[name] = {
                **{key: value for key, value in details.items() if key != "mask"},
                "removed_alive_fraction": float(
                    (erase_mask & alive).sum() / max(int(alive.sum()), 1)
                ),
            }
        states.append(state)
    return states, metadata, np.stack(masks)


def paired_updates(
    model,
    states: list,
    reference: torch.Tensor,
    system: str,
    cfg: Settings,
    shared_rng,
):
    next_states = []
    next_shared = shared_rng
    for state in states:
        if state is None:
            next_states.append(None)
            continue
        restore_rng(shared_rng, state.device)
        result = update(model, state, reference, system, cfg)
        next_shared = capture_rng(state.device)
        next_states.append(result if bool(torch.isfinite(result).all()) else None)
    return next_states, next_shared


def measure(state, control, target, cfg: Settings, local: LocalPCA):
    values = np.full(len(METRICS), np.nan)
    for name, value in zip(
        MORPHOLOGY_METRICS, morphology_metrics(state, target, cfg.alpha_threshold), strict=True
    ):
        values[METRIC_INDEX[name]] = value
    if control is not None:
        values[METRIC_INDEX["hidden_pair_rms"]] = rms(state[4:] - control[4:])
    scores, residual = local.project(state[4:])
    values[METRIC_INDEX["hidden_pca_residual"]] = residual
    return values, scores


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
    target,
    mature,
    system,
    replicate,
    regime,
    cfg,
    erasure_plan,
):
    """Record hidden projections and selected metrics without changing the paired simulation."""
    branches = condition_names(erasure_plan)
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
    pre_states = np.stack(pre_states)
    reference = pre_states[cfg.reference_start : cfg.perturb_step]
    local = fit_local_pca(reference[:, 4:], cfg.pca_components, cfg.seed + replicate)
    states, perturbation_metadata, erase_masks = perturbations(
        state,
        target_array,
        reference,
        cfg.seed + 3_000_000 + replicate,
        cfg.alpha_threshold,
        erasure_plan,
        branches,
    )
    shape = (len(branches), cfg.horizon + 1)
    curves = np.full((*shape, len(METRICS)), np.nan)
    control_morphology = np.full((*shape, len(CONTROL_MORPHOLOGY_METRICS)), np.nan)
    coordinates = np.full((*shape, len(local.components)), np.nan, dtype=np.float32)
    in_target = np.zeros(shape, dtype=bool)
    divergence = np.full(len(branches), -1, dtype=np.int64)
    snap_steps = snapshot_schedule(cfg) if replicate < cfg.snapshot_replicates else []
    snapshots = np.full(
        (len(branches), len(snap_steps), 4, *target.shape[-2:]), np.nan, dtype=np.float32
    )
    snap_indices = {step: index for index, step in enumerate(snap_steps)}

    def record(branch, step, array, control):
        values, scores = measure(array, control, target_array, cfg, local)
        curves[branch, step] = values
        control_morphology[branch, step] = control_morphology_metrics(
            array, control, cfg.alpha_threshold
        )
        coordinates[branch, step] = scores
        in_target[branch, step] = inside_regime(values, regime)
        if step in snap_indices:
            snapshots[branch, snap_indices[step]] = array[:4]

    for step in range(cfg.perturb_step):
        record(0, step, pre_states[step], pre_states[step])
    for branch in range(1, len(branches)):
        curves[branch, : cfg.perturb_step] = curves[0, : cfg.perturb_step]
        control_morphology[branch, : cfg.perturb_step] = control_morphology[0, : cfg.perturb_step]
        coordinates[branch, : cfg.perturb_step] = coordinates[0, : cfg.perturb_step]
        in_target[branch, : cfg.perturb_step] = in_target[0, : cfg.perturb_step]
        for step, index in snap_indices.items():
            if step < cfg.perturb_step:
                snapshots[branch, index] = snapshots[0, index]
    for step in range(cfg.perturb_step, cfg.horizon + 1):
        arrays = [None if state is None else as_array(state) for state in states]
        for branch, array in enumerate(arrays):
            if array is None:
                if divergence[branch] == -1:
                    divergence[branch] = step
                continue
            record(branch, step, array, arrays[0])
        if step == cfg.horizon or all(state is None for state in states):
            break
        states, shared_rng = paired_updates(model, states, mature, system, cfg, shared_rng)
    payload = {
        "steps": np.arange(cfg.horizon + 1),
        "branch_names": np.asarray(branches),
        "metric_names": np.asarray(METRICS),
        "metrics": curves,
        "control_morphology": control_morphology,
        "hidden_scores": coordinates,
        "hidden_pca_mean": local.mean,
        "hidden_pca_components": local.components,
        "hidden_scale": local.scale,
        "hidden_variance_ratio": local.variance_ratio,
        "in_target": in_target,
        "divergence_step": divergence,
        "snapshot_steps": np.asarray(snap_steps, dtype=np.int64),
        "snapshots": snapshots,
        "mature_state": mature_array.astype(np.float32),
        "pre_pulse_state": pre_states[-1],
        "erase_masks": erase_masks,
    }
    split_step = cfg.reference_start + len(reference) // 2
    return payload, {
        "status": "complete",
        "system": system,
        "replicate": replicate,
        "perturbations": perturbation_metadata,
        "local_pca": {
            "hidden": {
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
        },
        "fit_steps_inclusive": [cfg.reference_start, split_step - 1],
        "validation_steps_inclusive": [split_step, cfg.perturb_step - 1],
        "in_target_before_pulse": bool(
            inside_regime(
                morphology_metrics(pre_states[-1], target_array, cfg.alpha_threshold), regime
            )
        ),
        "timing": "t=0 is target attainment; t=pulse contains the post-pulse state.",
        "limitations": "Pre-reference stationarity is a short-window heuristic. PCA proximity "
        "and finite recovery tests do not establish a mathematical attractor.",
    }


@torch.no_grad()
def replay_snapshots(
    model,
    target: torch.Tensor,
    mature: torch.Tensor,
    system: str,
    replicate: int,
    cfg: Settings,
    expected_pre_pulse: np.ndarray,
    expected_metrics: np.ndarray,
    branches,
    erasure_plan: dict,
    *,
    control_metrics: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay saved trajectories for snapshots and optional stepwise control comparisons."""
    steps = np.asarray(snapshot_schedule(cfg), dtype=np.int64)
    indices = {int(step): index for index, step in enumerate(steps)}
    snapshots = np.full(
        (len(branches), len(steps), 4, *target.shape[-2:]), np.nan, dtype=np.float32
    )
    seed_nca(cfg.seed + 1_000_000 + replicate, target.device)
    state = mature.clone()
    reference = []
    target_array = as_array(target)

    def record(branch, step, array, control):
        measured = morphology_metrics(array, target_array, cfg.alpha_threshold)
        saved = expected_metrics[branch, step, : len(MORPHOLOGY_METRICS)]
        if not np.allclose(measured, saved, rtol=1e-5, atol=1e-6, equal_nan=True):
            raise ValueError(f"Replayed t={step} differs from saved metrics for {branches[branch]}")
        if control_metrics is not None:
            distance = rms(array[4:] - control[4:]) if control is not None else np.nan
            if not np.isclose(
                distance,
                expected_metrics[branch, step, METRIC_INDEX["hidden_pair_rms"]],
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
            ):
                raise ValueError(f"Paired replay differs for {branches[branch]} at t={step}")
            control_metrics[branch, step] = control_morphology_metrics(
                array, control, cfg.alpha_threshold
            )
        if step in indices:
            snapshots[branch, indices[step]] = array[:4]

    for step in range(cfg.perturb_step):
        if step in indices or control_metrics is not None:
            array = as_array(state)
            for branch in range(len(branches)):
                record(branch, step, array, array)
        if step >= cfg.reference_start:
            reference.append(as_array(state).astype(np.float32))
        state = update(model, state, mature, system, cfg)
        if not bool(torch.isfinite(state).all()):
            raise ValueError(f"Non-finite pre-pulse state at t={step + 1}")
    pre_pulse = as_array(state).astype(np.float32)
    if not np.array_equal(pre_pulse, expected_pre_pulse):
        raise ValueError(
            f"Replayed t={cfg.perturb_step} differs from the saved run; use the original checkpoint and device"
        )
    shared_rng = capture_rng(target.device)
    states, _, _ = perturbations(
        state,
        as_array(target),
        np.stack(reference),
        cfg.seed + 3_000_000 + replicate,
        cfg.alpha_threshold,
        erasure_plan,
        branches,
    )
    for step in range(cfg.perturb_step, cfg.horizon + 1):
        if step in indices or control_metrics is not None:
            arrays = [None if state is None else as_array(state) for state in states]
            for branch, array in enumerate(arrays):
                if array is None:
                    if np.isfinite(expected_metrics[branch, step, : len(MORPHOLOGY_METRICS)]).any():
                        raise ValueError(f"Replayed {branches[branch]} diverged early at t={step}")
                    continue
                record(branch, step, array, arrays[0])
        if step == cfg.horizon or all(branch_state is None for branch_state in states):
            break
        states, shared_rng = paired_updates(model, states, mature, system, cfg, shared_rng)
    return steps, snapshots


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
                "erase_fraction",
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
        target_path = Path(args.data_dir) / f"{morphology}.png"
        target = load_distill_target(target_path, cfg.target_size, cfg.padding, device)
        erasure_plan = primitive_erasure_plan(as_array(target), morphology, cfg)
        folder = Path(args.output_root) / morphology / args.checkpoint_tag
        folder.mkdir(parents=True, exist_ok=True)
        paths = {
            system: checkpoint_path(
                Path(args.checkpoint_root), morphology, system, args.checkpoint_tag
            )
            for system in SYSTEMS
        }
        manifest = {
            "protocol": (
                "persistent_vs_mature_restore_primitive_erasure_v4"
                if morphology == "butterfly"
                else "persistent_vs_mature_restore_primitive_erasure_v3"
            ),
            "recorded_metrics": list(METRICS),
            "branch_names": list(condition_names(erasure_plan)),
            "erasure_regions": {
                name: {key: value for key, value in detail.items() if key != "mask"}
                for name, detail in erasure_plan.items()
            },
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
            recorded = json.loads(manifest_path.read_text())
            # The script hash includes plotting code; simulation changes require a protocol bump.
            changed = sorted(
                key
                for key in manifest
                if key != "implementation_sha256" and recorded.get(key) != manifest.get(key)
            )
            if changed:
                raise ValueError(
                    f"Protocol/inputs/environment changed ({', '.join(changed)}). "
                    f"Use the original run options to resume {folder}, "
                    "or choose a different --output-root for the new protocol."
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
        plot_erasure_plan(as_array(target), erasure_plan, folder / "primitive_erase_masks.png")
        regime_path = folder / "calibration.json"
        if regime_path.exists():
            regime = json.loads(regime_path.read_text())
        else:
            regime, records = calibrate(models, target, cfg)
            save_npz(
                folder / "calibration_metrics.npz",
                records=records,
                systems=np.asarray(SYSTEMS),
                metric_names=np.asarray(MORPHOLOGY_METRICS),
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
                require_disk_space(result_dir / "trajectories.npz")
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
                        model, target, mature, system, replicate, regime, cfg, erasure_plan
                    )
                metadata.update(
                    {
                        "system": system,
                        "replicate": replicate,
                        "mature_step": mature_step,
                        "development_seed": cfg.seed + replicate,
                        "continuation_seed": cfg.seed + 1_000_000 + replicate,
                        "perturbation_seed": cfg.seed + 3_000_000 + replicate,
                    }
                )
                if payload:
                    save_npz(result_dir / "trajectories.npz", **payload)
                save_json(metadata_path, metadata)
                print(f"Saved {result_dir}", flush=True)


def recovery_rows(payload: dict, metadata: dict, cfg: Settings) -> list[dict]:
    pulse = cfg.perturb_step
    late = slice(max(pulse + 1, cfg.horizon - 199), cfg.horizon + 1)
    metrics = payload["metrics"]
    local = metadata["local_pca"]["hidden"]
    reference_stationary = bool(local["approximately_stationary_heuristic"])
    morphology_indices = [METRIC_INDEX[name] for name in MORPHOLOGY_METRICS]
    control_finite = bool(
        payload["divergence_step"][0] < 0
        and np.isfinite(metrics[0, late][:, morphology_indices]).all()
    )
    pre_target = bool(payload["in_target"][0, pulse])
    rows = []
    for branch, name in enumerate(payload["branch_names"]):
        target = payload["in_target"][branch]
        flags = target[pulse:]
        exits = np.flatnonzero(~flags)
        first_exit = int(exits[0]) if exits.size else None
        recovery = None
        if first_exit is not None:
            returned = first_sustained(flags[first_exit + 1 :], cfg.recovery_patience)
            if returned is not None:
                recovery = first_exit + 1 + returned
        divergence = int(payload["divergence_step"][branch])
        morphology_finite = bool(
            divergence < 0 and np.isfinite(metrics[branch, late][:, morphology_indices]).all()
        )
        target_fraction = float(target[late].mean())
        never_left = bool(pre_target and first_exit is None)
        recovered = bool(
            pre_target and recovery is not None and morphology_finite and target_fraction >= 0.95
        )
        distance = metrics[branch, :, METRIC_INDEX["hidden_pair_rms"]]
        initial = float(distance[pulse])
        pair_eligible = bool(
            pre_target and control_finite and np.isfinite(initial) and initial > EPS
        )
        close = np.isfinite(distance) & (distance <= 0.1 * initial)
        pair_return = (
            first_sustained(close[pulse:], cfg.recovery_patience) if pair_eligible else None
        )
        finite_distance = distance[late][np.isfinite(distance[late])]
        rows.append(
            {
                "branch": str(name),
                "in_target_before_pulse": pre_target,
                "reference_stationary_heuristic": reference_stationary,
                "reference_explained_variance": local["explained_variance"],
                "reference_reached_95_percent": local["reached_95_percent"],
                "first_target_exit_delay": first_exit if first_exit is not None else "",
                "first_morphology_recovery_delay": recovery if recovery is not None else "",
                "morphology_never_left_target": never_left,
                "morphology_recovered_after_exit": recovered,
                "morphology_retained_or_recovered": bool(
                    morphology_finite and (never_left or recovered)
                ),
                "last_200_target_fraction": target_fraction,
                "paired_hidden_return_eligible": pair_eligible,
                "first_paired_hidden_return_delay": pair_return if pair_return is not None else "",
                "paired_hidden_return": bool(
                    pair_eligible
                    and divergence < 0
                    and np.isfinite(distance[late]).all()
                    and pair_return is not None
                    and close[late].mean() >= 0.95
                ),
                "last_200_hidden_pair_rms": float(np.median(finite_distance))
                if finite_distance.size
                else float("nan"),
                "final_target_mse": float(metrics[branch, -1, METRIC_INDEX["target_mse"]]),
                "final_alpha_iou": float(metrics[branch, -1, METRIC_INDEX["alpha_iou"]]),
                "final_hidden_pair_rms": float(distance[-1]),
                "numerical_divergence_step": divergence if divergence >= 0 else "",
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


def aggregate(rows: list[dict], bootstrap_samples: int, seed: int):
    from scipy.stats import binomtest

    endpoints = {
        "morphology_retained_or_recovered": "in_target_before_pulse",
        "morphology_never_left_target": "in_target_before_pulse",
        "morphology_recovered_after_exit": "exited_target",
        "paired_hidden_return": "paired_hidden_return_eligible",
    }
    rows = [
        {
            **row,
            "exited_target": bool(
                row["in_target_before_pulse"] and not row["morphology_never_left_target"]
            ),
        }
        for row in rows
    ]
    summaries, comparisons = [], []
    rng = np.random.default_rng(seed)
    branches = tuple(dict.fromkeys(row["branch"] for row in rows))
    for branch in branches:
        for endpoint, eligibility in endpoints.items():
            by_system = {}
            for system in SYSTEMS:
                selected = [r for r in rows if r["system"] == system and r["branch"] == branch]
                eligible = [r for r in selected if r[eligibility]]
                by_system[system] = {r["replicate"]: r for r in eligible}
                successes = sum(bool(r[endpoint]) for r in eligible)
                low, high = wilson(successes, len(eligible))
                delays = [
                    r["first_morphology_recovery_delay"]
                    for r in eligible
                    if r["morphology_recovered_after_exit"]
                ]
                summaries.append(
                    {
                        "system": system,
                        "branch": branch,
                        "endpoint": endpoint,
                        "n_recorded": len(selected),
                        "n_eligible": len(eligible),
                        "n_numerical_failures": sum(
                            r["numerical_divergence_step"] != "" for r in eligible
                        ),
                        "successes": successes,
                        "fraction": successes / len(eligible) if eligible else float("nan"),
                        "ci95_low": low,
                        "ci95_high": high,
                        "median_recovery_delay_after_exit": float(np.median(delays))
                        if delays and endpoint.startswith("morphology_")
                        else "",
                        "n_recovered_delays": len(delays)
                        if endpoint.startswith("morphology_")
                        else "",
                    }
                )
            pairs = sorted(by_system[SYSTEMS[0]].keys() & by_system[SYSTEMS[1]].keys())
            if not pairs:
                continue
            delta = np.asarray(
                [
                    int(by_system[SYSTEMS[1]][i][endpoint])
                    - int(by_system[SYSTEMS[0]][i][endpoint])
                    for i in pairs
                ]
            )
            boot = np.empty(bootstrap_samples)
            for start in range(0, bootstrap_samples, 500):
                indices = rng.integers(
                    len(delta), size=(min(500, bootstrap_samples - start), len(delta))
                )
                boot[start : start + len(indices)] = delta[indices].mean(axis=1)
            positive, negative = int((delta > 0).sum()), int((delta < 0).sum())
            comparisons.append(
                {
                    "branch": branch,
                    "endpoint": endpoint,
                    "n_matched_evaluation_seeds": len(pairs),
                    "growing_minus_persistent_fraction": float(delta.mean()),
                    "paired_bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                    "paired_bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                    "mcnemar_exact_p_value": float(binomtest(positive, positive + negative).pvalue)
                    if positive + negative
                    else 1.0,
                    "bootstrap_samples": bootstrap_samples,
                }
            )
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["mcnemar_exact_p_value"])
    adjusted = 0.0
    for rank, index in enumerate(order):
        adjusted = max(
            adjusted, min(1.0, (len(order) - rank) * comparisons[index]["mcnemar_exact_p_value"])
        )
        comparisons[index]["p_value_holm_within_morphology_checkpoint"] = adjusted
    return summaries, comparisons


def analyze(args) -> None:
    regime = {
        "target_mse_max": getattr(args, "target_mse_max", 0.001),
        "alpha_iou_min": getattr(args, "target_iou_min", 0.95),
        "area_ratio_min": getattr(args, "target_area_min", 0.9),
        "area_ratio_max": getattr(args, "target_area_max", 1.1),
    }
    if not (
        np.isfinite(list(regime.values())).all()
        and regime["target_mse_max"] > 0
        and 0 <= regime["alpha_iou_min"] <= 1
        and 0 < regime["area_ratio_min"] <= regime["area_ratio_max"]
    ):
        raise ValueError("Invalid morphology evaluation thresholds")
    found = False
    for morphology, folder in result_folders(args):
        found = True
        cfg = Settings(**json.loads((folder / "manifest.json").read_text())["settings"])
        output = folder / "analysis"
        output.mkdir(exist_ok=True)
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
                payload = load_payload(result_dir)
                payload["in_target"] = inside_regime(payload["metrics"], regime)
                rows.extend(
                    {
                        "morphology": morphology,
                        "checkpoint_tag": folder.name,
                        "system": system,
                        "replicate": metadata["replicate"],
                        **row,
                    }
                    for row in recovery_rows(payload, metadata, cfg)
                )
                print(f"Analyzed {result_dir}", flush=True)
        save_json(output / "exclusions.json", exclusions)
        if not rows:
            print(f"No completed trajectories in {folder}", flush=True)
            continue
        summaries, comparisons = aggregate(rows, args.bootstrap_samples, cfg.seed + 4_000_000)
        write_csv(output / "recovery_by_replica.csv", rows)
        write_csv(output / "recovery_summary.csv", summaries)
        write_csv(output / "paired_model_comparison.csv", comparisons)
        save_json(
            output / "analysis_settings.json",
            {
                "metrics": list(METRICS),
                "morphology_evaluation": regime,
                "morphology_definition": "Raw RGBA MSE over the full grid; alpha > 0.1 for IoU and area. "
                "All three thresholds must pass. Development calibration is unchanged.",
                "morphology_recovery": f"After first exit: {cfg.recovery_patience} consecutive target states "
                "and >=95% of final 200 states in target. Never-exited trajectories are reported separately.",
                "hidden_pair_return": f"Absolute hidden RMS <=10% of initial pulse distance for "
                f"{cfg.recovery_patience} consecutive steps and >=95% of final 200. Undefined for zero initial "
                "distance, as in the unperturbed control.",
                "hidden_pca": "Same hidden-only PCA for every branch of each model/replica. First temporal "
                "half fits PCA/scales; second half sets a q99 residual threshold for projection diagnostics. "
                "Residual and explained variance are not used to classify latent-regime recovery.",
                "reference_stationarity": "Pre-reference mean shift <=1 radius and speed ratio 0.5-2. "
                "These short-window checks are heuristics, not hypothesis tests.",
                "recovery_time": "Delay from pulse, conditional on observed sustained recovery after exit; "
                "unrecovered trajectories are counted, not assigned zero recovery time.",
                "uncertainty": "95% Wilson intervals over endpoint-eligible rollout replicas; numerical "
                "failures are nonreturns. Paired bootstrap and exact McNemar on common eligible IDs; "
                "Holm correction across binary comparisons within this morphology/checkpoint pair.",
                "bootstrap_samples": args.bootstrap_samples,
                "scope": "One checkpoint pair; rollout replicas are not independent trainings. Morphologies "
                "are not pooled. Raw latent distances may differ in scale across trained models.",
                "figures": "Medians and IQR of finite replicas; missing/numerically diverged values are not "
                "imputed. PCA geometry alone does not prove an attractor or latent-regime recovery.",
            },
        )
    if not found:
        raise FileNotFoundError("No Experiment 2 manifests found under the requested output root")


def load_payload(result_dir: Path) -> dict:
    """Read retained conditions and metrics from both current and older archives."""
    with np.load(result_dir / "trajectories.npz", allow_pickle=False) as archive:
        names = archive["metric_names"].tolist()
        missing = set(METRICS) - set(names)
        if missing:
            raise ValueError(f"{result_dir}: missing metrics {sorted(missing)}")
        payload = {
            "metrics": archive["metrics"][..., [names.index(name) for name in METRICS]],
            "metric_names": np.asarray(METRICS),
        }
        for key in (
            "steps",
            "branch_names",
            "hidden_scores",
            "hidden_variance_ratio",
            "divergence_step",
            "snapshot_steps",
            "snapshots",
            "pre_pulse_state",
            "mature_state",
            "erase_masks",
            "control_morphology",
        ):
            if key in archive:
                payload[key] = archive[key]
    indices = retained_branch_indices(payload["branch_names"])
    for key in (
        "branch_names",
        "metrics",
        "hidden_scores",
        "divergence_step",
        "snapshots",
        "control_morphology",
    ):
        if key in payload:
            payload[key] = payload[key][indices]
    return payload


def load_control_comparison(result_dir, payload):
    if "control_morphology" in payload:
        values = payload["control_morphology"]
    else:
        path = result_dir / "morphology_control_comparison.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as archive:
            if (
                str(archive["source_sha256"]) != file_sha256(result_dir / "trajectories.npz")
                or not np.array_equal(archive["steps"], payload["steps"])
                or not np.array_equal(archive["branch_names"], payload["branch_names"])
                or tuple(archive["metric_names"]) != CONTROL_MORPHOLOGY_METRICS
            ):
                raise ValueError(f"Control comparison does not match the saved run: {path}")
            values = archive["metrics"]
    expected = (*payload["metrics"].shape[:2], len(CONTROL_MORPHOLOGY_METRICS))
    if values.shape != expected:
        raise ValueError(f"Invalid control comparison shape in {result_dir}")
    return values


def select_replicas(directories: list[Path]) -> list[Path]:
    """Use the same selected rollout for every perturbation, not a best-case image per branch."""
    candidates, features = [], []
    for path in directories:
        values = load_payload(path)["metrics"][0]
        selected = values[:, [METRIC_INDEX[name] for name in ("target_mse", "alpha_iou")]]
        if np.isfinite(selected).all():
            candidates.append(path)
            features.append([np.log1p(selected[:, 0]).mean(), selected[:, 1].mean()])
    if not candidates:
        return directories[:3]
    features = np.asarray(features)
    spread = np.maximum(np.diff(np.quantile(features, (0.25, 0.75), axis=0), axis=0)[0], 1e-6)
    distance = np.linalg.norm((features - np.median(features, axis=0)) / spread, axis=1)
    return [candidates[index] for index in np.argsort(distance, kind="stable")[:3]]


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


def plot_erasure_plan(target: np.ndarray, plan: dict, path: Path) -> None:
    plt = plotting()
    figure, axes = plt.subplots(2, len(plan), figsize=(3 * len(plan), 6), squeeze=False)
    for column, ((name, detail), label) in enumerate(
        zip(plan.items(), condition_labels(plan), strict=True)
    ):
        mask = detail["mask"]
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = (0.9, 0.1, 0.15, 0.8)
        axes[0, column].imshow(display_rgba(target))
        axes[0, column].imshow(overlay)
        erased = target.copy()
        erased[:, mask] = 0
        axes[1, column].imshow(display_rgba(erased))
        axes[0, column].set_title(
            f"{label}\n{detail['removed_target_pixels']} pixels ({detail['actual_target_fraction']:.2%} of target)",
            fontsize=10,
        )
        axes[1, column].set_title("Immediately after erasure", fontsize=10)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Primitive erasure: identical target area per region; red = removed cells")
    figure.tight_layout()
    finish_figure(plt, figure, path)


def preview_erasure(args) -> None:
    cfg = Settings(erase_fraction=args.erase_fraction)
    cfg.validate()
    output = Path(args.output_root) / "mask_preview"
    output.mkdir(parents=True, exist_ok=True)
    for morphology in args.morphologies:
        target = as_array(load_distill_target(Path(args.data_dir) / f"{morphology}.png"))
        plan = primitive_erasure_plan(target, morphology, cfg)
        path = output / f"{morphology}.png"
        plot_erasure_plan(target, plan, path)
        print(f"Saved {path}", flush=True)


def plot_snapshots(plt, payload: dict, result_dir: Path, cfg: Settings) -> None:
    branches = payload["branch_names"]
    snap_steps = payload["snapshot_steps"]
    if not snap_steps.size:
        return
    snapshot_columns = [(int(step), index) for index, step in enumerate(snap_steps)]
    snapshot_columns.insert(
        int(np.searchsorted(snap_steps, cfg.perturb_step)), (cfg.perturb_step, None)
    )
    figure, axes = plt.subplots(
        len(branches),
        len(snap_steps) + 1,
        figsize=(1.7 * (len(snap_steps) + 1), 1.6 * len(branches)),
        squeeze=False,
    )
    for branch, label in enumerate(condition_labels(branches)):
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


def plot_replica(result_dir: Path, cfg: Settings, output_dir: Path) -> None:
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt = plotting()
    payload = load_payload(result_dir)
    scores, steps = payload["hidden_scores"], payload["steps"]
    branches = payload["branch_names"]
    ratio = payload["hidden_variance_ratio"]
    indices = np.unique(
        np.r_[np.arange(0, len(steps), 10), cfg.perturb_step, cfg.perturb_step + 1, cfg.horizon]
    )
    reference = (steps >= cfg.reference_start) & (steps < cfg.perturb_step)
    norm = Normalize(0, cfg.horizon)
    columns, rows = 3, (len(branches) + 2) // 3
    output_dir.mkdir(parents=True, exist_ok=True)
    for dimension in (2, 3):
        figure = plt.figure(figsize=(5.2 * columns, 3.9 * rows + 1.2), layout="constrained")
        axes = []
        for branch, label in enumerate(condition_labels(branches)):
            axis = figure.add_subplot(
                rows, columns, branch + 1, projection="3d" if dimension == 3 else None
            )
            axes.append(axis)
            base = scores[0, indices, :dimension]
            ref = scores[0, reference, :dimension]
            points = scores[branch, indices, :dimension]
            axis.plot(*base.T, color="0.6", linestyle="--", linewidth=0.9, zorder=1)
            axis.scatter(*ref.T, color="#72b7cc", s=9, alpha=0.65, zorder=2)
            axis.plot(*points.T, color="0.7", linewidth=0.5, alpha=0.5, zorder=1)
            scatter = axis.scatter(
                *points.T, c=steps[indices], cmap="plasma", norm=norm, s=9, zorder=3
            )
            for point, marker, color in (
                (scores[0, cfg.perturb_step, :dimension], "o", "black"),
                (scores[branch, cfg.perturb_step, :dimension], "D", "#e87916"),
                (scores[branch, -1, :dimension], "x", "#007ea7"),
            ):
                if np.isfinite(point).all():
                    axis.scatter(*point, marker=marker, color=color, s=38, zorder=5)
            axis.set_xlabel(f"PC1 ({ratio[0]:.1%})", fontsize=9)
            axis.set_ylabel(f"PC2 ({ratio[1]:.1%})", fontsize=9)
            if dimension == 3:
                axis.set_zlabel(f"PC3 ({ratio[2]:.1%})", fontsize=9)
                axis.zaxis.set_major_locator(MaxNLocator(4))
                axis.view_init(elev=23, azim=-55)
            else:
                axis.grid(alpha=0.2)
            axis.set_title(label, fontsize=10)
            axis.xaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.tick_params(labelsize=8)
        figure.colorbar(scatter, ax=axes, fraction=0.025, label="Steps after target attainment")
        figure.legend(
            handles=[
                Line2D([], [], color="0.6", linestyle="--", label="Unperturbed control"),
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    color="#72b7cc",
                    label="Pre-pulse reference (fit + validation)",
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    color="black",
                    label="Immediately before pulse",
                ),
                Line2D(
                    [],
                    [],
                    marker="D",
                    linestyle="none",
                    color="#e87916",
                    label="Immediately after pulse",
                ),
                Line2D(
                    [],
                    [],
                    marker="x",
                    linestyle="none",
                    color="#007ea7",
                    label="Final state (if finite)",
                ),
            ],
            loc="outside lower center",
            ncol=3,
            fontsize=9,
        )
        figure.suptitle(
            f"{result_dir.parent.name}, {result_dir.name}: hidden-state PCA\n"
            "Axes fitted only before perturbation; geometry does not prove an attractor",
            fontsize=12,
        )
        finish_figure(plt, figure, output_dir / f"hidden_pca_{dimension}d.png")


def plot_population(folder: Path, cfg: Settings, *, control_only=False) -> None:
    import warnings

    plt = plotting()
    data, comparisons, branches = {}, {}, None
    for system in SYSTEMS:
        replicas, control_replicas = [], []
        for path in sorted((folder / system).glob("replicate_*/trajectories.npz")):
            payload = load_payload(path.parent)
            names = payload["branch_names"].tolist()
            if branches is not None and branches != names:
                raise ValueError("Cannot pool replicas with different perturbation conditions")
            branches = names
            replicas.append(payload["metrics"])
            control_values = load_control_comparison(path.parent, payload)
            if control_values is not None:
                control_replicas.append(control_values)
        data[system] = np.stack(replicas) if replicas else None
        comparisons[system] = (
            np.stack(control_replicas)
            if control_replicas and len(control_replicas) == len(replicas)
            else None
        )
    if branches is None:
        return
    output = folder / "analysis"
    output.mkdir(exist_ok=True)
    figures = (
        (
            "morphological_stability_recovery.png",
            ("target_mse", "alpha_iou"),
            ("Global RGBA target MSE", "Alpha-mask IoU"),
            "Morphological Stability and Recovery",
            data,
            METRIC_INDEX,
        ),
        (
            "hidden_divergence_from_control.png",
            ("hidden_pair_rms",),
            ("Hidden-state RMS distance to paired control",),
            "Hidden-State Deviation from the Paired Unperturbed Control",
            data,
            METRIC_INDEX,
        ),
    )
    if control_only:
        figures = ()
    if all(data[system] is None or comparisons[system] is not None for system in SYSTEMS):
        figures += (
            (
                "morphological_deviation_from_control.png",
                CONTROL_MORPHOLOGY_METRICS,
                ("Global RGBA MSE to paired control", "Alpha-mask IoU with paired control"),
                "Morphological Deviation from the Time-Matched Unperturbed Control",
                comparisons,
                {name: index for index, name in enumerate(CONTROL_MORPHOLOGY_METRICS)},
            ),
        )
    else:
        print(f"Missing stepwise visible comparisons in {folder}; run compare-control", flush=True)
    titles = ("Persistent NCA", "Growing NCA + mature restore (beta=0.1)")
    for filename, names, labels, title, figure_data, metric_indices in figures:
        figure, axes = plt.subplots(
            len(names),
            2,
            figsize=(13, 3.6 * len(names) + 1.3),
            squeeze=False,
            sharex=True,
            layout="constrained",
        )
        handles = []
        for column, system in enumerate(SYSTEMS):
            values = figure_data[system]
            if values is None:
                for axis in axes[:, column]:
                    axis.set_title(titles[column] + ": no recorded replicas")
                continue
            for row, (name, label) in enumerate(zip(names, labels, strict=True)):
                axis = axes[row, column]
                for branch, (branch_label, color) in enumerate(
                    zip(condition_labels(branches), COLORS)
                ):
                    curves = values[:, branch, :, metric_indices[name]].copy()
                    curves[~np.isfinite(curves)] = np.nan
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        low, median, high = np.nanpercentile(curves, (25, 50, 75), axis=0)
                    (line,) = axis.plot(
                        np.arange(cfg.horizon + 1),
                        median,
                        color=color,
                        label=branch_label,
                        linewidth=1.3,
                    )
                    axis.fill_between(
                        np.arange(cfg.horizon + 1), low, high, color=color, alpha=0.12
                    )
                    if not column and not row:
                        handles.append(line)
                axis.axvline(cfg.perturb_step, color="0.4", linestyle="--", linewidth=1)
                if name in ("alpha_iou", "alpha_control_iou"):
                    axis.set_ylim(0, 1.02)
                elif name in ("target_mse", "hidden_pair_rms", "rgba_control_mse"):
                    axis.set_yscale("symlog", linthresh=1e-6)
                axis.set_ylim(bottom=0)
                axis.set_ylabel(label, fontsize=10)
                axis.grid(alpha=0.2)
                if row == 0:
                    axis.set_title(f"{titles[column]} (n={len(values)})", fontsize=11)
                if row == len(names) - 1:
                    axis.set_xlabel("Steps after target attainment")
        if not handles:
            handles = [
                plt.Line2D([], [], color=color, label=label)
                for label, color in zip(condition_labels(branches), COLORS)
            ]
        handles.append(
            plt.Line2D(
                [], [], color="0.4", linestyle="--", label=f"Perturbation (t={cfg.perturb_step})"
            )
        )
        figure.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8)
        figure.suptitle(
            f"{folder.parent.name}: {title}\nMedian and IQR across rollout replicates",
            fontsize=12,
        )
        finish_figure(plt, figure, output / filename)


def compare_control(args) -> None:
    from threadpoolctl import threadpool_limits

    found = False
    for morphology, folder in result_folders(args):
        found = True
        manifest = json.loads((folder / "manifest.json").read_text())
        cfg = Settings(**manifest["settings"])
        target = None
        counts = {}
        for system in SYSTEMS:
            model, count = None, 0
            for archive_path in sorted((folder / system).glob("replicate_*/trajectories.npz")):
                result_dir = archive_path.parent
                metadata = json.loads((result_dir / "metadata.json").read_text())
                if metadata["status"] != "complete":
                    continue
                payload = load_payload(result_dir)
                if load_control_comparison(result_dir, payload) is None:
                    source_hash = file_sha256(archive_path)
                    if model is None:
                        environment = {
                            "python": platform.python_version(),
                            "torch": str(torch.__version__),
                            "numpy": np.__version__,
                        }
                        if any(manifest.get(key) != value for key, value in environment.items()):
                            raise ValueError(
                                "Control replay requires the original numerical environment"
                            )
                        checkpoint = Path(manifest["checkpoint_paths"][system])
                        target_path = Path(manifest["target_path"])
                        if file_sha256(checkpoint) != manifest["checkpoint_hashes"][system]:
                            raise ValueError(f"Checkpoint changed: {checkpoint}")
                        if file_sha256(target_path) != manifest["target_hash"]:
                            raise ValueError(f"Target changed: {target_path}")
                        device = get_device(manifest["device"])
                        model = load_growing_checkpoint(
                            checkpoint,
                            channels=cfg.channels,
                            hidden_size=cfg.hidden_size,
                            update_probability=cfg.fire_rate,
                            device=device,
                        )
                        if target is None:
                            target = load_distill_target(
                                target_path, cfg.target_size, cfg.padding, device
                            )
                    branches = payload["branch_names"].tolist()
                    plan = {
                        name: {**metadata["perturbations"][name], "mask": mask}
                        for name, mask in zip(
                            [name for name in branches if name.startswith("erase_")],
                            payload["erase_masks"],
                            strict=True,
                        )
                    }
                    values = np.full(
                        (*payload["metrics"].shape[:2], len(CONTROL_MORPHOLOGY_METRICS)), np.nan
                    )
                    print(
                        f"Control comparison replay: {morphology}/{system}/{result_dir.name}",
                        flush=True,
                    )
                    with threadpool_limits(limits=1):
                        replay_snapshots(
                            model,
                            target,
                            torch.as_tensor(payload["mature_state"], device=device).unsqueeze(0),
                            system,
                            int(metadata["replicate"]),
                            cfg,
                            payload["pre_pulse_state"],
                            payload["metrics"],
                            branches,
                            plan,
                            control_metrics=values,
                        )
                    if file_sha256(archive_path) != source_hash:
                        raise ValueError(f"Saved trajectory changed during replay: {archive_path}")
                    save_npz(
                        result_dir / "morphology_control_comparison.npz",
                        steps=payload["steps"],
                        branch_names=payload["branch_names"],
                        metric_names=np.asarray(CONTROL_MORPHOLOGY_METRICS),
                        metrics=values,
                        source_sha256=np.asarray(source_hash),
                    )
                count += 1
            counts[system] = count
        if not any(counts.values()):
            print(f"No completed trajectories in {folder}", flush=True)
            continue
        plot_population(folder, cfg, control_only=True)
        output = folder / "analysis"
        output.mkdir(exist_ok=True)
        save_json(
            output / "control_comparison_settings.json",
            {
                "metrics": list(CONTROL_MORPHOLOGY_METRICS),
                "reference": "Paired unperturbed trajectory at the SAME simulation step, not a fixed snapshot",
                "rgba_control_mse": "Raw RGBA MSE over the full grid, including background; zero means identical RGBA",
                "alpha_control_iou": f"IoU between alpha masks > {cfg.alpha_threshold}; one means identical masks (including two empty masks)",
                "steps": [0, cfg.horizon],
                "perturbation_step": cfg.perturb_step,
                "replicates": counts,
                "aggregation": "Median and 25th-75th percentiles of finite paired comparisons at each step; missing states are not imputed",
                "limitation": "A match to the control does not imply target recovery: the control may itself lose the morphology. Missing control states make both comparisons undefined",
            },
        )
        print(f"Saved {output / 'morphological_deviation_from_control.png'}", flush=True)
    if not found:
        raise FileNotFoundError("No Experiment 2 manifests found")


def plot(args) -> None:
    found = False
    for _, folder in result_folders(args):
        found = True
        cfg = Settings(**json.loads((folder / "manifest.json").read_text())["settings"])
        (folder / "analysis").mkdir(exist_ok=True)
        plot_population(folder, cfg)
        selection = {}
        for system in SYSTEMS:
            directories = [
                path.parent
                for path in sorted((folder / system).glob("replicate_*/trajectories.npz"))
            ]
            selected = (
                select_replicas(directories)
                if args.replicates is None
                else [
                    path for path in directories if int(path.name.split("_")[-1]) in args.replicates
                ]
            )
            selection[system] = [int(path.name.split("_")[-1]) for path in selected]
            for result_dir in selected:
                output = folder / "analysis" / system / result_dir.name
                plot_replica(result_dir, cfg, output)
                print(f"Plotted {output}", flush=True)
        save_json(
            folder / "analysis" / "figure_replicas.json",
            {
                "replicas": selection,
                "selection": "Explicit IDs"
                if args.replicates is not None
                else "Three finite control trajectories closest to median log1p MSE and mean IoU, IQR-scaled; "
                "the same replica is used for all its perturbations. Selection is illustrative, not inference.",
            },
        )
    if not found:
        raise FileNotFoundError("No Experiment 2 manifests found")


def regenerate_snapshots(args) -> None:
    from threadpoolctl import threadpool_limits

    plt = plotting()
    found = False
    for morphology, folder in result_folders(args):
        found = True
        manifest = json.loads((folder / "manifest.json").read_text())
        cfg = Settings(**manifest["settings"])
        device = get_device(manifest["device"])
        target_path = Path(manifest["target_path"])
        target = None
        for system in args.systems:
            model = None
            directories = [
                path.parent
                for path in sorted((folder / system).glob("replicate_*/trajectories.npz"))
            ]
            for result_dir in directories:
                replicate = int(result_dir.name.split("_")[-1])
                if args.replicates is not None and replicate not in args.replicates:
                    continue
                image_path = result_dir / "perturbation_snapshots.png"
                if image_path.exists() and not args.overwrite:
                    print(f"Already plotted: {image_path}", flush=True)
                    continue
                with np.load(result_dir / "trajectories.npz", allow_pickle=False) as archive:
                    indices = retained_branch_indices(archive["branch_names"])
                    branches = archive["branch_names"][indices].tolist()
                    pre_pulse = archive["pre_pulse_state"]
                    steps = archive["snapshot_steps"]
                    if steps.size:
                        snapshots = archive["snapshots"][indices]
                    else:
                        replay_path = result_dir / "snapshot_replay.npz"
                        if replay_path.exists():
                            with np.load(replay_path, allow_pickle=False) as replay:
                                steps, snapshots = replay["snapshot_steps"], replay["snapshots"]
                                replay_branches = (
                                    replay["branch_names"]
                                    if "branch_names" in replay
                                    else archive["branch_names"]
                                )
                                if len(replay_branches) != len(snapshots):
                                    raise ValueError(
                                        f"Snapshot condition count differs: {replay_path}"
                                    )
                                replay_indices = retained_branch_indices(replay_branches)
                                if replay_branches[replay_indices].tolist() != branches:
                                    raise ValueError(f"Snapshot conditions differ: {replay_path}")
                                snapshots = snapshots[replay_indices]
                        else:
                            checkpoint = Path(manifest["checkpoint_paths"][system])
                            if file_sha256(checkpoint) != manifest["checkpoint_hashes"][system]:
                                raise ValueError(f"Checkpoint changed: {checkpoint}")
                            if file_sha256(target_path) != manifest["target_hash"]:
                                raise ValueError(f"Target changed: {target_path}")
                            if model is None:
                                model = load_growing_checkpoint(
                                    checkpoint,
                                    channels=cfg.channels,
                                    hidden_size=cfg.hidden_size,
                                    update_probability=cfg.fire_rate,
                                    device=device,
                                )
                            if target is None:
                                target = load_distill_target(
                                    target_path, cfg.target_size, cfg.padding, device
                                )
                            mature = torch.as_tensor(
                                archive["mature_state"], device=device
                            ).unsqueeze(0)
                            print(f"Replaying {morphology}/{system}/{replicate:03d}", flush=True)
                            metadata = json.loads((result_dir / "metadata.json").read_text())
                            erasure_plan = {
                                name: {**metadata["perturbations"][name], "mask": mask}
                                for name, mask in zip(
                                    [name for name in branches if name.startswith("erase_")],
                                    archive["erase_masks"],
                                    strict=True,
                                )
                            }
                            with threadpool_limits(limits=1):
                                steps, snapshots = replay_snapshots(
                                    model,
                                    target,
                                    mature,
                                    system,
                                    replicate,
                                    cfg,
                                    pre_pulse,
                                    load_payload(result_dir)["metrics"],
                                    branches,
                                    erasure_plan,
                                )
                            save_npz(
                                replay_path,
                                branch_names=np.asarray(branches),
                                snapshot_steps=steps,
                                snapshots=snapshots,
                            )
                plot_snapshots(
                    plt,
                    {
                        "branch_names": branches,
                        "snapshot_steps": steps,
                        "snapshots": snapshots,
                        "pre_pulse_state": pre_pulse,
                    },
                    result_dir,
                    cfg,
                )
                print(f"Saved {image_path}", flush=True)
    if not found:
        raise FileNotFoundError("No Experiment 2 manifests found")


def analyze_attractor(args) -> None:
    analyze(args)
    plot(args)


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
    running.add_argument(
        "--erase-fraction",
        type=float,
        help="Override equal-area erasure; default: full gecko hind leg (~10%% of target), 10%% for butterfly",
    )
    running.set_defaults(handler=run)
    analysis = commands.add_parser(
        "analyze", help="Morphological and hidden-state recovery statistics from recorded data"
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
        help="Replica IDs for PCA figures; default: three closest to median control morphology",
    )
    plots.set_defaults(handler=plot)
    comparison = commands.add_parser(
        "compare-control",
        help="Compare RGBA and alpha masks to the paired control at every step; replay missing data without overwriting trajectories",
    )
    comparison.set_defaults(handler=compare_control)
    snapshots = commands.add_parser(
        "snapshots", help="Recreate missing per-replica morphology snapshots from saved runs"
    )
    snapshots.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    snapshots.add_argument("--replicates", nargs="+", type=int, help="Replica IDs; default: all")
    snapshots.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing snapshot figures using saved data when available",
    )
    snapshots.set_defaults(handler=regenerate_snapshots)
    preview = commands.add_parser(
        "preview-erasure", help="Plot anatomical erasure masks without NCA simulation"
    )
    preview.add_argument(
        "--erase-fraction", type=float, help="Override default full-leg/equal-area erasure"
    )
    preview.add_argument("--data-dir", default="data")
    preview.set_defaults(handler=preview_erasure)
    attractor = commands.add_parser(
        "attractor",
        help="Summarize recovery and plot hidden PCA plus the two population comparisons",
    )
    attractor.add_argument("--bootstrap-samples", type=int, default=5000)
    attractor.add_argument(
        "--replicates",
        nargs="+",
        type=int,
        help="Replica IDs for detailed figures only; statistics use all recorded replicas",
    )
    attractor.set_defaults(handler=analyze_attractor)
    for command in (analysis, attractor):
        command.add_argument("--target-mse-max", type=float, default=0.001)
        command.add_argument("--target-iou-min", type=float, default=0.95)
        command.add_argument("--target-area-min", type=float, default=0.9)
        command.add_argument("--target-area-max", type=float, default=1.1)
    for command in (training, running, analysis, plots, comparison, snapshots, preview, attractor):
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
    for command in (running, analysis, plots, comparison, snapshots, preview, attractor):
        command.add_argument("--output-root", default="reports/experiment_2")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "bootstrap_samples", 1) <= 0:
        raise ValueError("bootstrap-samples must be positive")
    tag = getattr(args, "checkpoint_tag", "default")
    if tag in ("", ".", "..") or "/" in tag or "\\" in tag:
        raise ValueError("checkpoint-tag must be a single nonempty folder name")
    args.handler(args)


if __name__ == "__main__":
    main()
