"""Paired hidden transplants with temporary freezing and fixed restoration references."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import affine_transform, binary_dilation, label
from threadpoolctl import threadpool_limits

from scripts import experiment_2 as e2

PROTOCOL = "experiment_3_paired_transplant_references_v2"
MORPHOLOGIES = ("gecko", "planaria")
TRANSPLANTS = {
    "gecko": {"leg": "tail", "head": "leg", "tail": "head"},
    "planaria": {"head": "tail", "tail": "head"},
}
METRICS = (*e2.MORPHOLOGY_METRICS, "target_retention", "ectopic_growth")
MODEL_FIELDS = (
    "channels",
    "hidden_size",
    "target_size",
    "padding",
    "fire_rate",
    "beta",
    "alpha_threshold",
    "mature_start",
    "mature_stop",
    "development_limit",
    "attainment_patience",
    "recovery_patience",
    "calibration_replicates",
    "acceptance_quantile",
    "seed",
)


@dataclass(frozen=True)
class Condition:
    name: str
    label: str
    wound: str | None = None
    donor: str | None = None
    reference: str = "original"


def primitive_masks(target, morphology, cfg):
    foreground = target[3] > cfg.alpha_threshold
    if not foreground.any():
        raise ValueError("The target foreground is empty")
    if morphology == "planaria":
        points = np.argwhere(foreground)
        _, _, directions = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
        axis = directions[0]
        axis *= (
            1 if axis[0] > 0 else -1
        )  # The supplied planaria target has its head above its tail.
        position = points @ axis
        cuts = position.min() + np.ptp(position) * np.array([1 / 3, 2 / 3])
        sections = np.searchsorted(cuts, position, side="right")
        regions = {}
        for index, name in enumerate(("head", "body", "tail")):
            region = np.zeros_like(foreground)
            region[tuple(points[sections == index].T)] = True
            regions[name] = region
    else:
        regions = {}
        scale = cfg.target_size / e2.PRIMITIVE_SOURCE_SIZE[morphology]
        for name in TRANSPLANTS[morphology]:
            polygon, center = e2.PRIMITIVE_REGIONS[morphology][name]
            image = Image.new("1", (foreground.shape[1], foreground.shape[0]))
            ImageDraw.Draw(image).polygon(
                [(x * scale + cfg.padding, y * scale + cfg.padding) for x, y in polygon], fill=1
            )
            region = np.asarray(image, dtype=bool) & foreground
            points = np.argwhere(region)
            if not len(points):
                raise ValueError(f"Empty anatomical region: {morphology}/{name}")
            center_yx = np.array([center[1], center[0]]) * scale + cfg.padding
            anchor = points[np.argmin(np.sum((points - center_yx) ** 2, axis=1))]
            components, _ = label(region, structure=np.ones((3, 3)))
            regions[name] = components == components[tuple(anchor)]
        regions["body"] = foreground & ~np.logical_or.reduce(list(regions.values()))
    if any(not mask.any() for mask in regions.values()):
        raise ValueError("An anatomical region is empty at this target resolution")
    return regions


def attachment(mask, body):
    contact = mask & binary_dilation(body, structure=np.ones((3, 3)))
    if not contact.any():
        raise ValueError("A primitive does not contact the body; inspect the anatomical masks")
    anchor = np.argwhere(contact).mean(axis=0)
    center = np.argwhere(mask).mean(axis=0)
    direction = center - anchor
    if np.linalg.norm(direction) < 1e-8:
        direction = center - np.argwhere(body).mean(axis=0)
    if np.linalg.norm(direction) < 1e-8:
        raise ValueError("Cannot determine the primitive's outward direction")
    return anchor, direction


def transplant_plans(morphology, regions):
    plans = {}
    for wound, donor in TRANSPLANTS[morphology].items():
        source = np.argwhere(regions[donor])
        donor_anchor, donor_direction = attachment(regions[donor], regions["body"])
        wound_anchor, wound_direction = attachment(regions[wound], regions["body"])
        angle = np.arctan2(wound_direction[0], wound_direction[1]) - np.arctan2(
            donor_direction[0], donor_direction[1]
        )
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        shape = regions[wound].shape
        corners = source[:, None, :] + np.array(
            [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]
        )
        extent = (corners - donor_anchor) @ rotation.T + wound_anchor
        if np.any(extent < -0.5 - 1e-8) or np.any(extent > np.array(shape) - 0.5 + 1e-8):
            raise ValueError(
                f"{wound} <- {donor} leaves the grid; increase padding, do not crop the donor"
            )
        # Rotate integer source labels, then copy exact hidden vectors without interpolating channels.
        source_labels = np.zeros(shape, dtype=np.int32)
        source_labels[tuple(source.T)] = np.arange(1, len(source) + 1)
        rotated = affine_transform(
            source_labels,
            rotation.T,
            offset=donor_anchor - rotation.T @ wound_anchor,
            output_shape=shape,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        footprint = rotated > 0
        destination = np.argwhere(footprint)
        if not len(destination):
            raise ValueError(f"Empty rotated graft: {wound} <- {donor}")
        sampled_source = source[rotated[tuple(destination.T)] - 1]
        plans[wound] = {
            "donor": donor,
            "wound": regions[wound],
            "source_yx": sampled_source,
            "destination_yx": destination,
            "footprint": footprint,
            "rotation_degrees": float(np.degrees(angle)),
            "rotation_yx": rotation.tolist(),
            "donor_attachment_yx": donor_anchor.tolist(),
            "wound_attachment_yx": wound_anchor.tolist(),
            "donor_pixels": len(source),
        }
    return plans


def conditions(plans, system="persistent", persistent_local_beta=0.0):
    result = [Condition("control", "Control")]
    for wound, plan in plans.items():
        donor = plan["donor"]
        if wound.startswith("insertion_"):
            result.append(
                Condition(
                    f"insert_{donor}",
                    f"Insert second {donor}"
                    + (" | free after freeze" if system == "persistent" else ""),
                    wound,
                    donor,
                    "donor" if system == "growing_restore" else "original",
                )
            )
            if system == "persistent" and persistent_local_beta > 0:
                result.append(
                    Condition(
                        f"insert_{donor}_local_restore",
                        f"Insert second {donor} | local restore beta={persistent_local_beta:g}",
                        wound,
                        donor,
                        "donor_local",
                    )
                )
            continue
        name = f"transplant_{wound}_from_{donor}"
        title = f"{wound.capitalize()} <- {donor}"
        result.append(
            Condition(
                name,
                title
                + (
                    " | original reference"
                    if system == "growing_restore"
                    else " | free after freeze"
                ),
                wound,
                donor,
            )
        )
        if system == "growing_restore":
            result.append(
                Condition(
                    name + "_donor_reference", title + " | donor reference", wound, donor, "donor"
                )
            )
    return tuple(result)


def prepare_state(mature, condition, plans):
    state = mature.clone()
    if condition.wound is None:
        return state, None
    plan = plans[condition.wound]
    state[:, :, torch.as_tensor(plan["wound"], device=state.device)] = 0
    source = torch.as_tensor(plan["source_yx"], device=state.device)
    destination = torch.as_tensor(plan["destination_yx"], device=state.device)
    values = mature[0, 4:, source[:, 0], source[:, 1]].clone()
    state[0, 4:, destination[:, 0], destination[:, 1]] = values
    return state, (destination, values)


def modified_reference(mature, condition, plans):
    state, frozen = prepare_state(mature, condition, plans)
    reference = mature.clone()
    if frozen is not None:
        destination, values = frozen
        reference[0, 4:, destination[:, 0], destination[:, 1]] = values
    return state, frozen, reference


def modified_target(target, condition, plans):
    """Analytical composite for evaluation only; never pasted into the NCA."""
    result = target.copy()
    if condition.donor is not None:
        plan = plans[condition.wound]
        source, destination = plan["source_yx"], plan["destination_yx"]
        result[:, plan["wound"]] = 0
        result[:, destination[:, 0], destination[:, 1]] = target[:, source[:, 0], source[:, 1]]
    return result


def restore_graft_hidden(state, frozen, beta):
    """Restore only graft hidden vectors; leave RGBA and all other cells unchanged."""
    destination, values = frozen
    y, x = destination[:, 0], destination[:, 1]
    state[0, 4:, y, x] = torch.lerp(state[0, 4:, y, x], values, beta)
    return state


def advance(model, state, reference, system, cfg, frozen, step, freeze_steps,
            persistent_local_beta=0.0):
    result = model(state, update_probability=cfg.fire_rate)
    if not bool(torch.isfinite(result).all()):
        return None
    if system == "growing_restore":
        result[:, 4:] = torch.lerp(result[:, 4:], reference[:, 4:], cfg.beta)
    elif frozen is not None and step > freeze_steps and persistent_local_beta > 0:
        result = restore_graft_hidden(result, frozen, persistent_local_beta)
    if frozen is not None and step <= freeze_steps:
        destination, values = frozen
        result[0, 4:, destination[:, 0], destination[:, 1]] = values
    return result if bool(torch.isfinite(result).all()) else None


def measure(state, target, threshold):
    visible = state[0, :4].detach().cpu().numpy().astype(np.float64)
    active, foreground = visible[3] > threshold, target[3] > threshold
    area = int(foreground.sum())
    return np.r_[
        e2.morphology_metrics(visible, target, threshold),
        (active & foreground).sum() / area,
        (active & ~foreground).sum() / area,
    ]


@torch.no_grad()
def simulate(model, target, mature, system, plans, cfg, regime, replicate, horizon, freeze_steps,
             persistent_local_beta=0.0):
    specs = conditions(plans, system, persistent_local_beta)
    snapshot_steps = np.array(
        sorted(
            {0, horizon, freeze_steps, min(freeze_steps + 1, horizon)}
            | {t for t in (1, 10, 100, 500) if t <= horizon}
            | set(range(1000, horizon + 1, 1000))
        )
    )
    snapshots = np.full(
        (len(specs), len(snapshot_steps), 4, *mature.shape[-2:]), np.nan, np.float32
    )
    metrics = np.full((len(specs), horizon + 1, len(METRICS)), np.nan)
    modified_metrics = np.full_like(metrics, np.nan)
    modified_targets = []
    in_target = np.zeros((len(specs), horizon + 1), dtype=bool)
    failures = np.full(len(specs), -1, dtype=int)
    snapshot_indices = {int(step): index for index, step in enumerate(snapshot_steps)}
    target_array = target[0].detach().cpu().numpy().astype(np.float64)
    for index, condition in enumerate(specs):
        e2.seed_nca(cfg.seed + 1_000_000 + replicate, mature.device)
        state, frozen, donor_reference = modified_reference(mature, condition, plans)
        desired = modified_target(target_array, condition, plans)
        modified_targets.append(desired)
        for step in range(horizon + 1):
            if step:
                state = advance(
                    model,
                    state,
                    donor_reference
                    if condition.reference == "donor" and step > freeze_steps
                    else mature,
                    system,
                    cfg,
                    frozen,
                    step,
                    freeze_steps,
                    persistent_local_beta if condition.reference == "donor_local" else 0.0,
                )
            if state is None or not bool(torch.isfinite(state).all()):
                failures[index] = step
                break
            values = measure(state, target_array, cfg.alpha_threshold)
            composite_values = measure(state, desired, cfg.alpha_threshold)
            if not np.isfinite(values).all() or not np.isfinite(composite_values).all():
                failures[index] = step
                break
            metrics[index, step] = values
            modified_metrics[index, step] = composite_values
            in_target[index, step] = e2.inside_regime(values, regime)
            if step in snapshot_indices:
                snapshots[index, snapshot_indices[step]] = state[0, :4].detach().cpu().numpy()
    return {
        "steps": np.arange(horizon + 1),
        "condition_names": np.array([c.name for c in specs]),
        "condition_references": np.array([c.reference for c in specs]),
        "metric_names": np.array(METRICS),
        "metrics": metrics,
        "modified_target_metrics": modified_metrics,
        "modified_targets": np.asarray(modified_targets, dtype=np.float32),
        "in_target": in_target,
        "numerical_failure_step": failures,
        "snapshot_steps": snapshot_steps,
        "snapshots": snapshots,
        "mature_state": mature[0].detach().cpu().numpy(),
    }


def insertion_plan(regions, angle_degrees=None, donor="tail"):
    """Place an intact hidden graft on the right flank near the body midpoint."""
    source = np.argwhere(regions[donor])
    donor_anchor, direction = attachment(regions[donor], regions["body"])
    if angle_degrees is None:
        angle_degrees = (
            -90.0 if donor == "tail" else -float(np.degrees(np.arctan2(direction[0], direction[1])))
        )
    if not np.isfinite(angle_degrees) or abs(angle_degrees) > 180:
        raise ValueError("Insertion angle must be finite and between -180 and 180 degrees")
    foreground = np.logical_or.reduce(list(regions.values()))
    contact = binary_dilation(regions["body"])
    body_points = np.argwhere(regions["body"])
    anchors = [
        attachment(regions[name], regions["body"])[0]
        for name in ("head", "tail")
        if name in regions
    ]
    midpoint = np.mean(anchors, axis=0) if len(anchors) == 2 else body_points.mean(axis=0)
    center_row = body_points[
        np.abs(body_points[:, 0] - midpoint[0]) == np.min(np.abs(body_points[:, 0] - midpoint[0]))
    ]
    flank = center_row[np.argmax(center_row[:, 1])].astype(float) + [0, 1]
    shape = np.array(foreground.shape)
    angle = np.radians(angle_degrees)
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    corners = source[:, None, :] + np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    extent = (corners - donor_anchor) @ rotation.T + donor_anchor
    if np.any(extent < -0.5) or np.any(extent > shape - 0.5):
        raise ValueError("Rotated donor leaves the grid; increase padding or reduce the angle")
    source_labels = np.zeros(shape, dtype=np.int32)
    source_labels[tuple(source.T)] = np.arange(1, len(source) + 1)
    rotated = affine_transform(
        source_labels,
        rotation.T,
        offset=donor_anchor - rotation.T @ donor_anchor,
        output_shape=tuple(shape),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    rotated_points = np.argwhere(rotated > 0)
    if not len(rotated_points) or label(rotated > 0)[1] != 1:
        raise ValueError("Rotation produces an empty or disconnected graft; choose another angle")
    sampled_source = source[rotated[tuple(rotated_points.T)] - 1]
    proximal = binary_dilation(regions["body"], structure=np.ones((3, 3)))[tuple(sampled_source.T)]
    candidates = []
    for dy in range(-int(rotated_points[:, 0].min()), int(shape[0] - rotated_points[:, 0].max())):
        for dx in range(1, int(shape[1] - rotated_points[:, 1].max())):
            if np.any(extent + [dy, dx] < -0.5) or np.any(extent + [dy, dx] > shape - 0.5):
                continue
            destination = rotated_points + [dy, dx]
            indices = tuple(destination.T)
            base_contact = contact[indices] & proximal
            anchor = donor_anchor + [dy, dx]
            if donor == "tail" and anchor[0] >= donor_anchor[0]:
                continue
            if anchor[1] <= midpoint[1]:
                continue
            if (
                not foreground[indices].any()
                and base_contact.sum() >= min(2, int(proximal.sum()))
                and base_contact.any()
            ):
                distance = float(np.sum((anchor - flank) ** 2))
                candidates.append((distance, -int(base_contact.sum()), dy * dy + dx * dx, dy, dx))
    if not candidates:
        raise ValueError(
            "No uncropped right-flank insertion with proximal body contact fits the grid"
        )
    _, _, _, dy, dx = min(candidates)
    destination = rotated_points + [dy, dx]
    footprint = np.zeros_like(foreground)
    footprint[tuple(destination.T)] = True
    insertion_anchor = donor_anchor + [dy, dx]
    return {
        f"insertion_{donor}": {
            "donor": donor,
            "wound": np.zeros_like(foreground),
            "source_yx": sampled_source,
            "destination_yx": destination,
            "footprint": footprint,
            "rotation_degrees": float(angle_degrees),
            "rotation_yx": rotation.tolist(),
            "donor_attachment_yx": donor_anchor.tolist(),
            "wound_attachment_yx": insertion_anchor.tolist(),
            "donor_pixels": len(source),
        }
    }


def geometry(morphology, target, cfg, manipulation="transplant", insertion_angle=None):
    regions = primitive_masks(target, morphology, cfg)
    plans = (
        insertion_plan(regions, insertion_angle, "tail" if morphology == "gecko" else "head")
        if manipulation == "insertion"
        else transplant_plans(morphology, regions)
    )
    return regions, plans


def preview_figure(path, target, regions, plans):
    plt = e2.plotting()
    fig, axes = plt.subplots(1, len(plans) + 1, figsize=(4 * (len(plans) + 1), 4))
    for axis in axes:
        axis.imshow(e2.display_rgba(target))
        axis.set(xticks=[], yticks=[])
    colors = ("#d04d45", "#3d9d78", "#357abb", "#777777")
    for (name, mask), color in zip(regions.items(), colors):
        points = np.argwhere(mask)
        axes[0].scatter(
            points[:, 1], points[:, 0], color=color, marker="s", s=13, alpha=0.45, label=name
        )
    axes[0].set_title("Anatomical regions")
    axes[0].legend(fontsize=9)
    for axis, (wound, plan) in zip(axes[1:], plans.items()):
        removed = np.argwhere(plan["wound"])
        pasted = plan["destination_yx"]
        if len(removed):
            axis.scatter(
                removed[:, 1],
                removed[:, 0],
                color="#d04d45",
                marker="s",
                s=18,
                alpha=0.6,
                label="Wound",
            )
        axis.scatter(
            pasted[:, 1],
            pasted[:, 0],
            facecolors="none",
            edgecolors="#357abb",
            marker="s",
            s=25,
            label="Inserted hidden footprint"
            if wound.startswith("insertion_")
            else "Frozen hidden transplant",
        )
        anchor_y, anchor_x = plan["wound_attachment_yx"]
        axis.scatter(anchor_x, anchor_y, color="#161616", marker="x", s=35, label="Attachment")
        axis.set_title(
            f"Second {plan['donor']} insertion (hidden only)\nrotation {plan['rotation_degrees']:.1f} deg; no erasure"
            if wound.startswith("insertion_")
            else f"{wound.capitalize()} <- {plan['donor']}\nrotation {plan['rotation_degrees']:.1f} deg; no resizing"
        )
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    e2.finish_figure(plt, fig, path)


def preview(args):
    folder = Path(args.output_root) / "previews"
    folder.mkdir(parents=True, exist_ok=True)
    cfg = e2.Settings()
    for morphology in args.morphologies:
        target = e2.load_distill_target(
            Path(args.data_dir) / f"{morphology}.png", cfg.target_size, cfg.padding
        )
        array = target[0].numpy()
        regions, plans = geometry(morphology, array, cfg, args.manipulation, args.insertion_angle)
        path = folder / f"{morphology}.png"
        preview_figure(path, array, regions, plans)
        print(f"Saved anatomical preview: {path.resolve()}", flush=True)


def summarize(payload, metadata, patience):
    rows = []
    for index, name in enumerate(payload["condition_names"]):
        failure = int(payload["numerical_failure_step"][index])
        flags = payload["in_target"][index]
        row = {
            "system": metadata["system"],
            "replicate": metadata["replicate"],
            "condition": str(name),
            "hidden_reference": str(payload["condition_references"][index])
            if metadata["system"] == "growing_restore" or str(payload["condition_references"][index]) == "donor_local"
            else "none",
            "numerical_failure_step": failure if failure >= 0 else None,
            "first_sustained_target_step": e2.first_sustained(flags, patience),
            "fraction_final_200_in_target": float(flags[-min(200, len(flags)) :].mean()),
        }
        row.update(
            {
                f"final_{metric}": float(value) if np.isfinite(value) else None
                for metric, value in zip(METRICS, payload["metrics"][index, -1], strict=True)
            }
        )
        row.update(
            {
                f"final_composite_{metric}": float(value) if np.isfinite(value) else None
                for metric, value in zip(
                    METRICS, payload["modified_target_metrics"][index, -1], strict=True
                )
            }
        )
        rows.append(row)
    return rows


def recorded_systems(manifest):
    entries = {
        system: {"conditions": manifest["conditions"][system], "signature": manifest["signature"]}
        for system in manifest["systems"]
    }
    entries.update(manifest.get("additional_systems", {}))
    return entries


def extend_manifest(manifest, expected, all_conditions, calibration_path):
    """Add a model without changing the original run signature or its archives."""
    signed = {key: manifest[key] for key in expected}
    digest = hashlib.sha256(json.dumps(signed, sort_keys=True).encode()).hexdigest()
    if digest != manifest["signature"]:
        raise ValueError("The original manifest signature is invalid; do not extend this run")
    for key, value in expected.items():
        if key not in {"systems", "conditions"} and manifest[key] != value:
            raise ValueError(f"Inputs/protocol changed ({key}); choose a new output root")
    entries = recorded_systems(manifest)
    for system in manifest["systems"]:
        if entries[system]["conditions"] != all_conditions[system]:
            raise ValueError(f"Conditions changed for {system}; choose a new output root")
    updated = dict(manifest)
    additions = dict(manifest.get("additional_systems", {}))
    for system in expected["systems"]:
        if system in manifest["systems"]:
            continue
        if not calibration_path.is_file():
            raise ValueError("Extending a run requires its original calibration.json")
        specification = {
            **expected,
            "systems": [system],
            "conditions": {system: all_conditions[system]},
            "calibration_sha256": e2.file_sha256(calibration_path),
        }
        signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
        if system in additions:
            if (
                additions[system]["signature"] != signature
                or additions[system]["conditions"] != all_conditions[system]
            ):
                raise ValueError(
                    f"Inputs/calibration changed for {system}; choose a new output root"
                )
        else:
            additions[system] = {
                "conditions": all_conditions[system],
                "signature": signature,
                "calibration_sha256": specification["calibration_sha256"],
                "implementation_sha256": e2.file_sha256(Path(__file__)),
                "definition": "Same manipulation and freeze duration. "
                + (
                    "Persistent evolves freely after freeze without beta restoration."
                    if system == "persistent"
                    else "Growing retains the original beta-restoration protocol."
                ),
            }
    if additions:
        updated["additional_systems"] = additions
    return updated


def plot_folder(folder, snapshot_replicates=3):
    plt = e2.plotting()
    from scripts.experiment_1 import _set_mse_log_scale

    manifest = json.loads((folder / "manifest.json").read_text())
    output = folder / "analysis"
    output.mkdir(exist_ok=True)
    entries = recorded_systems(manifest)
    systems = [system for system in e2.SYSTEMS if system in entries]
    fig, axes = plt.subplots(4, len(systems), figsize=(6.5 * len(systems), 13), squeeze=False)
    palette = plt.get_cmap("tab10")
    summaries = []
    for column, system in enumerate(systems):
        specs = [Condition(**entry) for entry in entries[system]["conditions"]]
        wounds = list(dict.fromkeys(condition.wound for condition in specs))
        records = []
        for path in sorted((folder / system).glob("replicate_*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                record = {key: archive[key] for key in archive.files}
            meta = json.loads(str(record["metadata"]))
            if meta["signature"] != entries[system]["signature"]:
                raise ValueError(f"Different experiment protocol: {path}")
            summaries.extend(summarize(record, meta, manifest["model"]["recovery_patience"]))
            records.append((record, meta))
        for row, (field, metric, target_name) in enumerate(
            (
                ("metrics", "target_mse", "original target"),
                ("metrics", "alpha_iou", "original target"),
                ("modified_target_metrics", "target_mse", "analytical composite"),
                ("modified_target_metrics", "alpha_iou", "analytical composite"),
            )
        ):
            axis = axes[row, column]
            for index, condition in enumerate(specs):
                if not records:
                    continue
                values = np.stack(
                    [record[field][index, :, METRICS.index(metric)] for record, _ in records]
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    lower, median, upper = np.nanquantile(values, [0.25, 0.5, 0.75], axis=0)
                if metric == "target_mse":
                    lower, median, upper = [
                        np.ma.masked_less_equal(value, 0) for value in (lower, median, upper)
                    ]
                time = records[0][0]["steps"]
                color = palette(wounds.index(condition.wound))
                axis.plot(
                    time,
                    median,
                    color=color,
                    linestyle="--" if condition.reference == "donor" else "-",
                    label=condition.label,
                    linewidth=1.4,
                )
                axis.fill_between(
                    time,
                    lower,
                    upper,
                    where=np.isfinite(values).sum(axis=0) >= 2,
                    color=color,
                    alpha=0.15,
                )
            if metric == "target_mse":
                _set_mse_log_scale(axis)
            else:
                axis.set_ylim(-0.02, 1.02)
            axis.axvline(manifest["freeze_steps"], color="#777777", linestyle="--", linewidth=1)
            axis.set(
                title=f"{system} (n={len(records)}): {metric} vs {target_name}",
                xlabel="Steps after Manipulation",
            )
            axis.grid(alpha=0.2)
        if not records:
            continue
        control_means = np.array([np.nanmean(record["metrics"][0, :, 0]) for record, _ in records])
        selected = np.argsort(abs(control_means - np.median(control_means)), kind="stable")[
            :snapshot_replicates
        ]
        for index in selected:
            record, meta = records[index]
            image, grid = plt.subplots(
                len(specs),
                len(record["snapshot_steps"]) + 2,
                figsize=(2 * (len(record["snapshot_steps"]) + 2), 1.8 * len(specs)),
                squeeze=False,
            )
            for row, condition in enumerate(specs):
                frames = [
                    record["mature_state"][:4],
                    record["modified_targets"][row],
                    *record["snapshots"][row],
                ]
                for col, frame in enumerate(frames):
                    axis = grid[row, col]
                    if np.isfinite(frame).all():
                        axis.imshow(e2.display_rgba(frame))
                    else:
                        axis.text(
                            0.5, 0.5, "Numerically\ndiverged", ha="center", va="center", fontsize=8
                        )
                    axis.set(xticks=[], yticks=[])
                    if row == 0:
                        axis.set_title(
                            "Mature, pre-manipulation"
                            if col == 0
                            else "Analytical composite"
                            if col == 1
                            else f"t={record['snapshot_steps'][col - 2]}",
                            fontsize=9,
                        )
                    if col == 0:
                        axis.set_ylabel(condition.label, fontsize=9)
            image.suptitle(f"{folder.name}: {system} | paired replica {meta['replicate']:03d}")
            image.tight_layout(rect=(0, 0, 1, 0.97))
            destination = output / system
            destination.mkdir(exist_ok=True)
            e2.finish_figure(
                plt, image, destination / f"snapshots_replica_{meta['replicate']:03d}.png"
            )
    for column in range(len(systems)):
        axes[-1, column].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, fontsize=8, frameon=False
        )
    title = (
        "hidden insertion comparison"
        if manifest.get("manipulation") == "insertion"
        else "transplant reference comparison"
    )
    fig.suptitle(f"{folder.name}: {title} | median and IQR")
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.savefig(output / "morphology_metrics.png", dpi=250)
    plt.close(fig)
    e2.write_csv(output / "per_run_summary.csv", summaries)


def run(args):
    insertion = args.manipulation == "insertion"
    local_beta = getattr(args, "persistent_local_beta", 0.0)
    if not np.isfinite(local_beta) or not 0 <= local_beta <= 1:
        raise ValueError("persistent-local-beta must be finite and between 0 and 1")
    if local_beta and (not insertion or "persistent" not in args.systems):
        raise ValueError("persistent-local-beta requires insertion and the Persistent model")
    systems = tuple(args.systems)
    if (
        min(args.replicates, args.calibration_replicates, args.freeze_steps) < 1
        or args.horizon <= args.freeze_steps
    ):
        raise ValueError(
            "Use positive replicate/freeze counts and a horizon longer than the freeze"
        )
    cfg = e2.Settings(seed=args.seed, calibration_replicates=args.calibration_replicates)
    device = e2.get_device(args.device)
    paths = {
        (morphology, system): e2.checkpoint_path(
            Path(args.checkpoint_root), morphology, system, args.checkpoint_tag
        )
        for morphology in args.morphologies
        for system in e2.SYSTEMS
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Train the Persistent planaria with: "
                "PYTHONPATH=src .venv/bin/python -m scripts.experiment_3 train --morphologies planaria"
            )
    for morphology in args.morphologies:
        target_path = Path(args.data_dir) / f"{morphology}.png"
        target = e2.load_distill_target(target_path, cfg.target_size, cfg.padding, device)
        array = target[0].detach().cpu().numpy()
        regions, plans = geometry(morphology, array, cfg, args.manipulation, args.insertion_angle)
        all_conditions = {
            system: [vars(condition) for condition in conditions(plans, system, local_beta)]
            for system in e2.SYSTEMS
        }
        mask_hash = hashlib.sha256(
            b"".join(
                plan[key].tobytes()
                for plan in plans.values()
                for key in ("wound", "source_yx", "destination_yx")
            )
        ).hexdigest()
        expected = {
            "protocol": PROTOCOL,
            "manipulation": args.manipulation,
            "model": {key: getattr(cfg, key) for key in MODEL_FIELDS},
            "freeze_steps": args.freeze_steps,
            "horizon": args.horizon,
            "systems": list(systems),
            "conditions": {system: all_conditions[system] for system in systems},
            "geometry_sha256": mask_hash,
            "target_sha256": e2.file_sha256(target_path),
            "checkpoints": {
                system: e2.file_sha256(paths[morphology, system]) for system in e2.SYSTEMS
            },
            "device": str(device),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "python": platform.python_version(),
        }
        if insertion:
            expected.update(
                protocol="experiment_3_hidden_insertion_v4",
                systems=list(systems),
                insertion_angle=args.insertion_angle,
            )
        if local_beta:
            expected["persistent_local_beta"] = local_beta
        signature = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
        folder = Path(args.output_root) / morphology
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            manifest = extend_manifest(
                previous, expected, all_conditions, folder / "calibration.json"
            )
            if manifest != previous:
                e2.save_json(manifest_path, manifest)
        else:
            if any(folder.iterdir()):
                raise FileExistsError(f"Output is not empty and has no manifest: {folder}")
            e2.save_json(
                manifest_path,
                {
                    **expected,
                    "signature": signature,
                    "implementation_sha256": e2.file_sha256(Path(__file__)),
                    "geometry": {
                        wound: {
                            "donor": plan["donor"],
                            "rotation_degrees": plan["rotation_degrees"],
                            "rotation_yx": plan["rotation_yx"],
                            "donor_attachment_yx": plan["donor_attachment_yx"],
                            "wound_attachment_yx": plan["wound_attachment_yx"],
                            "wound_pixels": int(plan["wound"].sum()),
                            "donor_pixels": plan["donor_pixels"],
                            "graft_pixels": len(plan["destination_yx"]),
                            "graft_background_pixels": int(
                                (plan["footprint"] & ~(array[3] > cfg.alpha_threshold)).sum()
                            ),
                        }
                        for wound, plan in plans.items()
                    },
                    "definition": (
                        "t=0 follows target attainment. No RGBA or foreground region is erased. "
                        "The donor (gecko tail or planaria head) is rigidly rotated and placed at the right flank "
                        "near the midpoint between head and tail attachments, without foreground overlap or cropping. "
                        "Nearest-neighbor rotation copies exact hidden vectors without resizing or blending. "
                        "Hidden values are frozen "
                        "through freeze_steps. Growing then uses local beta=0.1 toward the copied donor "
                        "and the original mature reference elsewhere; Persistent normally evolves freely. "
                        "When persistent_local_beta is enabled, an additional paired condition restores only "
                        "the graft hidden vectors toward their fixed donor values after freeze. RGBA evolve freely."
                        if insertion
                        else "t=0 follows target attainment and contains the post-wound transplant. "
                        "Donor hidden values are copied from the same mature state with a rigid spatial rotation, "
                        "aligning outward directions and attachment points; no resizing. "
                        "Nearest-neighbor rasterization copies exact channel vectors but can change pixel counts. "
                        "Freeze overrides global restore for updates 1..freeze_steps. After release, Growing "
                        "compares the original mature reference with donor hidden vectors in the graft footprint only; "
                        "the reference outside that footprint remains the original mature state. Persistent has no restore. "
                        "RGBA composites are evaluation references only, never interventions; regime flags refer to the original target. "
                        "Planaria uses equal-length thirds along the target's longitudinal axis; head at its upper end."
                    ),
                },
            )
        preview_figure(
            folder / ("insertion_preview.png" if insertion else "transplant_preview.png"),
            array,
            regions,
            plans,
        )
        models = {
            system: e2.load_growing_checkpoint(
                paths[morphology, system],
                channels=cfg.channels,
                hidden_size=cfg.hidden_size,
                update_probability=cfg.fire_rate,
                device=device,
            )
            for system in e2.SYSTEMS
        }
        calibration = folder / "calibration.json"
        if calibration.exists():
            regime = json.loads(calibration.read_text())
        else:
            regime, _ = e2.calibrate(models, target, cfg)
            e2.save_json(calibration, regime)
        for system in systems:
            signature = recorded_systems(json.loads(manifest_path.read_text()))[system]["signature"]
            model = models[system]
            destination = folder / system
            destination.mkdir(exist_ok=True)
            for replicate in range(args.replicates):
                path = destination / f"replicate_{replicate:03d}.npz"
                excluded = path.with_suffix(".json")
                if path.exists() or excluded.exists():
                    if path.exists():
                        with np.load(path, allow_pickle=False) as archive:
                            meta = json.loads(str(archive["metadata"]))
                    else:
                        meta = json.loads(excluded.read_text())
                    if meta["signature"] != signature:
                        raise ValueError(f"Incompatible saved replica: {path}")
                    print(f"Already recorded: {morphology}/{system}/{replicate:03d}", flush=True)
                    continue
                print(
                    f"{args.manipulation}: {morphology}/{system}, replica {replicate:03d}",
                    flush=True,
                )
                mature, mature_step, reason = e2.develop(
                    model, target, regime, cfg, cfg.seed + replicate
                )
                metadata = {
                    "signature": signature,
                    "system": system,
                    "replicate": replicate,
                    "development_seed": cfg.seed + replicate,
                    "continuation_seed": cfg.seed + 1_000_000 + replicate,
                    "mature_step": mature_step,
                    "reason": reason,
                }
                if mature is None:
                    e2.save_json(excluded, metadata)
                    continue
                result = simulate(
                    model,
                    target,
                    mature,
                    system,
                    plans,
                    cfg,
                    regime,
                    replicate,
                    args.horizon,
                    args.freeze_steps,
                    local_beta,
                )
                e2.save_npz(path, **result, metadata=np.array(json.dumps(metadata)))
        plot_folder(folder, args.snapshot_replicates)


def plot(args):
    for morphology in args.morphologies:
        plot_folder(Path(args.output_root) / morphology, args.snapshot_replicates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser(
        "train", help="Train missing Persistent checkpoints with the existing protocol"
    )
    training.add_argument("--training-seed", type=int, default=0)
    training.set_defaults(handler=e2.train, systems=("persistent",))
    running = commands.add_parser(
        "run", help="Record control and paired graft-reference conditions, then plot"
    )
    running.add_argument("--replicates", type=int, default=30)
    running.add_argument("--calibration-replicates", type=int, default=32)
    running.add_argument("--freeze-steps", type=int, default=300)
    running.add_argument(
        "--persistent-local-beta", type=float, default=0.0,
        help="Add a Persistent insertion condition with donor restore after freeze, only in the graft (0 disables)",
    )
    running.add_argument("--horizon", type=int, default=2000)
    running.add_argument("--seed", type=int, default=50000)
    running.add_argument(
        "--systems",
        nargs="+",
        choices=e2.SYSTEMS,
        default=e2.SYSTEMS,
        help="Models to record; completed replicas are skipped (default: both)",
    )
    running.set_defaults(handler=run)
    previewing = commands.add_parser(
        "preview", help="Inspect anatomical masks and complete donor footprints; no simulation"
    )
    previewing.set_defaults(handler=preview)
    plotting = commands.add_parser("plot", help="Regenerate plots from saved data; no simulation")
    plotting.set_defaults(handler=plot)
    for sub in (training, running, previewing, plotting):
        sub.add_argument("--morphologies", nargs="+", choices=MORPHOLOGIES, default=MORPHOLOGIES)
    for sub in (training, running, previewing):
        sub.add_argument("--data-dir", default="data")
    for sub in (training, running):
        sub.add_argument("--checkpoint-root", default="checkpoints")
        sub.add_argument("--checkpoint-tag", default="default")
        sub.add_argument("--device", default="auto")
    for sub in (running, previewing, plotting):
        sub.add_argument("--output-root", default=None)
        sub.add_argument(
            "--manipulation", choices=("transplant", "insertion"), default="transplant"
        )
        sub.add_argument(
            "--insertion-angle",
            type=float,
            default=None,
            help="Optional insertion rotation override; defaults: gecko tail -90 deg, planaria head aligned outward",
        )
    for sub in (running, plotting):
        sub.add_argument("--snapshot-replicates", type=int, default=3)
    args = parser.parse_args()
    if hasattr(args, "manipulation"):
        if args.output_root is None:
            args.output_root = (
                "reports/experiment_3_insertion_local_restore"
                if getattr(args, "persistent_local_beta", 0.0)
                else "reports/experiment_3_insertion"
                if args.manipulation == "insertion"
                else "reports/experiment_3"
            )
    if getattr(args, "snapshot_replicates", 0) < 0:
        parser.error("snapshot-replicates cannot be negative")
    with threadpool_limits(limits=1):
        args.handler(args)


if __name__ == "__main__":
    main()
