from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend("Agg")


def _display_rgb(state: torch.Tensor) -> np.ndarray:
    rgba = state[:4].detach().cpu().permute(1, 2, 0).numpy()
    alpha = np.clip(rgba[..., 3:4], 0.0, 1.0)
    return np.clip(rgba[..., :3] + 1.0 - alpha, 0.0, 1.0)


def _condition_labels(results_dir: Path) -> dict[str, str]:
    path = results_dir / "conditions.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["condition"]: row["label"] for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot selected NCA intervention snapshots.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    snapshot_path = results_dir / "snapshots.pt"
    payload = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    snapshots = payload["intervention"]
    missing = [name for name in args.conditions if name not in snapshots]
    if missing:
        available = ", ".join(sorted(snapshots))
        raise ValueError(f"unknown conditions: {missing}; available: {available}")

    steps = args.steps or sorted(snapshots[args.conditions[0]])
    for name in args.conditions:
        missing_steps = [step for step in steps if step not in snapshots[name]]
        if missing_steps:
            raise ValueError(f"condition {name} has no snapshots at steps {missing_steps}")

    labels = _condition_labels(results_dir)
    figure, axes = plt.subplots(
        len(args.conditions),
        len(steps),
        figsize=(2.4 * len(steps), 2.2 * len(args.conditions)),
        squeeze=False,
    )
    for row, name in enumerate(args.conditions):
        for column, step in enumerate(steps):
            axis = axes[row, column]
            state = snapshots[name][step]
            if state is None:
                axis.text(
                    0.5,
                    0.5,
                    "Numerically\ndiverged",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            else:
                axis.imshow(_display_rgb(state))
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"t={step}")
            if column == 0:
                axis.set_ylabel(labels.get(name, name))

    output = Path(args.output) if args.output else results_dir / "selected_snapshots.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
