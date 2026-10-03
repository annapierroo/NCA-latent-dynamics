from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from latent_nca.experiment_1 import load_experiment_config, run_experiment

MORPHOLOGIES = ("gecko", "butterfly", "spider", "planaria")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paired Growing NCA hidden-state interventions for one or more "
            "target morphologies."
        )
    )
    parser.add_argument("--config", default="configs/experiment_1.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-root", default="checkpoints/growing")
    parser.add_argument(
        "--output-root",
        default="reports/experiment_1",
    )
    parser.add_argument(
        "--morphologies",
        nargs="+",
        choices=MORPHOLOGIES,
        default=MORPHOLOGIES,
    )
    parser.add_argument("--device")
    args = parser.parse_args()

    base = load_experiment_config(args.config)
    for morphology in args.morphologies:
        config = replace(
            base,
            checkpoint=str(Path(args.checkpoint_root) / morphology / "model.pt"),
            target=str(Path(args.data_dir) / f"{morphology}.png"),
            output_dir=str(Path(args.output_root) / morphology),
            device=base.device if args.device is None else args.device,
        )
        output_dir = run_experiment(config)
        print(f"Experiment 1 completed for {morphology}: {output_dir}")


if __name__ == "__main__":
    main()
