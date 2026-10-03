from __future__ import annotations

import argparse
from pathlib import Path

from latent_nca.experiment_1.channel_dynamics import render_clear_channel_dynamics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render robust median heatmaps for hidden and visible channel dynamics "
            "from existing Experiment 1 metric files."
        )
    )
    parser.add_argument(
        "--results-root",
        default="reports/experiment_1",
    )
    parser.add_argument(
        "--morphologies",
        nargs="+",
        default=("gecko", "butterfly", "spider", "planaria"),
    )
    parser.add_argument("--output-name", default="channel_dynamics_clear")
    parser.add_argument("--early-stop", type=int, default=1000)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    for morphology in args.morphologies:
        results_dir = results_root / morphology
        result = render_clear_channel_dynamics(
            results_dir,
            results_dir / args.output_name,
            early_stop=args.early_stop,
        )
        print(f"Clear channel-dynamics heatmaps completed for {morphology}: {result}")


if __name__ == "__main__":
    main()
