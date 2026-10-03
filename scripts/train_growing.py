from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from latent_nca.data.device import get_device, seed_everything
from latent_nca.data.images import load_distill_target
from latent_nca.models import GrowingNCA
from latent_nca.training.growing import TrainingConfig, train_growing, train_persistent

MORPHOLOGIES = ("gecko", "butterfly", "spider", "planaria")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Growing NCA per morphology.")
    parser.add_argument("--config", default="configs/growing.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="checkpoints/growing")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--morphologies",
        nargs="+",
        choices=MORPHOLOGIES,
        default=MORPHOLOGIES,
        help="Morphologies to train (default: all).",
    )
    args = parser.parse_args()

    settings = yaml.safe_load(Path(args.config).read_text())
    device = get_device(args.device)
    seed_everything(args.seed)
    config = TrainingConfig(
        **{key: settings[key] for key in TrainingConfig.__dataclass_fields__ if key in settings}
    )

    for morphology in args.morphologies:
        target_path = Path(args.data_dir) / f"{morphology}.png"
        target = load_distill_target(
            target_path,
            size=settings.get("target_size", 40),
            padding=settings.get("target_padding", 16),
            device=device,
        )
        model = GrowingNCA(
            channels=settings.get("channels", 16),
            hidden_size=settings.get("hidden_size", 128),
            fire_rate=settings.get("update_probability", 0.5),
        ).to(device)
        output = Path(args.output_dir) / morphology
        trainer = train_persistent if config.mode == "persistent" else train_growing
        trainer(model, target, config, output / "model.pt", output / "loss.txt")
        print(f"finished {morphology} on {device}")


if __name__ == "__main__":
    main()
