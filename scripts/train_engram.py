from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from latent_nca.data.device import get_device, seed_everything
from latent_nca.data.images import load_engram_image
from latent_nca.models import GeneCA, GenePropCA
from latent_nca.training.engram import EngramTrainingConfig, train_engram

MORPHOLOGIES = ("gecko", "butterfly", "spider")
PRIMITIVE_SETS = {
    "shapes": (("square", "circle", "triangle"), ([0], [1], [2])),
    "body_parts": (
        ("torso", "head", "tail", "leg1", "leg2", "leg3", "leg4"),
        ([0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]),
    ),
    "lines": (("vertical", "horizontal"), ([0], [1])),
}
MORPHOLOGY_GENES = {
    "gecko": [1],
    "butterfly": [0],
    "spider": [2],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GeneCA and GenePropCA per morphology.")
    parser.add_argument("--config", default="configs/engram.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--primitive-dir", default="data/engram_primitives")
    parser.add_argument("--output-dir", default="checkpoints/engram")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = yaml.safe_load(Path(args.config).read_text())
    device = get_device(args.device)
    seed_everything(args.seed)
    config = EngramTrainingConfig(
        **{
            key: settings[key]
            for key in EngramTrainingConfig.__dataclass_fields__
            if key in settings
        }
    )
    primitive_names, primitive_genes = PRIMITIVE_SETS[settings.get("primitive_set", "body_parts")]
    primitive_targets = [
        load_engram_image(
            Path(args.primitive_dir) / f"{primitive}.png",
            size=settings.get("primitive_size", 30),
            padding=settings.get("primitive_padding", 5),
            device=device,
        )
        for primitive in primitive_names
    ]

    for morphology in MORPHOLOGIES:
        target = load_engram_image(
            Path(args.data_dir) / f"{morphology}.png",
            size=settings.get("morphology_size", 60),
            padding=settings.get("morphology_padding", 5),
            device=device,
        )
        channels = settings.get("channels", 16)
        gene_size = settings.get("gene_size", 8)
        train_engram(
            GeneCA(
                channels=channels,
                hidden_size=settings.get("gene_hidden_size", 64),
                gene_size=gene_size,
            ).to(device),
            GenePropCA(
                channels=channels,
                hidden_size=settings.get("gene_prop_hidden_size", 128),
                gene_size=gene_size,
            ).to(device),
            primitive_targets,
            primitive_genes,
            target,
            MORPHOLOGY_GENES[morphology],
            config,
            Path(args.output_dir) / morphology,
        )
        print(f"finished {morphology} on {device}")


if __name__ == "__main__":
    main()
