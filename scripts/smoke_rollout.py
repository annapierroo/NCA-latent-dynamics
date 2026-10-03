from __future__ import annotations

import argparse

from latent_nca.data.device import get_device, seed_everything
from latent_nca.models import EngramNCA, GeneCA, GenePropCA, GrowingNCA
from latent_nca.training.rollout import rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate baseline NCA tensor flow.")
    parser.add_argument("--model", choices=("growing", "engram"), default="growing")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    if args.model == "growing":
        model = GrowingNCA().to(device)
        state = model.seed(1, 32, 32, device)
    else:
        gene_ca = GeneCA().to(device)
        model = EngramNCA(gene_ca, GenePropCA().to(device)).to(device)
        state = gene_ca.seed(1, 32, 32, device, genes=[0])
    final, frames = rollout(model, state, args.steps, record=True)
    print(
        f"model={args.model} device={device} frames={tuple(frames.shape)} final={tuple(final.shape)}"
    )


if __name__ == "__main__":
    main()
