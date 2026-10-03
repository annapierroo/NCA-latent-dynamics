from __future__ import annotations

import torch

from latent_nca.models import GrowingNCA, load_growing_checkpoint

from .config import ExperimentConfig


def load_model(config: ExperimentConfig, device: torch.device) -> GrowingNCA:
    return load_growing_checkpoint(
        config.checkpoint,
        channels=config.model.channels,
        hidden_size=config.model.hidden_size,
        update_probability=config.model.update_probability,
        device=device,
    )


__all__ = ["load_model"]
