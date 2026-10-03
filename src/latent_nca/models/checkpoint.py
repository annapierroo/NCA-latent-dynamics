from __future__ import annotations

from pathlib import Path

import torch

from .growing import GrowingNCA


def load_growing_checkpoint(
    path: str | Path,
    *,
    channels: int,
    hidden_size: int,
    update_probability: float,
    device: torch.device,
) -> GrowingNCA:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model = GrowingNCA(
        channels=channels,
        hidden_size=hidden_size,
        fire_rate=update_probability,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


__all__ = ["load_growing_checkpoint"]
