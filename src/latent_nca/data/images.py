from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _resampling_filter() -> int:
    return getattr(Image, "Resampling", Image).LANCZOS


def load_rgba_target(
    path: str | Path,
    size: int = 40,
    padding: int = 16,
    device: torch.device | None = None,
    preserve_aspect: bool = False,
) -> torch.Tensor:
    """Load an RGBA target with premultiplied RGB, matching the NCA papers."""
    image = Image.open(path).convert("RGBA")
    if preserve_aspect:
        image.thumbnail((size, size), _resampling_filter())
    else:
        image = image.resize((size, size), _resampling_filter())
    values_np = np.asarray(image).astype(np.float32) / 255.0
    values_np[..., :3] *= values_np[..., 3:4]
    values = torch.from_numpy(values_np)
    height, width = values.shape[:2]
    target = torch.zeros(height + 2 * padding, width + 2 * padding, 4)
    target[padding : padding + height, padding : padding + width] = values
    return target.permute(2, 0, 1).unsqueeze(0).to(device=device)


def load_distill_target(
    path: str | Path, size: int = 40, padding: int = 16, device: torch.device | None = None
) -> torch.Tensor:
    return load_rgba_target(path, size=size, padding=padding, device=device, preserve_aspect=True)


def load_engram_image(
    path: str | Path, size: int, padding: int = 0, device: torch.device | None = None
) -> torch.Tensor:
    return load_rgba_target(path, size=size, padding=padding, device=device, preserve_aspect=False)
