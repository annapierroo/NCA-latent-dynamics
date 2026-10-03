from __future__ import annotations

import torch


def make_seed_pool(seed: torch.Tensor, pool_size: int) -> torch.Tensor:
    return seed.repeat(pool_size, 1, 1, 1).clone()


def sample_pool(pool: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(pool.shape[0], device=pool.device)[:batch_size]
    return indices, pool[indices].clone()


def replace_pool(pool: torch.Tensor, indices: torch.Tensor, states: torch.Tensor) -> None:
    pool[indices] = states.detach()


def damage_circles(states: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        return states
    damaged = states.clone()
    height, width = damaged.shape[-2:]
    n = min(count, states.shape[0])
    x = torch.linspace(-1.0, 1.0, width, device=states.device)[None, None, :]
    y = torch.linspace(-1.0, 1.0, height, device=states.device)[None, :, None]
    center = torch.empty(2, n, 1, 1, device=states.device).uniform_(-0.5, 0.5)
    radius = torch.empty(n, 1, 1, device=states.device).uniform_(0.1, 0.4)
    x = (x - center[0]) / radius
    y = (y - center[1]) / radius
    mask = (x.square() + y.square() < 1.0).to(states.dtype)
    damaged[:n] *= 1.0 - mask[:, None]
    return damaged
