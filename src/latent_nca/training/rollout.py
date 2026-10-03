from __future__ import annotations

from collections.abc import Callable

import torch


def rollout(
    model: Callable,
    state: torch.Tensor,
    steps: int,
    update_probability: float = 0.5,
    record: bool = False,
):
    frames = [state] if record else None
    for _ in range(steps):
        state = model(state, update_probability=update_probability)
        if record:
            frames.append(state)
    return (state, torch.stack(frames, dim=1)) if record else state
