from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GrowingNCA(nn.Module):
    """PyTorch reproduction of Distill's original 16-channel Growing NCA rule."""

    def __init__(self, channels: int = 16, hidden_size: int = 128, fire_rate: float = 0.5) -> None:
        super().__init__()
        self.channels = channels
        self.fire_rate = fire_rate
        self.perception = nn.Conv2d(channels * 3, hidden_size, 1)
        self.update = nn.Conv2d(hidden_size, channels, 1)
        nn.init.zeros_(self.update.weight)
        nn.init.zeros_(self.update.bias)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x[None, None].repeat(channels, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y[None, None].repeat(channels, 1, 1, 1))

    def get_living_mask(self, state: torch.Tensor) -> torch.Tensor:
        alpha = state[:, 3:4]
        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, state: torch.Tensor, angle: float | torch.Tensor = 0.0) -> torch.Tensor:
        grad_x = F.conv2d(state, self.sobel_x, padding=1, groups=self.channels)
        grad_y = F.conv2d(state, self.sobel_y, padding=1, groups=self.channels)
        if isinstance(angle, torch.Tensor):
            c, s = torch.cos(angle), torch.sin(angle)
        else:
            c, s = math.cos(angle), math.sin(angle)
        return torch.cat((state, c * grad_x - s * grad_y, s * grad_x + c * grad_y), dim=1)

    def forward(
        self,
        state: torch.Tensor,
        update_probability: float | None = None,
        angle: float | torch.Tensor = 0.0,
        step_size: float = 1.0,
    ) -> torch.Tensor:
        pre_life_mask = self.get_living_mask(state)
        delta = self.update(F.relu(self.perception(self.perceive(state, angle)))) * step_size
        fire_rate = self.fire_rate if update_probability is None else update_probability
        update_mask = (torch.rand_like(state[:, :1]) <= fire_rate).to(state.dtype)
        state = state + delta * update_mask
        post_life_mask = self.get_living_mask(state)
        life_mask = pre_life_mask & post_life_mask
        return state * life_mask.to(state.dtype)

    def seed(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        state = torch.zeros(batch_size, self.channels, height, width, device=device)
        cy, cx = height // 2, width // 2
        state[:, 3:, cy, cx] = 1.0
        return state
