from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class _Perception(nn.Module):
    """EngramNCA perception: identity plus Sobel-x, Sobel-y, and Laplacian."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        laplacian = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]])
        filters = torch.stack((sobel_x, sobel_x.t(), laplacian))
        self.channels = channels
        self.register_buffer("filters", filters[:, None].repeat(channels, 1, 1, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        padded = F.pad(state, (1, 1, 1, 1), mode="circular")
        observed = F.conv2d(padded, self.filters, groups=self.channels)
        return torch.cat((state, observed), dim=1)


def _pre_life_mask(state: torch.Tensor) -> torch.Tensor:
    alpha = F.pad(state[:, 3:4], (1, 1, 1, 1), mode="circular")
    return F.max_pool2d(alpha, kernel_size=3, stride=1) > 0.1


def _update_mask(state: torch.Tensor, update_probability: float) -> torch.Tensor:
    batch, _, height, width = state.shape
    mask = torch.rand(batch, 1, height, width, device=state.device, dtype=state.dtype)
    return torch.floor(mask + update_probability)


class GeneCA(nn.Module):
    """Engram GeneCA: update all non-gene channels and clone private genes."""

    def __init__(
        self,
        channels: int = 16,
        hidden_size: int = 64,
        gene_size: int = 8,
        visible_channels: int | None = None,
        public_hidden_channels: int | None = None,
        private_channels: int | None = None,
    ) -> None:
        super().__init__()
        if private_channels is not None:
            gene_size = private_channels
        if visible_channels is not None and public_hidden_channels is not None:
            channels = visible_channels + public_hidden_channels + gene_size
        self.channels = channels
        self.gene_size = gene_size
        self.public_channels = channels - gene_size
        self.perception = _Perception(channels)
        self.w1 = nn.Conv2d(channels * 4, hidden_size, 1)
        self.w2 = nn.Conv2d(hidden_size, self.public_channels, 1, bias=False)
        nn.init.zeros_(self.w2.weight)

    @property
    def visible(self) -> int:
        return 4

    @property
    def public(self) -> int:
        return self.public_channels - self.visible

    @property
    def private(self) -> int:
        return self.gene_size

    def forward(self, state: torch.Tensor, update_probability: float = 0.5) -> torch.Tensor:
        genes = state[:, -self.gene_size :]
        delta = self.w2(torch.relu(self.w1(self.perception(state))))
        alive = _pre_life_mask(state).to(state.dtype)
        mask = _update_mask(state, update_probability)
        public_state = state[:, : self.public_channels] + delta * mask * alive
        return torch.cat((public_state, genes), dim=1)

    def seed(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        genes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        state = torch.zeros(batch_size, self.channels, height, width, device=device)
        cy, cx = height // 2, width // 2
        state[:, 3 : self.public_channels, cy, cx] = 1.0
        for gene in genes or ():
            state[:, self.channels - 1 - gene, cy, cx] = 1.0
        return state


class GenePropCA(nn.Module):
    """Engram GenePropCA: update only private gene channels."""

    def __init__(
        self,
        channels: int = 16,
        hidden_size: int = 128,
        gene_size: int = 8,
        visible_channels: int | None = None,
        public_hidden_channels: int | None = None,
        private_channels: int | None = None,
    ) -> None:
        super().__init__()
        if private_channels is not None:
            gene_size = private_channels
        if visible_channels is not None and public_hidden_channels is not None:
            channels = visible_channels + public_hidden_channels + gene_size
        self.channels = channels
        self.gene_size = gene_size
        self.perception = _Perception(channels)
        self.w1 = nn.Conv2d(channels * 4, hidden_size, 1)
        self.w2 = nn.Conv2d(hidden_size, gene_size, 1, bias=False)
        nn.init.zeros_(self.w2.weight)

    def forward(
        self,
        state: torch.Tensor,
        update_probability: float = 0.5,
        is_dual: bool = False,
    ) -> torch.Tensor:
        if is_dual:
            genes = state[:, state.shape[1] - self.gene_size - 1 : -1]
            meta_gene = state[:, -1:]
            prefix = state[:, : state.shape[1] - self.gene_size - 1]
        else:
            genes = state[:, -self.gene_size :]
            meta_gene = None
            prefix = state[:, : state.shape[1] - self.gene_size]

        delta = self.w2(torch.relu(self.w1(self.perception(state))))
        alive = _pre_life_mask(state).to(state.dtype)
        mask = _update_mask(state, update_probability)
        genes = genes + delta * mask * alive
        if meta_gene is None:
            return torch.cat((prefix, genes), dim=1)
        return torch.cat((prefix, genes, meta_gene), dim=1)

    def seed(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        genes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        state = torch.zeros(batch_size, self.channels, height, width, device=device)
        cy, cx = height // 2, width // 2
        state[:, 3 : self.channels - self.gene_size, cy, cx] = 1.0
        for gene in genes or ():
            state[:, self.channels - 1 - gene, cy, cx] = 1.0
        return state


class EngramNCA(nn.Module):
    """The paper's ensemble step: GeneCA followed by GenePropCA."""

    def __init__(self, gene_ca: GeneCA, gene_prop_ca: GenePropCA) -> None:
        super().__init__()
        self.gene_ca = gene_ca
        self.gene_prop_ca = gene_prop_ca

    def forward(
        self,
        state: torch.Tensor,
        update_probability: float = 0.5,
        is_dual: bool = False,
    ) -> torch.Tensor:
        state = self.gene_ca(state, update_probability=update_probability)
        return self.gene_prop_ca(state, update_probability=update_probability, is_dual=is_dual)
