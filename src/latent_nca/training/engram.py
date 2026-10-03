from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from latent_nca.models import GeneCA, GenePropCA


@dataclass
class EngramTrainingConfig:
    gene_training_iterations: int = 15000
    prop_training_iterations: int = 14000
    gene_pool_size: int = 256
    prop_pool_size: int = 1000
    batch_size: int = 12
    train_steps_min: int = 32
    train_steps_max: int = 92
    update_probability: float = 0.5
    learning_rate: float = 1e-3
    lr_step_size: int = 5000
    lr_gamma: float = 0.3
    perception_loss_weight: float = 0.1


class GenePoolBank:
    """Separate state pools, one pool for each binary gene encoding."""

    def __init__(self, pools: list[torch.Tensor], seeds: list[torch.Tensor]) -> None:
        self.pools = pools
        self.seeds = seeds

    @classmethod
    def from_genes(
        cls,
        genes: Sequence[Sequence[int]],
        pool_size: int,
        channels: int,
        gene_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> GenePoolBank:
        pools: list[torch.Tensor] = []
        seeds: list[torch.Tensor] = []
        for gene in genes:
            seed = torch.zeros(channels, height, width, device=device)
            cy, cx = height // 2, width // 2
            seed[3 : channels - gene_size, cy, cx] = 1.0
            for gene_loc in gene:
                seed[channels - 1 - gene_loc, cy, cx] = 1.0
            seeds.append(seed)
            pools.append(seed.unsqueeze(0).repeat(pool_size, 1, 1, 1))
        return cls(pools, seeds)

    def sample_balanced(
        self, batch_size: int
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[int]]:
        base = batch_size // len(self.pools)
        remainder = batch_size % len(self.pools)
        indices: list[torch.Tensor] = []
        batches: list[torch.Tensor] = []
        partitions: list[int] = []
        for pool_index, (pool, seed) in enumerate(zip(self.pools, self.seeds, strict=True)):
            count = base + int(pool_index < remainder)
            partitions.append(count)
            if count == 0:
                indices.append(torch.empty(0, dtype=torch.long, device=pool.device))
                continue
            selected = torch.randperm(pool.shape[0], device=pool.device)[:count]
            batch = pool[selected].clone()
            batch[:1] = seed
            indices.append(selected)
            batches.append(batch)
        return indices, torch.cat(batches, dim=0), partitions

    def update(
        self, indices: list[torch.Tensor], states: torch.Tensor, partitions: list[int]
    ) -> None:
        offset = 0
        for pool, selected, count in zip(self.pools, indices, partitions, strict=True):
            if count:
                pool[selected] = states[offset : offset + count].detach()
                offset += count


def _perchannel_conv(state: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = state.shape
    state = state.reshape(batch * channels, 1, height, width)
    state = F.pad(state, (1, 1, 1, 1), mode="circular")
    observed = F.conv2d(state, filters[:, None])
    return observed.reshape(batch, -1, height, width)


def _loss(states: torch.Tensor, target: torch.Tensor, config: EngramTrainingConfig) -> torch.Tensor:
    pixel = (states[:, :4] - target).square().mean()
    if config.perception_loss_weight == 0:
        return pixel
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=states.device,
        dtype=states.dtype,
    )
    laplacian = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]],
        device=states.device,
        dtype=states.dtype,
    )
    filters = torch.stack((sobel_x, sobel_x.t(), laplacian))
    target = target.expand_as(states[:, :4])
    perception = (
        (_perchannel_conv(target, filters) - _perchannel_conv(states[:, :4], filters))
        .square()
        .mean()
    )
    return pixel + config.perception_loss_weight * perception


def _normalize_gradients(model: nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad /= parameter.grad.norm() + 1e-8


def _target_batch(targets: Sequence[torch.Tensor], partitions: Sequence[int]) -> torch.Tensor:
    return torch.cat(
        [
            target.expand(count, -1, -1, -1)
            for target, count in zip(targets, partitions, strict=True)
            if count
        ]
    )


def _train_pool_stage(
    model: nn.Module,
    bank: GenePoolBank,
    targets: Sequence[torch.Tensor],
    iterations: int,
    config: EngramTrainingConfig,
    pre_model: nn.Module | None = None,
) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
    )
    losses: list[float] = []
    model.train()
    if pre_model is not None:
        pre_model.eval()
        for parameter in pre_model.parameters():
            parameter.requires_grad_(False)

    for _ in range(iterations + 1):
        indices, state, partitions = bank.sample_balanced(config.batch_size)
        target = _target_batch(targets, partitions)
        steps = random.randrange(config.train_steps_min, config.train_steps_max)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(steps):
            if pre_model is not None:
                state = pre_model(state, update_probability=config.update_probability)
            state = model(state, update_probability=config.update_probability)
        loss = _loss(state, target, config)
        loss.backward()
        _normalize_gradients(model)
        optimizer.step()
        scheduler.step()
        bank.update(indices, state, partitions)
        losses.append(float(loss.detach().cpu().log()))
    return losses


def train_engram(
    gene_ca: GeneCA,
    gene_prop_ca: GenePropCA,
    primitive_targets: Sequence[torch.Tensor],
    primitive_genes: Sequence[Sequence[int]],
    morphology_target: torch.Tensor,
    morphology_gene: Sequence[int],
    config: EngramTrainingConfig,
    output_dir: str | Path,
) -> None:
    """Two-stage EngramNCA training matching the upstream notebooks."""
    if len(primitive_targets) != len(primitive_genes):
        raise ValueError("primitive_targets and primitive_genes must have the same length")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    primitive_height, primitive_width = primitive_targets[0].shape[-2:]
    primitive_device = primitive_targets[0].device
    gene_bank = GenePoolBank.from_genes(
        primitive_genes,
        config.gene_pool_size,
        gene_ca.channels,
        gene_ca.gene_size,
        primitive_height,
        primitive_width,
        primitive_device,
    )
    gene_losses = _train_pool_stage(
        gene_ca,
        gene_bank,
        primitive_targets,
        config.gene_training_iterations,
        config,
    )
    torch.save({"model": gene_ca.state_dict(), "config": config.__dict__}, output / "gene_ca.pt")
    (output / "gene_ca_loss.txt").write_text("\n".join(map(str, gene_losses)) + "\n")

    morphology_height, morphology_width = morphology_target.shape[-2:]
    morphology_bank = GenePoolBank.from_genes(
        [morphology_gene],
        config.prop_pool_size,
        gene_prop_ca.channels,
        gene_prop_ca.gene_size,
        morphology_height,
        morphology_width,
        morphology_target.device,
    )
    prop_losses = _train_pool_stage(
        gene_prop_ca,
        morphology_bank,
        [morphology_target],
        config.prop_training_iterations,
        config,
        pre_model=gene_ca,
    )
    torch.save(
        {"model": gene_prop_ca.state_dict(), "config": config.__dict__},
        output / "gene_prop_ca.pt",
    )
    (output / "gene_prop_ca_loss.txt").write_text("\n".join(map(str, prop_losses)) + "\n")
