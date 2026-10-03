from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from latent_nca.models import GrowingNCA
from latent_nca.training.pool import damage_circles, make_seed_pool, replace_pool, sample_pool
from latent_nca.training.rollout import rollout


@dataclass
class TrainingConfig:
    training_iterations: int = 8000
    pool_size: int = 1024
    batch_size: int = 8
    train_steps_min: int = 64
    train_steps_max: int = 96
    update_probability: float = 0.5
    damage_samples: int = 0
    learning_rate: float = 2e-3
    lr_decay_step: int = 2000
    lr_decay_factor: float = 0.1
    mode: str = "growing"


def _losses(states: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (states[:, :4] - target).square().mean(dim=(1, 2, 3))


def _normalize_gradients(model: nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad /= parameter.grad.norm() + 1e-8


def _make_distill_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: config.lr_decay_factor if step >= config.lr_decay_step else 1.0,
    )


def _finish_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: torch.Tensor,
    target: torch.Tensor,
    checkpoint_path: str | Path,
    log_path: str | Path,
    losses: list[float],
) -> list[float]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, checkpoint_path)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(str(value) for value in losses) + "\n")
    return losses


def train_growing(
    model: GrowingNCA,
    target: torch.Tensor,
    config: TrainingConfig,
    checkpoint_path: str | Path,
    log_path: str | Path,
) -> list[float]:
    """Original Growing NCA training: every optimizer step starts from seed."""
    device = target.device
    height, width = target.shape[-2:]
    seed = model.seed(1, height, width, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = _make_distill_scheduler(optimizer, config)
    losses: list[float] = []

    model.train()
    for _ in range(config.training_iterations):
        batch = seed.repeat(config.batch_size, 1, 1, 1)
        steps = random.randint(config.train_steps_min, config.train_steps_max)
        optimizer.zero_grad(set_to_none=True)
        state = rollout(model, batch, steps, config.update_probability)
        loss = _losses(state, target).mean()
        loss.backward()
        _normalize_gradients(model)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

    return _finish_training(
        model, optimizer, scheduler, state, target, checkpoint_path, log_path, losses
    )


def train_persistent(
    model: GrowingNCA,
    target: torch.Tensor,
    config: TrainingConfig,
    checkpoint_path: str | Path,
    log_path: str | Path,
) -> list[float]:
    """Distill persistent training with a state pool."""
    device = target.device
    height, width = target.shape[-2:]
    seed = model.seed(1, height, width, device)
    pool = make_seed_pool(seed, config.pool_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = _make_distill_scheduler(optimizer, config)
    losses: list[float] = []

    model.train()
    for _ in range(config.training_iterations):
        indices, batch = sample_pool(pool, config.batch_size)
        with torch.no_grad():
            order = torch.argsort(_losses(batch, target), descending=True)
            batch = batch[order]
            batch[0] = seed[0]
            if config.damage_samples:
                batch[-config.damage_samples :] = damage_circles(
                    batch[-config.damage_samples :], config.damage_samples
                )
        steps = random.randint(config.train_steps_min, config.train_steps_max)
        optimizer.zero_grad(set_to_none=True)
        state = rollout(model, batch, steps, config.update_probability)
        loss = _losses(state, target).mean()
        loss.backward()
        _normalize_gradients(model)
        optimizer.step()
        scheduler.step()
        replace_pool(pool, indices, state)
        losses.append(float(loss.detach().cpu()))

    return _finish_training(
        model, optimizer, scheduler, state, target, checkpoint_path, log_path, losses
    )
