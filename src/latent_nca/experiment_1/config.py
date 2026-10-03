from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 16
    hidden_size: int = 128
    target_size: int = 40
    target_padding: int = 16
    update_probability: float = 0.5
    alpha_threshold: float = 0.1


@dataclass(frozen=True)
class CalibrationConfig:
    replicates: int = 32
    seed_start: int = 10_000
    mature_start: int = 64
    mature_stop: int = 96
    acceptance_quantile: float = 0.95


@dataclass(frozen=True)
class EvaluationConfig:
    replicates: int = 30
    seed_start: int = 20_000
    continuation_seed_offset: int = 1_000_000
    horizon: int = 2_000
    attainment_patience: int = 5
    failure_patience: int = 10


@dataclass(frozen=True)
class InterventionConfig:
    include_standard_conditions: bool = True
    partial_freeze_counts: tuple[int, ...] = ()
    restore_strengths: tuple[float, ...] = (0.01, 0.03, 0.1, 0.25, 0.5, 0.75)
    periodic_reset_intervals: tuple[int, ...] = ()
    development_reference_steps: tuple[int, ...] = ()
    development_restore_strengths: tuple[float, ...] = ()
    development_reset_intervals: tuple[int, ...] = ()


@dataclass(frozen=True)
class AnalysisConfig:
    hidden_channel_record_every: int = 10
    visible_channel_record_every: int = 10
    snapshot_steps: tuple[int, ...] = (0, 100, 500, 1_000, 2_000)
    snapshot_all_conditions: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment_1"
    checkpoint: str = "checkpoints/growing/gecko/model.pt"
    target: str = "data/gecko.png"
    output_dir: str = "reports/experiment_1/gecko"
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    interventions: InterventionConfig = field(default_factory=InterventionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.name.startswith("experiment_1"):
            raise ValueError("name must start with 'experiment_1'")
        if self.model.channels <= 4:
            raise ValueError("the model must have at least one hidden channel")
        if not 0.0 < self.model.update_probability <= 1.0:
            raise ValueError("update_probability must be in (0, 1]")
        if not 0.0 < self.model.alpha_threshold < 1.0:
            raise ValueError("alpha_threshold must be in (0, 1)")
        if self.calibration.replicates <= 0 or self.evaluation.replicates <= 0:
            raise ValueError("replicate counts must be positive")
        if self.calibration.mature_start < 0:
            raise ValueError("mature_start must be non-negative")
        if self.calibration.mature_stop < self.calibration.mature_start:
            raise ValueError("mature_stop must be at least mature_start")
        if not 0.5 < self.calibration.acceptance_quantile < 1.0:
            raise ValueError("acceptance_quantile must be in (0.5, 1)")
        if self.evaluation.horizon <= 0:
            raise ValueError("evaluation horizon must be positive")
        if self.evaluation.continuation_seed_offset <= 0:
            raise ValueError("continuation_seed_offset must be positive")
        if self.evaluation.attainment_patience <= 0 or self.evaluation.failure_patience <= 0:
            raise ValueError("patience values must be positive")
        hidden_channels = self.model.channels - 4
        if any(
            value <= 0 or value >= hidden_channels
            for value in self.interventions.partial_freeze_counts
        ):
            raise ValueError(
                "partial freeze counts must be positive and smaller than the hidden-channel count"
            )
        if any(not 0.0 < value < 1.0 for value in self.interventions.restore_strengths):
            raise ValueError("restore strengths must be in (0, 1); freeze supplies beta=1")
        if any(value <= 0 for value in self.interventions.periodic_reset_intervals):
            raise ValueError("periodic reset intervals must be positive")
        reference_steps = self.interventions.development_reference_steps
        if len(reference_steps) != len(set(reference_steps)):
            raise ValueError("development reference steps must be unique")
        if any(step <= 0 or step >= self.calibration.mature_start for step in reference_steps):
            raise ValueError(
                "development reference steps must be positive and precede mature_start"
            )
        if any(not 0.0 < value < 1.0 for value in self.interventions.development_restore_strengths):
            raise ValueError("development restore strengths must be in (0, 1)")
        if any(value <= 0 for value in self.interventions.development_reset_intervals):
            raise ValueError("development reset intervals must be positive")
        if (
            self.interventions.development_restore_strengths
            or self.interventions.development_reset_intervals
        ) and not reference_steps:
            raise ValueError(
                "development restoration requires at least one development reference step"
            )
        if (
            self.analysis.hidden_channel_record_every <= 0
            or self.analysis.visible_channel_record_every <= 0
        ):
            raise ValueError("hidden- and visible-channel sampling intervals must be positive")
        if any(step < 0 or step > self.evaluation.horizon for step in self.analysis.snapshot_steps):
            raise ValueError("snapshot steps must fall within the evaluation horizon")


def _section(cls, values: dict[str, Any] | None):
    values = dict(values or {})
    for key in (
        "partial_freeze_counts",
        "restore_strengths",
        "periodic_reset_intervals",
        "development_reference_steps",
        "development_restore_strengths",
        "development_reset_intervals",
        "snapshot_steps",
    ):
        if key in values:
            values[key] = tuple(values[key])
    return cls(**values)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    config = ExperimentConfig(
        name=raw.get("name", "experiment_1"),
        checkpoint=raw.get("checkpoint", ExperimentConfig.checkpoint),
        target=raw.get("target", ExperimentConfig.target),
        output_dir=raw.get("output_dir", ExperimentConfig.output_dir),
        device=raw.get("device", ExperimentConfig.device),
        model=_section(ModelConfig, raw.get("model")),
        calibration=_section(CalibrationConfig, raw.get("calibration")),
        evaluation=_section(EvaluationConfig, raw.get("evaluation")),
        interventions=_section(InterventionConfig, raw.get("interventions")),
        analysis=_section(AnalysisConfig, raw.get("analysis")),
    )
    config.validate()
    return config
