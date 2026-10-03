"""Experiment 1: hidden-state control of long-term Growing NCA degeneration."""

from .config import ExperimentConfig, load_experiment_config
from .metrics import MorphologyEvaluator, SurvivalOutcome, TargetRegime


def run_experiment(config: ExperimentConfig):
    """Load the plotting/runtime stack only when the full experiment is requested."""
    from .runner import run_experiment as _run_experiment

    return _run_experiment(config)


__all__ = [
    "ExperimentConfig",
    "MorphologyEvaluator",
    "SurvivalOutcome",
    "TargetRegime",
    "load_experiment_config",
    "run_experiment",
]
