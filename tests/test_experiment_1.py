import numpy as np
import pytest
import torch

from latent_nca.experiment_1.conditions import (
    ConditionSpec,
    apply_post_step_intervention,
    build_conditions,
    resolve_intervention_reference,
)
from latent_nca.experiment_1.config import (
    AnalysisConfig,
    EvaluationConfig,
    ExperimentConfig,
    InterventionConfig,
)
from latent_nca.experiment_1.metrics import (
    MorphologyEvaluator,
    TargetRegime,
    first_sustained_true,
    hidden_channel_metrics,
    survival_outcome,
    visible_channel_metrics,
)
from latent_nca.experiment_1.runner import _run_branch
from latent_nca.experiment_1.visualization import (
    _finite_mean_std_count,
    _finite_median_quantiles_count,
    plot_core_morphology_metrics,
    plot_core_morphology_metrics_robust,
    plot_visible_channel_dynamics,
)


def test_target_regime_calibration_uses_requested_tails():
    records = {
        "target_mse": np.arange(1, 11, dtype=float),
        "alpha_iou": np.arange(1, 11, dtype=float) / 10,
        "area_ratio": np.arange(1, 11, dtype=float) / 10,
    }
    regime = TargetRegime.calibrate(records, acceptance_quantile=0.9)
    assert np.isclose(regime.target_mse_max, np.quantile(records["target_mse"], 0.9))
    assert np.isclose(regime.alpha_iou_min, np.quantile(records["alpha_iou"], 0.1))
    assert np.isclose(regime.area_ratio_min, np.quantile(records["area_ratio"], 0.1))
    assert np.isclose(regime.area_ratio_max, np.quantile(records["area_ratio"], 0.9))


def test_attainment_and_survival_require_sustained_runs():
    assert first_sustained_true([False, True, True, False, True, True, True], 3) == 6
    assert first_sustained_true([True, False, True], 2) is None

    outcome = survival_outcome([True, False, True, False, False, False, True], patience=3)
    assert outcome.event_observed
    assert outcome.survival_time == 3
    assert outcome.first_failure_step == 3

    censored = survival_outcome([True, True, False, True], patience=2)
    assert not censored.event_observed
    assert censored.survival_time == 3


def test_terminal_numerical_failure_is_not_right_censored():
    outcome = survival_outcome(
        [True, True, True, True, False],
        patience=10,
        terminal_failure_step=4,
    )

    assert outcome.event_observed
    assert outcome.survival_time == 4
    assert outcome.first_failure_step == 4


def test_morphology_evaluator_matches_identical_target():
    target = torch.zeros(1, 4, 8, 8)
    target[:, :3, 2:6, 2:6] = 0.5
    target[:, 3, 2:6, 2:6] = 1.0
    state = torch.zeros(1, 16, 8, 8)
    state[:, :4] = target
    evaluator = MorphologyEvaluator(target)
    metrics = evaluator.evaluate(state)
    assert metrics["target_mse"] == 0.0
    assert metrics["target_region_mse"] == 0.0
    assert metrics["alpha_iou"] == 1.0
    assert metrics["area_ratio"] == 1.0


def test_target_region_mse_ignores_errors_outside_target_mask():
    target = torch.zeros(1, 4, 6, 6)
    target[:, :3, 2:4, 2:4] = 0.5
    target[:, 3, 2:4, 2:4] = 1.0
    state = torch.zeros(1, 16, 6, 6)
    state[:, :4] = target
    state[:, :4, :2, :2] = 1.0
    evaluator = MorphologyEvaluator(target)

    background_error = evaluator.evaluate(state)
    assert background_error["target_mse"] > 0.0
    assert background_error["target_region_mse"] == 0.0

    state[:, :4, 2:4, 2:4] += 0.25
    target_error = evaluator.evaluate(state)
    assert np.isclose(target_error["target_region_mse"], 0.25**2)


def test_large_finite_state_keeps_mse_and_channel_distances_finite():
    target = torch.zeros(1, 4, 6, 6)
    target[:, 3, 2:4, 2:4] = 1
    mature = torch.ones(1, 16, 6, 6)
    state = mature.clone()
    state[:, :4] = target
    state[:, :4, :2] = 1e20
    state[:, 4:] = 1e20
    values = MorphologyEvaluator(target, mature_state=mature).evaluate(state, mature)
    assert all(np.isfinite(value) for value in values.values())
    assert values["target_region_mse"] == 0
    assert values["target_mse"] > float(np.finfo(np.float32).max)
    channels = hidden_channel_metrics(state, mature, mature)
    assert all(torch.isfinite(value).all() for value in channels.values())


def test_finite_state_does_not_terminate_branch_when_float32_diagnostics_would_overflow():
    class LargeFiniteModel:
        def __call__(self, state, update_probability):
            return torch.full_like(state, 1e20)

    target = torch.ones(1, 4, 4, 4)
    mature = torch.ones(1, 16, 4, 4)
    cfg = ExperimentConfig(
        evaluation=EvaluationConfig(horizon=3, failure_patience=2),
        analysis=AnalysisConfig(
            hidden_channel_record_every=1, visible_channel_record_every=1, snapshot_steps=(0, 3)
        ),
    )
    result = _run_branch(
        LargeFiniteModel(),
        target,
        mature,
        mature,
        TargetRegime(0.1, 0.9, 0.9, 1.1, 0.95),
        ConditionSpec("baseline", "Control", "baseline"),
        cfg,
        0,
        1,
        False,
    )
    assert result.summary["numerical_divergence_step"] == ""
    assert result.summary["first_failure_step"] == 1
    assert np.isfinite(result.metrics["target_mse"]).all()
    assert result.metrics["target_mse"].dtype == np.float64
    assert np.isfinite(result.channel_metrics["relative_l2"]).all()


def test_nonfinite_calibration_is_rejected():
    records = {
        "target_mse": np.array([0.1, np.inf]),
        "alpha_iou": np.ones(2),
        "area_ratio": np.ones(2),
    }
    with pytest.raises(ValueError, match="calibration"):
        TargetRegime.calibrate(records, 0.95)


def test_core_morphology_plot_includes_target_region_mse(tmp_path):
    condition = ConditionSpec("baseline", "Control", "baseline")
    metrics = {
        "baseline": {
            "target_mse": np.full((2, 3), 0.01),
            "target_region_mse": np.full((2, 3), 0.02),
            "alpha_iou": np.ones((2, 3)),
            "area_ratio": np.ones((2, 3)),
        }
    }
    regime = TargetRegime(0.1, 0.9, 0.9, 1.1, 0.95)
    output = tmp_path / "core_metrics.png"
    robust_output = tmp_path / "core_metrics_robust.png"

    plot_core_morphology_metrics(output, metrics, (condition,), regime)
    plot_core_morphology_metrics_robust(robust_output, metrics, (condition,), regime)

    assert output.is_file()
    assert output.stat().st_size > 0
    assert robust_output.is_file()
    assert robust_output.stat().st_size > 0


def test_finite_mean_std_count_uses_sample_sd_and_tracks_missing_replicates():
    values = np.asarray(
        [
            [1.0, 2.0, np.nan],
            [3.0, 4.0, np.nan],
            [5.0, np.nan, 7.0],
        ]
    )

    mean, standard_deviation, count = _finite_mean_std_count(values)

    assert np.allclose(mean, [3.0, 3.0, 7.0])
    assert np.allclose(standard_deviation[:2], [2.0, np.sqrt(2.0)])
    assert np.isnan(standard_deviation[2])
    assert np.array_equal(count, [3, 2, 1])


def test_finite_median_quantiles_count_tracks_distribution_and_missing_replicates():
    values = np.asarray(
        [
            [1.0, 2.0, np.nan],
            [3.0, 4.0, np.nan],
            [5.0, np.nan, 7.0],
        ]
    )

    median, q25, q75, q05, q95, count = _finite_median_quantiles_count(values)

    assert np.allclose(median, [3.0, 3.0, 7.0])
    assert np.allclose(q25, [2.0, 2.5, 7.0])
    assert np.allclose(q75, [4.0, 3.5, 7.0])
    assert np.allclose(q05, [1.2, 2.1, 7.0])
    assert np.allclose(q95, [4.8, 3.9, 7.0])
    assert np.array_equal(count, [3, 2, 1])


def test_visible_channel_heatmap_handles_extremes_and_early_window(tmp_path):
    conditions = (
        ConditionSpec("baseline", "Control", "baseline"),
        ConditionSpec("unstable", "Unstable", "restore", "beta", 0.1),
    )
    baseline = np.linspace(0.0, 2.0, 20, dtype=np.float32).reshape(1, 5, 4)
    unstable = baseline.copy()
    unstable[:, -1] = 1e12
    metrics = {
        "baseline": {
            "relative_l2": np.repeat(baseline, 2, axis=0),
            "step_rms": np.repeat(baseline * 1e-3, 2, axis=0),
        },
        "unstable": {
            "relative_l2": np.repeat(unstable, 2, axis=0),
            "step_rms": np.repeat(unstable * 1e-3, 2, axis=0),
        },
    }
    output = tmp_path / "visible_channels_early.png"

    plot_visible_channel_dynamics(
        output,
        metrics,
        conditions,
        [0, 250, 500, 750, 1000],
        max_step=500,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_freeze_and_restore_only_replace_hidden_channels():
    mature = torch.zeros(1, 16, 4, 4)
    mature[:, 4:] = 2.0
    next_state = torch.ones_like(mature)
    visible = next_state[:, :4].clone()

    freeze = ConditionSpec("freeze", "Freeze", "freeze", "beta", 1.0)
    frozen = apply_post_step_intervention(next_state, mature, freeze, relative_step=1)
    assert torch.equal(frozen[:, :4], visible)
    assert torch.all(frozen[:, 4:] == 2.0)

    restore = ConditionSpec("restore", "Restore", "restore", "beta", 0.25)
    restored = apply_post_step_intervention(next_state, mature, restore, relative_step=1)
    assert torch.allclose(restored[:, 4:], torch.full_like(restored[:, 4:], 1.25))


def test_partial_freeze_replaces_only_selected_hidden_channels():
    mature = torch.zeros(1, 16, 4, 4)
    mature[:, 4:] = 2.0
    next_state = torch.ones_like(mature)
    condition = ConditionSpec(
        "hidden_freeze_first_6",
        "Freeze first 6 hidden",
        "partial_freeze",
        "frozen_channels",
        6,
        tuple(range(6)),
    )

    frozen = apply_post_step_intervention(next_state, mature, condition, relative_step=1)

    assert torch.equal(frozen[:, :4], next_state[:, :4])
    assert torch.all(frozen[:, 4:10] == 2.0)
    assert torch.all(frozen[:, 10:] == 1.0)


def test_hidden_channel_metrics_compare_each_channel_with_mature_state():
    mature = torch.ones(1, 16, 2, 2)
    state = mature.clone()
    state[:, 4] = 2.0

    metrics = hidden_channel_metrics(state, mature, previous_state=mature)

    assert torch.isclose(metrics["relative_l2"][0], torch.tensor(1.0))
    assert torch.isclose(metrics["rms_delta"][0], torch.tensor(1.0))
    assert torch.isclose(metrics["step_rms"][0], torch.tensor(1.0))
    assert torch.all(metrics["relative_l2"][1:] == 0.0)


def test_visible_channel_metrics_compare_rgba_with_mature_state():
    mature = torch.ones(1, 16, 2, 2)
    state = mature.clone()
    state[:, 3] = 1.5

    metrics = visible_channel_metrics(state, mature, previous_state=mature)

    assert metrics["relative_l2"].shape == (4,)
    assert torch.all(metrics["relative_l2"][:3] == 0.0)
    assert torch.isclose(metrics["relative_l2"][3], torch.tensor(0.5))
    assert torch.isclose(metrics["rms_delta"][3], torch.tensor(0.5))
    assert torch.isclose(metrics["step_rms"][3], torch.tensor(0.5))


def test_numerical_divergence_terminates_only_the_current_branch():
    class DivergingModel:
        def __call__(self, state, update_probability):
            del update_probability
            return torch.full_like(state, float("nan"))

    target = torch.zeros(1, 4, 4, 4)
    mature = torch.zeros(1, 16, 4, 4)
    regime = TargetRegime(1.0, 0.0, 0.0, 1.0, 0.95)
    config = ExperimentConfig(
        evaluation=EvaluationConfig(replicates=1, horizon=5, failure_patience=2),
        analysis=AnalysisConfig(
            hidden_channel_record_every=1,
            visible_channel_record_every=1,
            snapshot_steps=(0, 2, 5),
        ),
    )
    result = _run_branch(
        DivergingModel(),
        target,
        mature,
        mature,
        regime,
        ConditionSpec("baseline", "Control", "baseline"),
        config,
        replicate=0,
        continuation_seed=1,
        collect_snapshots=True,
    )

    assert result.summary["numerical_divergence_step"] == 1
    assert np.isfinite(result.metrics["target_mse"][0])
    assert np.isnan(result.metrics["target_mse"][1:]).all()
    assert result.in_regime.tolist() == [True, False, False, False, False, False]
    assert result.channel_metrics["relative_l2"].shape == (6, 12)
    assert np.isnan(result.channel_metrics["relative_l2"][1:]).all()
    assert result.visible_channel_metrics["relative_l2"].shape == (6, 4)
    assert np.isnan(result.visible_channel_metrics["relative_l2"][1:]).all()
    assert result.snapshots[0] is not None
    assert result.snapshots[2] is None
    assert result.snapshots[5] is None


def test_late_numerical_divergence_is_an_observed_failure():
    class LateDivergingModel:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, state, update_probability):
            del update_probability
            self.calls += 1
            return torch.full_like(state, float("nan")) if self.calls == 5 else state

    target = torch.zeros(1, 4, 4, 4)
    mature = torch.zeros(1, 16, 4, 4)
    regime = TargetRegime(1.0, 0.0, 0.0, 1.0, 0.95)
    config = ExperimentConfig(
        evaluation=EvaluationConfig(replicates=1, horizon=5, failure_patience=10),
        analysis=AnalysisConfig(
            hidden_channel_record_every=1,
            visible_channel_record_every=1,
            snapshot_steps=(0, 5),
        ),
    )

    result = _run_branch(
        LateDivergingModel(),
        target,
        mature,
        mature,
        regime,
        ConditionSpec("baseline", "Control", "baseline"),
        config,
        replicate=0,
        continuation_seed=1,
        collect_snapshots=False,
    )

    assert result.summary["numerical_divergence_step"] == 5
    assert result.summary["event_observed"] is True
    assert result.summary["survival_time"] == 5


def test_condition_grid_has_no_duplicate_controls():
    conditions = build_conditions(InterventionConfig())
    assert sum(condition.name == "baseline" for condition in conditions) == 1
    assert sum(condition.name == "hidden_freeze" for condition in conditions) == 1
    assert not any(
        condition.family == "restore" and condition.parameter == 1.0 for condition in conditions
    )


def test_condition_grid_includes_requested_partial_freeze():
    conditions = build_conditions(InterventionConfig(partial_freeze_counts=(6,)))
    partial = next(condition for condition in conditions if condition.family == "partial_freeze")
    assert partial.name == "hidden_freeze_first_6"
    assert partial.hidden_indices == tuple(range(6))


def test_periodic_reset_grid_can_run_without_standard_conditions():
    conditions = build_conditions(
        InterventionConfig(
            include_standard_conditions=False,
            periodic_reset_intervals=(100,),
        )
    )
    assert [condition.name for condition in conditions] == ["baseline", "periodic_reset_100"]


def test_periodic_reset_is_a_hard_pulse_followed_by_free_evolution():
    mature = torch.zeros(1, 16, 4, 4)
    mature[:, 4:] = 2.0
    next_state = torch.ones_like(mature)
    condition = ConditionSpec(
        "periodic_reset_100",
        "Periodic reset P=100",
        "periodic_reset",
        "interval",
        100,
    )

    before_pulse = apply_post_step_intervention(next_state, mature, condition, relative_step=99)
    pulse = apply_post_step_intervention(next_state, mature, condition, relative_step=100)
    after_pulse = apply_post_step_intervention(next_state, mature, condition, relative_step=101)

    assert before_pulse is next_state
    assert after_pulse is next_state
    assert torch.equal(pulse[:, :4], next_state[:, :4])
    assert torch.all(pulse[:, 4:] == 2.0)


def test_developmental_conditions_cover_both_references_and_interventions():
    conditions = build_conditions(
        InterventionConfig(
            include_standard_conditions=False,
            development_reference_steps=(16, 48),
            development_restore_strengths=(0.1,),
            development_reset_intervals=(150,),
        )
    )

    assert [condition.name for condition in conditions] == [
        "baseline",
        "restore_development_step_16_beta_0p1",
        "restore_development_step_48_beta_0p1",
        "restore_development_step_16_period_150",
        "restore_development_step_48_period_150",
    ]
    assert [condition.reference_step for condition in conditions[1:]] == [16, 48, 16, 48]


def test_continuous_developmental_restore_changes_only_hidden_channels():
    reference = torch.zeros(1, 16, 3, 3)
    reference[:, 4:] = 4.0
    next_state = torch.ones_like(reference)
    condition = ConditionSpec(
        "restore_development_step_16_beta_0p1",
        "Restore dev t=16 beta=0.1",
        "development_restore",
        "beta",
        0.1,
        reference_step=16,
    )

    restored = apply_post_step_intervention(next_state, reference, condition, relative_step=1)

    assert torch.equal(restored[:, :4], next_state[:, :4])
    assert torch.allclose(restored[:, 4:], torch.full_like(restored[:, 4:], 1.3))


def test_periodic_developmental_reset_is_hard_only_every_150_steps():
    reference = torch.zeros(1, 16, 3, 3)
    reference[:, 4:] = 4.0
    next_state = torch.ones_like(reference)
    condition = ConditionSpec(
        "restore_development_step_48_period_150",
        "Dev reset t=48 P=150",
        "development_periodic_reset",
        "interval",
        150,
        reference_step=48,
    )

    free = apply_post_step_intervention(next_state, reference, condition, relative_step=149)
    reset = apply_post_step_intervention(next_state, reference, condition, relative_step=150)

    assert free is next_state
    assert torch.equal(reset[:, :4], next_state[:, :4])
    assert torch.all(reset[:, 4:] == 4.0)


def test_developmental_reference_is_loaded_from_the_same_replicate_record():
    mature = torch.zeros(1, 16, 3, 3)
    stored = torch.full((16, 3, 3), 2.0)
    condition = ConditionSpec(
        "restore_development_step_16_beta_0p1",
        "Restore dev t=16 beta=0.1",
        "development_restore",
        "beta",
        0.1,
        reference_step=16,
    )

    resolved = resolve_intervention_reference(mature, condition, {16: stored})

    assert resolved.shape == mature.shape
    assert torch.equal(resolved[0], stored)
    with pytest.raises(ValueError, match="missing developmental state"):
        resolve_intervention_reference(mature, condition, {})


def test_development_reference_steps_must_precede_target_calibration_window():
    config = ExperimentConfig(
        interventions=InterventionConfig(
            development_reference_steps=(64,),
            development_restore_strengths=(0.1,),
        )
    )

    with pytest.raises(ValueError, match="precede mature_start"):
        config.validate()


def test_hidden_reference_distance_is_distinct_from_distance_to_h_star():
    target = torch.zeros(1, 4, 3, 3)
    mature = torch.zeros(1, 16, 3, 3)
    mature[:, 4:] = 2.0
    reference = torch.zeros_like(mature)
    reference[:, 4:] = 1.0
    evaluator = MorphologyEvaluator(
        target,
        mature_state=mature,
        hidden_reference_state=reference,
    )

    metrics = evaluator.evaluate(mature)

    assert metrics["hidden_relative_l2"] == 0.0
    assert np.isclose(metrics["hidden_reference_relative_l2"], 1.0)
