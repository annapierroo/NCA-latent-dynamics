import csv
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from latent_nca.experiment_1.metrics import (
    MorphologyEvaluator,
    TargetRegime,
    first_sustained_true,
    hidden_channel_metrics,
    survival_outcome,
    visible_channel_metrics,
)
from scripts import experiment_1 as experiment
from scripts.experiment_1 import (
    AnalysisConfig,
    ConditionSpec,
    EvaluationConfig,
    ExperimentConfig,
    InterventionConfig,
    _finite_mean_std_count,
    _finite_median_quantiles_count,
    _run_branch,
    _set_mse_log_scale,
    apply_post_step_intervention,
    build_conditions,
    plot_core_morphology_metrics,
    plot_core_morphology_metrics_robust,
    plot_visible_channel_dynamics,
    resolve_intervention_reference,
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


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("magnitude", [1.0, 1e15, 1e20, 1e30])
def test_large_state_metrics_match_float64_reference_without_changing_state(device, magnitude):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    target = torch.ones(1, 4, 6, 6, device=device)
    mature = torch.ones(1, 16, 6, 6, device=device)
    state = torch.full_like(mature, magnitude)
    original, random_state = state.clone(), torch.get_rng_state()
    actual = MorphologyEvaluator(target, mature_state=mature).evaluate(state, mature)
    expected = np.square(state[:, :4].cpu().numpy().astype(np.float64) - 1).mean()
    assert actual["target_mse"] == pytest.approx(expected, rel=2e-6)
    assert actual["target_region_mse"] == pytest.approx(expected, rel=2e-6)
    assert actual["alpha_iou"] == 1
    assert actual["area_ratio"] == 1
    channels = hidden_channel_metrics(state, mature, mature)
    expected_rms = float(abs(state[0, 4, 0, 0].cpu()) - 1)
    np.testing.assert_allclose(channels["rms_delta"].cpu(), expected_rms, rtol=2e-6)
    assert torch.equal(state, original)
    assert torch.equal(torch.get_rng_state(), random_state)


def test_finite_empty_morphology_keeps_real_zero_iou_and_area():
    target = torch.ones(1, 4, 4, 4)
    state = torch.zeros(1, 16, 4, 4)
    metrics = MorphologyEvaluator(target).evaluate(state)
    assert metrics["alpha_iou"] == 0
    assert metrics["area_ratio"] == 0
    assert metrics["target_mse"] == 1
    assert metrics["target_region_mse"] == 1


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_evaluator_rejects_nonfinite_hidden_state_without_reference(invalid):
    target = torch.ones(1, 4, 4, 4)
    state = torch.ones(1, 16, 4, 4)
    state[:, 4, 0, 0] = invalid
    assert all(np.isnan(value) for value in MorphologyEvaluator(target).evaluate(state).values())


@pytest.mark.parametrize("revision", [None, "old-unsafe-transfer"])
def test_corrupted_legacy_mps_measurements_are_not_silently_replotted(
    tmp_path, monkeypatch, revision
):
    monkeypatch.setattr(
        experiment,
        "read_json",
        lambda path: {
            "protocol": experiment.PROTOCOL,
            "device": "mps",
            "measurement_revision": revision,
        },
    )
    with pytest.raises(ValueError, match="old CPU/float64 transfer can corrupt"):
        experiment._load_population(tmp_path)


@pytest.mark.parametrize(
    "condition", build_conditions(InterventionConfig()), ids=lambda case: case.name
)
def test_validity_checks_preserve_trajectory_metrics_and_random_stream(condition):
    class StochasticStep:
        def __call__(self, state, update_probability):
            return state + (torch.rand_like(state[:, :1]) <= update_probability) * 0.01

    model = StochasticStep()
    mature = torch.ones(1, 16, 4, 4)
    target = mature[:, :4].clone()
    reference = mature * 0.9 if condition.reference_step is not None else mature
    config = ExperimentConfig(
        evaluation=EvaluationConfig(horizon=5),
        analysis=AnalysisConfig(
            hidden_channel_record_every=1, visible_channel_record_every=1, snapshot_steps=(0, 5)
        ),
    )
    evaluator = MorphologyEvaluator(target, mature_state=mature, hidden_reference_state=reference)
    expected = []
    state, previous = mature.clone(), None
    torch.manual_seed(123)
    for step in range(6):
        expected.append(evaluator.evaluate(state, previous))
        if step < 5:
            previous = state
            state = apply_post_step_intervention(
                model(state, config.model.update_probability),
                reference,
                condition,
                step + 1,
                target_mask=evaluator.target_mask,
            )
    random_state = torch.get_rng_state()
    result = _run_branch(
        model,
        target,
        mature,
        reference,
        TargetRegime(10, 0, 0, 10, 0.95),
        condition,
        config,
        0,
        123,
        True,
    )
    for name in experiment.METRIC_NAMES:
        np.testing.assert_array_equal(result.metrics[name], [row[name] for row in expected])
    assert torch.equal(torch.get_rng_state(), random_state)
    np.testing.assert_array_equal(result.snapshots[5], state[0, :4])


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


@pytest.mark.parametrize("minimum,maximum", [(1e-300, 1e300), (1e-5, 1e75), (0.0, 0.0)])
def test_mse_log_axis_keeps_limits_and_ticks_finite(tmp_path, minimum, maximum):
    figure, axis = plt.subplots(figsize=(4, 3))
    try:
        values = np.array([minimum, maximum])
        axis.plot([0, 1], values)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _set_mse_log_scale(axis)
            figure.tight_layout()
            figure.savefig(tmp_path / "log_axis.png")
        lower, upper = axis.get_ylim()
        assert np.isfinite([lower, upper]).all()
        assert 0 < lower < upper
        if minimum > 0:
            assert lower <= minimum
            assert upper >= maximum
        for locations in (axis.yaxis.get_majorticklocs(), axis.yaxis.get_minorticklocs()):
            assert np.isfinite(locations).all()
            assert (locations > 0).all()
        assert axis.yaxis.get_majorticklocs().size <= 8
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), values)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("robust", [False, True])
def test_core_morphology_plot_handles_zeros_divergence_and_wide_log_range(
    tmp_path, monkeypatch, robust
):
    condition = ConditionSpec("baseline", "Control", "baseline")
    errors = np.array([[0.0, 1e-300, np.nan], [0.0, 1e-300, np.nan], [1e75, 1e75, np.nan]])
    metrics = {
        "baseline": {
            "target_mse": errors.copy(),
            "target_region_mse": errors.copy(),
            "alpha_iou": np.array([[1, 0.9, np.nan]] * 3),
            "area_ratio": np.array([[1, 1.1, np.nan]] * 3),
        }
    }
    original = errors.copy()
    figures, savefig = [], plt.Figure.savefig

    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture)
    plotter = plot_core_morphology_metrics_robust if robust else plot_core_morphology_metrics
    output = tmp_path / f"extreme_{robust}.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plotter(output, metrics, (condition,), TargetRegime(1e-3, 0.9, 0.9, 1.1, 0.95))
    assert output.is_file() and output.stat().st_size > 0
    np.testing.assert_array_equal(metrics["baseline"]["target_mse"], original)
    if robust:
        for axis in figures[-1].axes[:2]:
            line = axis.lines[0]
            assert line.get_marker() == "."
            assert line.get_markevery() == [1]
            assert np.ma.getmaskarray(line.get_ydata()).tolist() == [True, False]


def test_finite_mean_std_count_uses_sample_sd_and_tracks_missing_replicates():
    values = np.asarray(
        [
            [1.0, 2.0, np.nan],
            [3.0, 4.0, np.nan],
            [5.0, np.nan, 7.0],
        ]
    )

    mean, standard_deviation, count = _finite_mean_std_count(values)

    assert np.allclose(mean[:2], [3.0, 3.0])
    assert np.isnan(mean[2])
    assert np.allclose(standard_deviation[:2], [2.0, np.sqrt(2.0)])
    assert np.isnan(standard_deviation[2])
    assert np.array_equal(count, [3, 2, 0])


def test_finite_median_quantiles_count_tracks_distribution_and_missing_replicates():
    values = np.asarray(
        [
            [1.0, 2.0, np.nan],
            [3.0, 4.0, np.nan],
            [5.0, np.nan, 7.0],
        ]
    )

    median, q25, q75, q05, q95, count = _finite_median_quantiles_count(values)

    for actual, expected in (
        (median, [3.0, 3.0]),
        (q25, [2.0, 2.5]),
        (q75, [4.0, 3.5]),
        (q05, [1.2, 2.1]),
        (q95, [4.8, 3.9]),
    ):
        assert np.allclose(actual[:2], expected)
        assert np.isnan(actual[2])
    assert np.array_equal(count, [3, 2, 0])


@pytest.mark.parametrize("failed_metric", experiment.METRIC_NAMES)
def test_invalidity_is_cumulative_and_shared_across_all_metrics(failed_metric):
    metrics = {name: np.ones((3, 5)) for name in experiment.METRIC_NAMES}
    metrics["target_mse"] = np.array(
        [[0, 1, 1000, 1000, 1000], [0, 2, 2, np.nan, 1000], [0, 4, 4, 4, 4]], dtype=float
    )
    metrics[failed_metric][0, 1] = np.inf
    originals = {name: values.copy() for name, values in metrics.items()}
    masked, valid = experiment._mask_invalid_replicates(metrics)
    assert valid.sum(axis=0).tolist() == [3, 2, 2, 1, 1]
    for name, values in metrics.items():
        assert np.isnan(masked[name][0, 1:]).all()
        assert np.isnan(masked[name][1, 3:]).all()
        np.testing.assert_array_equal(values, originals[name])
    median, q25, q75, q05, q95, counts = _finite_median_quantiles_count(masked["target_mse"])
    np.testing.assert_allclose(median, [0, 3, 3, 4, 4])
    np.testing.assert_allclose(q25, [0, 2.5, 2.5, 4, 4])
    np.testing.assert_allclose(q75, [0, 3.5, 3.5, 4, 4])
    np.testing.assert_allclose(q05, [0, 2.1, 2.1, 4, 4])
    np.testing.assert_allclose(q95, [0, 3.9, 3.9, 4, 4])
    np.testing.assert_array_equal(counts, [3, 2, 2, 1, 1])


def test_saved_state_metric_and_channel_failures_mask_only_their_own_condition():
    payload = {
        "condition_names": np.array([f"case_{index}" for index in range(5)]),
        "metrics": np.ones((5, 6, len(experiment.METRIC_NAMES))),
        "mature_state": np.ones((16, 2, 3)),
        "hidden_steps": np.array([0, 3, 5]),
        "visible_steps": np.array([0, 3, 5]),
        "hidden_channels": np.ones((5, 3, 12, 3)),
        "visible_channels": np.ones((5, 3, 4, 3)),
        "snapshot_steps": np.array([0, 1, 4, 5]),
        "snapshots": np.ones((5, 4, 4, 2, 3)),
        "metadata": {
            "summaries": [
                {"condition": "case_1", "numerical_divergence_step": 2},
                {"condition": "case_4", "first_failure_step": 1, "numerical_divergence_step": ""},
            ]
        },
    }
    payload["metrics"][0, 1, 0] = np.nan
    payload["hidden_channels"][2, 1, 0, 0] = np.inf
    payload["snapshots"][3, 2, 0, 0, 0] = np.nan
    original = payload["metrics"].copy()
    experiment._mask_invalid_payload(payload)
    for index, stop in enumerate((1, 2, 3, 4, 6)):
        assert np.isfinite(payload["metrics"][index, :stop]).all()
        assert np.isnan(payload["metrics"][index, stop:]).all()
        for field, steps in (
            ("hidden_channels", "hidden_steps"),
            ("visible_channels", "visible_steps"),
            ("snapshots", "snapshot_steps"),
        ):
            assert np.isnan(payload[field][index, payload[steps] >= stop]).all()
    assert np.isfinite(original[1:]).all()
    assert np.isfinite(
        payload["metrics"][4]
    ).all()  # Morphological failure is not numerical failure.


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_hidden_state_is_rejected_even_if_reported_metrics_are_finite(
    monkeypatch, invalid
):
    class BrokenHidden:
        def __call__(self, state, update_probability):
            result = state.clone()
            result[:, 4, 0, 0] = invalid
            return result

    monkeypatch.setattr(
        MorphologyEvaluator, "evaluate", lambda *args: dict.fromkeys(experiment.METRIC_NAMES, 1.0)
    )
    target = torch.ones(1, 4, 4, 4)
    mature = torch.ones(1, 16, 4, 4)
    result = _run_branch(
        BrokenHidden(),
        target,
        mature,
        mature,
        TargetRegime(2, 0, 0, 2, 0.95),
        ConditionSpec("baseline", "Control", "baseline"),
        ExperimentConfig(
            evaluation=EvaluationConfig(horizon=4),
            analysis=AnalysisConfig(
                hidden_channel_record_every=2,
                visible_channel_record_every=2,
                snapshot_steps=(0, 1, 4),
            ),
        ),
        0,
        1,
        True,
    )
    assert result.summary["numerical_divergence_step"] == 1
    assert all(np.isnan(values[1:]).all() for values in result.metrics.values())
    assert result.snapshots[1] is None and result.snapshots[4] is None
    assert np.isnan(result.channel_metrics["step_rms"][1:]).all()


@pytest.mark.parametrize("scope", ["hidden", "visible"])
def test_nonfinite_channel_metric_retires_all_measurements(monkeypatch, scope):
    original = getattr(experiment, f"{scope}_channel_metrics")

    def broken(state, mature, previous):
        values = original(state, mature, previous)
        if previous is not None:
            values["step_rms"][0] = float("inf")
        return values

    monkeypatch.setattr(experiment, f"{scope}_channel_metrics", broken)
    target = torch.ones(1, 4, 4, 4)
    mature = torch.ones(1, 16, 4, 4)
    result = _run_branch(
        lambda state, **kwargs: state.clone(),
        target,
        mature,
        mature,
        TargetRegime(1, 0, 0, 2, 0.95),
        ConditionSpec("baseline", "Control", "baseline"),
        ExperimentConfig(
            evaluation=EvaluationConfig(horizon=3),
            analysis=AnalysisConfig(
                hidden_channel_record_every=1, visible_channel_record_every=1, snapshot_steps=(0, 3)
            ),
        ),
        0,
        1,
        True,
    )
    assert result.summary["numerical_divergence_step"] == 1
    assert result.metrics["target_mse"][0] == 0
    assert all(np.isnan(values[1:]).all() for values in result.metrics.values())


@pytest.mark.parametrize("robust", [False, True])
def test_every_condition_stops_at_last_valid_step_and_exports_counts(tmp_path, monkeypatch, robust):
    conditions = build_conditions(InterventionConfig())
    metrics = {}
    for condition in conditions:
        values = {name: np.ones((3, 6)) for name in experiment.MORPHOLOGY_METRIC_NAMES}
        for name in ("target_mse", "target_region_mse"):
            values[name] = np.array(
                [
                    [0, 0.01, 0.02, 999, 999, 999],
                    [0, 0.03, 0.04, 999, 999, 999],
                    [0, 0.05, 0.06, 0.07, 999, 999],
                ]
            )
        values["alpha_iou"][:2, 3] = np.nan
        values["alpha_iou"][2, 4] = np.inf
        metrics[condition.name] = values
    figures, close = [], plt.close

    def capture(figure):
        if isinstance(figure, plt.Figure):
            figures.append(figure)
        close(figure)

    monkeypatch.setattr(plt, "close", capture)
    path = tmp_path / "core.png"
    plotter = plot_core_morphology_metrics_robust if robust else plot_core_morphology_metrics
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plotter(path, metrics, conditions, TargetRegime(0.1, 0.9, 0.9, 1.1, 0.95))
    for axis in figures[-1].axes:
        lines = [line for line in axis.lines if "(n=" in line.get_label()]
        assert len(lines) == len(conditions)
        for line in lines:
            np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2, 3])
            assert "n=3->0" in line.get_label()
        if "MSE" in axis.get_title():
            assert axis.get_yscale() == "log"
            for line in lines:
                values = line.get_ydata()
                assert np.ma.getmaskarray(values).tolist() == [True, False, False, False]
                np.testing.assert_allclose(np.ma.getdata(values), [0, 0.03, 0.04, 0.07])
    with path.with_name("core_n_valid.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition.name]
        assert [int(row["n_valid"]) for row in selected] == [3, 3, 3, 1, 0, 0]
        assert all(int(row["n_total"]) == 3 for row in selected)
    assert metrics[conditions[0].name]["target_mse"][0, -1] == 999


def test_all_invalid_population_produces_no_morphology_curve(tmp_path, monkeypatch):
    condition = ConditionSpec("baseline", "Control", "baseline")
    metrics = {"baseline": {name: np.ones((2, 3)) for name in experiment.MORPHOLOGY_METRIC_NAMES}}
    metrics["baseline"]["target_mse"][:, 0] = np.nan
    figures, close = [], plt.close

    def capture(figure):
        if isinstance(figure, plt.Figure):
            figures.append(figure)
        close(figure)

    monkeypatch.setattr(plt, "close", capture)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plot_core_morphology_metrics_robust(
            tmp_path / "empty.png", metrics, (condition,), TargetRegime(0.1, 0.9, 0.9, 1.1, 0.95)
        )
    for axis in figures[-1].axes:
        line = next(line for line in axis.lines if "(n=" in line.get_label())
        assert line.get_xdata().size == 0
        assert line.get_ydata().size == 0


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
            include_morphology_freeze=False,
            periodic_reset_intervals=(100,),
            development_reference_steps=(),
            development_restore_strengths=(),
            development_reset_intervals=(),
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


def test_developmental_conditions_cover_step_16_restore_and_reset():
    conditions = build_conditions(
        InterventionConfig(
            include_standard_conditions=False,
            include_morphology_freeze=False,
            periodic_reset_intervals=(),
            development_reference_steps=(16,),
            development_restore_strengths=(0.1,),
            development_reset_intervals=(150,),
        )
    )

    assert [condition.name for condition in conditions] == [
        "baseline",
        "restore_development_step_16_beta_0p1",
        "restore_development_step_16_period_150",
    ]
    assert [condition.reference_step for condition in conditions[1:]] == [16, 16]


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
        "restore_development_step_16_period_150",
        "Dev reset t=16 P=150",
        "development_periodic_reset",
        "interval",
        150,
        reference_step=16,
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
