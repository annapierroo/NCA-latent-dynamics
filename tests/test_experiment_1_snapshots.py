import numpy as np
import pytest
import torch
from PIL import Image

from scripts import experiment_1 as experiment


def test_snapshot_selection_uses_all_conditions_and_penalizes_missing_values():
    constant = {name: np.ones((3, 5)) for name in ("target_region_mse", "alpha_iou", "area_ratio")}
    varying = {
        "target_region_mse": np.repeat([[0.01], [0.02], [0.5]], 5, axis=1),
        "alpha_iou": np.repeat([[0.2], [0.7], [0.9]], 5, axis=1),
        "area_ratio": np.repeat([[0.4], [1.0], [5.0]], 5, axis=1),
    }
    for values in varying.values():
        values[:, -1] = np.nan
        values[0, 2:4] = np.nan
    scores = experiment._snapshot_replica_scores({"baseline": constant, "restore": varying})
    assert np.isfinite(scores).all()
    assert np.argmin(scores) == 1
    assert scores[0] > scores[1]


def test_snapshot_gallery_reads_t10000_and_marks_divergence(tmp_path, monkeypatch):
    rgba = torch.zeros(4, 8, 8)
    rgba[1, 2:6, 2:6] = 0.7
    rgba[3, 2:6, 2:6] = 1
    payload = {
        "snapshot_steps": np.array([0, 10000]),
        "snapshots": np.stack(
            [np.stack([rgba.numpy(), np.full((4, 8, 8), np.nan, dtype=np.float32)])]
        ),
    }
    frames = experiment._saved_snapshot_frames(payload, 0, (0, 10000))
    torch.testing.assert_close(frames[0], rgba)
    assert frames[10000] is None
    with pytest.raises(ValueError, match="not recorded"):
        experiment._saved_snapshot_frames(payload, 0, (500,))
    from matplotlib.axes import Axes

    labels = []
    original_text = Axes.text

    def record_text(axis, x, y, text, *args, **kwargs):
        labels.append(text)
        return original_text(axis, x, y, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", record_text)
    path = tmp_path / "final_gallery.png"
    experiment._plot_final_gallery(path, {0: rgba, 1: None, 2: None}, "Synthetic", 10000)
    assert "Numerically\ndiverged" in labels
    with Image.open(path) as image:
        assert image.width > image.height
    experiment.plot_snapshots(
        tmp_path / "representative.png",
        {"baseline": {0: rgba, 10000: None}},
        (experiment.ConditionSpec("baseline", "Control", "baseline"),),
        (0, 10000),
        include_all=True,
        title="Actual saved replica",
    )


@pytest.mark.parametrize(
    "command", ("tables", "refresh", "run-morphology-freeze", "plot-channels", "plot-core")
)
def test_obsolete_commands_are_unavailable(command):
    with pytest.raises(SystemExit) as caught:
        experiment.build_parser().parse_args([command])
    assert caught.value.code == 2
