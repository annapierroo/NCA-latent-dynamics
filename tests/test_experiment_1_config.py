import errno
from pathlib import Path

import pytest

from scripts.experiment_1 import load_experiment_config


@pytest.mark.parametrize("code", (errno.ECANCELED, errno.ETIMEDOUT))
def test_unavailable_config_reports_exact_path_without_changing_settings(
    tmp_path, monkeypatch, code
):
    path = tmp_path / "gecko" / "resolved_config.json"
    failure = OSError(code, "Operation canceled")

    def canceled_read(self):
        assert self == path
        raise failure

    monkeypatch.setattr(Path, "read_text", canceled_read)
    with pytest.raises(OSError) as caught:
        load_experiment_config(path)

    assert caught.value is failure
    assert caught.value.errno == code
    assert str(path) in caught.value.__notes__[0]
    assert "Keep Downloaded" in caught.value.__notes__[1]
    assert "must not be replaced" in caught.value.__notes__[1]


def test_missing_config_remains_a_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment_config(tmp_path / "missing.json")
