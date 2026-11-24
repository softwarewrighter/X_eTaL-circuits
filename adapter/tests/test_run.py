"""The run directory format: run.toml plus each array in both files, shapes recorded."""

import tomllib

import numpy as np
import pytest

from circuits_adapter import run, text_arrays, xta

INFO = {k: ["x"] for k in run.REQUIRED}


def test_a_run_writes_its_metadata_and_arrays(tmp_path):
    a = np.arange(6, dtype=np.float64).reshape(2, 3) / 7
    out = run.write(tmp_path / "r", INFO, {"d": (("layer", "head"), a)})
    meta = tomllib.loads((out / "run.toml").read_text())
    assert meta["format"] == [run.FORMAT]
    assert meta["arrays"] == ["d"]
    assert (meta["d_axes"], meta["d_shape"]) == (["layer", "head"], ["2", "3"])
    assert np.array_equal(xta.read(out / "d.xta"), a.astype(np.float32))
    assert (out / "d.txt").read_text() == text_arrays.dumps(a.astype(np.float32))


def test_an_existing_directory_is_refused(tmp_path):
    (tmp_path / "r").mkdir()
    with pytest.raises(FileExistsError):
        run.write(tmp_path / "r", INFO, {})


def test_missing_metadata_and_wrong_axes_are_refused(tmp_path):
    with pytest.raises(run.RunError, match="missing: seed"):
        run.write(tmp_path / "a", {k: v for k, v in INFO.items() if k != "seed"}, {})
    with pytest.raises(run.RunError, match="axis names"):
        run.write(tmp_path / "b", INFO, {"d": (("layer",), np.zeros((2, 2)))})
