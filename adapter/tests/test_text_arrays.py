"""Text copies of a capture's arrays, read by X_eTaL's `n_umbers []N_GET` (docs/design.md, D10)."""

import tomllib

import numpy as np

from circuits_adapter import capture, text_arrays, xta
from tests.test_capture import sample


def test_each_f32_is_written_as_its_exact_f64_decimal():
    a = np.array([[0.1, -2.5e-7], [3.0, 0.0]], dtype=np.float32)
    text = text_arrays.dumps(a)
    assert text.splitlines()[0].split()[0] == "0.10000000149011612"
    back = np.array([float(x) for x in text.split()], dtype=np.float64)
    assert np.array_equal(back, a.astype(np.float64).ravel())


def test_one_line_per_last_axis_row():
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    lines = text_arrays.dumps(a).splitlines()
    assert len(lines) == 6
    assert lines[1].split() == ["4.0", "5.0", "6.0", "7.0"]


def test_vector_is_one_line_and_scalar_one_number():
    assert text_arrays.dumps(np.array([1, 2], dtype=np.float32)) == "1.0 2.0\n"
    assert text_arrays.dumps(np.float32(2.5)) == "2.5\n"


def test_capture_writes_text_copies_and_names_them(tmp_path):
    out = tmp_path / "cap"
    c = sample()
    capture.write(out, c)
    meta = tomllib.loads((out / "capture.toml").read_text())
    assert meta["text_arrays"] == ["attn", "logits"]
    for name in meta["text_arrays"]:
        nums = np.array((out / f"{name}.txt").read_text().split(), dtype=np.float64)
        assert np.array_equal(nums, xta.read(out / f"{name}.xta").astype(np.float64).ravel())


def test_text_copies_are_byte_identical_across_runs(tmp_path):
    for name in ("a", "b"):
        capture.write(tmp_path / name, sample())
    for f in ("attn.txt", "logits.txt"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()
