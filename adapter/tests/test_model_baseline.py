"""The baseline on GPT-2 Small: induction gives a large margin on the clean prompt and
none on its twin (needs the model extra and the weights)."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import experiment

pytestmark = pytest.mark.model
INDUCTION = Path(__file__).resolve().parents[2] / "experiments" / "induction.toml"


@pytest.fixture(scope="module")
def gpt2():
    from circuits_adapter import model

    return model.load("gpt2")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_clean_margin_is_large_and_the_twin_has_none(gpt2, seed):
    from circuits_adapter import baseline

    b = baseline.compute(gpt2, experiment.load(INDUCTION), seed)
    assert b["clean_ld"].mean() > 10
    assert abs(b["corrupt_ld"].mean()) < 3
    assert (b["clean_ld"] > b["corrupt_ld"]).all()


def test_a_baseline_run_is_written_with_its_metadata_and_is_deterministic(gpt2, tmp_path):
    from circuits_adapter import baseline

    exp = experiment.load(INDUCTION)
    for name in ("a", "b"):
        baseline.write(tmp_path / name, gpt2, exp, 0)
    meta = tomllib.loads((tmp_path / "a" / "run.toml").read_text())
    assert meta["model"] == ["gpt2"] and meta["seed"] == ["0"]
    assert meta["clean_ld_shape"] == ["15"] and meta["clean_ld_axes"] == ["position"]
    assert len(meta["clean_ids"]) == len(meta["corrupt_ids"]) == 33
    for f in ("clean_ld.xta", "corrupt_ld.xta", "run.toml"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()
    assert np.isfinite(np.loadtxt(tmp_path / "a" / "clean_ld.txt")).all()
