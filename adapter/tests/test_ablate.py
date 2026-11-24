"""Head ablation's hooks (torch, no weights) and the committed ablation fixture."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import xta

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "gpt2-induction-ablation"


def hooks():
    """torch and the ablation module, or a skip where the model extra is not installed."""
    torch = pytest.importorskip("torch")
    from circuits_adapter import ablate

    return torch, ablate


def test_zero_sets_only_that_heads_slice():
    torch, ablate = hooks()
    z = torch.ones(1, 5, 4, 3)
    out = ablate.set_head(2)(z.clone(), None)
    assert out[0, :, 2, :].abs().sum() == 0
    assert out[0, :, [0, 1, 3], :].eq(1).all()


def test_mean_sets_that_heads_slice_to_the_value():
    torch, ablate = hooks()
    z = torch.zeros(1, 5, 4, 3)
    value = torch.arange(15.0).reshape(5, 3)
    out = ablate.set_head(1, value)(z.clone(), None)
    assert torch.equal(out[0, :, 1, :], value)
    assert out[0, :, [0, 2, 3], :].eq(0).all()


def test_the_fixture_has_a_run_per_finding_seed_with_both_matrices():
    seeds = sorted(int(p.name.removeprefix("seed-")) for p in FIXTURE.iterdir())
    assert seeds == list(range(10))
    for seed in seeds:
        d = FIXTURE / f"seed-{seed}"
        meta = tomllib.loads((d / "run.toml").read_text())
        assert meta["seed"] == [str(seed)]
        assert meta["intervention"] == ["zero-ablation", "mean-ablation"]
        assert float(meta["baseline"][0]) > 10
        for name in ("zero_effect", "mean_effect"):
            assert meta[f"{name}_shape"] == ["12", "12"]
            assert meta[f"{name}_axes"] == ["layer", "head"]
            assert np.isfinite(xta.read(d / f"{name}.xta")).all()
