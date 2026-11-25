"""Matched random controls: directions of the real change's length (torch, no weights),
and the committed controls fixture."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import xta

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "gpt2-induction-controls"


def controls_module():
    pytest.importorskip("torch")
    from circuits_adapter import controls

    return controls


def test_directions_have_the_changes_length_at_each_position():
    import torch

    controls = controls_module()
    change = torch.arange(1.0, 1 + 5 * 64).reshape(5, 64) / 100
    r = controls.directions(change, 7, seed=0, layer=5, head=1)
    assert r.shape == (7, 5, 64)
    assert torch.allclose(r.norm(dim=-1), change.norm(dim=-1).expand(7, 5))


def test_directions_are_deterministic_and_differ_by_head_and_seed():
    import torch

    controls = controls_module()
    change = torch.ones(5, 64)
    a = controls.directions(change, 3, seed=0, layer=5, head=1)
    assert torch.equal(a, controls.directions(change, 3, seed=0, layer=5, head=1))
    assert not torch.equal(a, controls.directions(change, 3, seed=0, layer=5, head=2))
    assert not torch.equal(a, controls.directions(change, 3, seed=1, layer=5, head=1))


def test_add_to_head_adds_to_that_head_only():
    import torch

    controls = controls_module()
    r = torch.ones(2, 5, 3)
    out = controls.add_to_head(1, r)(torch.zeros(2, 5, 4, 3), None)
    assert out[:, :, 1].eq(1).all() and out[:, :, [0, 2, 3]].eq(0).all()


def test_the_fixture_has_controls_per_finding_seed():
    seeds = sorted(int(p.name.removeprefix("seed-")) for p in FIXTURE.iterdir())
    assert seeds == list(range(10))
    for seed in seeds:
        d = FIXTURE / f"seed-{seed}"
        meta = tomllib.loads((d / "run.toml").read_text())
        assert meta["controls"] == ["20"]
        for name in ("mean_control", "denoise_control"):
            assert meta[f"{name}_shape"] == ["20", "12", "12"]
            assert np.isfinite(xta.read(d / f"{name}.xta")).all()
