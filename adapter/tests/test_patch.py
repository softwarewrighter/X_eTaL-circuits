"""Patching's hook and ranking (torch, no weights) and the committed patching fixture."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import xta

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "gpt2-induction-patch"
SHAPES = {
    "resid_denoise": ["12", "33"],
    "head_denoise": ["12", "12"],
    "head_noise": ["12", "12"],
    "position_denoise": ["6", "33"],
}


def patch_module():
    pytest.importorskip("torch")
    from circuits_adapter import patch

    return patch


def test_put_copies_one_head_at_one_position_only():
    import torch

    patch = patch_module()
    source = torch.arange(1 * 5 * 4 * 3, dtype=torch.float32).reshape(1, 5, 4, 3)
    out = patch.put(source, head=2, position=3)(torch.zeros(1, 5, 4, 3), None)
    assert torch.equal(out[0, 3, 2], source[0, 3, 2])
    out[0, 3, 2] = 0
    assert out.eq(0).all()


def test_put_copies_a_position_or_everything():
    import torch

    patch = patch_module()
    source = torch.ones(1, 5, 8)
    assert patch.put(source, position=1)(torch.zeros(1, 5, 8), None).sum() == 8
    assert patch.put(source)(torch.zeros(1, 5, 8), None).sum() == 40


def test_leading_ranks_by_restoration():
    patch = patch_module()
    d = np.zeros((3, 4))
    d[2, 1], d[0, 3], d[1, 0] = 0.5, 0.4, 0.4
    assert patch.leading(d, 3) == [(2, 1), (0, 3), (1, 0)]


def test_the_fixture_has_a_run_per_finding_seed_with_every_result():
    seeds = sorted(int(p.name.removeprefix("seed-")) for p in FIXTURE.iterdir())
    assert seeds == list(range(10))
    for seed in seeds:
        d = FIXTURE / f"seed-{seed}"
        meta = tomllib.loads((d / "run.toml").read_text())
        assert meta["intervention"] == ["patch-denoise", "patch-noise"]
        assert float(meta["ld_clean"][0]) > float(meta["ld_corrupt"][0]) + 8
        assert len(meta["leading_heads"]) == 6
        for name, shape in SHAPES.items():
            assert meta[f"{name}_shape"] == shape
            assert np.isfinite(xta.read(d / f"{name}.xta")).all()
