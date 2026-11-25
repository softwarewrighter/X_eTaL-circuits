"""Path patching's hooks (torch, no weights) and the committed path-patching fixture."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import graph, xta

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "gpt2-induction-paths"


def module():
    pytest.importorskip("torch")
    from circuits_adapter import pathpatch

    return pathpatch


def test_the_adder_adds_and_the_keeper_keeps():
    import torch

    pathpatch = module()
    x = torch.ones(2, 3, 4)
    assert torch.equal(
        pathpatch._adder(torch.full((2, 3, 4), 2.0))(x, None), torch.full((2, 3, 4), 3.0)
    )
    store = {}
    out = pathpatch._keeper(store, 5)(x, None)
    assert out is x and torch.equal(store[5], x) and store[5] is not x


def test_layer_norm_matches_torch():
    import torch

    pathpatch = module()
    ln = torch.nn.LayerNorm(8, eps=1e-5)
    torch.nn.init.normal_(ln.weight)
    torch.nn.init.normal_(ln.bias)
    x = torch.randn(3, 5, 8)
    assert torch.allclose(pathpatch.layer_norm(x, ln), ln(x), atol=1e-6)


def test_each_receiver_has_its_twin():
    pathpatch = module()
    assert pathpatch.TWIN_OF == {"q": "second", "k": "first", "v": "second", "": "second"}


def test_the_fixture_is_the_sender_by_receiver_matrix():
    d = FIXTURE / "seed-0"
    meta = tomllib.loads((d / "run.toml").read_text())
    senders = [graph.sender(s, 12, 12) for s in meta["senders"]]
    receivers = [graph.receiver(r, 12, 12) for r in meta["receivers"]]
    assert (len(senders), len(receivers)) == (145, 433)
    assert meta["edge_effect_shape"] == ["145", "433"]
    e = xta.read(d / "edge_effect.xta")
    no_edge = np.array([[r.layer <= s.layer for r in receivers] for s in senders])
    assert (e[no_edge] == 0).all()
    assert np.isfinite(e).all()
    assert (no_edge == False).sum() == 29089  # noqa: E712


def test_4_11_leads_every_induction_heads_key_in_the_fixture():
    """Seed 0 (a result, checked across seeds and against controls later)."""
    d = FIXTURE / "seed-0"
    meta = tomllib.loads((d / "run.toml").read_text())
    e = xta.read(d / "edge_effect.xta")
    for head in ("5.1", "5.5", "6.9", "7.2", "7.10"):
        j = meta["receivers"].index(f"{head}.k")
        assert meta["senders"][int(np.argmax(e[:, j]))] == "4.11", head
