"""Edge path patching with controls (torch, no weights) and its committed fixture (D22)."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import graph, xta

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "gpt2-induction-edges"
PATHS = ROOT / "fixtures" / "gpt2-induction-paths" / "seed-0"


def module():
    pytest.importorskip("torch")
    from circuits_adapter import edgepatch

    return edgepatch


def test_control_directions_keep_each_positions_length_and_are_seeded():
    import torch

    edgepatch = module()
    d = torch.randn(7, 16)
    r = edgepatch.directions(d, 5, 3, "4.11 5.1.k")
    assert r.shape == (5, 7, 16)
    assert torch.allclose(r.norm(dim=-1), d.norm(dim=-1).expand(5, 7), atol=1e-5)
    assert torch.equal(r, edgepatch.directions(d, 5, 3, "4.11 5.1.k"))
    assert not torch.equal(r, edgepatch.directions(d, 5, 3, "4.11 5.5.k"))
    assert not torch.equal(r, edgepatch.directions(d, 5, 4, "4.11 5.1.k"))


def test_edges_are_grouped_by_receiver_in_order():
    edgepatch = module()
    es = [graph.edge(t, 12, 12) for t in ("4.11 5.1.k", "3.7 5.1.k", "4.11 6.9.k")]
    got = edgepatch.by_receiver(es)
    assert list(got) == [es[0][1], es[2][1]]
    assert got[es[0][1]] == [es[0][0], es[1][0]]


def seed_dirs() -> list[Path]:
    return sorted(FIXTURE.glob("seed-*"), key=lambda p: int(p.name.removeprefix("seed-")))


def test_the_fixture_holds_the_chosen_edges_with_controls_for_every_finding_seed():
    from circuits_adapter import choose, edges

    spec = edges.load(ROOT / "experiments" / "induction-edges.toml")
    s, r, a, g = choose.read(ROOT / "fixtures" / "gpt2-induction-attribution")
    want = [graph.name(*e) for e in choose.chosen(spec, s, r, choose.change(a, g))]
    assert [d.name for d in seed_dirs()] == [f"seed-{n}" for n in range(10)]
    for d in seed_dirs():
        meta = tomllib.loads((d / "run.toml").read_text())
        assert meta["edges"] == want and meta["controls"] == ["20"]
        assert meta["shape"] == ["12", "12"]
        assert xta.read(d / "edge_effect.xta").shape == (400,)
        assert xta.read(d / "edge_control.xta").shape == (20, 400)
        assert xta.read(d / "edge_gap.xta").shape == (400,)


def test_seed_0_effects_are_the_full_path_patching_runs():
    meta = tomllib.loads((seed_dirs()[0] / "run.toml").read_text())
    pm = tomllib.loads((PATHS / "run.toml").read_text())
    pp = xta.read(PATHS / "edge_effect.xta")
    si = {s: i for i, s in enumerate(pm["senders"])}
    rj = {r: j for j, r in enumerate(pm["receivers"])}
    want = [pp[si[e.split()[0]], rj[e.split()[1]]] for e in meta["edges"]]
    assert np.array_equal(xta.read(seed_dirs()[0] / "edge_effect.xta"), np.array(want))
