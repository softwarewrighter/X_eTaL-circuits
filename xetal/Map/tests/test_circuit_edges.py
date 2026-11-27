"""experiments/circuits/gpt2-induction.toml's edges are what the edge runs give: every
edge between the circuit's heads that passes on the finding seeds, with its input,
level and basis by the rules of docs/design.md (D18, D22, D23), computed with numpy."""

import tomllib
from pathlib import Path

import numpy as np

from circuits_adapter import xta

ROOT = Path(__file__).resolve().parents[3]
F = ROOT / "fixtures"
CIRCUIT = ROOT / "experiments" / "circuits" / "gpt2-induction.toml"
WORDS = {"q": "query", "k": "key", "v": "value"}


def verdict(d: Path):
    ds = sorted(d.glob("seed-*"), key=lambda p: int(p.name.removeprefix("seed-")))
    edges = tomllib.loads((ds[0] / "run.toml").read_text())["edges"]
    e = 100.0 * np.stack([xta.read(x / "edge_effect.xta") for x in ds]).astype(float)
    c = 100.0 * np.stack([xta.read(x / "edge_control.xta") for x in ds]).astype(float)
    g = np.stack([xta.read(x / "edge_gap.xta") for x in ds]).astype(float)
    m = e.mean(0)
    passes = (g > 0).all(0) & (np.abs(m) > np.abs(c.mean(0)).max(0)) & (np.abs(m) >= 5.0)
    return edges, m, passes


def signed(x: float) -> str:
    r = int(np.floor(0.5 + x))
    return f"+{r}" if r >= 0 else str(r)


def expected(heads: list[str]) -> list[tuple[str, str, str, str]]:
    edges, f, pf = verdict(F / "gpt2-induction-edges")
    held, g, ph = verdict(F / "gpt2-induction-heldout-edges")
    assert held == edges
    out = []
    for i, e in enumerate(edges):
        s, r = e.split()
        if r == "logits" or not pf[i]:
            continue
        head, kind = r.rsplit(".", 1)
        if s in heads and head in heads:
            level = "confirmed" if ph[i] else "tested"
            said = "confirmed" if ph[i] else "tested only"
            basis = f"path patching into the {WORDS[kind]}: {said} ({signed(f[i])}%, held-out {signed(g[i])}%)"  # noqa: E501
            out.append((f"{s} {head}", kind, level, basis))
    return out


def test_the_circuits_edges_follow_from_the_edge_runs():
    c = tomllib.loads(CIRCUIT.read_text())
    got = list(zip(c["edges"], c["edge_inputs"], c["evidence"], c["edge_basis"], strict=True))
    assert got == expected(c["heads"])


def test_the_edges_the_literature_drew_are_confirmed_through_the_key():
    c = tomllib.loads(CIRCUIT.read_text())
    have = set(zip(c["edges"], c["edge_inputs"], c["evidence"], strict=True))
    for head in ("5.1", "5.5", "6.9", "7.2", "7.10"):
        assert (f"4.11 {head}", "k", "confirmed") in have
