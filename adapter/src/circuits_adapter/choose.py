"""Which edges to path-patch, from edge attribution runs (docs/design.md, D21).

An edge's attribution is a fraction of its receiver's gap, as path patching's
effect is. Fractions of different receivers do not compare: a head whose
induction score barely differs between the twins has a gap near 0, and any
change is a large fraction of it. Edges are therefore ranked by the change
they make in their receiver's metric, in that metric's units (the attribution
times the receiver's gap, the mean over seeds), and separately for the two
metrics: attention scores (q and k receivers) and the logit difference (v and
logits). An edges experiment's ["top", N] takes the N largest of each metric,
then adds its `always` edges.
"""

import tomllib
from pathlib import Path

import numpy as np

from circuits_adapter import edges, graph, xta

FAMILIES = ("score", "logit")


def family(r: graph.Receiver) -> str:
    return "score" if r.input in ("q", "k") else "logit"


def read(runs_dir) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Senders, receivers, attribution (seed x sender x receiver), gap (seed x receiver)."""
    runs = sorted(Path(runs_dir).glob("seed-*"), key=lambda p: int(p.name.removeprefix("seed-")))
    if not runs:
        raise ValueError(f"{runs_dir}: no seed-N runs")
    meta = tomllib.loads((runs[0] / "run.toml").read_text())
    n_layers, n_heads = edges.SHAPE[meta["model"][0]]
    senders = [graph.sender(s, n_layers, n_heads) for s in meta["senders"]]
    receivers = [graph.receiver(r, n_layers, n_heads) for r in meta["receivers"]]
    attr = np.stack([xta.read(d / "edge_attribution.xta") for d in runs]).astype(np.float64)
    gap = np.stack([xta.read(d / "receiver_gap.xta") for d in runs]).astype(np.float64)
    return senders, receivers, attr, gap


def change(attr: np.ndarray, gap: np.ndarray) -> np.ndarray:
    """Each edge's change in its receiver's metric, the mean over seeds: sender x receiver."""
    return (attr * gap[:, None, :]).mean(0)


def chosen(spec, senders: list, receivers: list, delta: np.ndarray) -> list:
    """The edges of spec (edges.Edges) to path-patch, in graph.all_edges's order."""
    candidates = spec.candidates()
    if spec.top is None:
        keep = set(candidates)
    else:
        si = {s: i for i, s in enumerate(senders)}
        rj = {r: j for j, r in enumerate(receivers)}
        keep = set()
        for fam in FAMILIES:
            mine = [e for e in candidates if family(e[1]) == fam]
            size = np.array([abs(delta[si[s], rj[r]]) for s, r in mine])
            order = np.argsort(-size, kind="stable")[: spec.top]
            keep |= {mine[k] for k in order}
    keep |= set(spec.always)
    n_layers, n_heads = edges.SHAPE[spec.base.model]
    return [e for e in graph.all_edges(n_layers, n_heads) if e in keep]
