"""Path patching of chosen edges, each with matched random controls (docs/design.md, D22).

The edges come from an edges experiment and its attribution runs (choose,
D21). An edge's effect is path patching's (D20). Its controls follow D17: the
sender's difference (position x d_model) is replaced by a vector of the same
length at each position in a uniformly random direction, and added to the same
receiver on the same twin, measured in the same units (a fraction of the
receiver's gap). K controls per edge and seed.
"""

import zlib

import numpy as np
import torch

from circuits_adapter import graph, pathpatch

K = 20
CHUNK = 256  # copies of the twin per forward pass


def directions(d: torch.Tensor, k: int, seed: int, edge: str) -> torch.Tensor:
    """k x position x d_model: random directions, each position's length as in d."""
    g = torch.Generator().manual_seed(zlib.crc32(f"{seed} {edge}".encode()))
    r = torch.randn(k, *d.shape, generator=g, dtype=d.dtype)
    return r / r.norm(dim=-1, keepdim=True) * d.norm(dim=-1)[None, :, None]


def by_receiver(edges: list) -> dict:
    """The edges' senders grouped by receiver, in the edges' order."""
    out: dict = {}
    for s, r in edges:
        out.setdefault(r, []).append(s)
    return out


class Run:
    """One seed's twins (pathpatch.Seed) and the chosen edges' effects and controls."""

    def __init__(self, m, length: int, seed: int, k: int = K):
        self.seed, self.k = seed, k
        self.s = pathpatch.Seed(m, length, seed)

    def _measure(self, r: graph.Receiver, d: torch.Tensor) -> np.ndarray:
        return np.concatenate(
            [self.s.effects_of(r, d[i : i + CHUNK]) for i in range(0, len(d), CHUNK)]
        )

    def receiver(self, r: graph.Receiver, senders: list) -> tuple[np.ndarray, np.ndarray]:
        """The effect of each sender on r, and k controls each (k x sender)."""
        p = self.s.paths[pathpatch.TWIN_OF[r.input]]
        real = [p.difference(s).detach() for s in senders]
        pairs = zip(senders, real, strict=True)
        ctrl = [directions(d, self.k, self.seed, graph.name(s, r)) for s, d in pairs]
        got = self._measure(r, torch.cat([torch.stack(real), *ctrl]))
        n = len(senders)
        return got[:n], got[n:].reshape(n, self.k).T

    def gap(self, r: graph.Receiver) -> float:
        """r's gap in its metric's units: attention score (q, k), logit difference (v, logits)."""
        if r.input in ("q", "k"):
            base = self.s.base[pathpatch.TWIN_OF[r.input]][r.layer][0, r.head]
            return float(self.s.clean[r.layer][0, r.head] - base)
        p = self.s.paths["second"]
        return float(p.ld_clean - p.ld_corrupt)

    def all(self, edges: list) -> tuple[np.ndarray, np.ndarray]:
        """Effects (edge) and controls (control x edge), in the edges' order."""
        effect, control = np.zeros(len(edges)), np.zeros((self.k, len(edges)))
        at = {e: i for i, e in enumerate(edges)}
        for r, senders in by_receiver(edges).items():
            e, c = self.receiver(r, senders)
            cols = [at[(s, r)] for s in senders]
            effect[cols], control[:, cols] = e, c
        return effect, control


def write(out, m, exp, seed: int, edges: list, shape: tuple, chosen_from: str):
    """A run directory with the chosen edges' path-patching effects and their controls."""
    from circuits_adapter import model, run

    x = Run(m, exp.length, seed)
    effect, control = x.all(edges)
    p = x.s.paths["second"]
    info = model.run_info(m, str(exp.path), seed, exp.length, p.clean, p.corrupt)
    info["intervention"] = ["path-patch-denoise", "path-patch-control"]
    info["key_twin_ids"] = [str(t) for t in x.s.paths["first"].run_ids]
    info["metric_q_k"] = ["fraction of the receiver's attention-score gap at the induction target"]
    info["metric_v_logits"] = ["fraction of the clean-twin logit-difference gap"]
    info["controls"] = [str(x.k)]
    info["chosen_from"] = [chosen_from]
    info["shape"] = [str(v) for v in shape]
    info["edges"] = [graph.name(s, r) for s, r in edges]
    info["gap_units"] = ["q, k: attention score; v, logits: logit difference"]
    arrays = {
        "edge_effect": (("edge",), effect),
        "edge_control": (("control", "edge"), control),
        "edge_gap": (("edge",), np.array([x.gap(r) for _, r in edges])),
    }
    return run.write(out, info, arrays)
