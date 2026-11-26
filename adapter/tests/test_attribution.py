"""Edge attribution (torch, no weights), its committed fixture, and how it agrees with
path patching on the committed seed-0 runs (docs/design.md, D21).

The agreement is measured, not assumed: these tests state what the fixtures show.
"""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import graph, metric, xta

ROOT = Path(__file__).resolve().parents[2]
ATTRIBUTION = ROOT / "fixtures" / "gpt2-induction-attribution"
PATHS = ROOT / "fixtures" / "gpt2-induction-paths" / "seed-0"
KEYS = ["5.1.k", "5.5.k", "6.9.k", "7.2.k", "7.10.k"]


def module():
    pytest.importorskip("torch")
    from circuits_adapter import attribution

    return attribution


def test_the_torch_logit_difference_is_the_metrics():
    import torch

    attribution = module()
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(9, 20)).astype(np.float32)
    clean = list(rng.integers(0, 20, 9))
    corrupt = list(rng.integers(0, 20, 9))
    want = metric.logit_difference(logits, clean, corrupt, 4).mean()
    got = attribution.ld_torch(torch.tensor(logits), clean, corrupt, 4)
    assert float(got) == pytest.approx(want, abs=1e-5)


def test_each_heads_input_gradient_is_its_own():
    import torch

    attribution = module()
    torch.manual_seed(0)
    ln = torch.nn.LayerNorm(8, eps=1e-5)
    torch.nn.init.normal_(ln.weight)
    x, w, b = torch.randn(5, 8), torch.randn(3, 8, 4), torch.randn(3, 4)
    g = torch.randn(3, 5, 4)
    got = attribution.head_input_grads(x, ln, w, b, g)
    for h in range(3):
        xh = x.clone().requires_grad_(True)
        y = attribution.pathpatch.layer_norm(xh, ln) @ w[h] + b[h]
        (want,) = torch.autograd.grad(y, xh, g[h])
        assert torch.allclose(got[h], want, atol=1e-5)


def seed_dirs() -> list[Path]:
    return sorted(ATTRIBUTION.glob("seed-*"), key=lambda p: int(p.name.removeprefix("seed-")))


def test_the_fixture_has_every_finding_seed_as_a_sender_by_receiver_matrix():
    assert [d.name for d in seed_dirs()] == [f"seed-{s}" for s in range(10)]
    for d in seed_dirs():
        meta = tomllib.loads((d / "run.toml").read_text())
        assert meta["intervention"] == ["edge-attribution"]
        senders = [graph.sender(s, 12, 12) for s in meta["senders"]]
        receivers = [graph.receiver(r, 12, 12) for r in meta["receivers"]]
        e = xta.read(d / "edge_attribution.xta")
        gap = xta.read(d / "receiver_gap.xta")
        assert e.shape == (145, 433) and gap.shape == (433,)
        no_edge = np.array([[r.layer <= s.layer for r in receivers] for s in senders])
        assert (e[no_edge] == 0).all() and np.isfinite(e).all()
        logit_like = [j for j, r in enumerate(receivers) if r.input in ("v", "")]
        assert np.ptp(gap[logit_like]) == 0  # one logit-difference gap per seed


def both() -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray]:
    d = seed_dirs()[0]
    meta = tomllib.loads((d / "run.toml").read_text())
    assert meta["receivers"] == tomllib.loads((PATHS / "run.toml").read_text())["receivers"]
    senders = [graph.sender(s, 12, 12) for s in meta["senders"]]
    receivers = [graph.receiver(r, 12, 12) for r in meta["receivers"]]
    gap = xta.read(d / "receiver_gap.xta").astype(np.float64)
    attr = xta.read(d / "edge_attribution.xta").astype(np.float64) * gap
    pp = xta.read(PATHS / "edge_effect.xta").astype(np.float64) * gap
    return senders, receivers, attr, pp, gap


def family_mask(senders, receivers, inputs) -> np.ndarray:
    return np.array([[r.layer > s.layer and r.input in inputs for r in receivers] for s in senders])


@pytest.mark.parametrize("inputs,r_min", [(("q", "k"), 0.98), (("v", ""), 0.995)])
def test_attribution_tracks_path_patching_in_the_metrics_units(inputs, r_min):
    """Seed 0: Pearson 0.982 over the 19,296 score edges, 0.999 over the 9,793 logit edges.
    The 200 largest path-patched edges all have attribution's sign and are all among
    attribution's 300 largest."""
    senders, receivers, attr, pp, _ = both()
    mk = family_mask(senders, receivers, inputs)
    x, y = attr[mk], pp[mk]
    assert np.corrcoef(x, y)[0, 1] > r_min
    top = np.argsort(-np.abs(y))[:200]
    assert (np.sign(x[top]) == np.sign(y[top])).all()
    rank = np.empty(len(x), int)
    rank[np.argsort(-np.abs(x))] = np.arange(len(x))
    assert rank[top].max() < 300


def test_the_4_11_key_edges_rank_alike_and_near_the_top_by_both():
    senders, receivers, attr, pp, _ = both()
    mk = family_mask(senders, receivers, ("q", "k"))
    idx = np.argwhere(mk)
    x, y = attr[mk], pp[mk]

    def ranks(v):
        r = np.empty(len(v), int)
        r[np.argsort(-np.abs(v))] = np.arange(len(v))
        return r

    i = senders.index(graph.Sender(4, 11))
    js = [receivers.index(graph.receiver(k, 12, 12)) for k in KEYS]
    ks = [np.flatnonzero((idx[:, 0] == i) & (idx[:, 1] == j))[0] for j in js]
    assert list(ranks(x)[ks]) == list(ranks(y)[ks]) == [0, 3, 2, 9, 5]


def test_fractions_do_not_rank_edges_across_receivers():
    """Why edges are ranked in the metric's units: the largest fractions belong to
    heads whose induction score barely differs between the twins."""
    senders, receivers, _, _, gap = both()
    pp = xta.read(PATHS / "edge_effect.xta")
    worst = np.unravel_index(np.argmax(np.abs(pp)), pp.shape)
    assert abs(pp[worst]) > 50 and abs(gap[worst[1]]) < 0.1
    assert abs(gap[receivers.index(graph.receiver("5.1.k", 12, 12))]) > 10
