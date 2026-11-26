"""Choosing the edges to path-patch from attribution runs (docs/design.md, D21)."""

import dataclasses
from pathlib import Path

import numpy as np

from circuits_adapter import choose, edges, graph

ROOT = Path(__file__).resolve().parents[2]
EDGES = ROOT / "experiments" / "induction-edges.toml"
RUNS = ROOT / "fixtures" / "gpt2-induction-attribution"


def names(es) -> list[str]:
    return [graph.name(s, r) for s, r in es]


def test_the_change_is_attribution_times_gap_averaged_over_seeds():
    attr = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])  # seed x sender x receiver
    gap = np.array([[10.0, 1.0], [20.0, 1.0]])
    assert np.allclose(choose.change(attr, gap), [[35.0, 3.0]])


def test_each_metric_gets_its_top_n_by_size_in_its_units():
    spec = dataclasses.replace(edges.load(EDGES), top=2, always=())
    senders, receivers, _, _ = choose.read(RUNS)
    delta = np.zeros((len(senders), len(receivers)))
    at = {(str(s), str(r)): (i, j) for i, s in enumerate(senders) for j, r in enumerate(receivers)}
    for edge, v in {
        "4.11 5.1.k": 6.0, "1.8 4.11.k": -5.0, "3.0 6.9.q": 1.0,
        "9.6 logits": 1.2, "10.7 logits": -1.5, "5.1 9.9.v": 0.1,
    }.items():  # fmt: skip
        delta[at[tuple(edge.split())]] = v
    got = names(choose.chosen(spec, senders, receivers, delta))
    assert got == ["1.8 4.11.k", "4.11 5.1.k", "9.6 logits", "10.7 logits"]


def test_always_edges_are_added_and_all_means_every_candidate():
    spec = edges.load(EDGES)
    senders, receivers, _, _ = choose.read(RUNS)
    zero = np.zeros((len(senders), len(receivers)))
    some = dataclasses.replace(spec, top=1)
    got = choose.chosen(some, senders, receivers, zero)
    assert set(spec.always) <= set(got) and len(got) == 2 + len(spec.always)
    every = dataclasses.replace(spec, top=None)
    assert len(choose.chosen(every, senders, receivers, zero)) == 29089


def test_the_induction_edges_chosen_from_the_finding_seeds():
    spec = edges.load(EDGES)
    senders, receivers, attr, gap = choose.read(RUNS)
    assert attr.shape == (10, 145, 433) and gap.shape == (10, 433)
    got = choose.chosen(spec, senders, receivers, choose.change(attr, gap))
    fams = [choose.family(r) for _, r in got]
    assert len(got) == 400 and fams.count("score") == fams.count("logit") == 200
    assert set(spec.always) <= set(got)
    assert got == sorted(got, key=graph.all_edges(12, 12).index)
