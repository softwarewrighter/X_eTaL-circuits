"""The graph model: senders, receivers and edges are named, parsed and counted (D19)."""

from pathlib import Path

import pytest

from circuits_adapter import edges, graph

ROOT = Path(__file__).resolve().parents[2]
EDGES = ROOT / "experiments" / "induction-edges.toml"


@pytest.mark.parametrize("text", ["embed", "0.0", "4.11", "11.11"])
def test_senders_round_trip(text):
    assert str(graph.sender(text, 12, 12)) == text


@pytest.mark.parametrize("text", ["logits", "5.1.k", "0.0.q", "11.11.v"])
def test_receivers_round_trip(text):
    assert str(graph.receiver(text, 12, 12)) == text


def test_edges_round_trip():
    s, r = graph.edge("4.11 5.1.k", 12, 12)
    assert (s, r) == (graph.Sender(4, 11), graph.Receiver(5, 1, "k"))
    assert graph.name(s, r) == "4.11 5.1.k"
    assert graph.name(*graph.edge("embed logits", 12, 12)) == "embed logits"


@pytest.mark.parametrize(
    "text,says",
    [
        ("5.1 5.2.q", "not in a later layer"),
        ("7.2 5.1.k", "not in a later layer"),
        ("4.11 5.1", "not layer.head.q|k|v"),
        ("4.11 5.1.x", "not layer.head.q|k|v"),
        ("4.12 5.1.k", "outside the model"),
        ("12.0 logits", "outside the model"),
        ("4 5.1.k", "not layer.head or embed"),
        ("a.b 5.1.k", "not layer.head numbers"),
        ("4.11", 'not "sender receiver"'),
    ],
)
def test_bad_edges_are_refused(text, says):
    with pytest.raises(graph.GraphError, match=says):
        graph.edge(text, 12, 12)


def test_gpt2_small_has_this_many_edges():
    # embed: 12 layers x 12 heads x 3 inputs + logits; head in layer l: later
    # layers' 36 inputs each + logits.
    want = (432 + 1) + sum(12 * ((11 - la) * 36 + 1) for la in range(12))
    every = graph.all_edges(12, 12)
    assert len(every) == want == 29089
    assert len(set(every)) == len(every)
    assert every[0] == (graph.Sender(-1), graph.Receiver(0, 0, "q"))
    assert every[-1] == (graph.Sender(11, 11), graph.Receiver(12))
    assert len(graph.all_edges(12, 12, ("k",))) == (144 + 1) + sum(
        12 * ((11 - la) * 12 + 1) for la in range(12)
    )


def test_the_induction_edges_experiment_loads():
    e = edges.load(EDGES)
    assert e.base.name == "induction, GPT-2 Small"
    assert (e.senders, e.receivers, e.top) == (None, None, 200)
    assert len(e.candidates()) == 29089
    assert [graph.name(*a) for a in e.always][0] == "4.11 5.1.k"


def edited(tmp_path: Path, old: str, new: str) -> Path:
    text = EDGES.read_text()
    assert old in text
    (tmp_path / "induction.toml").write_text((ROOT / "experiments/induction.toml").read_text())
    p = tmp_path / "e.toml"
    p.write_text(text.replace(old, new))
    return p


def test_a_list_of_senders_and_receivers_narrows_the_candidates(tmp_path):
    e = edges.load(
        edited(tmp_path, 'senders = ["all"]\nreceivers = ["all"]',
               'senders = ["4.11", "embed"]\nreceivers = ["5.1.k", "logits"]')
    )  # fmt: skip
    assert [graph.name(*c) for c in e.candidates()] == [
        "embed 5.1.k", "embed logits", "4.11 5.1.k", "4.11 logits"
    ]  # fmt: skip


@pytest.mark.parametrize(
    "old,new,says",
    [
        ('path_patch = ["top", "200"]', 'path_patch = ["top", "x"]', "path_patch"),
        ('inputs = ["q", "k", "v"]', 'inputs = ["q", "o"]', "inputs"),
        ('always = ["4.11 5.1.k"', 'always = ["5.1 4.11.k"', "later layer"),
        ('base = ["induction.toml"]\n', "", "missing base"),
    ],
)
def test_bad_edge_experiments_are_refused(tmp_path, old, new, says):
    with pytest.raises(edges.EdgesError, match=says):
        edges.load(edited(tmp_path, old, new))
