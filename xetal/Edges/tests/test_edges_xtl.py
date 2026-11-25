"""The Edges library (xetal/Edges/src/Edges.xtl) names edges as the adapter does (D19)."""

import subprocess
from pathlib import Path

import pytest

from circuits_adapter import graph

ROOT = Path(__file__).resolve().parents[3]
XT = ROOT / "scripts" / "xt"
SRC = ROOT / "xetal" / "Edges" / "src" / "Edges.xtl"

pytestmark = pytest.mark.skipif(
    not (ROOT / "bin" / "xetal").exists(), reason="bin/xetal not built (just xetal)"
)
INPUT = {"q": 1, "k": 2, "v": 3}


def run(body: str) -> subprocess.CompletedProcess:
    src = '"ed:" u_se< "Edges"\n' + body
    return subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)


def numbers(text: str) -> list[int]:
    s, r = graph.edge(text, 12, 12)
    return [s.layer, s.head, r.layer, r.head, INPUT.get(r.input, 0)]


@pytest.mark.parametrize(
    "text", ["4.11 5.1.k", "embed 0.0.q", "embed logits", "11.11 logits", "0.3 11.10.v"]
)
def test_edges_parse_as_the_adapter_parses_them(text):
    r = run(f'(12, 12) ed:e_dge "{text}"')
    assert r.returncode == 0, r.stderr
    assert [int(v) for v in r.stdout.split()] == numbers(text)


@pytest.mark.parametrize(
    "text",
    ["5.1 5.2.q", "7.2 5.1.k", "4.11 5.1", "4.11 5.1.x", "4.12 5.1.k", "12.0 logits",
     "4 5.1.k", "a.b 5.1.k", "4.11", "4.11 5.1.kk"],
)  # fmt: skip
def test_what_the_adapter_refuses_is_refused(text):
    with pytest.raises(graph.GraphError):
        graph.edge(text, 12, 12)
    r = run(f'(12, 12) ed:e_dge "{text}"')
    assert r.returncode != 0 and "error[graph]" in r.stderr, (text, r.stdout)


@pytest.mark.parametrize("inputs", [("q", "k", "v"), ("k",), ("q", "v")])
def test_the_count_equals_the_adapters(inputs):
    r = run(f"ed:c_ount (12, 12, {len(inputs)})")
    assert int(r.stdout) == len(graph.all_edges(12, 12, inputs))


def test_doc_examples_pass():
    r = subprocess.run(
        [str(XT), "doc", "--test", str(SRC)], cwd=ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_export_types_are_pinned():
    r = subprocess.run([str(XT), "type", str(SRC)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout == (Path(__file__).parent / "types.out").read_text()


def test_lint_is_clean():
    r = subprocess.run([str(XT), "lint", str(SRC)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr == "", r.stderr
