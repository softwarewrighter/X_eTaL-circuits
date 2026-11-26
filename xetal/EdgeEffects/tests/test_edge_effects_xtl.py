"""The EdgeEffects library (xetal/EdgeEffects/src/EdgeEffects.xtl) and scripts/edge-verdict.sh.

Edgepatch runs (one per seed) read into X_eTaL: each edge's effect in percent,
its controls, its change in its receiver's metric, the verdicts, a report and
grids. Checked against numpy on the committed fixture (docs/design.md, D22).
"""

import shutil
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import xta

ROOT = Path(__file__).resolve().parents[3]
XT = ROOT / "scripts" / "xt"
VERDICT = ROOT / "scripts" / "edge-verdict.sh"
SRC = ROOT / "xetal" / "EdgeEffects" / "src" / "EdgeEffects.xtl"
RUNS = ROOT / "fixtures" / "gpt2-induction-edges"
SVG = "{http://www.w3.org/2000/svg}"

pytestmark = pytest.mark.skipif(
    not (ROOT / "bin" / "xetal").exists(), reason="bin/xetal not built (just xetal)"
)


def seed_dirs(d: Path = RUNS) -> list[Path]:
    return sorted(d.glob("seed-*"), key=lambda p: int(p.name.removeprefix("seed-")))


def runs(d: Path = RUNS) -> str:
    return '"' + " ".join(str(p) for p in seed_dirs(d)) + '"'


def run(body: str) -> subprocess.CompletedProcess:
    src = '"ee:" u_se< "EdgeEffects"\n' + body
    return subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)


def out(body: str) -> str:
    r = run(body)
    assert r.returncode == 0, r.stderr
    return r.stdout


def numbers(body: str) -> np.ndarray:
    return np.array(out(body).split(), dtype=np.float64)


class Numpy:
    """The fixture, as the library defines it."""

    def __init__(self):
        ds = seed_dirs()
        self.edges = tomllib.loads((ds[0] / "run.toml").read_text())["edges"]
        self.e = 100.0 * np.stack([xta.read(d / "edge_effect.xta") for d in ds]).astype(float)
        c = np.stack([xta.read(d / "edge_control.xta") for d in ds]).astype(float)
        g = np.stack([xta.read(d / "edge_gap.xta") for d in ds]).astype(float)
        self.mean = self.e.mean(0)
        self.controls = 100.0 * c.mean(0)
        self.change = (self.e / 100.0 * g).mean(0)
        a, top = np.abs(self.mean), np.abs(self.controls).max(0)
        self.top = top
        self.judged = (g > 0).all(0)
        self.passes = self.judged & (a > top) & (a >= 5.0)
        self.score = np.array([e[-1] in "qk" for e in self.edges])


@pytest.fixture(scope="module")
def np_():
    return Numpy()


def test_effects_controls_and_changes_equal_numpys(np_):
    got = numbers(f"r_avel ee:e_ffects {runs()}").reshape(np_.e.shape)
    assert np.allclose(got, np_.e, rtol=1e-9, atol=1e-9)
    got = numbers(f"r_avel ee:c_ontrols {runs()}").reshape(np_.controls.shape)
    assert np.allclose(got, np_.controls, rtol=1e-6, atol=1e-9)
    assert np.allclose(numbers(f"ee:c_hanges {runs()}"), np_.change, rtol=1e-6, atol=1e-9)


def test_edges_into_a_gap_that_is_not_always_positive_are_not_judged(np_):
    got = numbers(f"ee:j_udged {runs()}").astype(int)
    assert list(got) == list(np_.judged.astype(int))
    unjudged = {np_.edges[i].split()[1] for i in np.flatnonzero(~np_.judged)}
    assert "4.11.k" in unjudged and "5.1.k" not in unjudged
    assert f"; {int((~np_.judged).sum())} are not judged" in out(f"ee:v_erdicts {runs()}")


def test_the_verdicts_are_numpys_two_sided_rule(np_):
    assert list(numbers(f"ee:p_asses {runs()}").astype(int)) == list(np_.passes.astype(int))
    assert list(numbers(f"ee:f_amilies {runs()}").astype(int)) == list(np_.score.astype(int))


def test_the_report_counts_and_ranks_by_change_in_each_metric(np_):
    text = out(f"ee:v_erdicts {runs()}")
    n = int(np_.passes.sum())
    assert f"over 10 seeds: {n} have an effect of at least 5%" in text
    assert "beat all 20 of their random controls" in text
    qk, vl = text.split("\n\ninto ")[1:]
    for part, fam in ((qk, np_.score), (vl, ~np_.score)):
        sel = np_.passes & fam
        lines = [ln for ln in part.splitlines()[1:] if ln]
        order = [i for i in np.argsort(-np.abs(np_.change), kind="stable") if sel[i]]
        assert [ln.split(" ")[0:2] for ln in lines] == [
            np_.edges[i].split() for i in order[: len(lines)]
        ]
        assert len(lines) == min(15, int(sel.sum()))
    first = qk.splitlines()[1]
    assert first.startswith("4.11 5.1.k +36% (controls up to 1%), attention score +5.34")


def grid(svg: str) -> tuple[list[str], list[str], np.ndarray]:
    root = ET.fromstring(svg)
    rows = [t.text for t in root.find(f".//{SVG}g[@class='rows']")]
    cols = [t.text for t in root.find(f".//{SVG}g[@class='cols']")]
    cells = [int(t.text) for t in root.find(f".//{SVG}g[@class='cells']").iter(f"{SVG}text")]
    return rows, cols, np.array(cells).reshape(len(rows), len(cols))


@pytest.mark.parametrize("kind", ["q", "k"])
def test_a_kinds_grid_holds_its_passing_edges_capped_at_100(np_, kind):
    rows, cols, cells = grid(out(f'{runs()} ee:s_vgPassing "{kind}"'))
    want = {}
    for i, e in enumerate(np_.edges):
        s, r = e.split()
        if np_.passes[i] and r.endswith("." + kind):
            v = int(np.floor(0.5 + np_.mean[i]))
            want[(s, r[:-2])] = max(-100, min(100, v))
    assert {s for s, _ in want} == set(rows) and {r for _, r in want} == set(cols)
    key = [tuple(int(x) for x in c.split(".")) for c in cols]
    assert key == sorted(key)
    for a, s in enumerate(rows):
        for b, r in enumerate(cols):
            assert cells[a, b] == want.get((s, r), 0), (s, r)


def test_the_logits_grid_is_a_layer_by_head_grid(np_):
    rows, cols, cells = grid(out(f'{runs()} ee:s_vgPassing "logits"'))
    assert rows == [f"L{n}" for n in range(12)] and cols == [f"H{n}" for n in range(12)]
    want = np.zeros((12, 12), int)
    for i, e in enumerate(np_.edges):
        s, r = e.split()
        if np_.passes[i] and r == "logits" and s != "embed":
            layer, head = map(int, s.split("."))
            want[layer, head] = int(np.floor(0.5 + np_.mean[i]))
    assert np.array_equal(cells, want)
    assert want[10, 7] < 0 and want[7, 2] > 0


def test_counts_into_each_kind(np_):
    for kind, ends in (("q", ".q"), ("k", ".k"), ("v", ".v"), ("logits", "logits")):
        n = sum(1 for i, e in enumerate(np_.edges) if np_.passes[i] and e.endswith(ends))
        assert int(out(f'{runs()} ee:c_ountInto "{kind}"')) == n


def test_a_kind_without_passing_edges_is_refused():
    r = run(f'{runs()} ee:s_vgPassing "v"')
    assert r.returncode != 0 and "no edge into values passes" in r.stderr


def test_runs_of_different_edges_are_refused(tmp_path):
    for d in seed_dirs()[:2]:
        shutil.copytree(d, tmp_path / d.name)
    toml = tmp_path / "seed-1" / "run.toml"
    toml.write_text(toml.read_text().replace('"4.11 5.1.k"', '"3.7 5.1.k"', 1))
    r = run(f"ee:n_ames {runs(tmp_path)}")
    assert r.returncode != 0 and "different edges" in r.stderr


def test_command_line_prints_verdicts_and_writes_grids(tmp_path):
    r = subprocess.run(
        [str(VERDICT), "fixtures/gpt2-induction-edges", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("path patching of 400 edges")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "edges-k.svg",
        "edges-logits.svg",
        "edges-q.svg",
    ]


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


def readme_section() -> str:
    text = (ROOT / "README.md").read_text()
    return text.split("## Which edges beat their controls", 1)[1].split("\n## ", 1)[0]


def test_readme_example_runs_and_matches(tmp_path):
    sec = readme_section()
    cmd = sec.split("```bash", 1)[1].split("```", 1)[0].strip()
    shown = sec.split("```text\n", 1)[1].split("```", 1)[0]
    _, _, runs_dir, out_dir = cmd.split()
    r = subprocess.run(
        [str(VERDICT), runs_dir, str(tmp_path)], cwd=ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == shown, "the README's report is out of date"
    for svg in tmp_path.glob("*.svg"):
        assert svg.read_text() == (ROOT / out_dir / svg.name).read_text(), svg.name
    assert sorted(p.name for p in tmp_path.glob("*.svg")) == sorted(
        p.name for p in (ROOT / out_dir).glob("*.svg")
    )
