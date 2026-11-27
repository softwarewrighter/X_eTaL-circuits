"""Edges checked on held-out seeds (xetal/EdgeEffects, scripts/edge-confirm.sh): an edge
is confirmed when it passes on the finding seeds and again on the held-out seeds
(docs/design.md, D18, D22). Checked against numpy on the committed fixtures."""

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
CONFIRM = ROOT / "scripts" / "edge-confirm.sh"
FOUND = ROOT / "fixtures" / "gpt2-induction-edges"
HELD = ROOT / "fixtures" / "gpt2-induction-heldout-edges"
SVG = "{http://www.w3.org/2000/svg}"

pytestmark = pytest.mark.skipif(
    not (ROOT / "bin" / "xetal").exists(), reason="bin/xetal not built (just xetal)"
)


def seed_dirs(d: Path) -> list[Path]:
    return sorted(d.glob("seed-*"), key=lambda p: int(p.name.removeprefix("seed-")))


def runs(d: Path) -> str:
    return '"' + " ".join(str(p) for p in seed_dirs(d)) + '"'


def run(body: str) -> subprocess.CompletedProcess:
    src = '"ee:" u_se< "EdgeEffects"\n' + body
    return subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)


def out(body: str) -> str:
    r = run(body)
    assert r.returncode == 0, r.stderr
    return r.stdout


def verdict(d: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Edges, mean percent and passes, by D22's rule."""
    ds = seed_dirs(d)
    edges = tomllib.loads((ds[0] / "run.toml").read_text())["edges"]
    e = 100.0 * np.stack([xta.read(x / "edge_effect.xta") for x in ds]).astype(float)
    c = 100.0 * np.stack([xta.read(x / "edge_control.xta") for x in ds]).astype(float)
    g = np.stack([xta.read(x / "edge_gap.xta") for x in ds]).astype(float)
    m = e.mean(0)
    passes = (g > 0).all(0) & (np.abs(m) > np.abs(c.mean(0)).max(0)) & (np.abs(m) >= 5.0)
    return edges, m, passes


def levels() -> tuple[list[str], np.ndarray, np.ndarray]:
    edges, _, f = verdict(FOUND)
    held, g, h = verdict(HELD)
    assert held == edges
    return edges, f.astype(int) + (f & h), g


def test_levels_are_numpys():
    _, want, _ = levels()
    got = out(f"ee:l_evels ({runs(FOUND)}, {runs(HELD)})").split()
    assert [int(v) for v in got] == list(want)


def test_the_held_out_seeds_are_not_the_finding_seeds():
    found = {tomllib.loads((d / "run.toml").read_text())["seed"][0] for d in seed_dirs(FOUND)}
    held = {tomllib.loads((d / "run.toml").read_text())["seed"][0] for d in seed_dirs(HELD)}
    assert len(found) == len(held) == 10 and not found & held
    r = run(f"ee:l_evels ({runs(FOUND)}, {runs(FOUND)})")
    assert r.returncode != 0 and "share a seed" in r.stderr


def test_held_out_runs_of_other_edges_are_refused(tmp_path):
    for d in seed_dirs(HELD):
        shutil.copytree(d, tmp_path / d.name)
        toml = tmp_path / d.name / "run.toml"
        toml.write_text(toml.read_text().replace('"4.11 5.1.k"', '"3.7 5.1.k"', 1))
    r = run(f"ee:l_evels ({runs(FOUND)}, {runs(tmp_path)})")
    assert r.returncode != 0 and "other edges" in r.stderr


def test_the_report_counts_confirmed_edges_and_names_the_tested_only():
    edges, lv, _ = levels()
    text = out(f"ee:c_onfirmations ({runs(FOUND)}, {runs(HELD)})")
    assert text.startswith(f"{(lv == 2).sum()} of the {(lv > 0).sum()} edges that pass")
    for i in np.flatnonzero(lv == 1):
        if edges[i].endswith("logits"):
            assert f"{edges[i]} finding" in text and "tested only" in text


@pytest.mark.parametrize("kind", ["q", "k"])
def test_a_confirmed_grid_holds_the_confirmed_edges_held_out_effects(kind):
    edges, lv, g = levels()
    root = ET.fromstring(out(f'({runs(FOUND)}, {runs(HELD)}) ee:s_vgConfirmed "{kind}"'))
    rows = [t.text for t in root.find(f".//{SVG}g[@class='rows']")]
    cols = [t.text for t in root.find(f".//{SVG}g[@class='cols']")]
    cells = [int(t.text) for t in root.find(f".//{SVG}g[@class='cells']").iter(f"{SVG}text")]
    cells = np.array(cells).reshape(len(rows), len(cols))
    want = {}
    for i, e in enumerate(edges):
        s, r = e.split()
        if lv[i] == 2 and r.endswith("." + kind):
            want[(s, r[:-2])] = max(-100, min(100, int(np.floor(0.5 + g[i]))))
    assert {s for s, _ in want} == set(rows) and {r for _, r in want} == set(cols)
    for a, s in enumerate(rows):
        for b, r in enumerate(cols):
            assert cells[a, b] == want.get((s, r), 0), (s, r)
    assert "confirmed on held-out seeds" in root.findall(f"{SVG}text")[0].text


def test_command_line_prints_and_writes_confirmed_grids(tmp_path):
    r = subprocess.run(
        [str(CONFIRM), "fixtures/gpt2-induction-edges", "fixtures/gpt2-induction-heldout-edges",
         str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "edges that pass on the 10 finding seeds" in r.stdout
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "confirmed-k.svg",
        "confirmed-logits.svg",
        "confirmed-q.svg",
    ]


def readme_section() -> str:
    text = (ROOT / "README.md").read_text()
    return text.split("## Edges confirmed on held-out seeds", 1)[1].split("\n## ", 1)[0]


def test_readme_example_runs_and_matches(tmp_path):
    sec = readme_section()
    cmd = sec.split("```bash", 1)[1].split("```", 1)[0].strip()
    shown = sec.split("```text\n", 1)[1].split("```", 1)[0]
    _, _, found, held, out_dir = cmd.split()
    r = subprocess.run(
        [str(CONFIRM), found, held, str(tmp_path)], cwd=ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == shown, "the README's report is out of date"
    for svg in tmp_path.glob("*.svg"):
        assert svg.read_text() == (ROOT / out_dir / svg.name).read_text(), svg.name
