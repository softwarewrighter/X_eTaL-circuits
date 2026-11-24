"""The Scores library (xetal/Scores/src/Scores.xtl) and scripts/scores.sh.

Previous-token and induction scores of every head, from attention patterns
only: candidates, not causal evidence. Checked against the same scores
computed with numpy from the binary capture, and against the GPT-2 Small
heads known from the literature (docs/reference.md).
"""

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import xta

ROOT = Path(__file__).resolve().parents[3]
XT = ROOT / "scripts" / "xt"
SCORES = ROOT / "scripts" / "scores.sh"
FULL = ROOT / "fixtures" / "gpt2-induction-full"
SRC = ROOT / "xetal" / "Scores" / "src" / "Scores.xtl"
D = f'"{FULL}"'
SVG = "{http://www.w3.org/2000/svg}"
INDUCTION = {(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)}
PREVIOUS = (4, 11)

pytestmark = pytest.mark.skipif(
    not (ROOT / "bin" / "xetal").exists(), reason="bin/xetal not built (just xetal)"
)


def run(body: str) -> subprocess.CompletedProcess:
    src = '"sc:" u_se< "Scores"\n' + body
    return subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)


def out(body: str) -> str:
    r = run(body)
    assert r.returncode == 0, r.stderr
    return r.stdout


def numbers(body: str) -> np.ndarray:
    return np.array(out(body).split(), dtype=np.float64)


def numpy_scores() -> tuple[np.ndarray, np.ndarray]:
    a = xta.read(FULL / "attn.xta").astype(np.float64)
    n = a.shape[-1]
    t = (n - 1) // 2
    i = np.arange(1, n)
    previous = a[:, :, i, i - 1].mean(-1)
    q = np.arange(t + 1, n)
    induction = a[:, :, q, q - t + 1].mean(-1)
    return previous, induction


def test_previous_token_scores_equal_numpys():
    got = numbers(f"r_avel sc:p_revious {D}").reshape(12, 12)
    assert np.allclose(got, numpy_scores()[0], rtol=1e-12, atol=1e-15)


def test_induction_scores_equal_numpys():
    got = numbers(f"r_avel sc:i_nduction {D}").reshape(12, 12)
    assert np.allclose(got, numpy_scores()[1], rtol=1e-12, atol=1e-15)


def ranked(score: str, k: int) -> list[tuple[int, int]]:
    flat = [int(v) for v in out(f"{k} t_ake {D} sc:r_anked sc:{score} {D}").split()]
    return list(zip(flat[0::2], flat[1::2], strict=True))


def test_the_known_induction_heads_lead():
    assert set(ranked("i_nduction", 5)) == INDUCTION


def test_the_known_previous_token_head_leads():
    assert ranked("p_revious", 1) == [PREVIOUS]


def test_ranking_follows_numpy():
    prev, ind = numpy_scores()
    for score, s in (("i_nduction", ind), ("p_revious", prev)):
        want = [divmod(int(k), 12) for k in np.argsort(-s.ravel(), kind="stable")[:10]]
        assert ranked(score, 10) == want


def test_a_prompt_that_does_not_repeat_is_refused(tmp_path):
    d = tmp_path / "cap"
    shutil.copytree(FULL, d)
    toml = (d / "capture.toml").read_text()
    lines = toml.splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("token_ids = "))
    ids = lines[i].split(", ")
    ids[-1] = '"1"]'
    lines[i] = ", ".join(ids)
    (d / "capture.toml").write_text("\n".join(lines) + "\n")
    r = run(f'sc:i_nduction "{d}"')
    assert r.returncode != 0 and "does not repeat" in r.stderr


def test_the_report_names_heads_and_says_candidates():
    text = out(f"sc:r_eport {D}")
    assert "candidates" in text and "not causal evidence" in text
    lines = text.splitlines()
    first_induction = lines[lines.index("induction") + 1].split()
    assert (int(first_induction[0].split(".")[0]), int(first_induction[0].split(".")[1])) in (
        INDUCTION
    )
    assert lines[lines.index("previous token") + 1].split()[0] == "4.11"


def test_the_grids_are_drawn_with_layer_and_head_labels():
    root = ET.fromstring(out(f'{D} sc:s_vg "induction"'))
    assert [t.text for t in root.find(f".//{SVG}g[@class='rows']")] == [f"L{n}" for n in range(12)]
    assert [t.text for t in root.find(f".//{SVG}g[@class='cols']")] == [f"H{n}" for n in range(12)]
    cells = [int(t.text) for t in root.find(f".//{SVG}g[@class='cells']").iter(f"{SVG}text")]
    want = np.floor(0.5 + 100 * numpy_scores()[1]).astype(int).ravel()
    assert cells == list(want)
    title, note = (t.text for t in root.findall(f"{SVG}text")[:2])
    assert "induction" in title and "candidates" in note


def test_command_line_prints_the_report_and_writes_both_grids(tmp_path):
    r = subprocess.run(
        [str(SCORES), "fixtures/gpt2-induction-full", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == out(f"sc:r_eport {D}").rstrip("\n") + "\n"
    for name in ("induction", "previous"):
        assert ET.parse(tmp_path / f"{name}.svg").getroot().tag == f"{SVG}svg"


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
    return text.split("## Scoring every head", 1)[1].split("\n## ", 1)[0]


def test_readme_examples_run_and_match(tmp_path):
    sec = readme_section()
    cmds = sec.split("```bash", 1)[1].split("```", 1)[0].strip().splitlines()
    shown = sec.split("```text\n", 1)[1].split("```", 1)[0]
    assert [c.split()[:3] for c in cmds] == [["just", "scores", "fixtures/gpt2-induction-full"]] * 2
    r = subprocess.run(
        [str(SCORES), "fixtures/gpt2-induction-full"], cwd=ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == shown, "the README's printed report is out of date"
    out_dir = cmds[1].split()[3]
    r = subprocess.run(
        [str(SCORES), "fixtures/gpt2-induction-full", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    for name in ("induction", "previous"):
        current = (tmp_path / f"{name}.svg").read_text()
        assert current == (ROOT / out_dir / f"{name}.svg").read_text(), "rerun the README command"
