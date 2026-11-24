"""The Heatmap library (xetal/Heatmap/src/Heatmap.xtl) and scripts/heatmap.sh on the fixture.

Runs X_eTaL through scripts/xt; skipped when bin/xetal is not built (`just xetal`).
"""

import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import xta

ROOT = Path(__file__).resolve().parents[3]
XT = ROOT / "scripts" / "xt"
HEATMAP = ROOT / "scripts" / "heatmap.sh"
FIXTURE = ROOT / "fixtures" / "gpt2-induction"
SRC = ROOT / "xetal" / "Heatmap" / "src" / "Heatmap.xtl"
D = f'"{FIXTURE}"'
SVG = "{http://www.w3.org/2000/svg}"

pytestmark = pytest.mark.skipif(
    not (ROOT / "bin" / "xetal").exists(), reason="bin/xetal not built (just xetal)"
)


def run(body: str) -> subprocess.CompletedProcess:
    src = '"hm:" u_se< "Heatmap"\n' + body
    return subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)


def out(body: str) -> str:
    r = run(body)
    assert r.returncode == 0, r.stderr
    return r.stdout


def attn() -> np.ndarray:
    return xta.read(FIXTURE / "attn.xta").astype(np.float64)


def test_pattern_is_chosen_by_model_layer_and_head():
    # Model numbering, from 0: layer 5 is the capture's second layer; head 1 its second head.
    got = np.array(out(f"r_avel {D} hm:p_attern (5, 1)").split(), dtype=np.float64)
    assert np.array_equal(got, attn()[1, 1].ravel())


@pytest.mark.parametrize("pick", ["(5, 12)", "(5, -1)", "(3, 0)"])
def test_a_head_or_layer_not_captured_is_refused(pick):
    r = run(f"{D} hm:p_attern {pick}")
    assert r.returncode != 0 and "error[" in r.stderr


def test_percent_rounds_to_whole_numbers():
    assert out("hm:p_ercent 0.0 0.004 0.005 0.5 0.996 1.0").split() == [
        "0",
        "0",
        "1",
        "50",
        "100",
        "100",
    ]


def test_xml_escapes_token_text():
    assert out('hm:x_ml "<|endoftext|> & \\"x\\""').strip() == (
        "&lt;|endoftext|&gt; &amp; &quot;x&quot;"
    )
    assert out('t_ally hm:x_ml ""').strip() == "0"


def svg(layer: int, head: int) -> str:
    return out(f"{D} hm:s_vg ({layer}, {head})")


def test_svg_is_well_formed_with_one_cell_per_query_key():
    root = ET.fromstring(svg(5, 1))
    assert root.tag == f"{SVG}svg"
    cells = [r for r in root.iter(f"{SVG}rect") if r.get("width") == "24"]
    assert len(cells) == 17 * 17


def test_cells_carry_the_heads_percentages():
    root = ET.fromstring(svg(5, 1))
    grid = root.find(f".//{SVG}g[@class='cells']")
    numbers = [t.text for t in grid.iter(f"{SVG}text")]
    want = np.floor(0.5 + 100 * attn()[1, 1]).astype(int).ravel()
    assert [int(n) for n in numbers] == list(want)


def test_tokens_label_both_axes():
    root = ET.fromstring(svg(5, 1))
    tokens = [line for line in (FIXTURE / "capture.toml").read_text().splitlines()]
    toks = next(t for t in tokens if t.startswith("tokens = "))
    want = re.findall(r'"((?:[^"\\]|\\.)*)"', toks)
    for axis in ("rows", "cols"):
        g = root.find(f".//{SVG}g[@class='{axis}']")
        assert [t.text for t in g.iter(f"{SVG}text")] == want


def test_title_names_the_head_and_says_observation():
    root = ET.fromstring(svg(5, 1))
    title, note = (t.text for t in root.findall(f"{SVG}text")[:2])
    assert "gpt2" in title and "layer 5 head 1" in title
    assert "observation, not causal evidence" in note


def test_command_line_writes_the_svg(tmp_path):
    target = tmp_path / "h.svg"
    r = subprocess.run(
        [str(HEATMAP), "fixtures/gpt2-induction", "5", "1", str(target)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert target.read_text() == svg(5, 1).rstrip("\n") + "\n"


def test_same_inputs_same_bytes():
    assert svg(5, 5) == svg(5, 5)


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


def readme_heatmap_commands() -> list[list[str]]:
    text = (ROOT / "README.md").read_text()
    block = text.split("## Drawing an attention head", 1)[1].split("```bash", 1)[1]
    lines = block.split("```", 1)[0].strip().splitlines()
    return [line.split() for line in lines]


def test_readme_examples_run_and_the_readme_image_is_current(tmp_path):
    cmds = readme_heatmap_commands()
    assert [c[:2] for c in cmds] == [["just", "heatmap"], ["just", "heatmap"]]
    for args in cmds:
        given = args[2:]
        out_file = given[4] if len(given) > 4 else None  # `> file` or a path
        written = tmp_path / "out.svg"
        argv = [str(HEATMAP), *given[:3]]
        if given[3:4] == [">"]:
            r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            written.write_text(r.stdout)
        else:
            out_file = given[3]
            r = subprocess.run([*argv, str(written)], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            assert written.read_text() == (ROOT / out_file).read_text(), "rerun the README command"
        assert ET.fromstring(written.read_text()).tag == f"{SVG}svg"
        assert out_file
