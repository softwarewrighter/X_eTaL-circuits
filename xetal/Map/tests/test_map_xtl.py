"""The Map library (xetal/Map/src/Map.xtl) and scripts/map.sh: the model map.

One picture of the model: a row per layer (the first at the bottom), a cell per
head colored by its attention-pattern scores and an MLP cell, the residual
stream, and a circuit drawn over it with the evidence key.
"""

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import xta

ROOT = Path(__file__).resolve().parents[3]
XT = ROOT / "scripts" / "xt"
MAP = ROOT / "scripts" / "map.sh"
FULL = ROOT / "fixtures" / "gpt2-induction-full"
CIRCUIT = ROOT / "experiments" / "circuits" / "gpt2-induction.toml"
SRC = ROOT / "xetal" / "Map" / "src" / "Map.xtl"
SVG = "{http://www.w3.org/2000/svg}"
BLUE = np.array([59, 130, 246])
ORANGE = np.array([245, 158, 11])

pytestmark = pytest.mark.skipif(
    not (ROOT / "bin" / "xetal").exists(), reason="bin/xetal not built (just xetal)"
)


def run(body: str) -> subprocess.CompletedProcess:
    src = '"mp:" u_se< "Map"\n' + body
    return subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)


def draw(circuit: Path = CIRCUIT, capture: Path = FULL) -> ET.Element:
    r = run(f'"{capture}" mp:s_vg "{circuit}"')
    assert r.returncode == 0, r.stderr
    return ET.fromstring(r.stdout)


@pytest.fixture(scope="module")
def picture() -> ET.Element:
    return draw()


def scores() -> tuple[np.ndarray, np.ndarray]:
    a = xta.read(FULL / "attn.xta").astype(np.float64)
    n = a.shape[-1]
    t = (n - 1) // 2
    i = np.arange(1, n)
    q = np.arange(t + 1, n)
    return a[:, :, i, i - 1].mean(-1), a[:, :, q, q - t + 1].mean(-1)


def color(prev: float, ind: float) -> str:
    s = max(prev, ind)
    if s < 0.2:
        return "#f3f4f6"
    hue = BLUE if ind >= prev else ORANGE
    rgb = np.floor(0.5 + 245 + s * (hue - 245)).astype(int)
    return "#" + "".join(f"{v:02x}" for v in rgb)


def by_class(root: ET.Element, cls: str) -> list[ET.Element]:
    return [e for e in root.iter() if e.get("class") == cls]


def test_a_cell_per_head_and_an_mlp_per_layer(picture):
    assert len(by_class(picture, "head")) == 144
    assert len(by_class(picture, "mlp")) == 12


def test_head_colors_come_from_the_scores(picture):
    prev, ind = scores()
    for g in by_class(picture, "head"):
        layer, head = (int(v) for v in g.get("id").removeprefix("head-").split("."))
        rect = g.find(f"{SVG}rect")
        assert rect.get("fill") == color(prev[layer, head], ind[layer, head]), g.get("id")


def test_hovering_a_head_shows_its_numbers(picture):
    g = next(g for g in by_class(picture, "head") if g.get("id") == "head-5.5")
    title = g.find(f"{SVG}title").text
    assert title.startswith("head 5.5: induction 95%, previous token ")
    mlp = by_class(picture, "mlp")[0].find(f"{SVG}title").text
    assert mlp == "MLP 0: no score yet"


def test_the_first_layer_is_at_the_bottom(picture):
    def y(layer: int) -> float:
        g = next(g for g in by_class(picture, "head") if g.get("id") == f"head-{layer}.0")
        return float(g.find(f"{SVG}rect").get("y"))

    assert y(0) > y(1) > y(11)


def edge_point(root: ET.Element, name: str, side: str) -> tuple[float, float]:
    """The middle of a head's top or bottom edge."""
    g = next(g for g in by_class(root, "head") if g.get("id") == f"head-{name}")
    r = g.find(f"{SVG}rect")
    y = float(r.get("y")) + (float(r.get("height")) if side == "bottom" else 0)
    return float(r.get("x")) + float(r.get("width")) / 2, y


def test_each_edge_runs_up_from_head_to_head_and_is_dashed_when_observed(picture):
    edges = by_class(picture, "edge")
    assert [(e.get("data-from"), e.get("data-to")) for e in edges] == [
        ("4.11", t) for t in ("5.1", "5.5", "6.9", "7.2", "7.10")
    ]
    for e in edges:
        d = e.get("d").replace("M", "").replace("Q", "").split()
        start, end = tuple(map(float, d[0:2])), tuple(map(float, d[4:6]))
        assert start == edge_point(picture, e.get("data-from"), "top")
        assert end == edge_point(picture, e.get("data-to"), "bottom")
        assert e.get("data-evidence") == "observed"
        assert e.get("stroke-dasharray") == "6 4"


def edited(tmp_path: Path, old: str, new: str) -> Path:
    p = tmp_path / "circuit.toml"
    p.write_text(CIRCUIT.read_text().replace(old, new))
    return p


def test_evidence_levels_change_the_line(tmp_path):
    levels = '["observed", "tested", "confirmed", "observed", "observed"]'
    root = draw(
        edited(tmp_path, '["observed", "observed", "observed", "observed", "observed"]', levels)
    )
    edges = by_class(root, "edge")
    assert [e.get("stroke-dasharray") for e in edges[:3]] == ["6 4", None, None]
    assert [e.get("stroke-width") for e in edges[:3]] == ["2", "2", "4"]


@pytest.mark.parametrize(
    "old,new,says",
    [
        ('"observed", "observed"]', '"observed", "guessed"]', "evidence"),
        ('"4.11 7.10"', '"4.11 12.0"', "not captured"),
        ('"4.11 7.10"', '"4.11 7.12"', "no head"),
        ('"4.11 7.10"', '"7.10 4.11"', "later layer"),
        ('evidence = ["observed", ', "evidence = [", "one evidence level per edge"),
        (
            'head_evidence = [\n    "confirmed",',
            'head_evidence = [\n    "proven",',
            "head evidence",
        ),
        ('head_evidence = [\n    "confirmed", ', "head_evidence = [\n    ", "one head_evidence"),
        ('"10.1", "0.1"]', '"10.1", "12.1"]', "was not captured"),
    ],
)
def test_a_bad_circuit_is_refused(tmp_path, old, new, says):
    r = run(f'"{FULL}" mp:s_vg "{edited(tmp_path, old, new)}"')
    assert r.returncode != 0 and "error[map]" in r.stderr and says in r.stderr, r.stderr


def test_the_map_carries_both_keys_and_says_candidates(picture):
    text = " ".join(t.text or "" for t in picture.iter(f"{SVG}text"))
    for words in (
        "observed",
        "tested",
        "confirmed",
        "induction",
        "previous token",
        "candidates",
        "induction, GPT-2 Small",
        "embed",
        "unembed",
    ):
        assert words in text, words
    assert len(by_class(picture, "key-edge")) == 3


def test_circuit_heads_are_outlined_by_their_evidence(picture, tmp_path):
    outlines = by_class(picture, "head-evidence")
    assert [g.get("data-head") for g in outlines] == [
        "4.11", "5.1", "5.5", "6.9", "7.2", "7.10", "7.6", "8.5", "9.9", "9.6", "10.1", "0.1"
    ]  # fmt: skip
    assert {g.get("data-evidence") for g in outlines} == {"confirmed"}
    assert {g.find(f"{SVG}rect").get("stroke-width") for g in outlines} == {"3.5"}
    title = next(g for g in outlines if g.get("data-head") == "6.9").find(f"{SVG}title").text
    assert title.startswith("head 6.9: confirmed (mean ablation tested only")
    levels = '"tested", "observed", "confirmed",'
    root = draw(edited(tmp_path, '"confirmed", "confirmed", "confirmed",', levels))
    rects = [g.find(f"{SVG}rect") for g in by_class(root, "head-evidence")[:3]]
    assert [(r.get("stroke-width"), r.get("stroke-dasharray")) for r in rects] == [
        ("1.5", None), ("1.5", "4 3"), ("3.5", None)
    ]  # fmt: skip
    assert len(by_class(picture, "key-head")) == 3


def test_the_residual_stream_and_layer_labels(picture):
    assert len(by_class(picture, "residual")) == 1
    labels = [t.text for t in by_class(picture, "layer-label")]
    assert labels == [f"L{n}" for n in range(12)]


def test_command_line_writes_the_map_and_the_readme_image_is_current(tmp_path):
    out = tmp_path / "map.svg"
    r = subprocess.run(
        [
            str(MAP),
            "fixtures/gpt2-induction-full",
            "experiments/circuits/gpt2-induction.toml",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert out.read_text() == (ROOT / "images" / "gpt2-model-map.svg").read_text()
    readme = (ROOT / "README.md").read_text()
    assert (
        "just map fixtures/gpt2-induction-full experiments/circuits/gpt2-induction.toml "
        "images/gpt2-model-map.svg"
    ) in readme


def test_same_inputs_same_bytes():
    a = run(f'"{FULL}" mp:s_vg "{CIRCUIT}"').stdout
    assert a == run(f'"{FULL}" mp:s_vg "{CIRCUIT}"').stdout


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
