"""The Effects library (xetal/Effects/src/Effects.xtl) and scripts/effects.sh.

Ablation and patching runs (one per seed) read into X_eTaL: each method's
layer x head effects in percent, their mean and spread over the seeds, the
ranking, a report and diverging grids. Checked against numpy on the fixtures.
"""

import subprocess
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import xta

ROOT = Path(__file__).resolve().parents[3]
XT = ROOT / "scripts" / "xt"
EFFECTS = ROOT / "scripts" / "effects.sh"
SRC = ROOT / "xetal" / "Effects" / "src" / "Effects.xtl"
ABLATION = ROOT / "fixtures" / "gpt2-induction-ablation"
PATCH = ROOT / "fixtures" / "gpt2-induction-patch"
SVG = "{http://www.w3.org/2000/svg}"
METHODS = {
    "zero": (ABLATION, "zero_effect"),
    "mean": (ABLATION, "mean_effect"),
    "denoise": (PATCH, "head_denoise"),
    "noise": (PATCH, "head_noise"),
}

pytestmark = pytest.mark.skipif(
    not (ROOT / "bin" / "xetal").exists(), reason="bin/xetal not built (just xetal)"
)


def runs(d: Path) -> str:
    seeds = sorted(int(p.name.removeprefix("seed-")) for p in d.iterdir())
    return '"' + " ".join(str(d / f"seed-{s}") for s in seeds) + '"'


def run(body: str) -> subprocess.CompletedProcess:
    src = '"ef:" u_se< "Effects"\n' + body
    return subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)


def out(body: str) -> str:
    r = run(body)
    assert r.returncode == 0, r.stderr
    return r.stdout


def numbers(body: str) -> np.ndarray:
    return np.array(out(body).split(), dtype=np.float64)


def numpy_percent(method: str) -> np.ndarray:
    """seeds x layer x head, in percent, as the library defines it."""
    d, name = METHODS[method]
    seeds = sorted(int(p.name.removeprefix("seed-")) for p in d.iterdir())
    out = []
    for s in seeds:
        a = xta.read(d / f"seed-{s}" / f"{name}.xta").astype(np.float64)
        if method in ("zero", "mean"):
            meta = tomllib.loads((d / f"seed-{s}" / "run.toml").read_text())
            a = 100.0 / float(meta["baseline"][0]) * a
        else:
            a = 100.0 * a
        out.append(a)
    return np.stack(out)


@pytest.mark.parametrize("method", list(METHODS))
def test_effects_mean_and_spread_equal_numpys(method):
    d = METHODS[method][0]
    want = numpy_percent(method)
    got = numbers(f'r_avel {runs(d)} ef:e_ffects "{method}"').reshape(want.shape)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-12)
    mean = numbers(f'r_avel ef:m_ean {runs(d)} ef:e_ffects "{method}"').reshape(12, 12)
    assert np.allclose(mean, want.mean(0), rtol=1e-9, atol=1e-12)
    spread = numbers(f'r_avel ef:s_pread {runs(d)} ef:e_ffects "{method}"').reshape(12, 12)
    assert np.allclose(spread, want.std(0), rtol=1e-6, atol=1e-9)


def ranked(method: str, k: int) -> list[tuple[int, int]]:
    d = METHODS[method][0]
    flat = [
        int(v)
        for v in out(f'{k} t_ake ef:r_anked ef:m_ean {runs(d)} ef:e_ffects "{method}"').split()
    ]
    return list(zip(flat[0::2], flat[1::2], strict=True))


@pytest.mark.parametrize("method", list(METHODS))
def test_ranking_follows_numpy(method):
    m = numpy_percent(method).mean(0).ravel()
    want = [divmod(int(k), 12) for k in np.argsort(-m, kind="stable")[:10]]
    assert ranked(method, 10) == want


def test_the_known_induction_heads_lead_denoising():
    assert set(ranked("denoise", 5)) == {(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)}


def test_a_method_the_runs_do_not_hold_is_refused():
    r = run(f'{runs(ABLATION)} ef:e_ffects "denoise"')
    assert r.returncode != 0 and "error[" in r.stderr
    r = run(f'{runs(ABLATION)} ef:e_ffects "guess"')
    assert r.returncode != 0 and "no method" in r.stderr


def test_a_relative_run_directory_is_refused():
    r = run('"fixtures/gpt2-induction-ablation/seed-0" ef:e_ffects "mean"')
    assert r.returncode != 0 and "absolute path" in r.stderr


def test_the_report_names_heads_with_mean_and_spread_and_says_candidates():
    text = out(f'{runs(PATCH)} ef:r_eport "denoise"')
    assert "not yet compared with random controls" in text
    lines = text.splitlines()
    head, pct, pm, spread = lines[2].split()
    assert head == "6.9" and pct.endswith("%") and pm == "+-"
    m = numpy_percent("denoise")
    assert int(pct[:-1]) == int(np.floor(0.5 + m.mean(0)[6, 9]))
    assert int(spread) == int(np.floor(0.5 + m.std(0)[6, 9]))


@pytest.mark.parametrize("method", ["mean", "denoise"])
def test_the_grid_is_diverging_with_layer_and_head_labels(method):
    d = METHODS[method][0]
    root = ET.fromstring(out(f'{runs(d)} ef:s_vg "{method}"'))
    assert [t.text for t in root.find(f".//{SVG}g[@class='rows']")] == [f"L{n}" for n in range(12)]
    cells = [int(t.text) for t in root.find(f".//{SVG}g[@class='cells']").iter(f"{SVG}text")]
    want = np.floor(0.5 + numpy_percent(method).mean(0)).astype(int).ravel()
    assert cells == list(want)
    fills = {r.get("fill") for r in root.find(f".//{SVG}g[@class='cells']").iter(f"{SVG}rect")}
    assert len(fills) > 3
    title, note = (t.text for t in root.findall(f"{SVG}text")[:2])
    assert "10 seeds" in title and "not yet compared with random controls" in note


def test_command_line_prints_reports_and_writes_grids(tmp_path):
    r = subprocess.run(
        [str(EFFECTS), "fixtures/gpt2-induction-patch", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "denoising" in r.stdout and "noising" in r.stdout
    for name in ("denoise", "noise"):
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
    return text.split("## Testing heads by intervention", 1)[1].split("\n## ", 1)[0]


def test_readme_examples_run_and_match(tmp_path):
    sec = readme_section()
    cmds = sec.split("```bash", 1)[1].split("```", 1)[0].strip().splitlines()
    shown = [b.split("```", 1)[0] for b in sec.split("```text\n")[1:]]
    assert len(cmds) == len(shown) == 2
    for cmd, text in zip(cmds, shown, strict=True):
        _, _, runs_dir, out_dir = cmd.split()
        r = subprocess.run(
            [str(EFFECTS), runs_dir, str(tmp_path)], cwd=ROOT, capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout == text, "the README's report is out of date"
        for svg in tmp_path.glob("*.svg"):
            assert svg.read_text() == (ROOT / out_dir / svg.name).read_text(), svg.name
