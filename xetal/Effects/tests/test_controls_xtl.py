"""Effects' controls: which heads beat every matched random control (xetal/Effects),
checked against numpy on the fixtures, and scripts/verdict.sh."""

import subprocess
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import xta

ROOT = Path(__file__).resolve().parents[3]
XT = ROOT / "scripts" / "xt"
VERDICT = ROOT / "scripts" / "verdict.sh"
F = ROOT / "fixtures"
ABLATION, PATCH, CONTROLS = (F / f"gpt2-induction-{k}" for k in ("ablation", "patch", "controls"))
SVG = "{http://www.w3.org/2000/svg}"
REAL = {"mean": (ABLATION, "mean_effect"), "denoise": (PATCH, "head_denoise")}

pytestmark = pytest.mark.skipif(
    not (ROOT / "bin" / "xetal").exists() or not CONTROLS.exists(), reason="bin/xetal or fixture"
)


def runs(d: Path) -> str:
    seeds = sorted(int(p.name.removeprefix("seed-")) for p in d.iterdir())
    return '"' + " ".join(str(d / f"seed-{s}") for s in seeds) + '"'


def out(body: str) -> str:
    src = '"ef:" u_se< "Effects"\n' + body
    r = subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def percent(d: Path, name: str, method: str) -> np.ndarray:
    seeds = sorted(int(p.name.removeprefix("seed-")) for p in d.iterdir())
    out = []
    for s in seeds:
        a = xta.read(d / f"seed-{s}" / f"{name}.xta").astype(np.float64)
        if method == "mean":
            meta = tomllib.loads((d / f"seed-{s}" / "run.toml").read_text())
            a = 100.0 / float(meta["baseline"][0]) * a
        else:
            a = 100.0 * a
        out.append(a)
    return np.stack(out).mean(0)


def numpy_passes(method: str) -> np.ndarray:
    d, name = REAL[method]
    real = percent(d, name, method)
    top = percent(CONTROLS, f"{method}_control", method).max(0)
    return ((real > top) & (real >= 5.0)).astype(int)


@pytest.mark.parametrize("method", ["mean", "denoise"])
def test_controls_equal_numpys(method):
    got = np.array(out(f'r_avel {runs(CONTROLS)} ef:c_ontrols "{method}"').split(), float)
    want = percent(CONTROLS, f"{method}_control", method)
    assert np.allclose(got.reshape(want.shape), want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method", ["mean", "denoise"])
def test_passes_follow_the_rule(method):
    d = REAL[method][0]
    got = np.array(out(f'r_avel ({runs(d)}, {runs(CONTROLS)}) ef:p_asses "{method}"').split(), int)
    assert np.array_equal(got.reshape(12, 12), numpy_passes(method))


def test_runs_of_other_seeds_are_refused():
    one = '"' + str(CONTROLS / "seed-0") + '"'
    src = f'"ef:" u_se< "Effects"\nr_avel ({runs(PATCH)}, {one}) ef:p_asses "denoise"'
    r = subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "different seeds" in r.stderr


def test_verdicts_list_the_passing_heads_best_first():
    text = out(f'({runs(PATCH)}, {runs(CONTROLS)}) ef:v_erdicts "denoise"')
    passes = numpy_passes("denoise")
    assert f"{passes.sum()} of 144 heads have an effect of at least 5% and beat all 20" in text
    heads = [line.split()[0] for line in text.splitlines()[2:] if line.strip()]
    real = percent(PATCH, "head_denoise", "denoise") * passes
    order = [divmod(int(k), 12) for k in np.argsort(-real.ravel(), kind="stable")[: passes.sum()]]
    assert heads == [f"{a}.{b}" for a, b in order]


def test_the_passing_grid_shows_only_passing_heads():
    root = ET.fromstring(out(f'({runs(PATCH)}, {runs(CONTROLS)}) ef:s_vgPassing "denoise"'))
    cells = np.array(
        [int(t.text) for t in root.find(f".//{SVG}g[@class='cells']").iter(f"{SVG}text")]
    )
    real = np.floor(0.5 + percent(PATCH, "head_denoise", "denoise")).astype(int)
    assert np.array_equal(cells.reshape(12, 12), real * numpy_passes("denoise"))


def test_command_line_prints_both_verdicts_and_writes_grids(tmp_path):
    dirs = [str(d.relative_to(ROOT)) for d in (ABLATION, PATCH, CONTROLS)]
    r = subprocess.run(
        [str(VERDICT), *dirs, str(tmp_path)], cwd=ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert "mean ablation" in r.stdout and "denoising" in r.stdout
    for m in ("mean", "denoise"):
        assert ET.parse(tmp_path / f"passing-{m}.svg").getroot().tag == f"{SVG}svg"


def test_the_readme_verdicts_and_grids_are_current(tmp_path):
    text = (ROOT / "README.md").read_text()
    sec = text.split("## Which heads beat their controls", 1)[1].split("\n## ", 1)[0]
    cmd = sec.split("```bash", 1)[1].split("```", 1)[0].split()
    shown = sec.split("```text\n", 1)[1].split("```", 1)[0]
    assert cmd[:2] == ["just", "verdict"]
    r = subprocess.run(
        [str(VERDICT), *cmd[2:5], str(tmp_path)], cwd=ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == shown, "the README's verdicts are out of date"
    for svg in tmp_path.glob("*.svg"):
        assert svg.read_text() == (ROOT / cmd[5] / svg.name).read_text(), svg.name
