"""Held-out validation (xetal/Effects l_evels, c_onfirmations; scripts/confirm.sh):
heads that pass on the finding seeds are confirmed when they pass again on the
held-out seeds. Checked against numpy on the fixtures."""

import subprocess
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import experiment, xta

ROOT = Path(__file__).resolve().parents[3]
XT = ROOT / "scripts" / "xt"
CONFIRM = ROOT / "scripts" / "confirm.sh"
F = ROOT / "fixtures"
FIND, HELD = "gpt2-induction", "gpt2-induction-heldout"
REAL = {"mean": ("ablation", "mean_effect"), "denoise": ("patch", "head_denoise")}
SVG = "{http://www.w3.org/2000/svg}"

pytestmark = pytest.mark.skipif(
    not (ROOT / "bin" / "xetal").exists() or not (F / f"{HELD}-controls").exists(),
    reason="bin/xetal or the held-out fixtures",
)


def seeds(d: Path) -> list[int]:
    return sorted(int(p.name.removeprefix("seed-")) for p in d.iterdir())


def runs(d: Path) -> str:
    return '"' + " ".join(str(d / f"seed-{s}") for s in seeds(d)) + '"'


def sets(method: str) -> str:
    kind = REAL[method][0]
    dirs = (
        F / f"{FIND}-{kind}",
        F / f"{FIND}-controls",
        F / f"{HELD}-{kind}",
        F / f"{HELD}-controls",
    )
    return "(" + ", ".join(runs(d) for d in dirs) + ")"


def out(body: str) -> str:
    src = '"ef:" u_se< "Effects"\n' + body
    r = subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def percent(d: Path, name: str, method: str) -> np.ndarray:
    out = []
    for s in seeds(d):
        a = xta.read(d / f"seed-{s}" / f"{name}.xta").astype(np.float64)
        meta = tomllib.loads((d / f"seed-{s}" / "run.toml").read_text())
        scale = 100.0 / float(meta["baseline"][0]) if method == "mean" else 100.0
        out.append(scale * a)
    return np.stack(out).mean(0)


def passes(prefix: str, method: str) -> np.ndarray:
    kind, name = REAL[method]
    real = percent(F / f"{prefix}-{kind}", name, method)
    top = percent(F / f"{prefix}-controls", f"{method}_control", method).max(0)
    return ((real > top) & (real >= 5.0)).astype(int)


def test_the_held_out_fixtures_are_the_experiments_held_out_seeds():
    exp = experiment.load(ROOT / "experiments" / "induction.toml")
    for kind in ("ablation", "patch", "controls"):
        assert seeds(F / f"{HELD}-{kind}") == list(exp.heldout_seeds)
        assert not set(seeds(F / f"{HELD}-{kind}")) & set(exp.finding_seeds)


@pytest.mark.parametrize("method", ["mean", "denoise"])
def test_levels_follow_the_rule(method):
    got = np.array(out(f'r_avel {sets(method)} ef:l_evels "{method}"').split(), int)
    f, g = passes(FIND, method), passes(HELD, method)
    assert np.array_equal(got.reshape(12, 12), f + f * g)


def test_confirmations_report_every_tested_head_with_its_verdict():
    text = out(f'{sets("denoise")} ef:c_onfirmations "denoise"')
    f, g = passes(FIND, "denoise"), passes(HELD, "denoise")
    assert f"{int((f * g).sum())} of the {int(f.sum())} heads that pass" in text
    lines = [line for line in text.splitlines()[2:] if line.strip()]
    assert len(lines) == f.sum()
    for line in lines:
        layer, head = (int(v) for v in line.split()[0].split("."))
        verdict = "confirmed" if g[layer, head] else "tested only"
        assert line.split(": ", 1)[1].startswith(verdict), line


def test_command_line_and_readme_are_current(tmp_path):
    text = (ROOT / "README.md").read_text()
    sec = text.split("## Confirmed on held-out seeds", 1)[1].split("\n## ", 1)[0]
    cmd = sec.split("```bash", 1)[1].split("```", 1)[0].split()
    shown = sec.split("```text\n", 1)[1].split("```", 1)[0]
    assert cmd[:2] == ["just", "confirm"]
    r = subprocess.run(
        [str(CONFIRM), *cmd[2:4], str(tmp_path)], cwd=ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == shown, "the README's verdicts are out of date"
    for m in ("mean", "denoise"):
        svg = tmp_path / f"confirmed-{m}.svg"
        assert ET.parse(svg).getroot().tag == f"{SVG}svg"
        assert svg.read_text() == (ROOT / cmd[4] / svg.name).read_text()
