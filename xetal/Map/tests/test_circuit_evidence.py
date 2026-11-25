"""experiments/circuits/gpt2-induction.toml's head evidence is what the runs give:
each head's level and basis follow from the finding and held-out runs by the
rule of docs/design.md (D17, D18), computed here with numpy."""

import tomllib
from pathlib import Path

import numpy as np

from circuits_adapter import xta

ROOT = Path(__file__).resolve().parents[3]
F = ROOT / "fixtures"
CIRCUIT = ROOT / "experiments" / "circuits" / "gpt2-induction.toml"
METHODS = {
    "mean": ("ablation", "mean_effect", "mean ablation"),
    "denoise": ("patch", "head_denoise", "denoising"),
}


def percent(d: Path, name: str, method: str) -> np.ndarray:
    out = []
    for p in sorted(d.iterdir(), key=lambda p: int(p.name.removeprefix("seed-"))):
        meta = tomllib.loads((p / "run.toml").read_text())
        scale = 100.0 / float(meta["baseline"][0]) if method == "mean" else 100.0
        out.append(scale * xta.read(p / f"{name}.xta").astype(np.float64))
    return np.stack(out).mean(0)


def verdicts(method: str):
    kind, name, _ = METHODS[method]
    found, held = {}, {}
    for prefix, store in (("gpt2-induction", found), ("gpt2-induction-heldout", held)):
        real = percent(F / f"{prefix}-{kind}", name, method)
        top = percent(F / f"{prefix}-controls", f"{method}_control", method).max(0)
        store["effect"] = real
        store["pass"] = (real > top) & (real >= 5.0)
    return found, held


def basis(layer: int, head: int) -> tuple[int, str]:
    """The head's level (0 none, 1 tested, 2 confirmed) and its basis text."""
    level, said, under = 0, [], []
    for method in ("mean", "denoise"):
        found, held = verdicts(method)
        words = METHODS[method][2]
        f = int(np.floor(0.5 + found["effect"][layer, head]))
        g = int(np.floor(0.5 + held["effect"][layer, head]))
        if not found["pass"][layer, head]:
            under.append(f"{words} under 5%")
            continue
        both = held["pass"][layer, head]
        level = max(level, 2 if both else 1)
        verdict = "confirmed" if both else "tested only"
        said.append(f"{words} {verdict} ({f}%, held-out {g}%)")
    return level, "; ".join(said + under)


def test_every_circuit_heads_evidence_follows_from_the_runs():
    c = tomllib.loads(CIRCUIT.read_text())
    names = {0: "observed", 1: "tested", 2: "confirmed"}
    for head, ev, text in zip(c["heads"], c["head_evidence"], c["head_basis"], strict=True):
        layer, h = (int(v) for v in head.split("."))
        level, want = basis(layer, h)
        assert ev == names[level], head
        if level:
            assert text == want, head


def test_every_confirmed_head_is_in_the_circuit():
    c = tomllib.loads(CIRCUIT.read_text())
    for method in METHODS:
        found, held = verdicts(method)
        for layer, h in zip(*np.nonzero(found["pass"] & held["pass"]), strict=True):
            assert f"{layer}.{h}" in c["heads"], (method, layer, h)
