"""X_eTaL (bin/xetal, at XETAL_COMMIT: `just xetal`) reads the fixture's text arrays exactly.

Skipped when bin/xetal is not built.
"""

import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import xta

ROOT = Path(__file__).resolve().parents[2]
XETAL = ROOT / "bin" / "xetal"
FIXTURE = Path("fixtures") / "gpt2-induction"

pytestmark = pytest.mark.skipif(not XETAL.exists(), reason="bin/xetal not built (just xetal)")


def xetal(expr: str) -> str:
    r = subprocess.run(
        [str(XETAL), "eval", "-e", expr], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return r.stdout


def read_expr(name: str) -> str:
    meta = tomllib.loads((ROOT / FIXTURE / "capture.toml").read_text())
    shape = " ".join(meta[f"{name}_shape"])
    return f'{shape} r_eshape n_umbers []N_GET "{FIXTURE / name}.txt"'


def test_attention_shape_from_text():
    assert xetal(f"s_hape {read_expr('attn')}").split() == ["2", "12", "17", "17"]


def test_every_attention_value_equals_the_binary_files_f64_widening():
    printed = np.array(xetal(f"r_avel {read_expr('attn')}").split(), dtype=np.float64)
    expected = xta.read(ROOT / FIXTURE / "attn.xta").astype(np.float64).ravel()
    assert np.array_equal(printed, expected)


def test_logits_read_whole():
    printed = np.array(xetal(f"r_avel {read_expr('logits')}").split(), dtype=np.float64)
    expected = xta.read(ROOT / FIXTURE / "logits.xta").astype(np.float64)
    assert np.array_equal(printed, expected)
