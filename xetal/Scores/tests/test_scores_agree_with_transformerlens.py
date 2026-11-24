"""X_eTaL's head scores agree with the same scores computed from TransformerLens's cache.

GPT-2 Small at the pinned revision is run again on the full fixture's tokens;
both scores are computed in Python from its attention patterns and compared
with the X_eTaL scores read from the fixture, within 1e-5 relative. Needs the
model extra and the weights (ADAPTER_MODEL_TESTS=1 just xetal-test).
"""

import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
XT = ROOT / "scripts" / "xt"
FULL = ROOT / "fixtures" / "gpt2-induction-full"
D = f'"{FULL}"'

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not (ROOT / "bin" / "xetal").exists(), reason="bin/xetal not built"),
]


def xetal(body: str) -> np.ndarray:
    src = '"sc:" u_se< "Scores"\n' + body
    r = subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return np.array(r.stdout.split(), dtype=np.float64).reshape(12, 12)


@pytest.fixture(scope="module")
def tl_scores():
    import torch

    from circuits_adapter import model

    info = tomllib.loads((FULL / "capture.toml").read_text())
    m = model.load(info["model"][0], revision=info["revision"][0])
    ids = [int(t) for t in info["token_ids"]]
    with torch.no_grad():
        _, c = m.bridge.run_with_cache(
            torch.tensor([ids]), names_filter=lambda h: h.endswith("hook_pattern")
        )
    a = np.stack([c[f"blocks.{n}.attn.hook_pattern"][0].double().numpy() for n in range(12)])
    n = a.shape[-1]
    t = (n - 1) // 2
    i = np.arange(1, n)
    q = np.arange(t + 1, n)
    return {"p_revious": a[:, :, i, i - 1].mean(-1), "i_nduction": a[:, :, q, q - t + 1].mean(-1)}


@pytest.mark.parametrize("score", ["p_revious", "i_nduction"])
def test_scores_agree_with_transformerlens(tl_scores, score):
    got = xetal(f"sc:{score} {D}")
    assert np.allclose(got, tl_scores[score], rtol=1e-5, atol=1e-7)
