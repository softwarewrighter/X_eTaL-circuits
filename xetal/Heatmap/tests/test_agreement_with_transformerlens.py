"""What X_eTaL reads and draws agrees with TransformerLens's own cache.

GPT-2 Small at the pinned revision is run again on the fixture's tokens; its
attention patterns are compared with the values X_eTaL reads from the fixture
(within 1e-5 relative), entry by entry for a handful of (layer, head, query,
key), and row by row for the sums. Needs the model extra and the weights
(ADAPTER_MODEL_TESTS=1 just xetal-test).
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
XT = ROOT / "scripts" / "xt"
FIXTURE = ROOT / "fixtures" / "gpt2-induction"
D = f'"{FIXTURE}"'
RTOL = 1e-5

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not (ROOT / "bin" / "xetal").exists(), reason="bin/xetal not built"),
]

# (layer, head, query, key), model numbering from 0: BOS columns, the
# induction stripe of 5.1 and 5.5, a previous-token-like entry of layer 0.
ENTRIES = [(5, 1, 0, 0), (5, 1, 11, 4), (5, 1, 15, 8), (5, 5, 12, 5), (5, 5, 16, 9), (0, 3, 9, 9)]


def xetal(body: str) -> np.ndarray:
    src = '"hm:" u_se< "Heatmap"\n' + body
    r = subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return np.array(r.stdout.split(), dtype=np.float64)


@pytest.fixture(scope="module")
def cache():
    import tomllib

    import torch

    from circuits_adapter import model

    info = tomllib.loads((FIXTURE / "capture.toml").read_text())
    m = model.load(info["model"][0], revision=info["revision"][0])
    ids = [int(t) for t in info["token_ids"]]
    layers = [int(n) for n in info["attn_layers"]]
    names = [f"blocks.{n}.attn.hook_pattern" for n in layers]
    with torch.no_grad():
        _, c = m.bridge.run_with_cache(torch.tensor([ids]), names_filter=lambda h: h in names)
    return {n: c[h][0].double().numpy() for n, h in zip(layers, names, strict=True)}


@pytest.mark.parametrize("entry", ENTRIES)
def test_entries_agree(cache, entry):
    layer, head, q, k = entry
    got = xetal(f"{D} hm:p_attern ({layer}, {head})").reshape(17, 17)[q, k]
    assert got == pytest.approx(cache[layer][head, q, k], rel=RTOL)
    assert got > 0.01  # the entries chosen carry weight, so relative is meaningful


@pytest.mark.parametrize("layer,head", [(0, 3), (5, 1), (5, 5)])
def test_row_sums_agree(cache, layer, head):
    got = xetal(f"'+ r_/_2 {D} hm:p_attern ({layer}, {head})")
    assert np.allclose(got, cache[layer][head].sum(-1), rtol=RTOL, atol=0)
