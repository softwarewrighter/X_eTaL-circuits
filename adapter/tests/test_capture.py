"""Writing a capture directory (docs/design.md, "Capture format"); no model needed."""

import tomllib

import numpy as np
import pytest
from safetensors.numpy import load_file

from circuits_adapter import capture, xta


def causal_attention(layers=2, heads=3, t=5, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((layers, heads, t, t))
    scores = np.where(np.tril(np.ones((t, t), dtype=bool)), scores, -np.inf)
    w = np.exp(scores - scores.max(-1, keepdims=True))
    return (w / w.sum(-1, keepdims=True)).astype(np.float32)


def sample(t=5):
    return capture.Capture(
        attn=causal_attention(t=t),
        logits=np.linspace(-1, 1, 11, dtype=np.float32),
        info={
            "model": ["gpt2"],
            "revision": ["abc"],
            "library": ["transformer-lens 4.4.0"],
            "device": ["cpu"],
            "dtype": ["float32"],
            "seed": ["0"],
            "prompt_kind": ["repeated-random"],
            "prompt": [""],
            "token_ids": [str(i) for i in range(t)],
            "tokens": [f"t{i}" for i in range(t)],
        },
        attn_layers=[0, 1],
    )


def test_writes_arrays_and_toml(tmp_path):
    out = tmp_path / "cap"
    capture.write(out, sample())
    assert sorted(p.name for p in out.iterdir()) == ["attn.xta", "capture.toml", "logits.xta"]
    meta = tomllib.loads((out / "capture.toml").read_text())
    assert meta["format"] == ["xetal-circuits-capture-1"]
    assert meta["arrays"] == ["attn", "logits"]
    assert meta["attn_axes"] == ["layer", "head", "query", "key"]
    assert meta["attn_shape"] == ["2", "3", "5", "5"]
    assert meta["attn_layers"] == ["0", "1"]
    assert meta["logits_axes"] == ["vocab"]
    assert meta["logits_shape"] == ["11"]
    assert meta["logits_position"] == ["4"]
    for name in meta["arrays"]:
        assert [str(d) for d in xta.shape(out / f"{name}.xta")] == meta[f"{name}_shape"]


def test_also_writes_safetensors_with_the_same_arrays(tmp_path):
    out = tmp_path / "cap"
    c = sample()
    capture.write(out, c, safetensors=True)
    tensors = load_file(out / "capture.safetensors")
    assert sorted(tensors) == ["attn", "logits"]
    assert np.array_equal(tensors["attn"], c.attn)
    assert np.array_equal(tensors["logits"], c.logits)


def test_same_capture_gives_byte_identical_files(tmp_path):
    for name in ("a", "b"):
        capture.write(tmp_path / name, sample(), safetensors=True)
    for f in ("attn.xta", "logits.xta", "capture.toml", "capture.safetensors"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def test_refuses_existing_directory(tmp_path):
    capture.write(tmp_path / "cap", sample())
    with pytest.raises(FileExistsError):
        capture.write(tmp_path / "cap", sample())


def test_check_attention_accepts_softmax_rows():
    capture.check_attention(causal_attention())


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda a: a[0], "rank"),
        (lambda a: a[..., :-1], "square"),
        (lambda a: a * 1.01, "sum"),
        (lambda a: a + np.triu(np.full(a.shape[-2:], 1e-3, dtype=np.float32), 1), "causal"),
    ],
)
def test_check_attention_refuses_bad_patterns(damage, message):
    with pytest.raises(capture.CaptureError, match=message):
        capture.check_attention(damage(causal_attention()))


def test_refuses_missing_metadata_and_mismatched_tokens(tmp_path):
    c = sample()
    del c.info["revision"]
    with pytest.raises(capture.CaptureError, match="revision"):
        capture.write(tmp_path / "a", c)
    c = sample()
    c.info["tokens"] = c.info["tokens"][:-1]
    with pytest.raises(capture.CaptureError, match="tokens"):
        capture.write(tmp_path / "b", c)
