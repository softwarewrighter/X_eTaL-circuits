"""The committed fixture captures (fixtures/, made by `just fixture`) stay valid."""

import tomllib
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from circuits_adapter import capture, xta

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "gpt2-induction"


def test_fixture_metadata_matches_its_arrays():
    meta = tomllib.loads((FIXTURE / "capture.toml").read_text())
    assert meta["format"] == [capture.FORMAT]
    assert all(k in meta for k in capture.REQUIRED)
    for name in meta["arrays"]:
        assert [str(d) for d in xta.shape(FIXTURE / f"{name}.xta")] == meta[f"{name}_shape"]


def test_fixture_attention_is_valid_and_both_formats_agree():
    attn = xta.read(FIXTURE / "attn.xta")
    capture.check_attention(attn)
    tensors = load_file(FIXTURE / "capture.safetensors")
    assert np.array_equal(tensors["attn"], attn)
    assert np.array_equal(tensors["logits"], xta.read(FIXTURE / "logits.xta"))


def test_fixture_second_copy_repeats_the_first():
    ids = tomllib.loads((FIXTURE / "capture.toml").read_text())["token_ids"]
    half = (len(ids) - 1) // 2
    assert ids[1 : 1 + half] == ids[1 + half :]


FULL = FIXTURE.parent / "gpt2-induction-full"


def test_full_fixture_has_every_layer_and_repeats():
    meta = tomllib.loads((FULL / "capture.toml").read_text())
    assert meta["attn_layers"] == [str(n) for n in range(12)]
    assert meta["attn_shape"] == ["12", "12", "33", "33"]
    attn = xta.read(FULL / "attn.xta")
    capture.check_attention(attn)
    ids = meta["token_ids"]
    assert ids[1:17] == ids[17:]
