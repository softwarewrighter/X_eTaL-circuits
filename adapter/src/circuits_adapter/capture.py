"""A capture: model internals plus the metadata that makes them reproducible.

Layout and keys: docs/design.md ("Capture format", D2-D4, D7).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from safetensors.numpy import save as safetensors_bytes

from circuits_adapter import text_arrays, toml_out, xta

FORMAT = "xetal-circuits-capture-1"
ROW_SUM_TOL = 1e-4
REQUIRED = (
    "model", "revision", "library", "device", "dtype", "seed",
    "prompt_kind", "prompt", "token_ids", "tokens",
)  # fmt: skip


class CaptureError(ValueError):
    """A capture that breaks the format's rules."""


@dataclass
class Capture:
    attn: np.ndarray  # f32, layer head query key
    logits: np.ndarray  # f32, vocab (the last position)
    info: dict = field(default_factory=dict)  # REQUIRED keys, lists of strings
    attn_layers: list = field(default_factory=list)  # model layer index of each attn row


def check_attention(attn: np.ndarray) -> None:
    """Refuse anything but causal softmax rows: layer head query key, rows summing to 1."""
    if attn.ndim != 4:
        raise CaptureError(f"attention rank {attn.ndim}, expected 4 (layer head query key)")
    if attn.shape[2] != attn.shape[3]:
        raise CaptureError(f"attention is not square in query and key: {attn.shape}")
    if np.any(np.triu(attn, 1) != 0):
        raise CaptureError("attention is not causal: a query attends to a later key")
    err = float(np.abs(attn.sum(-1, dtype=np.float64) - 1).max(initial=0))
    if err > ROW_SUM_TOL:
        raise CaptureError(f"attention rows sum to 1 only within {err:.3g}")


def _check_info(c: Capture) -> None:
    missing = [k for k in REQUIRED if k not in c.info]
    if missing:
        raise CaptureError(f"metadata missing: {', '.join(missing)}")
    t = c.attn.shape[-1]
    for key in ("token_ids", "tokens"):
        if len(c.info[key]) != t:
            raise CaptureError(f"{key} has {len(c.info[key])} entries for {t} positions")
    if len(c.attn_layers) != c.attn.shape[0]:
        raise CaptureError("attn_layers does not name every attention layer")


def metadata(c: Capture) -> dict:
    """capture.toml's table: format, the run's metadata, then each array's axes and shape."""
    shape = lambda a: [str(d) for d in a.shape]  # noqa: E731
    return {
        "format": [FORMAT],
        **{k: c.info[k] for k in REQUIRED},
        "arrays": ["attn", "logits"],
        "text_arrays": ["attn", "logits"],
        "attn_axes": ["layer", "head", "query", "key"],
        "attn_shape": shape(c.attn),
        "attn_layers": [str(n) for n in c.attn_layers],
        "logits_axes": ["vocab"],
        "logits_shape": shape(c.logits),
        "logits_position": [str(c.attn.shape[-1] - 1)],
    }


def _arrays(c: Capture) -> dict:
    return {
        "attn": np.ascontiguousarray(c.attn, dtype=np.float32),
        "logits": np.ascontiguousarray(c.logits, dtype=np.float32),
    }


def write(out, c: Capture, safetensors: bool = False) -> Path:
    """Write the capture to a new directory; an existing one is refused."""
    check_attention(c.attn)
    _check_info(c)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    arrays = _arrays(c)
    for name, a in arrays.items():
        xta.write(out / f"{name}.xta", a)
        text_arrays.write(out / f"{name}.txt", a)
    if safetensors:
        (out / "capture.safetensors").write_bytes(safetensors_bytes(arrays))
    (out / "capture.toml").write_text(toml_out.dumps(metadata(c)), encoding="ascii")
    return out
