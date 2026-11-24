"""A run directory: an experiment's results, with the metadata that makes them reproducible.

run.toml (every value a list of strings, read by X_eTaL with []L_IST) and each
array as XTLARR01 (<name>.xta) and as text numbers (<name>.txt, D10), shaped by
<name>_shape. Layout: docs/design.md ("Run format", D14).
"""

from pathlib import Path

import numpy as np

from circuits_adapter import text_arrays, toml_out, xta

FORMAT = "xetal-circuits-run-1"
REQUIRED = (
    "experiment", "model", "revision", "library", "device", "dtype", "seed",
    "length", "clean_ids", "corrupt_ids",
)  # fmt: skip


class RunError(ValueError):
    """A run that breaks the format's rules."""


def metadata(info: dict, arrays: dict) -> dict:
    missing = [k for k in REQUIRED if k not in info]
    if missing:
        raise RunError(f"metadata missing: {', '.join(missing)}")
    extra = {k: v for k, v in info.items() if k not in REQUIRED}
    table = {"format": [FORMAT], **{k: info[k] for k in REQUIRED}, **extra}
    table["arrays"] = list(arrays)
    for name, (axes, a) in arrays.items():
        if len(axes) != a.ndim:
            raise RunError(f"{name}: {len(axes)} axis names for rank {a.ndim}")
        table[f"{name}_axes"] = list(axes)
        table[f"{name}_shape"] = [str(d) for d in a.shape]
    return table


def write(out, info: dict, arrays: dict) -> Path:
    """Write a new run directory (an existing one is refused); arrays: name -> (axes, array)."""
    table = metadata(info, arrays)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name, (_, a) in arrays.items():
        a32 = np.ascontiguousarray(a, dtype=np.float32)
        xta.write(out / f"{name}.xta", a32)
        text_arrays.write(out / f"{name}.txt", a32)
    (out / "run.toml").write_text(toml_out.dumps(table), encoding="ascii")
    return out
