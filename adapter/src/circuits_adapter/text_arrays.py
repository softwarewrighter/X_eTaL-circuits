"""Arrays as text numbers for X_eTaL's `n_umbers []N_GET` (docs/design.md, D10).

Row-major, one line per last-axis row. Each value is the shortest decimal of its
f64 widening (Python's float repr), so X_eTaL reads exactly the f64 that a
binary reader widening the same f32 would give. The shape is in capture.toml.
"""

from pathlib import Path

import numpy as np


def _row(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def dumps(a) -> str:
    a = np.asarray(a)
    if a.ndim == 0:
        return repr(float(a)) + "\n"
    rows = a.reshape(-1, a.shape[-1]) if a.size else np.zeros((0, 0))
    return "".join(_row(r) + "\n" for r in rows)


def write(path, a) -> None:
    """Write the text copy to a new file; an existing file is refused."""
    with Path(path).open("x", encoding="ascii") as f:
        f.write(dumps(a))
