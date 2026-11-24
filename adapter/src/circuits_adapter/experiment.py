"""Experiment specifications (experiments/*.toml): every value a list of strings, so
X_eTaL reads them with []L_IST as Python reads them with tomllib."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

REQUIRED = (
    "name", "model", "revision", "prompt", "length", "corruption", "metric",
    "finding_seeds", "heldout_seeds", "interventions",
)  # fmt: skip
KNOWN = {
    "prompt": {"repeated-random"},
    "corruption": {"fresh-second-copy"},
    "metric": {"logit-difference"},
    "interventions": {"zero-ablation", "mean-ablation", "patch-denoise", "patch-noise"},
}


class ExperimentError(ValueError):
    """An experiment file that cannot be run as written."""


@dataclass(frozen=True)
class Experiment:
    path: Path
    name: str
    model: str
    revision: str
    length: int
    finding_seeds: tuple[int, ...]
    heldout_seeds: tuple[int, ...]
    interventions: tuple[str, ...]


def _check_known(t: dict) -> None:
    for key, allowed in KNOWN.items():
        bad = [v for v in t[key] if v not in allowed]
        if bad:
            raise ExperimentError(f"{key}: {', '.join(bad)} not one of {sorted(allowed)}")


def _ints(t: dict, key: str) -> tuple[int, ...]:
    try:
        return tuple(int(v) for v in t[key])
    except ValueError as e:
        raise ExperimentError(f"{key}: {e}") from None


def _seeds(t: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    finding, heldout = _ints(t, "finding_seeds"), _ints(t, "heldout_seeds")
    if not finding or not heldout:
        raise ExperimentError("finding_seeds and heldout_seeds must both be given")
    if set(finding) & set(heldout):
        raise ExperimentError("a seed is both a finding seed and a held-out seed")
    return finding, heldout


def load(path) -> Experiment:
    path = Path(path)
    t = tomllib.loads(path.read_text())
    missing = [k for k in REQUIRED if k not in t]
    if missing:
        raise ExperimentError(f"{path}: missing {', '.join(missing)}")
    if any(not isinstance(v, list) or not all(isinstance(s, str) for s in v) for v in t.values()):
        raise ExperimentError(f"{path}: every value must be a list of strings")
    _check_known(t)
    finding, heldout = _seeds(t)
    (length,) = _ints(t, "length")
    return Experiment(
        path, t["name"][0], t["model"][0], t["revision"][0], length,
        finding, heldout, tuple(t["interventions"]),
    )  # fmt: skip
