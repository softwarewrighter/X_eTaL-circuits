"""Edge experiments (experiments/*-edges.toml, docs/design.md D19): which edges to test.

Every value is a list of strings. `base` names the experiment (beside this file)
whose prompts, metric, seeds and controls are used.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from circuits_adapter import experiment, graph

REQUIRED = ("name", "base", "senders", "receivers", "inputs", "path_patch", "always")
SHAPE = {"gpt2": (12, 12)}  # layers, heads of the models experiments use


class EdgesError(ValueError):
    """An edge experiment that cannot be run as written."""


@dataclass(frozen=True)
class Edges:
    path: Path
    name: str
    base: experiment.Experiment
    senders: tuple | None  # None: all
    receivers: tuple | None  # None: all
    inputs: tuple[str, ...]
    top: int | None  # None: path-patch all edges
    always: tuple

    def candidates(self) -> list:
        """The edges in scope, in graph.all_edges's order."""
        n_layers, n_heads = SHAPE[self.base.model]
        keep = graph.all_edges(n_layers, n_heads, self.inputs)
        if self.senders is not None:
            keep = [e for e in keep if e[0] in self.senders]
        if self.receivers is not None:
            keep = [e for e in keep if e[1] in self.receivers]
        return keep


def _listed(t: dict, key: str, parse):
    return None if t[key] == ["all"] else tuple(parse(v) for v in t[key])


def _path_patch(t: dict) -> int | None:
    spec = t["path_patch"]
    if spec == ["all"]:
        return None
    if len(spec) == 2 and spec[0] == "top" and spec[1].isdigit() and int(spec[1]) > 0:
        return int(spec[1])
    raise EdgesError(f'path_patch: {spec} is not ["all"] or ["top", N]')


def load(path) -> Edges:
    path = Path(path)
    t = tomllib.loads(path.read_text())
    missing = [k for k in REQUIRED if k not in t]
    if missing:
        raise EdgesError(f"{path}: missing {', '.join(missing)}")
    base = experiment.load(path.parent / t["base"][0])
    if base.model not in SHAPE:
        raise EdgesError(f"{path}: no layer and head counts for model {base.model}")
    n_layers, n_heads = SHAPE[base.model]
    bad = [i for i in t["inputs"] if i not in graph.INPUTS]
    if bad or not t["inputs"]:
        raise EdgesError(f"inputs: {bad or 'none'} not among {list(graph.INPUTS)}")
    try:
        senders = _listed(t, "senders", lambda v: graph.sender(v, n_layers, n_heads))
        receivers = _listed(t, "receivers", lambda v: graph.receiver(v, n_layers, n_heads))
        always = tuple(graph.edge(e, n_layers, n_heads) for e in t["always"])
    except graph.GraphError as e:
        raise EdgesError(f"{path}: {e}") from None
    return Edges(
        path, t["name"][0], base, senders, receivers, tuple(t["inputs"]), _path_patch(t), always
    )
