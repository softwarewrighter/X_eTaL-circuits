"""The graph of a transformer's paths (docs/design.md, D19): senders, receivers, edges.

A sender is a head "layer.head" or "embed" (the token and position embeddings).
A receiver is a head's query, key or value input "layer.head.q|k|v", or "logits"
(the final residual stream). An edge "sender receiver" exists when the receiver
reads the residual stream after the sender writes it: a head's input in a later
layer, or the logits. Layers and heads are numbered from 0, as the model numbers
them.
"""

from dataclasses import dataclass

INPUTS = ("q", "k", "v")
EMBED, LOGITS = "embed", "logits"


class GraphError(ValueError):
    """A sender, receiver or edge that does not name a path of the model."""


@dataclass(frozen=True, order=True)
class Sender:
    layer: int  # -1 for the embeddings
    head: int = 0

    def __str__(self) -> str:
        return EMBED if self.layer < 0 else f"{self.layer}.{self.head}"


@dataclass(frozen=True, order=True)
class Receiver:
    layer: int  # n_layers for the logits
    head: int = 0
    input: str = ""

    def __str__(self) -> str:
        return LOGITS if not self.input else f"{self.layer}.{self.head}.{self.input}"


def _ints(parts: list[str], text: str) -> list[int]:
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise GraphError(f"{text!r} is not layer.head numbers") from None
    if any(v < 0 for v in values):
        raise GraphError(f"{text!r} has a negative number")
    return values


def sender(text: str, n_layers: int, n_heads: int) -> Sender:
    if text == EMBED:
        return Sender(-1, 0)
    parts = text.split(".")
    if len(parts) != 2:
        raise GraphError(f"sender {text!r} is not layer.head or embed")
    layer, head = _ints(parts, text)
    _check_head(layer, head, n_layers, n_heads, text)
    return Sender(layer, head)


def receiver(text: str, n_layers: int, n_heads: int) -> Receiver:
    if text == LOGITS:
        return Receiver(n_layers)
    parts = text.split(".")
    if len(parts) != 3 or parts[2] not in INPUTS:
        raise GraphError(f"receiver {text!r} is not layer.head.q|k|v or logits")
    layer, head = _ints(parts[:2], text)
    _check_head(layer, head, n_layers, n_heads, text)
    return Receiver(layer, head, parts[2])


def _check_head(layer: int, head: int, n_layers: int, n_heads: int, text: str) -> None:
    if layer >= n_layers or head >= n_heads:
        raise GraphError(f"{text!r} is outside the model ({n_layers} layers, {n_heads} heads)")


def edge(text: str, n_layers: int, n_heads: int) -> tuple[Sender, Receiver]:
    """Parse "sender receiver", refusing a receiver that does not come after the sender."""
    parts = text.split()
    if len(parts) != 2:
        raise GraphError(f'edge {text!r} is not "sender receiver"')
    s = sender(parts[0], n_layers, n_heads)
    r = receiver(parts[1], n_layers, n_heads)
    if r.layer <= s.layer:
        raise GraphError(f"edge {text!r}: the receiver is not in a later layer than the sender")
    return s, r


def name(s: Sender, r: Receiver) -> str:
    return f"{s} {r}"


def all_edges(n_layers: int, n_heads: int, inputs=INPUTS) -> list[tuple[Sender, Receiver]]:
    """Every edge, senders in order (embed first), then receivers in order (logits last)."""
    senders = [Sender(-1)] + [Sender(la, h) for la in range(n_layers) for h in range(n_heads)]
    out = []
    for s in senders:
        later = [
            Receiver(la, h, i)
            for la in range(s.layer + 1, n_layers)
            for h in range(n_heads)
            for i in inputs
        ]
        out += [(s, r) for r in [*later, Receiver(n_layers)]]
    return out
