"""Activation patching (docs/design.md, D16): one activation copied between the twins.

Denoising runs the twin with one activation taken from the clean prompt and
gives the fraction of the clean-twin gap it restores:
    (ld_patched - ld_twin) / (ld_clean - ld_twin).
Noising runs the clean prompt with one activation from the twin and gives the
fraction of the gap it destroys:
    (ld_clean - ld_patched) / (ld_clean - ld_twin).
Both are positive when the output needs the activation. ld is the mean logit
difference (D14). Patched: the residual stream entering each layer at each
position (blocks.{l}.hook_resid_pre), each head's output at every position
(hook_z), and each of the leading heads' output at each position.
"""

import numpy as np
import torch

from circuits_adapter import metric, prompts, twins

LEADING = 6  # heads patched position by position: the most restoring by denoising


def resid_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_pre"


def z_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_z"


def put(source: torch.Tensor, head: int | None = None, position: int | None = None):
    """A hook copying source into the activation: all of it, or one head and/or position."""

    def hook(x, hook):
        idx = (slice(None), slice(None) if position is None else position)
        if head is not None:
            idx = (*idx, head)
        x[idx] = source[idx]
        return x

    return hook


class Twins:
    """A seed's clean prompt and twin, their caches and their mean logit differences."""

    def __init__(self, m, length: int, seed: int):
        self.m, self.length = m, length
        self.clean, self.corrupt = prompts.twins(seed, length, m.bos)
        twins.check(self.clean, self.corrupt, length)
        names = {resid_name(n) for n in range(m.n_layers)} | {z_name(n) for n in range(m.n_layers)}
        self.cache = {}
        for kind, ids in (("clean", self.clean), ("corrupt", self.corrupt)):
            with torch.no_grad():
                _, c = m.bridge.run_with_cache(self._tokens(ids), names_filter=lambda h: h in names)
            self.cache[kind] = {k: v.clone() for k, v in c.items()}
        self.ld = {k: self.run(k, []) for k in ("clean", "corrupt")}

    def _tokens(self, ids):
        return torch.tensor([ids], device=self.m.device)

    def run(self, kind: str, hooks) -> float:
        """The mean logit difference running the clean prompt or the twin with hooks."""
        ids = self.clean if kind == "clean" else self.corrupt
        with torch.no_grad():
            out = self.m.bridge.run_with_hooks(self._tokens(ids), fwd_hooks=list(hooks))
        lg = out[0].float().cpu().numpy()
        return float(metric.logit_difference(lg, self.clean, self.corrupt, self.length).mean())

    def fraction(self, direction: str, name: str, head=None, position=None) -> float:
        """Denoise or noise one activation: the fraction of the gap restored or destroyed."""
        gap = self.ld["clean"] - self.ld["corrupt"]
        if direction == "denoise":
            hook = put(self.cache["clean"][name], head, position)
            return (self.run("corrupt", [(name, hook)]) - self.ld["corrupt"]) / gap
        hook = put(self.cache["corrupt"][name], head, position)
        return (self.ld["clean"] - self.run("clean", [(name, hook)])) / gap


def residual(t: Twins) -> np.ndarray:
    """Denoising the residual stream entering each layer at each position: layer x position."""
    n = len(t.clean)
    out = np.zeros((t.m.n_layers, n))
    for layer in range(t.m.n_layers):
        for pos in range(n):
            out[layer, pos] = t.fraction("denoise", resid_name(layer), position=pos)
    return out


def heads(t: Twins, direction: str) -> np.ndarray:
    """Each head's output patched at every position: layer x head."""
    n_heads = t.m.bridge.cfg.n_heads
    out = np.zeros((t.m.n_layers, n_heads))
    for layer in range(t.m.n_layers):
        for h in range(n_heads):
            out[layer, h] = t.fraction(direction, z_name(layer), head=h)
    return out


def head_positions(t: Twins, ranked: list[tuple[int, int]]) -> np.ndarray:
    """Denoising each of the ranked heads at each position: head x position."""
    n = len(t.clean)
    out = np.zeros((len(ranked), n))
    for i, (layer, h) in enumerate(ranked):
        for pos in range(n):
            out[i, pos] = t.fraction("denoise", z_name(layer), head=h, position=pos)
    return out


def leading(denoise: np.ndarray, k: int = LEADING) -> list[tuple[int, int]]:
    """The k heads whose denoising restores the most, best first (ties by position)."""
    order = np.argsort(-denoise.ravel(), kind="stable")[:k]
    return [divmod(int(i), denoise.shape[1]) for i in order]


def write(out, m, exp, seed: int):
    """A run directory with every patching result for one seed."""
    from circuits_adapter import model, run

    t = Twins(m, exp.length, seed)
    denoise = heads(t, "denoise")
    ranked = leading(denoise)
    info = model.run_info(m, str(exp.path), seed, exp.length, t.clean, t.corrupt)
    info["intervention"] = ["patch-denoise", "patch-noise"]
    info["ld_clean"] = [repr(t.ld["clean"])]
    info["ld_corrupt"] = [repr(t.ld["corrupt"])]
    info["leading_heads"] = [f"{layer}.{h}" for layer, h in ranked]
    arrays = {
        "resid_denoise": (("layer", "position"), residual(t)),
        "head_denoise": (("layer", "head"), denoise),
        "head_noise": (("layer", "head"), heads(t, "noise")),
        "position_denoise": (("rank", "position"), head_positions(t, ranked)),
    }
    return run.write(out, info, arrays)
