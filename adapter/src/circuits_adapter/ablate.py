"""Head ablation (docs/design.md, D15): one head's output replaced, the metric's drop recorded.

A head's output is its slice of blocks.{l}.attn.hook_z (position x head x d_head,
before W_O). Zero ablation sets it to 0; mean ablation sets it to the head's mean
over the corrupted twins of the experiment's finding seeds, position by position.
The effect of a head is the clean prompt's mean logit difference minus the
ablated one: positive when the output needs the head.
"""

import numpy as np
import torch

from circuits_adapter import metric, model, prompts, twins

METHODS = ("zero", "mean")


def z_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_z"


def set_head(head: int, value=None):
    """A hook setting one head's output to 0 (value None) or to value (position x d_head)."""

    def hook(z, hook):
        z[:, :, head, :] = 0.0 if value is None else value
        return z

    return hook


def mean_ld(m, ids, clean, corrupt, length, hooks=()) -> float:
    """The mean logit difference of one run of ids, with the given forward hooks."""
    with torch.no_grad():
        out = m.bridge.run_with_hooks(torch.tensor([ids], device=m.device), fwd_hooks=list(hooks))
    return float(
        metric.logit_difference(out[0].float().cpu().numpy(), clean, corrupt, length).mean()
    )


def twin_means(m, exp) -> dict[int, torch.Tensor]:
    """Each layer's hook_z averaged over the finding seeds' twins: position x head x d_head."""
    names = [z_name(n) for n in range(m.n_layers)]
    total = {n: 0.0 for n in range(m.n_layers)}
    for seed in exp.finding_seeds:
        _, corrupt = prompts.twins(seed, exp.length, m.bos)
        with torch.no_grad():
            _, cache = m.bridge.run_with_cache(
                torch.tensor([corrupt], device=m.device), names_filter=lambda h: h in names
            )
        for n in total:
            total[n] = total[n] + cache[names[n]][0]
    return {n: t / len(exp.finding_seeds) for n, t in total.items()}


def effects(m, exp, seed: int, means: dict | None = None) -> dict:
    """The clean baseline and, per method, the layer x head matrix of effects."""
    clean, corrupt = prompts.twins(seed, exp.length, m.bos)
    twins.check(clean, corrupt, exp.length)
    base = mean_ld(m, clean, clean, corrupt, exp.length)
    heads = m.bridge.cfg.n_heads
    out = {"baseline": base}
    for method in METHODS:
        e = np.zeros((m.n_layers, heads))
        for layer in range(m.n_layers):
            for h in range(heads):
                value = None if method == "zero" else means[layer][:, h, :]
                hook = (z_name(layer), set_head(h, value))
                e[layer, h] = base - mean_ld(m, clean, clean, corrupt, exp.length, [hook])
        out[method] = e
    return out


def write(out, m, exp, seed: int, means: dict):
    """A run directory with both effect matrices and the clean baseline."""
    from circuits_adapter import run

    clean, corrupt = prompts.twins(seed, exp.length, m.bos)
    r = effects(m, exp, seed, means)
    info = model.run_info(m, str(exp.path), seed, exp.length, clean, corrupt)
    info["intervention"] = ["zero-ablation", "mean-ablation"]
    info["mean_over"] = [f"twins of finding seeds {' '.join(map(str, exp.finding_seeds))}"]
    info["baseline"] = [repr(r["baseline"])]
    axes = ("layer", "head")
    arrays = {"zero_effect": (axes, r["zero"]), "mean_effect": (axes, r["mean"])}
    return run.write(out, info, arrays)
