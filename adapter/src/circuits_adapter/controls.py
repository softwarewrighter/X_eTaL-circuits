"""Matched random controls (docs/design.md, D17).

For each head, the real intervention changes the head's output z (position x
d_head) by some vector at each position: mean ablation by (mean - z_clean),
denoising by (z_clean - z_twin). A control changes it by a vector of the same
length at each position in a uniformly random direction. K controls per head
and seed run as one batch. Control effects are in the real method's units:
mean ablation as the clean logit difference lost, denoising as the fraction of
the gap restored. A head passes (in X_eTaL) when its mean real effect over the
seeds exceeds every control's mean over the seeds.
"""

import numpy as np
import torch

from circuits_adapter import metric, prompts, twins

K = 20
METHODS = ("mean", "denoise")


def z_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_z"


def directions(norms: torch.Tensor, k: int, seed: int, layer: int, head: int) -> torch.Tensor:
    """k x position x d_head: random directions, each position's length as in norms."""
    g = torch.Generator().manual_seed(((seed * 1000 + layer) * 100 + head) * 7 + 1)
    r = torch.randn(k, *norms.shape, generator=g)
    return r / r.norm(dim=-1, keepdim=True) * norms.norm(dim=-1)[None, :, None]


def add_to_head(head: int, r: torch.Tensor):
    """A hook adding r (batch x position x d_head) to one head's output."""

    def hook(z, hook):
        z[:, :, head, :] = z[:, :, head, :] + r.to(z.dtype)
        return z

    return hook


def batch_ld(m, ids, clean, corrupt, length, k, hooks) -> np.ndarray:
    """The mean logit difference of each of k copies of ids, run with hooks."""
    tokens = torch.tensor([ids] * k, device=m.device)
    with torch.no_grad():
        out = m.bridge.run_with_hooks(tokens, fwd_hooks=list(hooks)).float().cpu().numpy()
    return np.array([metric.logit_difference(o, clean, corrupt, length).mean() for o in out])


class Seed:
    """A seed's twins, caches and baselines, and the controls of every head."""

    def __init__(self, m, length: int, seed: int, means: dict, k: int = K):
        self.m, self.length, self.seed, self.means, self.k = m, length, seed, means, k
        self.clean, self.corrupt = prompts.twins(seed, length, m.bos)
        twins.check(self.clean, self.corrupt, length)
        names = [z_name(n) for n in range(m.n_layers)]
        self.z = {}
        for kind, ids in (("clean", self.clean), ("corrupt", self.corrupt)):
            with torch.no_grad():
                _, c = m.bridge.run_with_cache(
                    torch.tensor([ids], device=m.device), names_filter=lambda h: h in names
                )
            self.z[kind] = {n: c[z_name(n)][0].clone() for n in range(m.n_layers)}
        self.ld = {
            kind: float(batch_ld(m, ids, self.clean, self.corrupt, length, 1, [])[0])
            for kind, ids in (("clean", self.clean), ("corrupt", self.corrupt))
        }

    def control(self, method: str, layer: int, head: int) -> np.ndarray:
        """k control effects of one head, in the method's units."""
        clean_z = self.z["clean"][layer][:, head, :]
        if method == "mean":
            change = self.means[layer][:, head, :] - clean_z
            ids, run_on = self.clean, "clean"
        else:
            change = clean_z - self.z["corrupt"][layer][:, head, :]
            ids, run_on = self.corrupt, "corrupt"
        r = directions(change, self.k, self.seed, layer, head)
        hook = (z_name(layer), add_to_head(head, r))
        ld = batch_ld(self.m, ids, self.clean, self.corrupt, self.length, self.k, [hook])
        if run_on == "clean":
            return self.ld["clean"] - ld
        return (ld - self.ld["corrupt"]) / (self.ld["clean"] - self.ld["corrupt"])

    def all(self, method: str) -> np.ndarray:
        """k x layer x head control effects."""
        heads = self.m.bridge.cfg.n_heads
        out = np.zeros((self.k, self.m.n_layers, heads))
        for layer in range(self.m.n_layers):
            for h in range(heads):
                out[:, layer, h] = self.control(method, layer, h)
        return out


def write(out, m, exp, seed: int, means: dict):
    """A run directory with both methods' controls for one seed."""
    from circuits_adapter import model, run

    s = Seed(m, exp.length, seed, means)
    info = model.run_info(m, str(exp.path), seed, exp.length, s.clean, s.corrupt)
    info["intervention"] = ["mean-ablation-control", "patch-denoise-control"]
    info["controls"] = [str(s.k)]
    info["mean_over"] = [f"twins of finding seeds {' '.join(map(str, exp.finding_seeds))}"]
    info["baseline"] = [repr(s.ld["clean"])]
    info["ld_corrupt"] = [repr(s.ld["corrupt"])]
    axes = ("control", "layer", "head")
    arrays = {"mean_control": (axes, s.all("mean")), "denoise_control": (axes, s.all("denoise"))}
    return run.write(out, info, arrays)
