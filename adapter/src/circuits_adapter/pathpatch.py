"""Path patching (docs/design.md, D20): a sender's clean output reaches one receiver only.

The run is the twin. A sender's difference is what it writes into the residual
stream on the clean prompt minus on the twin: a head's z @ W_O (its share of the
attention output), the embeddings' input to layer 0. A receiver reads the twin's
residual stream plus that difference, and nothing else changes:

  layer.head.q|k|v   that head's query, key or value, computed from
                     LayerNorm1(resid_pre + difference) (the other heads untouched)
  mlp layer          the MLP's input, LayerNorm2's input + difference (internal:
                     not an edge of the graph, used to check the method)
  logits             the final LayerNorm's input + difference

Everything after the receiver runs as usual. The effect is the fraction of the
clean-twin gap restored (D16). Senders are batched: one forward pass of B copies
of the twin gives one receiver's effect for B senders.
"""

import numpy as np
import torch

from circuits_adapter import graph, metric, prompts, twins

QKV = {"q": ("W_Q", "b_Q"), "k": ("W_K", "b_K"), "v": ("W_V", "b_V")}


def layer_norm(x: torch.Tensor, ln) -> torch.Tensor:
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + 1e-5) * ln.weight + ln.bias


class Paths:
    """A seed's twins, the senders' differences, and runs of the twin with receivers patched."""

    def __init__(self, m, length: int, seed: int, twin: str = "second"):
        self.m, self.length, self.twin = m, length, twin
        self.clean, self.corrupt = prompts.twins(seed, length, m.bos)
        twins.check(self.clean, self.corrupt, length)
        # The metric's wrong tokens come from the fresh-second-copy twin either way (D20).
        self.run_ids = self.corrupt
        if twin == "first":
            self.run_ids = prompts.key_twin(seed, length, m.bos)
            twins.check_key(self.clean, self.run_ids, self.corrupt, length)
        elif twin != "second":
            raise ValueError(f"twin {twin!r} is not second or first")
        names = {f"blocks.{n}.attn.hook_z" for n in range(m.n_layers)} | {"blocks.0.hook_resid_pre"}
        self.cache = {}
        for kind, ids in (("clean", self.clean), ("corrupt", self.run_ids)):
            with torch.no_grad():
                _, c = m.bridge.run_with_cache(
                    self._tokens(ids, 1), names_filter=lambda h: h in names
                )
            self.cache[kind] = {k: v[0].clone() for k, v in c.items()}
        self.ld_clean = self._ld(self.clean, 1, [])[0]
        self.ld_corrupt = self._ld(self.run_ids, 1, [])[0]

    def _tokens(self, ids, b):
        return torch.tensor([ids] * b, device=self.m.device)

    def _ld(self, ids, b, hooks) -> np.ndarray:
        with torch.no_grad():
            out = self.m.bridge.run_with_hooks(self._tokens(ids, b), fwd_hooks=list(hooks))
        lg = out.float().cpu().numpy()
        return np.array(
            [metric.logit_difference(o, self.clean, self.corrupt, self.length).mean() for o in lg]
        )

    def difference(self, s: graph.Sender) -> torch.Tensor:
        """What the sender writes on the clean prompt minus on the twin: position x d_model."""
        if s.layer < 0:
            key = "blocks.0.hook_resid_pre"
            return self.cache["clean"][key] - self.cache["corrupt"][key]
        z = f"blocks.{s.layer}.attn.hook_z"
        dz = self.cache["clean"][z][:, s.head] - self.cache["corrupt"][z][:, s.head]
        return dz @ self.m.bridge.W_O[s.layer, s.head]

    def hooks(self, patches: list) -> list:
        """Hooks for (receiver, difference) pairs; a difference is batch x position x d_model."""
        out, resid = [], {}
        by_layer: dict = {}
        for r, d in patches:
            if r == "logits":
                out.append(("ln_final.hook_in", _adder(d)))
            elif isinstance(r, tuple) and r[0] == "mlp":
                out.append((f"blocks.{r[1]}.ln2.hook_in", _adder(d)))
            else:
                by_layer.setdefault(r.layer, []).append((r, d))
        for layer, items in by_layer.items():
            out.append((f"blocks.{layer}.hook_resid_pre", _keeper(resid, layer)))
            for kind in "qkv":
                mine = [(r, d) for r, d in items if r.input == kind]
                if mine:
                    out.append(
                        (f"blocks.{layer}.attn.hook_{kind}", self._qkv(resid, layer, kind, mine))
                    )
        return out

    def _qkv(self, resid: dict, layer: int, kind: str, items: list):
        b = self.m.bridge
        w, bias = (getattr(b, n)[layer] for n in QKV[kind])
        ln = b.blocks[layer].ln1._original_component

        def hook(x, hook):
            for r, d in items:
                xn = layer_norm(resid[layer] + d, ln)
                x[:, :, r.head, :] = xn @ w[r.head] + bias[r.head]
            return x

        return hook

    def effect(self, patches: list, batch: int) -> np.ndarray:
        """The fraction of the gap restored by running the twin with the patches."""
        ld = self._ld(self.run_ids, batch, self.hooks(patches))
        return (ld - self.ld_corrupt) / (self.ld_clean - self.ld_corrupt)


def _adder(d):
    def hook(x, hook):
        return x + d.to(x.dtype)

    return hook


def _keeper(store: dict, layer: int):
    def hook(x, hook):
        store[layer] = x.clone()
        return x

    return hook


TWIN_OF = {"q": "second", "k": "first", "v": "second", "": "second"}  # by receiver input (D20)


def scores_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_attn_scores"


class Seed:
    """Both twins of a seed and every receiver's effects, by the metric of its input (D20)."""

    def __init__(self, m, length: int, seed: int):
        self.m, self.length, self.seed = m, length, seed
        self.paths = {t: Paths(m, length, seed, t) for t in ("second", "first")}
        q = metric.positions(length)
        self.q, self.target = q, q - length + 1
        self.base = {t: self._scores(p.run_ids, []) for t, p in self.paths.items()}
        self.clean = self._scores(self.paths["second"].clean, [])

    def _scores(self, ids, hooks, batch: int = 1, layer: int | None = None) -> dict:
        """Each layer's (or one layer's) target attention scores: batch x head."""
        layers = range(self.m.n_layers) if layer is None else [layer]
        store = {}

        def spy(n):
            def hook(x, hook):
                store[n] = x[:, :, self.q, self.target].mean(-1).detach().float().cpu().numpy()
                return x

            return hook

        spies = [(scores_name(n), spy(n)) for n in layers]
        with torch.no_grad():
            self.m.bridge.run_with_hooks(
                torch.tensor([ids] * batch, device=self.m.device), fwd_hooks=hooks + spies
            )
        return store

    def effects(self, r: graph.Receiver, senders: list) -> np.ndarray:
        """The effect of each sender on receiver r (a fraction of r's gap)."""
        twin = TWIN_OF[r.input]
        p = self.paths[twin]
        d = torch.stack([p.difference(s) for s in senders])
        if r.input in ("q", "k"):
            got = self._scores(p.run_ids, p.hooks([(r, d)]), len(senders), r.layer)[r.layer]
            base = self.base[twin][r.layer][0, r.head]
            return (got[:, r.head] - base) / (self.clean[r.layer][0, r.head] - base)
        target = "logits" if not r.input else r
        return p.effect([(target, d)], len(senders))

    def matrix(self, inputs=graph.INPUTS) -> tuple[list, list, np.ndarray]:
        """Senders, receivers and the sender x receiver effects (0 where no edge)."""
        n_layers, n_heads = self.m.n_layers, self.m.bridge.cfg.n_heads
        edges = graph.all_edges(n_layers, n_heads, inputs)
        senders = sorted({s for s, _ in edges})
        receivers = sorted({r for _, r in edges})
        out = np.zeros((len(senders), len(receivers)))
        for j, r in enumerate(receivers):
            mine = [i for i, s in enumerate(senders) if s.layer < r.layer]
            out[mine, j] = self.effects(r, [senders[i] for i in mine])
        return senders, receivers, out


def write(out, m, exp, seed: int):
    """A run directory with the sender x receiver path-patching effects for one seed."""
    from circuits_adapter import model, run

    s = Seed(m, exp.length, seed)
    senders, receivers, e = s.matrix()
    p = s.paths["second"]
    info = model.run_info(m, str(exp.path), seed, exp.length, p.clean, p.corrupt)
    info["intervention"] = ["path-patch-denoise"]
    info["key_twin_ids"] = [str(t) for t in s.paths["first"].run_ids]
    info["metric_q_k"] = ["fraction of the receiver's attention-score gap at the induction target"]
    info["metric_v_logits"] = ["fraction of the clean-twin logit-difference gap"]
    info["senders"] = [str(x) for x in senders]
    info["receivers"] = [str(x) for x in receivers]
    return run.write(out, info, {"edge_effect": (("sender", "receiver"), e)})
