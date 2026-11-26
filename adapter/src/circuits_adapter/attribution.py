"""Edge attribution patching (docs/design.md, D21): every edge's path-patching effect, linearized.

Path patching adds a sender's difference d (clean minus twin, pathpatch.Paths)
to one receiver's input and measures that receiver's metric (D20). Attribution
keeps the first-order term: d dotted with the gradient of the same metric with
respect to the receiver's input, on the same twin, over the same gap.

  q, k        the metric is the receiver head's attention score at the induction
              target, which reads only that head's query and key: the gradient
              is taken through LayerNorm1 and W_Q or W_K of that head alone, at the
              twin's residual stream (the key twin for keys)
  v, logits   one backward pass of the logit difference on the second-copy twin
              gives the gradient at every head's value and at the final
              LayerNorm's input; a value's is carried back through W_V and
              LayerNorm1 of that head

A seed costs a few forward passes and one backward pass, against one forward
pass per receiver for path patching.
"""

import numpy as np
import torch

from circuits_adapter import graph, metric, pathpatch


def ld_torch(logits: torch.Tensor, clean: list[int], corrupt: list[int], length: int):
    """The mean logit difference (metric.logit_difference) as a torch scalar."""
    q = metric.positions(length)
    right = torch.as_tensor(np.asarray(clean)[q + 1])
    wrong = torch.as_tensor(np.asarray(corrupt)[q + 1])
    q = torch.as_tensor(q)
    return (logits[q, right] - logits[q, wrong]).mean()


def head_input_grads(x: torch.Tensor, ln, w: torch.Tensor, b: torch.Tensor, g: torch.Tensor):
    """Each head's gradient at the residual stream x (pos x d_model), given its
    gradient g (head x pos x d_head) at LayerNorm(x) @ w + b (w: head x d_model x d_head)."""
    xs = x.detach().expand(w.shape[0], *x.shape).clone().requires_grad_(True)
    y = torch.einsum("hpd,hde->hpe", pathpatch.layer_norm(xs, ln), w) + b[:, None, :]
    (gx,) = torch.autograd.grad(y, xs, g)
    return gx


class Attribution:
    """A seed's twins (pathpatch.Seed) and the gradient of each receiver's metric."""

    def __init__(self, m, length: int, seed: int):
        self.m, self.length = m, length
        self.s = pathpatch.Seed(m, length, seed)
        self.resid = {t: self._resid(p.run_ids) for t, p in self.s.paths.items()}
        self.g_v, self.g_logits = self._backward()

    def _resid(self, ids) -> dict:
        names = {f"blocks.{n}.hook_resid_pre" for n in range(self.m.n_layers)}
        with torch.no_grad():
            _, c = self.m.bridge.run_with_cache(
                torch.tensor([ids], device=self.m.device), names_filter=lambda h: h in names
            )
        return {n: c[f"blocks.{n}.hook_resid_pre"][0].clone() for n in range(self.m.n_layers)}

    def _backward(self):
        """The logit difference's gradient at every layer's hook_v and at ln_final.hook_in."""
        p = self.s.paths["second"]
        names = [f"blocks.{n}.attn.hook_v" for n in range(self.m.n_layers)] + ["ln_final.hook_in"]
        zeros = {}

        def leaf(name):
            def hook(x, hook):
                zeros[name] = torch.zeros_like(x, requires_grad=True)
                return x + zeros[name]

            return hook

        with torch.enable_grad():
            out = self.m.bridge.run_with_hooks(
                torch.tensor([p.run_ids], device=self.m.device),
                fwd_hooks=[(n, leaf(n)) for n in names],
            )
            ld = ld_torch(out[0].float(), p.clean, p.corrupt, self.length)
            grads = torch.autograd.grad(ld, [zeros[n] for n in names])
        self.ld = float(ld.detach())
        g_v = {n: g[0].permute(1, 0, 2) for n, g in enumerate(grads[:-1])}  # head x pos x d_head
        return g_v, grads[-1][0]

    def scores(self, layer: int, twin: str):
        """Each head's target attention score at layer, from the twin's residual stream by hand."""
        b = self.m.bridge
        ln = b.blocks[layer].ln1._original_component
        x = self.resid[twin][layer]
        xq = x.expand(b.cfg.n_heads, *x.shape).clone().requires_grad_(True)
        xk = x.expand(b.cfg.n_heads, *x.shape).clone().requires_grad_(True)
        q = torch.einsum("hpd,hde->hpe", pathpatch.layer_norm(xq, ln), b.W_Q[layer])
        k = torch.einsum("hpd,hde->hpe", pathpatch.layer_norm(xk, ln), b.W_K[layer])
        q, k = q + b.b_Q[layer][:, None], k + b.b_K[layer][:, None]
        s = (q[:, self.s.q] * k[:, self.s.target]).sum(-1).mean(-1) / np.sqrt(q.shape[-1])
        return s, xq, xk

    def grads(self, layer: int, kind: str) -> torch.Tensor:
        """Every head's metric gradient at its kind input of layer: head x pos x d_model."""
        if kind == "v":
            b = self.m.bridge
            ln = b.blocks[layer].ln1._original_component
            x = self.resid["second"][layer]
            return head_input_grads(x, ln, b.W_V[layer], b.b_V[layer], self.g_v[layer])
        with torch.enable_grad():
            s, xq, xk = self.scores(layer, pathpatch.TWIN_OF[kind])
            (g,) = torch.autograd.grad(s.sum(), xq if kind == "q" else xk)
        return g

    def gap(self, layer: int, kind: str) -> np.ndarray:
        """Each head's gap, as path patching divides by (pathpatch.Seed.effects)."""
        if kind in ("q", "k"):
            base = self.s.base[pathpatch.TWIN_OF[kind]][layer][0]
            return self.s.clean[layer][0] - base
        p = self.s.paths["second"]
        return np.full(self.m.bridge.cfg.n_heads, p.ld_clean - p.ld_corrupt)

    def receiver_gaps(self, receivers: list) -> np.ndarray:
        """Each receiver's gap in its metric's units: attention score, or logit difference."""
        return np.array([self.gap(r.layer, r.input)[r.head] if r.input else self.gap(0, "v")[0]
                         for r in receivers])  # fmt: skip

    def matrix(self, inputs=graph.INPUTS) -> tuple[list, list, np.ndarray]:
        """Senders, receivers and the sender x receiver attributions (0 where no edge)."""
        n_layers, n_heads = self.m.n_layers, self.m.bridge.cfg.n_heads
        edges = graph.all_edges(n_layers, n_heads, inputs)
        senders = sorted({s for s, _ in edges})
        receivers = sorted({r for _, r in edges})
        d = {
            t: torch.stack([p.difference(s) for s in senders]).detach()
            for t, p in self.s.paths.items()
        }
        col = {r: j for j, r in enumerate(receivers)}
        out = np.zeros((len(senders), len(receivers)))
        layer_of = np.array([s.layer for s in senders])
        for layer in range(n_layers):
            for kind in inputs:
                twin = pathpatch.TWIN_OF[kind]
                a = torch.einsum("spd,hpd->sh", d[twin], self.grads(layer, kind)).cpu().numpy()
                a = a / self.gap(layer, kind)
                for h in range(n_heads):
                    j = col[graph.Receiver(layer, h, kind)]
                    out[:, j] = np.where(layer_of < layer, a[:, h], 0.0)
        p = self.s.paths["second"]
        a = torch.einsum("spd,pd->s", d["second"], self.g_logits).cpu().numpy()
        out[:, col[graph.Receiver(n_layers)]] = a / (p.ld_clean - p.ld_corrupt)
        return senders, receivers, out


def write(out, m, exp, seed: int):
    """A run directory with the sender x receiver edge attributions for one seed."""
    from circuits_adapter import model, run

    a = Attribution(m, exp.length, seed)
    senders, receivers, e = a.matrix()
    p = a.s.paths["second"]
    info = model.run_info(m, str(exp.path), seed, exp.length, p.clean, p.corrupt)
    info["intervention"] = ["edge-attribution"]
    info["key_twin_ids"] = [str(t) for t in a.s.paths["first"].run_ids]
    info["metric_q_k"] = ["fraction of the receiver's attention-score gap at the induction target"]
    info["metric_v_logits"] = ["fraction of the clean-twin logit-difference gap"]
    info["gap_units"] = ["q, k: attention score; v, logits: logit difference"]
    info["senders"] = [str(x) for x in senders]
    info["receivers"] = [str(x) for x in receivers]
    arrays = {
        "edge_attribution": (("sender", "receiver"), e),
        "receiver_gap": (("receiver",), a.receiver_gaps(receivers)),
    }
    return run.write(out, info, arrays)
