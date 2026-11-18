"""Runs a model through TransformerLens 4.x and captures its attention (docs/design.md, D6).

Needs the `model` extra (torch, transformer-lens).
"""

import importlib.metadata as md
from dataclasses import dataclass

import numpy as np
import torch
from transformer_lens.model_bridge import TransformerBridge

from circuits_adapter.capture import Capture

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
REVISIONS = {"gpt2": GPT2_REVISION}
LIBRARIES = ("transformer-lens", "torch", "transformers")


@dataclass
class Loaded:
    name: str
    revision: str
    device: str
    bridge: TransformerBridge

    @property
    def bos(self) -> int:
        return int(self.bridge.tokenizer.bos_token_id)

    @property
    def n_layers(self) -> int:
        return int(self.bridge.cfg.n_layers)


def load(name: str, revision: str | None = None, device: str = "cpu") -> Loaded:
    """Boot the model at a pinned revision (raw Hugging Face numerics, no compatibility mode)."""
    revision = revision or REVISIONS.get(name)
    if revision is None:
        raise ValueError(f"no pinned revision for {name}: pass one")
    bridge = TransformerBridge.boot_transformers(name, device=device, revision=revision)
    return Loaded(name, revision, device, bridge)


def _info(m: Loaded, ids: list[int], seed: int, prompt: str) -> dict:
    tok = m.bridge.tokenizer
    return {
        "model": [m.name],
        "revision": [m.revision],
        "library": [f"{lib} {md.version(lib)}" for lib in LIBRARIES],
        "device": [m.device],
        "dtype": ["float32"],
        "seed": [str(seed)],
        "prompt_kind": ["text" if prompt else "repeated-random"],
        "prompt": [prompt],
        "token_ids": [str(t) for t in ids],
        "tokens": [tok.decode([t]) for t in ids],
    }


def capture(m: Loaded, ids: list[int], seed: int, layers=None, prompt: str = "") -> Capture:
    """Run the tokens once and capture attention patterns and the last position's logits."""
    layers = list(range(m.n_layers)) if layers is None else list(layers)
    torch.manual_seed(seed)
    names = [f"blocks.{n}.attn.hook_pattern" for n in layers]
    tokens = torch.tensor([ids], device=m.device)
    with torch.no_grad():
        logits, cache = m.bridge.run_with_cache(tokens, names_filter=lambda h: h in names)
    attn = np.stack([cache[h][0].float().cpu().numpy() for h in names])
    last = logits[0, -1].float().cpu().numpy()
    return Capture(
        attn.astype(np.float32), last.astype(np.float32), _info(m, ids, seed, prompt), layers
    )


def encode(m: Loaded, text: str) -> list[int]:
    """BOS, then the text's tokens."""
    return [m.bos, *m.bridge.tokenizer.encode(text, add_special_tokens=False)]
