# Reference material

What later work builds on: the paper's method, the authors' code, and
the TransformerLens API for GPT-2 Small. Statements are marked
**[paper]** (checked against `paper/main.tex` in the authors'
repository), **[code]** (checked against their source), **[probe]**
(run on this machine) or **[unverified]** (from `docs/research.txt` or
memory, not yet checked).

## Sources

| Source | Where | Version |
| ------ | ----- | ------- |
| "A Circuit for In-Context Learning", Andrew Gao and William Gao (arXiv:2611.05344) | `paper/main.tex` in the repository below | as of the commit below |
| in-context-learning-circuit (MIT) | <https://github.com/andrewgcodes/in-context-learning-circuit>, cloned to `third_party/` (gitignored) | `26c44caf8f9b5a5a197988095f1f76b0562bbaa4` (2026-11-03) |
| TransformerLens | PyPI `transformer-lens` | 4.4.0 (with torch 2.15.0, transformers 5.23.0) |
| GPT-2 Small | Hugging Face `gpt2` | revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` |

To recreate the clone:

```bash
git clone https://github.com/andrewgcodes/in-context-learning-circuit.git \
  third_party/in-context-learning-circuit
git -C third_party/in-context-learning-circuit checkout 26c44caf8f9b5a5a197988095f1f76b0562bbaa4
```

## The paper's method

**Task.** Few-shot sentence classification with arbitrary labels
(`foo`/`bar`, rare words such as `zorp`/`quil`, digits, letters, and
natural words). Prompts are `x -> y` lines followed by a query
`x ->`. Up to twelve demonstrations (eighteen in one prediction test).
**[paper]**

**Models.** Qwen3-4B-Base (36 layers, 32 query heads, 8 KV groups,
QK-norm), SmolLM3-3B-Base (36 layers, 16 heads, 4 groups),
Llama-3.2-3B (28 layers, 24 heads, 8 groups). Intermediate checkpoints
of OLMo-2 1B and 7B are used for emergence, and Qwen3 0.6B-8B for scale.
**[paper]**

**Metric.** The margin: the log-probability of the correct label minus
that of the other label, each over its full token sequence, in nats.
An intervention's effect is the paired change in margin, with
bootstrap intervals. **[paper] [code: `icl/effects.py`]**

**Twin prompts.** Pairs that differ in one controlled way, with the
same token length and label positions: labels exchanged, one sentence
replaced by one of the other class, one sentence replaced by another of
the same class (a control), unique instead of repeated labels.
**[paper] [code: `tasks.add_twins`]**

**The circuit**, as the paper states it **[paper]**:

| Part | What it does |
| ---- | ------------ |
| slot | each label token's residual stream holds the label identity (value) and the class of the preceding sentence (key) |
| writers | about twenty heads in layers 10-23 attend from the label token back to its sentence and write the class in as the key |
| query code | heads at the query arrow encode the query sentence's class |
| readers | upper-layer heads match the query's class against the stored keys and copy out the matching labels |
| gate | early duplicate-token heads detect that a label has repeated; the signal is pooled into one residual direction at the query that lowers the readers' logit on the attention sink at position 0 |
| null action | without the gate the readers attend to the sink and no label is favored |
| brakes | heads that suppress the most recently seen label (recency bias) |
| copiers | late heads that write the answer logits |

**Keys against values.** In Qwen, patching the *values* of the 72 KV
groups of layers 21-29 at the label tokens, from a prompt with the
labels exchanged, carries 86% of the exchange's effect on the margin.
Replacing a sentence with one of the other class moves the margin by
-2.87 nats on the six-demonstration battery. Patching the changed
prompt's *keys* at the label tokens in layers 21-29 carries -2.35 of
that. A same-class replacement moves it by only +0.08. **[paper]**

**The read law.** For a reader, the attention mass on matching slots
is

```
m = n_m e^L / ( n_m e^L + (N - n_m) + n_rest + e^L_sink )
```

where `n_m` is the number of matching slots, `N` the number of slots,
`L` the matching-slot logit, `L_sink` the sink logit, and `n_rest` the
other mass. Fitted on twelve-demonstration prompts, it transfers to
fewer slots per class. It fails on eighteen demonstrations: median
errors were 0.15 (Qwen) and 0.13 (Llama), against a 0.07 criterion.
The cause is that a head's sink mass and class budget are fixed, so
readers must act together. **[paper]**

**Sufficiency.** Keeping only a 79-head set at the query, with the
others replaced by reference activations, keeps about 60% of the
margin. The 189-head set (adding natural-label readers and brakes)
beats the intact model. The choice of reference activations matters a
lot. These tests hold the query side; the slots are built normally.
**[paper]**

**Generality.** Similar roles appear in SmolLM3 and Llama, filled by
different heads. Across OLMo-2 checkpoints, copying a repeated label
appears first, then the null action, then the readers' class
selectivity. The 1B model never develops readers. **[paper]**

**What does not bind.** Word parity, the third letter of a word, and
digit rules never bind, because they are not represented linearly at
the arrow. Writing the class direction into the slots by hand raises
accuracy from 50% to 71%. **[paper]**

**From scratch.** A four-layer, eight-head Llama-architecture model
(3.4M parameters) trained on a synthetic task grows the same parts. It
grows the gate only when retrieval is penalized. **[paper]**

## The authors' code

It uses Hugging Face `transformers` with PyTorch forward hooks, *not*
TransformerLens. It loads models in bfloat16 on CUDA (float32 on CPU)
with `attn_implementation='eager'` so attention weights are returned.
**[code]**

Dependencies (`requirements.txt`): torch>=2.3, transformers>=4.51,
numpy, matplotlib, modal, datasets, scipy. Mean ablation uses
`NeelNanda/pile-10k` (300 documents) unless `ICL_DOCS` names a JSON
list of strings. CI runs on CPU: compile, `tests/`, figures. **[code]**

**Memory.** A 4B model in BF16 is about 8 GB of weights. A full
battery takes about an hour on one H100. Whether full sweeps fit in
24 GB on the RTX 3090 is **[unverified]**.

### Runner: `icl/run.py`

```bash
python icl/run.py <hf-model-id> <families> [key=value ...]
python icl/run.py Qwen/Qwen3-4B-Base concept12 seed=1 smoke out=runs/qwen-smoke
python icl/check.py runs/qwen-smoke          # run.py also calls it at the end
python icl/effects.py effects.tsv runs/qwen-smoke
```

The pipeline, in order: build the repeat-flag direction `U[l]` from
paired prompts; set the mean-ablation means and natural scale from
natural text; build the battery; compute look-back and induction
scores per head on every sequence; run every condition per prompt
(the `none` condition is intact); write rows. **[code]**

Options **[code]**:

| Option | Meaning |
| ------ | ------- |
| `out=` | output directory (or `$GRAIL_OUT`; default `runs/<model>`) |
| `seed=` | battery seed (a fresh prompt draw for held-out confirmation) |
| `smoke` | a few prompts per task (`SMOKE_TASKS`) |
| `intact` | no interventions |
| `rev=` | Hugging Face revision (training checkpoints) |
| `heads=l:h,...` | mean-ablate the named heads, plus size-matched random controls; `heads_each`, `heads_pairs` |
| `knockpos=query`, `knockzero` | where to ablate; zero instead of mean |
| `dir=mlp\|attn\|both[:bands]` | project the flag direction out of late outputs (plus a random-direction control) |
| `kpath=`, `patch=`, `psrc=`, `patchlayers=`, `patchcum=` | path-specific removal and activation patching of reader inputs |
| `twin=`, `twinset=` | Q/K/V/output handoff from a twin prompt |
| `circ=<heads>` with `circmean=battery\|kind` or `circrs` | keep only the named heads at the query; the others get mean or resampled activations |
| `circbands=`, `circscreen=` | cut or screen the complement one layer band at a time |
| `radd=<npz>:<layer>:<scale>`, `dadd=`, `sinkv=`, `rkcond=`, `raknock`, `hgain=` | steering, sink-value and gain interventions |
| `attrow=`, `srow=`, `arowk=` | attention-row loupes |
| `trace=<n>` | token-observatory shards (read with `icl/trace_view.py`) |
| `kinds=`, `every=`, `order=`, `ndemo=` | prompt subsets and order or count controls |

Outputs in `out/` **[code]**:

- `run_meta.json`: model, layers, softmax layers, head count, device,
  dtype, BOS id, argv.
- `tagscope.jsonl`: per sequence, its kind, token surprisal, flag
  readout, label spans.
- `knock.jsonl`: one row per (prompt, condition), with `tlp`
  (target log-prob), `alt_tlp`, `tlp1` (first token), and, depending on
  options, `qatt` (query attention on label spans), `arow`, `srow`,
  `dla` (per-head direct logit attribution), `mdla`, `geom`, and the
  top-5 predictions.
- `run_summary.json`: flag-direction statistics.
- `run.status`: progress.

### Modules

| File | Content **[code]** |
| ---- | ------------------ |
| `icl/tasks.py` | `build_battery(tok, families, smoke)` returns `Item(kind, ids, spans, target, alt, meta, twins)`. Families include `std`, `redund`, `correct`, `bind`, `ctl`, `concept`, `concept12`, `concept12n`, `labelswap`, `manyclass`, `markcorrect`, `func*`, `functask*`, `slotgate*`, `mixed`. Label sets: `sym` foo/bar, `num` 7/3, `word` good/bad, `rare` zorp/quil, `letter` A/B, plus reversed versions |
| `icl/tokspans.py` | label spans from the full-prompt tokenization (skips a bare-space token, asserts the decoded span equals the label); target ids as the model sees them after the prompt |
| `icl/hooks.py` | the `Hooks` class: captures key inputs, post-norm q/k and output-projection inputs; head mean ablation; Q/K/V/output patching; residual pushes and holds; direction removal. Handles Qwen/Llama (`k_proj`), GPT-2 (`c_attn`) and GPT-NeoX layouts |
| `icl/score.py` | `seq_logprob`, `surprisal`, `flag_readout`, `query_attention` (asserts each attention row sums to 1, with tolerance 1e-3 fp32 / 2e-2 bf16), `dup_head_scores` (look-back and induction scores), `query_row_top`, `mlp_direct`, `slot_rows` |
| `icl/loupes.py` | MLP neuron writers onto a direction, the split of a reader's attention logit along the flag direction, value readout through a head's OV |
| `icl/effects.py` | per run: intact margin and each condition's paired effect with a 4000-sample bootstrap 95% interval, written as TSV |
| `icl/check.py` | invariants: expected kinds present, equal rows per condition, every row has an intact twin, `tlp <= tlp1`, non-blank targets, attention-row positions inside the prompt |
| `icl/trace_view.py` | `trace_view.py OUT/trace/000_none.npz [second.npz] [--pos=-1] [--top=6]`: layer by layer, at one position, the residual norm, projections on the flag direction and gate, and the heads writing most and where they attend; with two shards, the differences |
| `icl/allheads_law.py`, `predict_ndemo18.py`, `fit_gate.py`, `analyze_*.py` | read-law fits, the eighteen-demonstration prediction, gate-direction fit, analyses of specific runs |
| `toy/toy_circuit.py` | a three-layer attention-only transformer with hand-set weights implementing the circuit; runs on CPU in seconds |
| `toy/train_toy.py`, `analyze_toy.py` | the from-scratch Llama toy (`--cond mix\|noone\|gate`); about ten minutes on one GPU |

`dup_head_scores(att, ids)` defines the scores Saga 1 needs. For every
token that occurred earlier, the look-back score is the attention mass
on its earlier copies, and the induction score is the mass on the
tokens right after those copies. Both are averaged over such tokens,
per head. **[code]**

## TransformerLens 4.x for GPT-2 Small

TransformerLens 4.0 removed `HookedTransformer.from_pretrained`.
Models load through `TransformerBridge`. By default its weights,
logits and activations match Hugging Face's.
`enable_compatibility_mode()` reproduces the old `HookedTransformer`
processing (LayerNorm folding, centered weights). **[probe]**

```python
from transformer_lens.model_bridge import TransformerBridge
bridge = TransformerBridge.boot_transformers("gpt2", device="cpu", revision=REV)
bridge.set_use_attn_result(True)          # only if per-head outputs are needed
logits, cache = bridge.run_with_cache(tokens)
```

Hook names and shapes for a batch of 1 and T tokens (GPT-2 Small:
12 layers, 12 heads, d_model 768, d_head 64). The legacy names still
work. **[probe]**

| Hook | Shape | Content |
| ---- | ----- | ------- |
| `blocks.{l}.attn.hook_pattern` | [1, H, T, T] | attention weights (query, key) |
| `blocks.{l}.attn.hook_attn_scores` | [1, H, T, T] | pre-softmax scores |
| `blocks.{l}.attn.hook_q`, `hook_k`, `hook_v` | [1, T, H, 64] | per-head projections |
| `blocks.{l}.attn.hook_z` | [1, T, H, 64] | per-head attention output before W_O |
| `blocks.{l}.attn.hook_result` | [1, T, H, 768] | per-head write into the residual stream; only with `set_use_attn_result(True)`; heads plus `b_O` sum exactly to `hook_attn_out` |
| `blocks.{l}.hook_resid_pre`, `hook_resid_mid`, `hook_resid_post` | [1, T, 768] | residual stream |
| `blocks.{l}.hook_attn_out`, `hook_mlp_out` | [1, T, 768] | sublayer outputs |
| `blocks.{l}.mlp.hook_pre`, `hook_post` | [1, T, 3072] | MLP hidden, before and after the activation |
| `hook_embed`, `hook_pos_embed` | [1, T, 768] | embeddings |
| `ln_final.hook_normalized` | [1, T, 768] | final normalized residual |
| logits (return value), `unembed.hook_out` | [1, T, 50257] | logits |

The bridge also exposes new-style names (`blocks.{l}.attn.q.hook_out`,
`blocks.{l}.hook_in`, `blocks.{l}.mlp.in.hook_out`, ...): 493 hooks in
all for GPT-2.

### Checking an install

The probe used a throwaway uv environment with
`transformer-lens==4.4.0`, on an Apple-silicon Mac. Its input was BOS
followed by 50 random tokens (ids 1000-9999, seed 0), repeated twice.
**[probe]**

- CPU, MPS (`torch.backends.mps.is_available()` true), and
  compatibility mode gave the same scores. Attention rows summed to 1
  within 5e-7.
- Induction score (attention from the second copy of a token to the
  token after its first copy): 7.10 0.913, 5.1 0.910, 5.5 0.910,
  6.9 0.896, 7.2 0.835, then 10.1 0.518.
- Previous-token score: 4.11 0.989, then 3.7 0.538.
- Mean log-probability on the second repeat: -0.268 nats, so the model
  predicts the repeat.

These match the GPT-2 Small induction heads (5.1, 5.5, 6.9, 7.2, 7.10)
and previous-token head (4.11) reported in the interpretability
literature **[unverified citation]**. They are attention-pattern scores
only: candidates, not causal evidence.
