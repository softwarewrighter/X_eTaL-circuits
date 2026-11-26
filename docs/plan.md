# X_eTaL-circuits -- Implementation Plan

X_eTaL-circuits is a workbench for finding, testing and drawing
*circuits* inside language models: the groups of attention heads,
MLP features and information paths that work together to produce a
particular behavior. X_eTaL does the analysis and drawing (tensor
slicing, causal-effect matrices, statistics, graph discovery,
heatmaps and circuit graphs). A thin model adapter runs the model
and exposes its internals.

Source material: `docs/research.txt` (an archived design
conversation, not normative). It covers arXiv:2611.05344, "A Circuit
for In-Context Learning" (Gao and Gao, 2026), and its MIT-licensed
code at <https://github.com/andrewgcodes/in-context-learning-circuit>.
It also covers induction heads in GPT-2 Small and the introspectable
Baby Dragon Hatchling (BDH) architecture
(<https://github.com/pathwaycom/bdh>).

Development is driven by agentrail sagas: one active saga in
`.agentrail/`, with finished ones archived to `.agentrail-archive/`.
Each saga delivers one milestone you can demonstrate. Every step is
test-first and ends with the checks and a commit before
`agentrail complete`. Only Saga 1 is loaded into agentrail now. Each
later saga is created when the one before it closes, and its step
list is revised then in light of what was learned.

## Principles

These come from the research and apply to every saga.

1. **Observation is not causation.** Attention weights, activation
   magnitudes and sparse "lit up" neurons are hypotheses. A component
   is in a circuit only after an intervention (patching, ablation,
   steering) shows its causal effect. That effect must be measured
   against matched random controls and checked on held-out prompts.
2. **A continuous metric over accuracy.** Measure the logit or
   log-probability margin, `m = log P(correct) - log P(incorrect)`,
   wherever the backend exposes it. Never treat generated-text
   "confidence" as a probability.
3. **Twin prompts.** Experiments use prompt pairs that differ in
   exactly one controlled way (label swap, class swap, query swap,
   unique labels) and whose tokens line up position by position.
   Record and check the tokenization.
4. **Reproducibility.** Every run records the model id and revision,
   dtype or quantization, backend, seed, sampling settings, context
   length and the exact prompt. The subject model never sees an
   agent's system prompt, tools or history.
5. **The analysis is in X_eTaL. The model hooks are not, at first.**
   Run the model through TransformerLens or the paper authors' Python
   hooks. Do not rewrite their mechanistic code until our results
   match theirs.
6. **Big tensors travel as binary.** Small summaries and scores go as
   JSON. Activation tensors go as safetensors (or Arrow IPC), with
   metadata naming the axes, dtype, token positions and model
   revision.
7. **Start small and known.** GPT-2 Small (12 layers x 12 heads,
   with well-studied induction heads) validates the tooling. Then
   Qwen3-4B-Base, the paper's reference model. No quantized or MoE
   models until the tooling is validated.

## Where things live

| Repository          | What belongs there                                                   |
| ------------------- | -------------------------------------------------------------------- |
| X_eTaL-circuits     | this workbench: the adapter, the experiment specifications, the studies, the explorer |
| X_eTaL-libraries    | general-purpose pieces found along the way: heatmaps, directed-graph layout, statistics |
| X_eTaL-extensions   | native Rust needed by X_eTaL: the array-file reader for XTLARR01 and safetensors (D1, D7), later an adapter client |
| X_eTaL-gpu          | the XTLARR01 format and its Rust reader; GPU runs of large sweeps and forward passes |
| X_eTaL-ML           | ML libraries reused here (Attention, Norm, NN) and the in-context-learning teaching demos (D5) |
| X_eTaL              | language changes, if a study needs one (decided with the user there) |

Code starts here. A piece moves to a library or an extension only
once a second user of it exists. The exception is the array-file
reader, which already has two users (D3).

In-context learning is split with X_eTaL-ML (D5). Its proposed
"in-context learning" saga keeps the teaching demos: a hand-built ICL
circuit on its Attention library, a toy trained in X_eTaL, and browser
pages. Sagas 4-7 here keep the experiments on real models and the
causal tooling, and point to those demos rather than repeating them.

This repository never edits another (D9). What it needs from one is
an ask in that repository's ledger here, which that repository's own
plan takes:

- [`xetal-extensions-asks.md`](xetal-extensions-asks.md): X1, the
  array-file extension (XTLARR01 and safetensors); X2, the shared
  crate.
- [`xetal-gpu-asks.md`](xetal-gpu-asks.md): G1, switch to the shared
  crate; G2, captures as GPU inputs.
- [`xetal-ml-asks.md`](xetal-ml-asks.md): M1, the in-context-learning
  split; M2, reuse of its libraries.
- [`xetal-libraries-asks.md`](xetal-libraries-asks.md): none yet.

## Hardware

| Host            | Role                                                             |
| --------------- | ---------------------------------------------------------------- |
| RTX 3090 24GB   | main mechanistic host: Qwen3-4B-Base in BF16, patching sweeps     |
| RTX 5060 16GB   | behavioral runs, smaller models                                   |
| RTX 3060 12GB   | quantized inference; the coding agent's own model                 |
| M1 Max 64GB     | GPT-2 Small work, larger-memory inference, slower mechanistic runs |
| M3 Pro 18GB     | client, X_eTaL analysis, secondary tests                          |

GPT-2 Small runs anywhere, including on a CPU, so Sagas 1-3 need no
GPU host.

## Saga 1 -- attention-viewer (Milestone 1)  [DONE]

Goal: capture GPT-2 Small's attention on a repeated-token induction
prompt, load it into X_eTaL, and show a layer/head-selectable heatmap.
Find the induction heads by an attention-pattern score. The numbers
must agree with TransformerLens. Draw the whole model as a map, with
the scores and a candidate circuit on it, and publish every picture as
live pages.

| #  | Step slug           | Delivers                                                                |
| -- | ------------------- | ----------------------------------------------------------------------- |
| 1  | scaffold            | repo layout, justfile, `.gitignore`, uv Python project for the adapter, checks script, CLAUDE.md project section |
| 2  | reference-survey    | `docs/reference.md`: the paper's method and the authors' repo layout and runner interface (pinned commit); TransformerLens hook names for GPT-2 |
| 3  | xetal-ingest        | decisions D1-D6 in `docs/design.md`: XTLARR01 arrays read through an X_eTaL-extensions extension, TOML metadata, f32 captures, the split with X_eTaL-ML |
| 4  | adapter-capture     | Python adapter: load GPT-2 Small, write a capture directory (attn.xta, logits.xta, capture.toml); an XTLARR01 writer; a committed fixture |
| 5  | arrayfile-extension | in ../X_eTaL-extensions: the extension reading XTLARR01 and safetensors files (D7: whole array, one first-axis slice, shape, tensor names), sharing X_eTaL-gpu's reader |
| 6  | text-arrays         | for now (D10): the adapter also writes each array as text numbers that X_eTaL reads with `n_umbers []N_GET`, exactly equal to the binary files; X_eTaL pinned (`XETAL_COMMIT`, `just xetal`) |
| 7  | xetal-load          | X_eTaL reads and validates a capture: axes layer head query key, token labels, metadata; selection helpers |
| 8  | attention-heatmap   | one layer/head drawn with `[]N_GRID` and token labels; checked against TransformerLens numbers |
| 9  | induction-score     | induction and previous-token scores swept over L x H in X_eTaL; the ranked heads match known GPT-2 induction heads |
| 10 | model-map           | one picture of the model: a row per layer, a cell per head and per MLP, scores as color, a circuit overlay drawn with the evidence key (D13) |
| 11 | pages               | GitHub Pages site built from committed captures: every heatmap with a layer/head picker, the score grids, the model map, the ELI5 (D12); a README screenshot linking to the demo; closes the saga |

Outcome. GPT-2 Small at the pinned revision gives, from attention
patterns alone, exactly the induction heads the literature names (5.1,
5.5, 6.9, 7.2, 7.10 lead the induction score) and 4.11 as the
previous-token head; X_eTaL's values agree with TransformerLens's cache
within 1e-5 relative. Step 5 (the array-file extension) stays reserved
for X_eTaL-extensions; the text route (D10) carried GPT-2 sizes. What
was learned for later sagas:

- Moving the X_eTaL pin can break the libraries (QD9 broke relative
  capture paths, D11): move it on purpose, with the tests.
- A 16-token repeat (33 positions) is enough to find the induction
  heads; the committed captures stay a few MB.
- Pictures stay data-driven: a circuit is a TOML file, and its edges
  change style as evidence comes in (D13). Every new picture goes on the
  live pages (D12) with a test that keeps the README's copy current.

## Saga 2 -- causal-experiments (Milestone 2)  [DONE]

Goal: test causal importance, not just patterns, on GPT-2 Small's
induction behavior. Twin prompts (clean and corrupted, token-aligned),
head ablation and activation patching give an L x H effect matrix `D`
in X_eTaL, compared with matched random controls and checked on
held-out seeds. Heads that pass become "tested" or "confirmed" on the
model map (D13). Edges stay "observed": ablating a head tests the
head, not which head feeds which; path patching (Saga 3) tests edges.

| # | Step slug          | Delivers |
| - | ------------------ | -------- |
| 1 | twin-prompt-spec   | an experiment file in experiments/; the corrupted twin (second copy replaced by fresh random tokens, aligned); the margin metric decided in design.md; per-position metric in the capture |
| 2 | adapter-ablate     | zero and mean ablation of each head's output; the L x H change in the metric per seed |
| 3 | adapter-patch      | clean-into-corrupted and corrupted-into-clean patching per layer, head and position, as the fraction of the gap recovered |
| 4 | effect-matrix      | xetal/Effects: D per method, mean and spread over seeds, ranking, diverging grids; `just effects`; README and live pages |
| 5 | random-controls    | matched random heads and directions; a head's effect is real only outside its controls (threshold in design.md) |
| 6 | heldout-validation | the passing heads rerun on held-out seeds: "confirmed" when they pass there too, "tested" otherwise |
| 7 | circuit-evidence   | per-head evidence in the circuit file and on the map; the verdicts written in; pages and screenshot updated; closes the saga |

Captures with logits per position and per-head outputs need the
adapter's capture extended; MLP cells get a score once MLP outputs are
captured.

Outcome. On GPT-2 Small's repeated-token task, with matched random
controls and a 5% floor (D17) and ten held-out seeds (D18):
mean ablation confirms 5.1 (22% of the clean logit difference), 4.11
(10-13%), 7.6, 8.5 and 5.5; denoising confirms the five induction
heads (6.9, 5.1, 7.2, 5.5, 7.10: 8-15% of the gap each) and 9.9, 9.6,
10.1 and 0.1. Effects agree within about a point between seed sets.
Single heads matter little except 5.1: the heads back each other up.
10.7 and 11.10 restore negatively (they work against copying). All
twelve heads are confirmed on the model map; its edges stay observed.
What was learned for later sagas:

- Beating random controls is not enough when they average to 0 over
  seeds: a minimum effect is needed, fixed before the held-out seeds.
- Batching the controls (20 copies of the prompt per forward pass)
  makes them cheap; a full set of runs for 20 seeds takes about an hour
  on CPU.
- The circuit file is checked against the runs by a test, so the map
  cannot claim more than the runs show.

## Saga 3 -- circuit-discovery (Milestone 3)  [ACTIVE]

Goal: test the edges of the induction circuit in GPT-2 Small, which
Saga 2 left observed: does 4.11 feed the induction heads, and through
which input (query, key or value), and which heads do 7.6, 8.5, 9.9,
9.6, 10.1 and 0.1 read from or feed? Edges are scored by path patching
(a sender head's output patched only into a receiver's query, key or
value input), with the same twins, metric, random controls, 5% floor
and held-out seeds as Saga 2. Edge attribution patching gives a cheap
ranking first. The resulting subgraph is checked for faithfulness
(the circuit alone recovers the behavior), completeness and minimality
against random subgraphs of the same size.

Steps (revised after Saga 2): graph-model (senders, receivers and
their q/k/v inputs as data), path-patching (adapter), edge-attribution
(adapter, a ranking to choose which edges to path-patch), edge-effects
(X_eTaL: effect per edge, controls, verdicts, as Effects does for
heads), heldout-edges, faithfulness (keep only the circuit, mean-ablate
the rest), circuit-graph-view (the map draws edge evidence, and a graph
view per input; README, pages and screenshot).

Progress: graph-model (D19); path-patching (D20: the key twin and the
attention-score metric; 4.11 restores 26-46% of each induction head's
key gap on seeds 0 and 1); pages-intro (the live demo's introduction
and glossary, inserted at the user's ask); edge-attribution (D21: all
edges in seconds, Pearson 0.98/0.999 with path patching in the metric's
units; edges chosen per metric, 400 for the induction experiment).

## Saga 4 -- icl-behavior

Goal: behavioral replication of the paper's few-shot classification
experiments through an OpenAI-compatible local API (Ollama,
llama.cpp), with no internals needed. Conditions: baseline (six
demonstrations), label reversal, demonstration counts 0/1/2/4/6/12,
recency and order, unique labels, unfamiliar single-token labels,
and 3-6 classes. Use 50-100 held-out queries per condition, several
seeds and balanced labels. Measure accuracy, reversal consistency
and log-probability margins where available, with paired bootstrap
confidence intervals.

Steps (draft): rust-workspace (`icl-core`, `icl-provider`,
`icl-batteries`, `icl-eval`, `icl-cli`, `icl-report`), six-shot-baseline
against one model, label-reversal, conditions-sweep, margins,
statistics, report (JSONL/CSV export, analyzed and plotted in
X_eTaL), compare models (Qwen, SmolLM and the local coding models;
base against instruction-tuned).

## Saga 5 -- icl-mechanistic

Needs X_eTaL to read Qwen-sized captures: the array-file extension
(Saga 1's step arrayfile-extension, reserved for X_eTaL-extensions,
asks X1 and X2) is carried here as this saga's first step, reserved
until those asks land.

Goal: run the authors' code unmodified on Qwen3-4B-Base on the RTX
3090: the `icl/run.py` smoke run on `concept12`, then `check.py` and
`effects.py`. Reproduce the label-swap versus class-swap experiment:
patching *values* at label positions (layers 21-29) carries the label
exchange, and patching *keys* carries the class change. Import the
effects into X_eTaL as heatmaps. Check selected writer, reader, gate
and brake interventions against the paper's head numbers.

## Saga 6 -- xetal-icl

Goal: carry the method over to programming-language content, using
only compiler-verified X_eTaL examples.

- Experiment A: arbitrary labels on X_eTaL constructs (reductions
  against elementwise transforms), then label reversal. This is close
  to the paper's task.
- Experiment B: learning X_eTaL idioms from examples. Score with the
  compiler and linter: parse, compile, test, idiom and complexity
  warnings. A control gives consistent but undesirable idioms.

Separate four kinds of failure: understanding the construct,
associating it with the solution, retrieving the association, and
generating valid code.

## Saga 7 -- circuit-emergence

Goal: train a tiny transformer on a synthetic induction (or
key-value) task, starting from the authors' `toy/`, and save
checkpoints. At each checkpoint measure accuracy, attention patterns,
ablation effects and causal edges. Animate the circuit graph over
training in X_eTaL. Compare the order in which components appear
with the paper's OLMo-2 checkpoint findings. The training runs in
PyTorch and is measured with this repository's causal tooling. A toy
trained in X_eTaL itself belongs to X_eTaL-ML (D5).

## Saga 8 -- bdh

Goal: use the paper's causal method to test BDH's claim to be
introspectable. Build an adapter for `pathwaycom/bdh` that records
neuron activity and sparse synaptic state with deltas, and that can
snapshot and restore state so replays are exact. Run the same
associative task (`A -> X, B -> Y, A -> ?`) on a small transformer
and a BDH model of similar size. Success needs all three:
an association forms, specific state changes go with it, and
reversing or suppressing those changes alters retrieval. Then build
a synaptic-state explorer and a *circuit discoverability* comparison:
interventions needed, circuit size, faithfulness and stability.

## Open questions (decide with the user, record in docs/design.md)

- Where the explorer runs: the command line first (`xetal-x`, native
  extensions); browser pages would carry their data with the page, as
  X_eTaL-ML's do.
- Who owns the shared XTLARR01 crate (X_eTaL-extensions or X_eTaL-gpu).
- Whether the adapter stays Python (TransformerLens) or a Rust
  runtime (candle, burn) is added once results match.
- Which hosts run which sagas, and how results return to the
  analysis machine.
