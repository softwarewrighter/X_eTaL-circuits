# experiments/

Declarative experiment specifications: prompts, prompt pairs,
interventions and metrics, shared by the adapter and the X_eTaL
analysis. Outputs go to `runs/`, which is not committed.

## circuits/

Circuits to draw on the model map (`just map`), one TOML file each,
every value a list of strings: `name`, `model`, `source` (where the
circuit comes from), `heads` (layer.head by the model's numbers, from
0) with their `roles`, and `edges` ("a b": a feeds b) with one
`evidence` level each:

| Level | Drawn | Means |
| ----- | ----- | ----- |
| observed | dashed | an attention pattern or attribution score only |
| tested | solid | an intervention moves the margin, against matched random controls |
| confirmed | thick | also on held-out prompts |

`circuits/gpt2-induction.toml` is the previous-token head 4.11
feeding the induction heads 5.1, 5.5, 6.9, 7.2 and 7.10; every edge
is observed.

## induction.toml

Does GPT-2 Small's output need its induction heads? The clean prompt
is a beginning token, then 16 distinct random tokens, then the same 16
again. Its corrupted twin replaces the second copy with 16 fresh
tokens, so the two line up position by position and the twin gives
nothing to copy. The metric, at each second-copy position, is the
logit of the clean prompt's next token minus the logit of the twin's
next token there, averaged over the positions. The finding seeds
choose heads; the held-out seeds only check them.

```bash
just baseline --experiment experiments/induction.toml --seed 0 --out runs/baseline-0
```

writes a run directory: `run.toml` (the model, revision, libraries,
seed and both prompts' tokens) and the metric per position for the
clean prompt (`clean_ld`) and its twin (`corrupt_ld`), each as `.xta`
and `.txt`. On GPT-2 Small the clean mean is about 14 to 18 and the
twin's about 0 to 2.5: the gap that ablating or patching a head can
close.

```bash
just ablate --experiment experiments/induction.toml --out runs/ablate            # the finding seeds
just ablate --experiment experiments/induction.toml --seeds heldout --out runs/ablate-heldout
```

switches off each head in turn on the clean prompt and records how much
the mean logit difference drops: `zero_effect` (the head's output set to
0) and `mean_effect` (set to its mean over the finding seeds' twins), a
layer x head matrix each, one run per seed (`seed-N`). A positive
effect means the output needs the head. These effects are not yet
compared with random controls, so they name candidates.
`fixtures/gpt2-induction-ablation` holds the finding seeds' runs.

```bash
just patch --experiment experiments/induction.toml --out runs/patch            # the finding seeds
```

copies one activation between the twins at a time. *Denoising* runs the
twin with one activation from the clean prompt and records the fraction
of the clean-twin gap it restores; *noising* runs the clean prompt with
one activation from the twin and records the fraction it destroys. Each
run (`seed-N`) holds `resid_denoise` (the residual stream entering each
layer, at each position), `head_denoise` and `head_noise` (each head's
output, layer x head), and `position_denoise` (the six most restoring
heads, named in `run.toml`'s `leading_heads`, position by position).
`fixtures/gpt2-induction-patch` holds the finding seeds' runs.

```bash
just controls --experiment experiments/induction.toml --out runs/controls
just verdict runs/ablate runs/patch runs/controls
```

`just controls` makes, for each head and seed, 20 changes to the head's
output of the same size as the real mean ablation's and denoising's, in
random directions (`mean_control`, `denoise_control`: control x layer x
head). `just verdict` lists the heads whose mean effect is at least 5%
and beats every control (docs/design.md, D17).
`fixtures/gpt2-induction-controls` holds the finding seeds' controls.

```bash
just confirm fixtures/gpt2-induction fixtures/gpt2-induction-heldout
```

reruns nothing: it reads the finding seeds' runs (`PREFIX-ablation`,
`PREFIX-patch`, `PREFIX-controls`) and the held-out seeds' runs of the
same three, and gives each head that passes on the finding seeds its
verdict: confirmed when it passes again on the held-out seeds, tested
only when it does not (docs/design.md, D18). The held-out runs are made
with `--seeds heldout`; `fixtures/gpt2-induction-heldout-*` hold them.
