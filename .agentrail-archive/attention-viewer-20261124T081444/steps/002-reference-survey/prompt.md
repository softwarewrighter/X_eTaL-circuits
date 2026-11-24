Write docs/reference.md: the reference material the later steps build on.

- Clone https://github.com/andrewgcodes/in-context-learning-circuit into a
  gitignored `third_party/` (do not commit it); record the commit hash.
  Read it in full before summarizing.
- Summarize: icl/run.py runner interface (model, battery, seed, smoke,
  out=), icl/tasks.py batteries, icl/hooks.py (patch/ablate/steer),
  icl/score.py, icl/effects.py, icl/check.py, icl/trace_view.py, toy/.
  Note dependencies and BF16 memory needs.
- Summarize the paper's method (twin prompts, margin metric, K vs V
  patching at label positions, writers/readers/gate/brakes, the softmax
  slot model and where it failed, faithfulness/completeness/minimality,
  cross-model and OLMo-2 checkpoint checks) from docs/research.txt;
  mark anything not verified against the paper or code.
- TransformerLens: the hook names for GPT-2 Small that Saga 1 needs
  (attention pattern, z/result per head, resid streams, logits), the
  version to pin, and how to verify it on this machine (CPU/MPS).
- No code changes beyond docs and .gitignore.
