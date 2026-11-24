Twin prompts and the experiment specification, test-first.

- A declarative experiment file in experiments/ (TOML, every value a
  list of strings, so X_eTaL reads it with []L_IST as well as Python):
  model and revision, the prompt family (BOS + t random tokens twice),
  t, the seeds used for finding effects and the held-out seeds, the
  metric, the corruption, the interventions to run.
- The corrupted twin: the second copy replaced by fresh random tokens
  (no repeat), same length, so every position lines up; the adapter
  checks the alignment from the tokenization (rule 4), never assumes.
- The metric (rule 3): at each second-copy position the logit
  difference or log-probability margin between the correct next token
  (the induction answer) and the corrupted twin's token there; averaged
  over positions. Decide and record the exact definition in
  docs/design.md (a D entry), with why.
- The adapter records per-position logits for the metric (extend the
  capture: logits for every position, or only the metric values, with
  the shapes in capture.toml), with all run metadata (rule 2).
- Tests: twins align; the clean margin is clearly positive and the
  corrupted one near zero on GPT-2 Small (model test); determinism.
