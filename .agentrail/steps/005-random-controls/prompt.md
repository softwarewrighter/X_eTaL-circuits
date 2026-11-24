Matched random controls, test-first.

- For each head effect, a control distribution: the same intervention
  on random heads of matched norm (and random directions of matched
  magnitude in the head's output space), so an effect is called real
  only when it lies outside the controls (state the threshold and why,
  in docs/design.md).
- X_eTaL compares D with the controls: per head, the effect, the
  control interval and whether it passes; shown on the grid (passing
  heads marked) and in the report.
- Tests: controls are deterministic per seed; a head with zero effect
  never passes; the known induction heads pass on GPT-2 Small (model
  test), stated as the result, not assumed.
