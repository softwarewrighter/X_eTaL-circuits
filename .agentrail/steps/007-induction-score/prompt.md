Induction and previous-token scores over L x H, in X_eTaL, test-first.

- On a repeated random-token sequence (length T repeated twice),
  previous-token score = mean attention from position i to i-1;
  induction score = mean attention from a token in the second repeat to
  the token AFTER its earlier occurrence. Compute both as L x H
  matrices in X_eTaL.
- Rank heads; render each matrix as a heatmap.
- Validate against TransformerLens (same scores computed in Python on
  the same capture) and against the GPT-2 Small induction heads known
  from the literature (docs/reference.md lists them: induction 5.1, 5.5, 6.9, 7.2, 7.10; previous-token 4.11); report agreement.
- State in the README and the output that these are attention-pattern
  scores (candidates), and that causal tests are Saga 2.
