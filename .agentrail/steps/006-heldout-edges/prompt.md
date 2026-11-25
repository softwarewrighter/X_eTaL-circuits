Held-out edges, test-first.

- Path patching and edge controls rerun on the held-out seeds; an edge
  is confirmed when it passes again (D18's rule), tested when it passes
  on the finding seeds only.
- experiments/circuits/gpt2-induction.toml's edges updated from the
  verdicts (each with its basis), checked by a test against the runs;
  new edges the runs confirm are added, edges they do not stay observed
  or are removed (say which and why).
