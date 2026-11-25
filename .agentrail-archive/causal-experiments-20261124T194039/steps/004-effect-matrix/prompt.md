The effect matrix in X_eTaL, test-first.

- An X_eTaL library (xetal/Effects) reads the ablation and patching
  runs: the L x H effect matrix D per method, the mean and spread over
  seeds, ranked heads, a report and grids drawn with hm:g_rid (a
  diverging scale where effects have sign: []S_CALE 3).
- just effects RUN_DIR [OUT_DIR]; README section with the grid and the
  report kept current by a test; the grid goes on the live pages
  (pages/build.py) from a committed run.
- Label: effects of interventions on this prompt family; not yet
  compared with random controls (next step).
