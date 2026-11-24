Circuit evidence on the map and the pages, test-first; closes the saga.

- Heads carry evidence too (D13 says components and edges): add a
  per-head evidence level to experiments/circuits/*.toml (observed,
  tested, confirmed) and draw it on the model map (cell outline:
  dashed, solid, thick), with the key. Update gpt2-induction.toml from
  this saga's verdicts, each level citing the run it comes from.
- Edges stay "observed": ablating a head tests the head, not which head
  feeds which; path patching in Saga 3 tests edges. Say so on the map
  and in the README.
- The live pages: the effect grids, the controls, the held-out report,
  the updated map; the README screenshot retaken (just screenshot).
- Close the saga: docs/plan.md (Saga 2 DONE with its outcome, Saga 3
  revised).
