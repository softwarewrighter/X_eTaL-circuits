Head ablation in the adapter, test-first.

- For each (layer, head): zero ablation and mean ablation (the mean
  over the corrupted twins' activations, at the same positions) of the
  head's output (TransformerLens 4.x hook names checked against
  docs/reference.md), on the clean prompt; record the change in the
  metric. One run gives the L x H matrix of effects for one seed.
- Written as a run directory (runs/, not committed) with its metadata
  and arrays the X_eTaL side reads (text route, D10); a small committed
  fixture of the effects for the tests.
- Tests: no-op ablation (scale 1) changes nothing; ablating every head
  of a layer equals ablating the layer's attention output; determinism;
  the model test that ablating 5.1/5.5 lowers the margin more than a
  layer-0 head (a sanity check, not the evidence).
