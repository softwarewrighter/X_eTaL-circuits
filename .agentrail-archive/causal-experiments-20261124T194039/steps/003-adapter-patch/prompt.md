Activation patching in the adapter, test-first.

- Clean-into-corrupted patching (denoising) and corrupted-into-clean
  (noising): per layer (residual stream), per head (head output), and
  per position for the heads that matter, recording the metric
  recovered as a fraction of the clean-corrupted gap.
- Same run-directory format as the ablations; a committed small
  fixture.
- Tests: patching everything recovers the clean metric exactly (and
  nothing leaves the corrupted one); determinism; model tests on the
  expected shape of the residual-patching result (late positions in
  layers 5-7).
