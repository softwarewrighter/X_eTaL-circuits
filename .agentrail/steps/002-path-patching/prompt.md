Path patching in the adapter, test-first.

- For an edge sender -> receiver input: run the twin; the receiver's
  q, k or v input is the twin's residual stream plus the difference
  between the sender's clean and twin outputs (its write into the
  residual stream, hook_result / z @ W_O); everything else as in the
  twin. Metric: the fraction of the clean-twin gap restored (D14, D16).
  Logits as receiver: the sender's difference added to the final
  residual stream only.
- Probe and record how TransformerLens 4.x exposes per-head q/k/v inputs
  (hook_q_input etc., use_split_qkv_input) and where LayerNorm applies;
  test the path against a hand computation.
- Batch edges where possible (one receiver, many senders) for speed.
- Run directories as before (one per seed), edge x effect arrays with
  the edge names in run.toml.
- Tests: patching a sender into every receiver input of all later
  layers (and the logits) equals patching the sender's output (D16's
  head denoise) within float tolerance; an edge from a head whose
  outputs are equal in both twins has no effect; determinism.
