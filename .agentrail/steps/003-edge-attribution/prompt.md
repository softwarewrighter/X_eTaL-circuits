Edge attribution patching in the adapter, test-first: a cheap ranking.

- The linear approximation: for each edge, (clean - twin sender output)
  dot the gradient of the metric with respect to the receiver input, on
  the twin. One forward and backward pass per seed gives every edge.
- Written as runs like path patching's; used to choose which edges to
  path-patch (the top N, recorded in the edges experiment file).
- Tests: on a handful of edges the attribution's sign and rank agree
  with path patching (stated as the result, not assumed); determinism.
