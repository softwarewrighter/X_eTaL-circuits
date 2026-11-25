The graph model, test-first: which edges exist and how they are named.

- Senders: heads (layer.head), and the embeddings. Receivers: a head's
  query, key or value input (layer.head.q|k|v), and the logits. An edge
  goes from a sender to a receiver in a later layer (the logits are
  after every layer).
- An experiment file for edges (experiments/induction-edges.toml):
  the base experiment it extends (experiments/induction.toml), the
  senders and receivers to consider (all heads by default, or a list),
  and which to path-patch (all, or the top N by attribution, step 3).
- Python reads it (adapter) and X_eTaL reads it ([]L_IST); both name
  edges the same way ("4.11 5.1.k", "9.9 logits").
- Record the decisions (D19): naming, which edges are in scope, why q/k/v
  are separate receivers (K-composition vs Q- and V-composition).
- Tests: names parse and round-trip; edges into an earlier or the same
  layer are refused; counts of all edges for GPT-2 Small are as computed.
