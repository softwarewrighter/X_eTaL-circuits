# Saga 3 -- circuit-discovery (Milestone 3)

Goal: test the edges of the induction circuit in GPT-2 Small, which
Saga 2 left observed. Does 4.11 feed the induction heads, and through
which input (query, key or value)? Which heads do 7.6, 8.5, 9.9, 9.6,
10.1 and 0.1 read from or feed? Edges are scored by path patching (a
sender head's output patched only into a receiver's query, key or
value input), with Saga 2's twins, metric, matched random controls,
5% floor and held-out seeds. Edge attribution patching ranks edges
cheaply first. The resulting subgraph is checked for faithfulness,
completeness and minimality against random subgraphs of the same size.

Roadmap, principles, Saga 1 and 2 outcomes: docs/plan.md. Rules:
CLAUDE.md (observation is not causation; the margin; twins aligned;
tests first). Decisions: docs/design.md (D14-D18 for the experiment,
the metric, controls and held-out seeds).

Steps: graph-model, path-patching, edge-attribution, edge-effects,
heldout-edges, faithfulness, circuit-graph-view.
