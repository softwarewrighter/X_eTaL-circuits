# Saga 2 -- causal-experiments (Milestone 2)

Goal: test causal importance, not just patterns, on GPT-2 Small's
induction behavior. Twin prompts (clean and corrupted, token-aligned),
head ablation and activation patching give an L x H effect matrix `D`
in X_eTaL, compared with matched random controls and checked on
held-out seeds. Heads that pass become "tested" or "confirmed" on the
model map (D13); edges stay "observed" until path patching (Saga 3).
Every new picture goes on the live pages.

Roadmap, principles, Saga 1's outcome: docs/plan.md. Rules: CLAUDE.md
(observation is not causation; the margin as the metric; twins line up
token by token; tests first).

Steps: twin-prompt-spec, adapter-ablate, adapter-patch, effect-matrix,
random-controls, heldout-validation, circuit-evidence.
