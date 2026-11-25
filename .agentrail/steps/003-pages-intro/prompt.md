The live demo explains itself to a first-time reader (the user's ask), test-first.

- A front page that starts with an introduction, before any picture:
  what a language model's layers, heads and MLPs are; what a circuit
  is; the induction behavior studied here and why it matters (the
  paper's in-context learning builds on it); what was done (the twins,
  the interventions, controls and held-out seeds); what was found so
  far, in plain words; and how to read each page. The model map moves
  below the introduction, or to its own page.
- A short "how to read this" note on every page (what the axes, colors
  and numbers mean; which evidence kind it shows), and a glossary of the
  terms the pages use (attention, induction head, previous-token head,
  logit difference, twin, ablation, patching, control, held-out,
  observed / tested / confirmed), linked from the pages.
- The text in a file the README can share (docs/eli5.md stays the long
  form); American spellings, ASCII markdown where it is markdown.
- Tests: every page links the introduction and the glossary; every
  glossary term used on a page is defined; links resolve; the build is
  still deterministic; README screenshot retaken.
