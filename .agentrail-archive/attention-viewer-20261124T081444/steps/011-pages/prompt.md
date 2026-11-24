Live pages on GitHub Pages (the user's choice), test-first: every
picture this saga draws, viewable in a browser without a clone.

- A static site built by a script (`just pages`) from committed
  captures only (fixtures/): the attention heatmap for every layer and
  head with a layer/head picker, the score grids, the model map with
  its evidence key, and docs/eli5.md. The SVGs are drawn by X_eTaL, as
  at the command line; the page itself only picks among them (no
  model, no server).
- Every page states the model, revision and prompt it shows, and that
  attention pictures are observations, not causal evidence.
- A GitHub Actions workflow builds and deploys it on a push to main.
  Turning Pages on in the repository's settings is the user's: say so.
- The README's header links to the live demo instead of "No live site
  yet", and the README shows a screenshot of the demo (the model map,
  committed under images/) that links to it, as the sibling
  repositories do (the user's ask). The screenshot is retaken by a
  script, so it matches the deployed pages.
- Tests: the build is deterministic (same inputs, same bytes), every
  page's links resolve, and the markdown stays ASCII.
- Close the saga: update docs/plan.md (Saga 1 DONE, Saga 2 steps
  revised from what was learned; later sagas add their pictures here).
