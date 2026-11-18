# Asks for X_eTaL-ML

What this repository needs from
[X_eTaL-ML](https://github.com/softwarewrighter/X_eTaL-ML)
(`../X_eTaL-ML`, read only from here: its own plan takes an ask).
Statuses as in [`xetal-extensions-asks.md`](xetal-extensions-asks.md).

| #  | Status | Kind | Ask | Needed by | Workaround |
| -- | ------ | ---- | --- | --------- | ---------- |
| M1 | not yet taken | plan | Its proposed "in-context learning" saga and this repository split the work (D5 in [`design.md`](design.md), the user's choice). X_eTaL-ML keeps the teaching demos: a hand-built ICL circuit on its Attention library, a toy trained in X_eTaL, browser pages. This repository does the experiments on real models (GPT-2 Small, Qwen3-4B) and the causal tooling. Asked: its plan says so and points here, as `docs/plan.md` here points there. Its "later: a real model once X_eTaL reads binary BF16 tensors" is then this repository's, served by X_eTaL-extensions' asks X1 and X2 | Sagas 4-7 here (no duplicate studies) | none needed: the split is recorded here |
| M2 | not yet taken | reuse | The analysis here reuses its `Attention` (`c_ausal`, `w_eights`, `s_cores`), `Norm` and `NN` libraries rather than redefining them, pinned by an `XETAL_ML_COMMIT` here, as its own pins work. Asked: nothing now; a note in its plan that these exports have a user outside it, so a change to their types is announced | Saga 1 step xetal-load onward | none needed |
