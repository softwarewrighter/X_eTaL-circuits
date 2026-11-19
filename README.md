<p align="center">
  <img src="https://raw.githubusercontent.com/softwarewrighter/X_eTaL/main/images/xetal-logo-yellow.png" alt="X_eTaL: X underlined, a raised e, T, a raised a, L" width="122">
</p>

<p align="center">
  <b><a href="https://softwarewrighter.github.io/X_eTaL/">Try X_eTaL live in your browser</a></b>
  -- the editor, running in WebAssembly
  <br>
  <a href="https://blog.softwarewrighter.com/series/#array-languages">Read the blog series on array languages</a>
  -- posts on X_eTaL and array languages
</p>

# X_eTaL-circuits

Find, test and draw *circuits* inside language models, with
[X_eTaL](https://github.com/softwarewrighter/X_eTaL) as the analysis
and visualization language.

A circuit is a set of model components, such as attention heads, MLP
features and the residual-stream paths between them, that cooperate
to produce one behavior. Known examples are an induction head
copying the token that followed an earlier match, and the
key-value associative memory that lets a model learn a mapping like
`good -> foo, bad -> bar` from a few examples in its prompt. Circuits
are not separate programs the model switches between. They are
descriptions of mechanisms inside one fixed, learned computation, and
they can only be found by experiment.

> An attention map shows where a model looks. A circuit shows what
> the model needs.

The work here follows the causal method of "A Circuit for In-Context
Learning" (arXiv:2611.05344):

1. **Twin prompts** that differ in exactly one controlled way (swap a
   label, swap a sentence's class), with their tokens lined up.
2. **A continuous measure**: the margin
   `log P(correct) - log P(incorrect)`, not just right or wrong.
3. **Interventions**: activation patching, head ablation and
   steering, swept over layers, heads, positions and Q/K/V. Each
   result is compared with matched random controls.
4. **A circuit graph**, whose edges are tested by path patching and
   validated for faithfulness, completeness and minimality on
   held-out prompts.

These are array computations, a natural fit for an array language. A
patching sweep is a layer x head effect matrix, and an attention
capture is an L x H x T x T tensor. X_eTaL does the slicing,
statistics, graph discovery and drawing. A small adapter (Python
with TransformerLens, at first) runs the model and exposes its
internals.

## Studies

| Model                    | Question                                                         |
| ------------------------ | ---------------------------------------------------------------- |
| GPT-2 Small              | where are the induction heads, and which ones are needed?         |
| local models via Ollama  | do they learn arbitrary labels from examples, and reverse them?   |
| Qwen3-4B-Base            | are the paper's writers, readers, gate and brakes found here too? |
| X_eTaL code              | does the same mechanism carry over to programming-language examples? |
| a tiny trained transformer | in what order do circuit parts appear during training?          |
| Baby Dragon Hatchling    | does a model designed to be inspected make circuits easier to find? |

## Capturing a model run

The adapter runs a model through TransformerLens and writes a
*capture*: a directory with the attention patterns (`attn.xta`, layer
x head x query x key), the last position's logits (`logits.xta`), and
`capture.toml`, which holds the model, its revision, library versions,
the seed, the prompt and its tokens. The arrays are XTLARR01 files
(f32), optionally also one `capture.safetensors`, and for now each
array also as text numbers (`attn.txt`, `logits.txt`), which X_eTaL
reads with `n_umbers []N_GET`. The format is in
[`docs/design.md`](docs/design.md).

```bash
just xetal                                           # X_eTaL at the pinned commit (bin/xetal)
just setup --model                                   # torch + TransformerLens
just capture --out runs/induction --length 50        # BOS + 50 random tokens, twice
just capture --out runs/hello --text "Hello world" --layers 0,5 --safetensors
```

`fixtures/gpt2-induction` is a small capture (GPT-2 Small, two layers,
17 positions) that the tests read. `just fixture` makes it again,
byte for byte.

## Status

Early: the adapter writes captures. Reading them in X_eTaL, the
heatmaps, the scores and the model map come next. See
[`docs/plan.md`](docs/plan.md).

## Documentation

- [`docs/plan.md`](docs/plan.md) -- the roadmap, principles and open
  questions
- [`docs/reference.md`](docs/reference.md) -- the paper's method, the
  authors' code and the TransformerLens hooks for GPT-2 Small
- [`docs/design.md`](docs/design.md) -- the capture format, the rules
  and the decisions behind them
- `docs/xetal-*-asks.md` -- what this repository needs from its sibling
  repositories, one ledger each
- [`docs/research.txt`](docs/research.txt) -- the archived design
  conversation behind the project (not normative)

## Development

Development is test-driven and tracked with agentrail sagas; see
[`CLAUDE.md`](CLAUDE.md) (also `AGENTS.md`) for the agent workflow
and rules.

```bash
just setup                # adapter/.venv (uv), the adapter and its dev tools
just setup --model        # also torch and TransformerLens
just check                # adapter lint and tests, markdown checks
agentrail status          # current saga state
agentrail next            # the current step and its context
```

The model adapter is a Python package in `adapter/`, always run from
its own uv environment (`adapter/.venv`), never the system Python.

## The X_eTaL repositories

- [X_eTaL](https://github.com/softwarewrighter/X_eTaL) -- the
  language: a terse, statically typed, functional array language in
  plain ASCII, decorated typographically.
- [X_eTaL-libraries](https://github.com/softwarewrighter/X_eTaL-libraries)
  -- libraries written in X_eTaL itself.
- [X_eTaL-extensions](https://github.com/softwarewrighter/X_eTaL-extensions)
  -- small native Rust libraries used from X_eTaL like any other
  library.
- [X_eTaL-demos](https://github.com/softwarewrighter/X_eTaL-demos) --
  small programs worth watching.
- [X_eTaL-games](https://github.com/softwarewrighter/X_eTaL-games) --
  games and puzzles whose rules are array-shaped.

## Related Projects

- [in-context-learning-circuit](https://github.com/andrewgcodes/in-context-learning-circuit)
  -- the code and prompt batteries for arXiv:2611.05344.
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
  -- hooks for capturing and patching transformer activations.
- [bdh](https://github.com/pathwaycom/bdh) -- the Baby Dragon
  Hatchling reference implementation.
- [sw-mlpl](https://github.com/sw-ml-study/sw-mlpl) -- Software
  Wrighter's Machine Learning Programming Language, a Rust array
  language inspired by APL, APL2, J, and BQN.

## Links

- Blog: [Software Wrighter Lab](https://software-wrighter-lab.github.io/)
- Discord: [Join the community](https://discord.com/invite/Ctzk5uHggZ)
- YouTube: [Software Wrighter](https://www.youtube.com/@SoftwareWrighter)

## Copyright

Copyright (c) 2026 Michael A Wright

## License

MIT. See [`LICENSE`](LICENSE) and [`COPYRIGHT`](COPYRIGHT).
