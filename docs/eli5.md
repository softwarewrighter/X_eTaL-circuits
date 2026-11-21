# Explain it like I'm five

The paper and this repository in plain words. The details, with
sources, are in [`reference.md`](reference.md) (the paper) and
[`plan.md`](plan.md) (this repository).

## First: what is a language model doing?

A language model reads some words and guesses the next one. Inside,
it is a tall stack of floors (**layers**). On every floor there are
two kinds of workers:

- **Attention heads** look back at earlier words and fetch something
  useful from them. GPT-2 Small has 12 floors with 12 heads each, so
  144 heads.
- **MLPs** think about what was fetched, one word at a time.

Every word has a notepad (the **residual stream**) that goes up the
stack. Each worker reads the notepad and writes a little more on it.
At the top, the notepad says what the next word should be.

Nobody wrote down which worker does what. The model learned it from
lots of text. Finding out afterwards is called **mechanistic
interpretability**. A group of workers that does one job together is
called a **circuit**.

## A first circuit: copying a pattern

Give a model a random list of words, then the same list again:

```
apple river seven cloud ... apple river seven cloud ...
```

The second time through, the model guesses the next word easily. It
looks back at what came after this word last time and copies it.
Two kinds of heads do this together:

- a **previous-token head** writes "the word before me was X" on each
  notepad;
- an **induction head** finds the earlier spot that follows the
  current word, and copies the word it finds there.

In GPT-2 Small, the heads known to do this are in layers 5, 6 and 7.
This repository's first job is to find them again and draw them.

## The paper: learning from examples in the prompt

"A Circuit for In-Context Learning" (Andrew Gao and William Gao,
arXiv:2611.05344) asks a harder question. Show a model a few examples
with made-up labels:

```
the food was great -> zorp
I hated the movie  -> quil
what a lovely day  -> zorp
the service was slow ->
```

The model answers `quil`. It was never trained on `zorp` and `quil`.
It worked out the rule from the examples in the prompt. This is
**in-context learning**. The paper says it found the workers that do
this in three models (Qwen3-4B, SmolLM3-3B, Llama-3.2-3B). Think of
a coat check:

- **Slots.** Each label word (`zorp`) becomes a coat-check ticket.
  It holds which label it is and what kind of sentence came before it
  (happy or sad).
- **Writers.** Some heads look from the label back at its sentence
  and write "this was a happy sentence" on the ticket.
- **Query code.** At the last arrow, other heads write down what kind
  of sentence the question is.
- **Readers.** Heads near the top compare the question's kind with
  every ticket and copy out the label from the tickets that match.
- **Gate.** Early heads notice that a label has repeated, so the
  prompt is a labeled list. They switch the readers on. Without the
  gate, the readers look at the very first word (a resting place
  called the **attention sink**) and pick no label.
- **Brakes.** Some heads push down the label seen most recently, so
  the model does not just repeat it.
- **Copiers.** Late heads write the final answer.

How do they know? They don't just look at where heads attend. They
**change** things and measure what happens. They swap two labels,
replace a sentence with one of the other kind, or copy one worker's
output from another prompt. Then they measure how much more the model
prefers the right label over the wrong one (the **margin**). They
also found the same parts growing during training, and in a tiny
model trained from scratch.

## Looking is not proving

Seeing a head attend to the right word is like seeing someone stand
near a broken vase. It makes them a suspect, not the culprit. A head
is called part of a circuit only after changing or removing it moves
the answer. The change is also compared with changing random heads,
and tested on prompts it was not tuned on. Everything here says which
kind of evidence it shows. Pictures of attention are always marked
as observation.

## What this repository does

It is a workbench for finding, testing and drawing circuits.

1. **Record.** A small Python program runs a model (GPT-2 Small
   through TransformerLens) on a prompt. It saves where every head
   looked (a **capture**), with everything needed to repeat the run:
   the model and its exact version, the seed, the prompt and the
   library versions.
2. **Read and check.** [X_eTaL](https://github.com/softwarewrighter/X_eTaL),
   an array language, reads the capture. It checks the numbers make
   sense: no head looks at a future word, and each head's attention
   adds up to 1.
3. **Draw.** Heatmaps of one head's attention, then scores that pick
   out the previous-token and induction heads, then one map of the
   whole model.

Later work, in [`plan.md`](plan.md):

- **Change things:** switch heads off, or swap in their output from
  another prompt, to test what each one does.
- **Find circuits:** draw the graph of which workers feed which.
- **Rerun the paper's study:** first the behavior through a local
  model server, then the authors' own code on Qwen3-4B.
- **Try it on code:** the same method on X_eTaL programs.
- **Watch circuits grow:** train a tiny model and save checkpoints.
- **Test other designs:** the same tests on a different kind of
  network (BDH).
