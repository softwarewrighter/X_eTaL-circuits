## What this is

A language model reads text and guesses the next word. Inside, GPT-2
Small is a stack of 12 [layers](glossary.html#layer). Each layer has 12
[attention heads](glossary.html#attention-head), which look back at
earlier words and move information from them, and one
[MLP](glossary.html#mlp), which works on each word where it stands.
Every word carries a running notepad up the stack, the
[residual stream](glossary.html#residual-stream): each head and MLP
reads it and adds to it, and the top of the stack turns it into a
guess.

Nobody wrote down which head does what; the model learned it from
text. A [circuit](glossary.html#circuit) is a small group of heads (and
MLPs) that together do one job. This site shows how we look for one,
and how sure we are about each part of it.

## The job studied: copying a repeat

Give the model a list of random words, then the same list again. The
second time through, it predicts each next word almost perfectly: it
looks back at what followed this word the first time, and copies it.
Two kinds of heads are known to do this together:

- a [previous-token head](glossary.html#previous-token-head) notes, at
  each word, which word came just before it;
- an [induction head](glossary.html#induction-head) finds the earlier
  place whose previous word matches the current word, and copies the
  word it finds there.

This is the simplest way a model learns from the text in front of it.
The paper behind this project, Gao and Gao's "A Circuit for In-Context
Learning", studies a richer version of it in larger models.

## How we test it

Looking is not proving. Where a head looks, its
[attention](glossary.html#attention), only names a suspect. So we:

1. pair each prompt with a [twin](glossary.html#twin) that differs in
   one controlled way (the second copy replaced by fresh words), lined
   up word by word;
2. measure how strongly the model prefers the right next word over the
   twin's, the [logit difference](glossary.html#logit-difference);
3. switch one head off ([ablation](glossary.html#ablation)), or copy
   its output from one prompt into the other
   ([patching](glossary.html#patching)), and see how far that measure
   moves;
4. compare each effect with 20 random changes of the same size
   ([controls](glossary.html#control)), and require at least 5%;
5. check the heads that pass on ten new prompts that played no part in
   choosing them ([held-out seeds](glossary.html#held-out-seeds)).

## How sure we are

Every head and every arrow between heads carries one of three
[evidence levels](glossary.html#evidence-level), drawn the same way in
every picture:

- **observed** (dashed): seen in attention patterns only;
- **tested** (solid): beats its random controls on the prompts used to
  find it;
- **confirmed** (thick): passes again on the held-out prompts.
