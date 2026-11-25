## Ablation

Switching one head off and measuring what the model loses. *Mean
ablation* replaces the head's output with its average over the twin
prompts, so the rest of the model still sees ordinary values; *zero
ablation* sets it to zero, which the model never sees otherwise and so
is noisier.

## Attention

Where a head looks: for each word, how much weight the head puts on
each earlier word (the weights add up to 1). A heatmap of a head's
attention has a row per word that looks (the query) and a column per
word looked at (the key). Attention shows where a head looks, not
whether the model needs it.

## Attention head

One of the parts of a layer that moves information between words. It
builds a *query* at each word, matches it against the *keys* of
earlier words, and copies their *values* in proportion to the match.
GPT-2 Small has 12 heads in each of its 12 layers, named layer.head
from 0 (5.1 is layer 5, head 1).

## Circuit

A small set of heads and MLPs, and the paths between them, that
together produce one behavior. A circuit is a description of part of
one fixed computation, found by experiment, not a separate program.

## Control

A random change to a head's output of the same size as the real
intervention's, in a random direction. If a head's real effect beats 20
such controls, its effect comes from what the head carries, not just
from disturbing it that much.

## Edge

A path from one part of the model to a later one: a head's output
read by a later head's query, key or value, or by the final guess.
Switching a head off tests the head; testing an edge takes *path
patching*, which lets a head's output reach only one later reader.

## Evidence level

How sure we are about a head or an edge: *observed* (attention
patterns only; dashed), *tested* (beats its random controls on the
prompts used to find it; solid), *confirmed* (passes again on held-out
prompts; thick).

## Held-out seeds

Prompts (made from random seeds) kept aside until the end. Heads and
the 5% threshold are chosen on the *finding seeds*; a head is confirmed
only if it passes again on the held-out seeds, so the result is not
tuned to the prompts that found it.

## Induction head

A head that, at a repeated word, attends to the word that followed the
same word earlier and copies it. GPT-2 Small's are 5.1, 5.5, 6.9, 7.2
and 7.10.

## Layer

One level of the model's stack: its attention heads, then its MLP.
GPT-2 Small has 12, numbered from 0 at the bottom (nearest the words)
to 11 at the top (nearest the guess).

## Logit difference

How much more the model favors the right next word than the twin's
word at the same place: the difference of their logits (the scores
before they are turned into probabilities). It measures preference in
fine steps, where "right or wrong" would not.

## MLP

The part of a layer that works on each word where it stands, without
looking at other words. Each of GPT-2 Small's layers has one. The map
shows them; they have no scores yet.

## Patching

Copying an activation from one prompt into a run of the other.
*Denoising* runs the twin with one head's output from the clean prompt
and asks how much of the behavior comes back; *noising* runs the clean
prompt with one head's output from the twin and asks how much is lost.
*Path patching* lets the copied output reach only one later reader.

## Previous-token head

A head that, at each word, attends to the word just before it, so its
output records which word that was. GPT-2 Small's clearest is 4.11.

## Residual stream

The running notepad each word carries up the model: every head and MLP
reads it and adds its output to it, and the top layer turns it into
the next-word guess.

## Twin

A second prompt that differs from the clean one in one controlled way,
lined up word by word. Here the clean prompt is a list of random words
repeated; the twin replaces the second copy with fresh words, so there
is nothing to copy. The *key twin* replaces the first copy instead, to
test what heads find through their keys.
