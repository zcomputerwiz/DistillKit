# PLE forensics, phases 0 and 1

Two questions, both answered. Phase 0 verifies that the addressing scheme this project
has been using is the one Flash-Next actually specifies. Phase 1 answers the only
measurement left that argued the table holds usable content.

## Phase 0: every addressing assumption holds

`NGramHashConfig` carries the whole scheme as *defaults*, and `from_pretrained_config`
— the function that would build it from a real Flash-Next config — is defined and never
called. `SidecarDataCollator` and `analyze_donor_reader` both construct a bare
`NGramHasher()`. So every row address in this project rests on those defaults.

`python scratch/ple_forensics/verify_tokenizer.py`:

| field | ours | Flash-Next |
| --- | ---: | ---: |
| vocab_size | 248320 | 248320 |
| ngram_size | 3 | 3 |
| heads_per_ngram | 8 | 8 |
| ngram_vocab_size_base | 20000000 | 20000000 |
| make_ngram_vocab_size_divisible_by | 128 | 128 |
| seed | 1234 | 1234 † |
| ple_embed_dim | 2560 | 2560 |
| eos_token_id | 248044 | 248044 |

† absent from `config.json`; `Qwen4ExpTextConfig` supplies 1234, which is what the
reference module would use. Reading the raw JSON alone reports a spurious mismatch.

The tokenizers are **byte-identical** — both vocabularies hash to
`139b05b661bc63cabb6c0a03da15c4d1559a00a2fd3157d44941fd571987b584` — and every id the
project names agrees: 198 `\n`, 248044 `<|endoftext|>`, 248045/248046 the im markers,
248068/248069 the think tags. The teacher cache's `vocab_size` matches.

This was worth checking because the failure mode is silent: a differing tokenizer would
make every address a correct hash of the wrong number, and the novelty stratification,
the frequency bins and the shuffled control's matched hit rate would all be meaningless.

## Phase 1: the capacity probe does not survive on content

The probe never saved its trained matrix and scored with `reduction="sum"`, so this
needed a fresh run, not a re-read. `--per-token` now emits NLL, target id and document
index for the baseline and every arm, beside the untouched aggregate path — the per-token
means reproduce the aggregates to 1e-8, which is float accumulation order.

The run reproduces the historical result: baseline 0.4797, table 0.4308, shuffled 0.4737,
table over control 0.042948 against the quoted 0.0472.

### The layout mask had to be rebuilt first

The project's `LAYOUT_TOKEN_IDS` names three ids. This corpus's assistant targets use
**35 whitespace-only token types** — `\n\n` (271), tabs, and eight distinct runs of
spaces among them. Scored against the three-id mask, `\n\n` alone was 53.9% of what was
being reported as the *content* gain. The mask is now derived from the tokenizer: any
token whose decoded text is entirely whitespace is layout, whatever its id.

### The result

48,963 assistant tokens, 124 documents, layout 13.7%. Negative is better; bootstrap over
documents.

| comparison | all | layout | content |
| --- | --- | --- | --- |
| table − baseline | −0.048926 [−0.057263, −0.041914] | **−0.345593** | **−0.001846 [−0.003624, −0.000376]** |
| shuffled − baseline | −0.005978 [−0.008742, −0.003802] | −0.046689 | +0.000482 [−0.000876, +0.001609] |
| table − shuffled | −0.042947 [−0.049438, −0.037362] | −0.298904 | −0.002328 [−0.003483, −0.001168] |

**Of the 2,395.5 nats the table gains over baseline, 2,317.5 — 96.7% — are layout
tokens, which are 13.7% of the scored positions.**

Content survives, and it is real: −0.001846 with an interval excluding zero. It is also
**26× smaller than the headline**, about 0.4% of the content baseline of 0.4469. The
shuffled control is neutral on content (interval spans zero), so the real-over-shuffled
content figure is not an artefact of a harmful control — but the honest number is the
one against baseline.

### Where the content gain sits

−78.0 nats over 42,257 tokens and 5,916 distinct target types:

| id | token | count | nats |
| ---: | --- | ---: | ---: |
| 3377 | ` problem` | 38 | −24.17 |
| 13 | `.` | 1124 | −23.12 |
| 1409 | `Time` | 22 | −12.10 |
| 248046 | `<\|im_end\|>` | 110 | −12.04 |
| 9764 | `Let` | 37 | −10.43 |
| 59 | `\` | 67 | −8.20 |
| … | | | |
| 760 | `The` | 131 | **+9.95** |
| 71093 | ` ``` ` | 107 | **+12.82** |
| 279 | ` the` | 860 | **+13.33** |

Cumulative, best first: top 1 −24.2, top 10 −109.5, top 100 −197.7, top 1000 −310.9,
against a total of −78.0. **The top ten token types more than account for the entire
content gain, and the tail is net harmful** — types 1001 onward contribute +232.9 nats.

The surviving gainers are not ordinary lexical content either. `<|im_end|>` is a control
token; `.` and `\` are punctuation; and ` problem`, `Time`, `Let`, `Need`, `As`,
` start` are the formulaic opening vocabulary of a chain-of-thought block. Meanwhile the
table actively *hurts* the most common function word in the corpus.

## Does this change the earlier C1 finding?

No. The published C1 split used the same incomplete three-id mask, so it was re-checked
with the derived one (41 whitespace types in the reply bundle, layout 12.8% of tokens):

| arm | content, narrow mask | content, derived mask |
| --- | --- | --- |
| C1 real | +0.003512 | **+0.003596** |
| C2 shuffled | +0.001405 | +0.001477 |

Unchanged. The extra whitespace types contribute ~nothing to C1's layout gain — that
benefit is specifically `\n` and `<think>`, not whitespace generally, which is itself a
difference from the capacity probe, where all whitespace gains.

## Answers to the questions this phase was asked

**Q6 — does the −0.0539 survive on content-only?** No, not meaningfully. It becomes
−0.001846, 3.3% of the gain, with 96.7% on 13.7% of positions.

**Q5 — which token ids account for the gain?** Layout overall. Within content, ten types
account for more than the whole of it, led by a control token, a full stop, and
chain-of-thought preamble words.

**Q7 — distributed across lexical content, or concentrated in formatting?**
Overwhelmingly concentrated in formatting and control.

## What this means

The capacity probe was the last measurement arguing the table holds content this student
can use. It does — 0.0018 nats of it, at the head, with the transport problem entirely
bypassed and a free 2560×2560 matrix reading it. That is the ceiling, not the floor: at
layer 1 the same features have to survive 31 more frozen layers, and the C1 arm measures
+0.0036 *against* content there.

Phase 2 (n-gram identity, corpus frequency, continuation entropy, collision counts
joined to the existing per-token NLL) and phase 3 (the corpus-scale index) were
conditional on this. Entropy as an admission feature would still need the within-content
control — the `‖v‖/‖h‖` result scored 0.5362 on "next token is layout" and was flat
inside content — but there is now very little content signal for any admission rule to
gate toward.

---

# Phase 0 of the offload experiment: the gradient gate

The offload hypothesis is different from everything measured above. It does not claim
the sidecar predicts content better — it claims that if a cheap external memory takes
over repetitive structural prediction, the backbone stops spending capacity there and
can put it into content. That is falsifiable before any training.

`python scratch/ple_forensics/gradient_conflict.py --documents 64`, on the frozen
pre-retrofit student, 64 held-out documents, 27,531 assistant targets. Gradients are
taken against a labelled *sample* of each layer's parameters — the attention output
projection and the MLP down projection, 40 matrices over 32 layers, 965M parameters —
because all 4B will not fit twice over beside the model.

Parameter gradients rather than activation gradients: parameters are shared across
positions, so `cos(grad L_layout, grad L_content)` is exactly "do these two objectives
want to move the same weights the same way". Activation gradients live at different
positions for the two classes and have no natural pairing.

## Where the backbone's update energy goes

| class | share of tokens | share of update energy | over/under |
| --- | ---: | ---: | ---: |
| lexical | 70.3% | 68.0% | 0.97× |
| **whitespace** | **11.6%** | **21.7%** | **1.87×** |
| punctuation | 17.4% | 8.6% | 0.49× |
| control | 0.7% | 1.7% | 2.42× |

Per unit of its own mean loss, whitespace is 11.8× as gradient-dense as content and
control is 270×; both figures are inflated for a rare class by the 1/n averaging inside
the class mean, which is why the share-weighted column is the one that answers the
question. Weighted, **whitespace consumes 21.7% of the backbone's update energy on 11.6%
of the positions.** That is the "meaningful gradient energy" the gate asked for.

## But the gradients are orthogonal, not conflicting

| class | A = cos with content gradient |
| --- | ---: |
| whitespace | **−0.0050** |
| control | +0.0006 |
| punctuation | +0.0110 |

Random vectors in 965M dimensions sit at 3.5e-5, so these are 20–300× the chance floor
and not noise — and they are still, for practical purposes, orthogonal. Whitespace turns
slightly negative with depth (+0.011 at layer 0, −0.008 at layer 28, −0.066 at 31) while
its energy ratio rises monotonically (5.2 → 30.7), so what conflict exists is late and
small.

## Verdict: the gate passes, with a caveat that shapes the arms

By the stated criterion — meaningful energy, weak alignment — this proceeds. Whitespace
is taking a fifth of the update on an eighth of the tokens.

The caveat is what orthogonality implies. Three cases the diagnostic could have found:

* **Strongly negative A.** Layout actively fights content; removing it frees direction.
  The strongest case to proceed, and not what was found.
* **A near 1.** Layout is already doing content's work; removing it takes that away.
  Would have stopped the experiment. Also not what was found.
* **A ≈ 0.** Neither. Removing layout frees *capacity* — a fifth of the update budget —
  but no direction.

The third is what is here, and it matters for the design: if the mechanism is purely
capacity reallocation, then **arm B (masked loss, no sidecar) already implements the
entire hypothesised mechanism.** The sidecar in arm D does not change what the backbone
optimises; it only preserves the layout competence that B throws away.

So the predicted signature is `D − A ≈ B − A` on content, with D and B separating on
layout rather than on content. That is still a result worth having — "layout can be
offloaded to a cheap memory at no cost to layout capability, and it buys X nats of
content" is an engineering claim — but the content gain would be attributable to
reweighting, not to the sidecar. Codex is right that B is mandatory; on this evidence B
is closer to the main arm than to a control.
