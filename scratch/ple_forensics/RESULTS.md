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


---

# Branch specialisation in the donor: what could be tested, and the answer

## Most of it cannot be run here

Flash-Next is **360 GB in bf16** — 48 layers, 512 experts, top-10 routing, per its own
`model.safetensors.index.json`. What is on this machine is a **93.7 GB IQ4_XS GGUF** and
an HF snapshot carrying config, index and tokenizer and **no weights at all** (13 MB).
The hardware is 2×24 GB VRAM and 128 GB RAM, and running llama.cpp is out under a
standing constraint.

The branchwise gradient matrix and the causal branch ablation — the two pieces named as
decisive — both need a full forward, and the gradient one a full backward, through that
model. Neither is reachable. The same applies to the gate statistics and the
HyperConnection routing, which depend on the donor's residual stream.

## Two of the proposed metrics are the same metric

For the direct PLE write, `delta h_s = g_s v`, every branch receives the same 2560-D
value, so

    e_s = ||g_s v||^2 / sum_j ||g_j v||^2 = g_s^2 / sum_j g_j^2

The branch energy fraction is a pure function of the gate vector and does not involve `v`
at all. **Direct branch-energy specialisation and gate-amplitude specialisation are one
measurement, not two** — and neither can be computed without the donor's hidden state.

## The convolution path is reachable, and it is the informative one

The convolution's input is `v = value_proj(rows)`, which depends only on the n-gram table
and the token ids — no hidden state anywhere. And the four filter banks are genuinely
distinct (flattened cosine −0.216 to +0.506, from the donor preflight), so they are the
one component that could specialise *without* the gate.

12,000 held-out screen positions, donor `value_proj` and all four donor conv banks, CPU:

| class | tokens | branch 0 | branch 1 | branch 2 | branch 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| whitespace | 1,380 | 0.0091 | 0.8628 | 0.0600 | 0.0681 |
| control | 93 | 0.0097 | 0.8582 | 0.0617 | 0.0704 |
| punctuation | 2,152 | 0.0091 | 0.8622 | 0.0603 | 0.0684 |
| lexical | 8,375 | 0.0091 | 0.8639 | 0.0596 | 0.0674 |

whitespace minus lexical, per branch: −0.00005, −0.00111, +0.00043, +0.00073. **All four
span zero.** The largest class gap anywhere is 0.0011 against an even split of 0.25.

**The stop rule fires: branch specialisation is not supported in the convolution path.**

The banks are wildly *uneven* — branch 1 carries 86.3% of the energy, matching the
preflight's PCA finding that one donor stream explains 91.05% — but that imbalance is
identical for every token class. The bank is near rank-one, and it is near rank-one the
same way for whitespace as for content.

## Scope of this null: the conv banks only

**Donor GR/HC specialisation is unresolved, not refuted.** The measurement above covers
one of the four mechanisms specialisation could come from — branch-specific conv filters
— and says nothing about the other three. Class-dependent gate amplitudes,
HyperConnection read/write routing, and causal branch effects all need the donor's
residual stream, and the donor cannot be run here.

An earlier draft of this section argued the gate path was "independently unlikely" from
the reader-visibility finding that upstream's trained keys are nearly collinear (pairwise
cosine 0.81–0.96) with the mean key reproducing 65–93% of the gate. That inference does
not hold and is withdrawn. It concerns the *PLE gate*, which is not the GR/HC routing;
and 65–93% reproduced by a constant leaves 7–35% that is not, which is exactly where
class-dependence would live. A null in the conv banks is not evidence about the gate.

## Decision

Proceed to a **whitespace-only** A/B/D offload experiment, with donor GR/HC
specialisation recorded as unresolved rather than settled. Nothing below infers anything
about the donor's routing from the conv-bank null.

The gradient gate's caveat is what shapes how the arms are read: whitespace and content
gradients are orthogonal, so the mechanism under test is capacity reallocation, arm B
implements all of it, and **B versus A on content is the first stop gate** — if masking
the loss alone buys nothing, there is nothing for a sidecar to preserve on top of.

Two constraints on the build, both from the routing problem rather than from any donor
finding:

* Arm D's sidecar is a **sparse whitespace expert plus a context-only mixture router**,
  `P = pi(context) P_expert + (1 - pi(context)) P_backbone`, over the full vocabulary.
  Routing on the ground-truth class would leak the answer, and a reduced-vocabulary head
  under oracle routing does not produce an NLL comparable to arm A's.
* Arms B and D must normalise the backbone loss identically. If B averages over content
  tokens and D averages over a different denominator, the two are not comparable and the
  mandatory B-vs-D contrast is meaningless.


---

# Router predictability, and whether there is work to offload

Arm D needs `P = pi(context) P_expert + (1 - pi(context)) P_backbone` over the full
vocabulary. Routing on the ground-truth class would leak the answer, so `pi` has to come
from context — and the cheapest candidate costs nothing to build: the frozen backbone's
own whitespace mass, `pi = sum_{w in whitespace} P_backbone(w)`. 440 whitespace token
types across the full 248,320-token vocabulary; 64 held-out documents, 27,531 assistant
positions, 11.6% whitespace.

## The router is essentially free

| | |
| --- | ---: |
| AUC of `pi` against the actual class | **0.9994** |
| mean `pi` at whitespace | 0.9559 |
| mean `pi` elsewhere | 0.0075 |

| threshold | fires on | precision | whitespace recalled |
| --- | ---: | ---: | ---: |
| `pi` >= 0.5 | 11.7% | 95.5% | 96.8% |
| `pi` >= 0.8 | 11.0% | 98.2% | 93.2% |
| `pi` >= 0.9 | 10.5% | 99.1% | 90.1% |

The backbone already knows when whitespace is coming, almost perfectly, and that
knowledge is available without any oracle and without training anything.

## And most of the whitespace NLL is the part an expert can take

At a whitespace position the loss factors exactly:

    -log P(w) = -log P(whitespace) + -log P(w | whitespace)
                 \_____ detection ____/   \_____ selection ____/

Detection cannot be offloaded — the router has to do it anyway, and the router is the
backbone. Selection can.

| | nats | share | per token |
| --- | ---: | ---: | ---: |
| total whitespace NLL | 1477.68 | | 0.4641 |
| detection | 255.70 | 17.3% | 0.0803 |
| **selection** | **1221.97** | **82.7%** | **0.3838** |

**82.7% of the whitespace NLL is offloadable.** Detection is nearly free precisely
because the router is so good; what costs is choosing *which* whitespace token — which
run of spaces, how many newlines — among 440 types. That is local, structural, repetitive
work, and it is exactly what the capacity probe showed the table doing: layout NLL
1.0002 to 0.1744.

Both gates pass. Arm D has a well-posed target: 0.3838 nats per whitespace token, on
11.6% of positions, reachable by a router that already works.

## Normalisation, before the arms are built

Arms B and D must normalise the backbone loss identically or the mandatory B-vs-D
contrast is meaningless. Two defensible conventions, and they test different things:

* **Same denominator as A** (total assistant tokens, whitespace terms simply absent).
  B and D are then exactly A minus the whitespace contributions, which is the clean
  counterfactual for "the backbone stops spending gradient there". It also means B and D
  take slightly smaller steps than A, so part of any difference is an effective learning
  rate change.
* **Own denominator** (mean over content tokens). Step magnitude matches A, but B and D
  are now also *upweighting* content, which is a second intervention.

The first is used, because the hypothesis is about what the gradient points at rather
than how big it is, and because it makes B and D identical to A except for the removed
terms. The effective-step confound is real and is the first thing to vary if B minus A
comes out marginal.


---

# Is selection a share of the *update*, or only of the loss?

The router check put selection at 82.7% of the whitespace NLL. Arm B removes a gradient,
not a loss, and the two can diverge — a term can dominate the loss and contribute little
to the update, or the reverse. So this is measured over the exact window arms A and B
will train: decoder layers 20–28, the tuned A3 window, **all 1.01B parameters in it**
rather than the sampled matrices used earlier, because that is the set that actually
moves.

48 documents, 2,179 whitespace and 17,381 content targets.

| term | per objective | share-weighted | tokens |
| --- | ---: | ---: | ---: |
| detect `−log P(WS)` | 2.2626 | 0.0356 | 2,179 |
| **select `−log P(w\|WS)`** | **21.3586** | **0.3357** | 2,179 |
| content (full-vocab CE) | 1.0000 | 1.0000 | 17,381 |

**Of the whitespace contribution to the update, selection is 90.4%** — higher than its
82.7% share of the NLL, not lower. Per unit of its own objective, selection is 21.4× as
gradient-dense as content while detection is only 2.3×.

Removing the selection term takes **24.5%** of the window's total update energy
(0.3357 of 1.3713). That is what arm B is doing, and it is not a rounding error.

Gradient cosines are consistent with the whole-model measurement — everything close to
orthogonal:

| pair | cosine |
| --- | ---: |
| select vs content | +0.0147 |
| detect vs content | −0.0136 |
| select vs detect | −0.0154 |

Note these figures are not directly comparable to the earlier whole-model
`whitespace R_share = 0.3184`: that used a sampled matrix set across all 32 layers and
normalised against the lexical class, this uses every parameter in layers 20–28 and
normalises against all non-whitespace targets.

## Where that leaves the arms

Every pre-training gate now passes:

* whitespace takes a fifth of the update, orthogonally to content;
* the router is free and near-perfect (AUC 0.9994);
* 82.7% of the whitespace loss and **90.4% of its gradient** is the selection term an
  expert could take;
* removing it is 24.5% of what the trainable window spends.

Arms A and B are next, and B versus A on content is the stop gate. The claim under test
is deliberately narrow: *removing conditional whitespace-selection training from the
backbone improves content optimisation, and the n-gram sidecar can take that selection
task over without losing whitespace prediction.* Not semantic capacity offload —
optimisation-budget transfer, until something supports the stronger reading.
