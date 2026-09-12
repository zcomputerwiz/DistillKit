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

## Where the backbone's gradient energy goes

| class | share of tokens | share of gradient energy | over/under |
| --- | ---: | ---: | ---: |
| lexical | 70.3% | 68.0% | 0.97× |
| **whitespace** | **11.6%** | **21.7%** | **1.87×** |
| punctuation | 17.4% | 8.6% | 0.49× |
| control | 0.7% | 1.7% | 2.42× |

Per unit of its own mean loss, whitespace is 11.8× as gradient-dense as content and
control is 270×; both figures are inflated for a rare class by the 1/n averaging inside
the class mean, which is why the share-weighted column is the one that answers the
question. Weighted, **whitespace consumes 21.7% of the backbone's gradient energy on
11.6% of the positions.** That is the "meaningful gradient energy" the gate asked for.

Gradient energy, not optimiser update magnitude. AdamW normalises per coordinate and
moves roughly `lr` per element almost regardless of gradient size -- a fact this project
established when it found `sharpness` bit-identical across 72 steps -- so a share of the
gradient does not translate into the same share of the step. Every "share of the update"
phrasing below has been corrected to say gradient.

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
is taking a fifth of the gradient energy on an eighth of the tokens.

The caveat is what orthogonality implies. Three cases the diagnostic could have found:

* **Strongly negative A.** Layout actively fights content; removing it frees direction.
  The strongest case to proceed, and not what was found.
* **A near 1.** Layout is already doing content's work; removing it takes that away.
  Would have stopped the experiment. Also not what was found.
* **A ≈ 0.** Neither. Removing layout removes a workload — a fifth of the gradient
  energy — but no direction, and a gradient is not a conserved budget that something
  else then inherits.

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

Removing the selection term takes **24.5% of the measured share-weighted gradient
energy in the trainable window** (0.3357 of 1.3713). That phrasing is the strict one and
should be used verbatim: it is measured gradient energy, *not* optimizer-update energy —
see the final section, where the two are shown to come apart. That is what arm B is
doing, and it is not a rounding error. It does not follow that content's share grows to
fill it: gradient magnitude is not conserved.

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

* whitespace takes a fifth of the gradient energy, orthogonally to content;
* the router is free and near-perfect (AUC 0.9994);
* 82.7% of the whitespace loss and **90.4% of its gradient** is the selection term an
  expert could take;
* removing it is 24.5% of what the trainable window spends.

Arms A and B are next, and B versus A on content is the stop gate. The claim under test
is deliberately narrow: *removing conditional whitespace-selection training from the
backbone improves content optimisation, and the n-gram sidecar can take that selection
task over without losing whitespace prediction.* Not semantic capacity offload —
optimisation-budget transfer, until something supports the stronger reading.


---

# Arms A and B: the stop gate fires

Plain CE, no teacher. Layers 20–28, LR 3e-5, constant with 20-step warmup, 512 training
documents at 1,024 tokens, batch 2, 256 steps, seed 42 — identical in both arms. The only
difference is whether `-log P(w | WS)` contributes at whitespace targets. Scored on 124
held-out documents, 48,963 assistant tokens, 13.2% whitespace.

## Absolute NLL

| arm | content | whitespace | ws detect | ws select |
| --- | ---: | ---: | ---: | ---: |
| baseline (untrained) | 0.48779 | 0.42634 | 0.07014 | 0.35620 |
| A | 0.51361 | 0.16327 | 0.07190 | 0.09136 |
| B | 0.51353 | 0.36666 | 0.07285 | 0.29380 |

## The gate

    B - A on content:  -0.000080  [-0.000719, +0.000603]   spans zero

**Removing whitespace selection from the backbone's objective bought nothing on
content.** By the pre-registered rule, this stops here: arm D is not built, because there
is no content benefit for a sidecar to preserve.

The arm is not broken — it did exactly what it was designed to do:

| whitespace term, B − A | | |
| --- | --- | --- |
| detect | +0.000952 [−0.000136, +0.002048] | preserved, as intended |
| select | +0.202440 [+0.167810, +0.247054] | lost, as intended |

B stopped training selection and lost selection; it kept detection. The intervention
landed cleanly and produced no content effect.

## The part that complicates the reading

Both arms are **worse on content than the untrained model**:

| | content | whitespace |
| --- | --- | --- |
| A − baseline | **+0.025826** [+0.006911, +0.041060] | −0.263071 |
| B − baseline | **+0.025747** [+0.006860, +0.040844] | −0.059679 |

256 steps of plain CE on this window makes the model dramatically better at whitespace
(A gains 0.263 nats) and measurably worse at content. **That degradation is not caused by
whitespace learning.** A and B lose content to within 0.00008 of each other, and B barely
trained whitespace selection at all. Whatever degrades content here does so whether or not
whitespace selection is in the objective — which is what Phase 0's near-orthogonal
gradients already predicted.

An earlier draft said the backbone spends training on whitespace "at content's expense".
The data does not support that and it is withdrawn.

What B does show is that the removed workload simply disappears. B gained 0.060 on
whitespace against A's 0.263, for the same content cost — less total useful work, not
different work.

That is a clean negative for *natural* reallocation, with one honest caveat about the
platform: it is measured in a regime where content degrades in both arms, so what was
tested is whether removing selection slows the degradation, not whether it accelerates
improvement.

### Natural against explicit reallocation

The distinction everything downstream depends on. For A the gradient is
`g_A = g_C + g_WS`; for B it is `g_B = g_C`. Because `g_C` and `g_WS` are near-orthogonal,
removing `g_WS` does **not** imply `g_C → 1.25 g_C`. Gradient magnitude is not a conserved
budget that the remaining term inherits.

So arm B as run tests: *does removing whitespace-selection work naturally improve content
optimisation?* Answer so far, no.

A step-matched or LR-raised B tests something different — *if the removed update magnitude
is **explicitly reallocated to content**, does content improve?* That is a worthwhile
engineering question and it is **not** the automatic-capacity-offload hypothesis. Should
such an arm win, the claim it supports is "offloading whitespace permits more aggressive
content-directed optimisation at the same overall update magnitude", never "the backbone
repurposes freed capacity".

The pre-registered confound sat inside that distinction: B uses A's denominator and so
was assumed to take a smaller step. **It does not** -- measured later at 97.8% of A's
displacement, because AdamW is approximately scale-insensitive per coordinate. See the
final section; the caveat is withdrawn.

## What would change the verdict

1. **An arm-A-only LR sweep, first and cheapest.** Find a regime where held-out content
   NLL is flat or improving rather than degrading — 3e-6, 1e-5, 2e-5, 3e-5, everything
   else held. Logging train content NLL alongside held-out separates the two
   explanations: train improving while held-out worsens is overfitting or domain
   adaptation; train worsening too is optimisation instability.
2. **Repeat matched A/B** once A actually learns content. B ≈ A there is a strong negative
   for natural reallocation; B < A means interference exists at longer horizon or lower
   rate and D is back on the table; B > A means the whitespace gradient was regularising.
3. **Step-matched B, optionally and last**, labelled as explicit reallocation and never as
   evidence that removal frees capacity by itself.

**Distillation replication is deliberately held.** The 0.7 sparse KL + 0.3 hidden-state
cosine objective breaks the clean CE decomposition, because the hidden-state term can keep
teaching whitespace-selection *representations* even with the token-level selection loss
masked. Establish whether any A/B separation exists under CE first.

The defensible statement meanwhile: *under the tested plain-CE regime, conditional
whitespace selection is a large, nearly orthogonal gradient workload, but removing it does
not naturally redirect optimisation toward content — the removed gradient magnitude simply
disappears.*


---

# The LR sweep, and matched A/B in a regime that learns

The A/B pilot ran at 3e-5, where arm A *degrades* held-out content. That is a weak
platform: what it tested was whether removing whitespace selection slows a degradation,
not whether it accelerates an improvement. Arm A alone, across rates, everything else
held — same window, corpus, splits, horizon, optimiser, denominator.

## Finding a regime

Full held-out content NLL against the untrained model (0.48779):

| lr | content | vs baseline | ws select |
| --- | ---: | ---: | ---: |
| 3e-6 | 0.46757 | −0.02022 | 0.08340 |
| **1e-5** | **0.45918** | **−0.02861** | 0.07350 |
| 2e-5 | 0.47779 | −0.01000 | 0.08022 |
| 3e-5 | 0.51361 | **+0.02583** | 0.09136 |

**3e-5 was past the edge.** At 1e-5 arm A improves content by 0.0286 nats.

## Overfitting, not instability

Logging train content beside held-out separates the two explanations:

| lr | held-out content, s32 → s256 | train content, s32 → s256 |
| --- | --- | --- |
| 3e-6 | 0.5079 → 0.4957 (down) | 0.5619 → 0.4355 (down) |
| 1e-5 | 0.4973 → 0.4934 (down, then flat) | 0.5532 → 0.4138 (down) |
| 2e-5 | 0.5005 → **0.5176** (up) | 0.5472 → 0.4227 (down) |
| 3e-5 | 0.5157 → **0.5548** (up) | 0.5462 → 0.4571 (down) |

Train content improves at **every** rate; held-out diverges only at 2e-5 and above. That
is overfitting or domain adaptation, not optimisation instability. (The train column
sawtooths because each point covers a different 32-step window of documents; the trend is
what matters.)

Separately: whitespace selection collapses from 0.35620 to 0.073–0.091 at *every* rate,
including 3e-6. It is learned fast and cheaply regardless — consistent with the large,
independent workload Phase 0 measured.

## Matched A/B at 1e-5

Both arms re-run with identical settings, differing only in whether `-log P(w | WS)`
contributes at whitespace targets.

| arm | content | whitespace | ws detect | ws select |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.48779 | 0.42634 | 0.07014 | 0.35620 |
| A | 0.45918 | 0.14187 | 0.06837 | 0.07350 |
| B | 0.45891 | 0.41289 | 0.06828 | 0.34461 |

Both arms now genuinely learn content:

| | content vs baseline |
| --- | --- |
| A | −0.028605 [−0.040481, −0.019068] |
| B | −0.028881 [−0.040758, −0.019382] |

    B - A on content:  -0.000276  [-0.000551, +0.000005]   spans zero

The arm landed as designed again — B lost selection (+0.271112 [+0.228936, +0.325898])
and kept detection (−0.000089 [−0.000421, +0.000236]).

## Verdict

This is the pre-registered **B ≈ A** outcome, now measured in a regime where arm A
improves content by 0.0286 nats rather than degrading it. Removing a workload that is
90.4% of the whitespace gradient energy and roughly a quarter of everything the window
spends changes content by −0.000276 — about **1% of A's own content gain**, with an
interval touching zero.

So the negative is not an artefact of the bad 3e-5 platform. It holds where the backbone
is demonstrably learning content.

**Whitespace-selection learning is a large, nearly orthogonal workload. Removing it
removes the workload; the optimiser does not redirect the freed gradient magnitude toward
content. The removed magnitude simply disappears.**

Arm D remains unbuilt: there is no natural content benefit for a sidecar to preserve.

## What is left

Only the optional arm, and it tests a different claim. A step-matched or LR-raised B —
**explicitly reallocating the removed update magnitude to content** — would ask whether
content improves when the freed magnitude is *put* there rather than expected to migrate
on its own. If it wins, the supported claim is "offloading whitespace permits more
aggressive content-directed optimisation at the same overall update magnitude", never
"the backbone repurposes freed capacity".

Distillation replication stays held, for the reason it was held: the hidden-state term can
keep teaching whitespace-selection representations even with the token-level selection
loss masked, which would break the decomposition this whole design rests on.


---

# Explicit reallocation, and a pre-registered confound that turned out not to exist

## The confound was wrong

The ledger has been carrying a caveat that arm B "takes a roughly 24.5% smaller effective
step", because it uses A's denominator with the whitespace terms absent. Every run now
snapshots the window before training and reports the parameter displacement, so that is
measured rather than asserted. At 1e-5, over 256 steps:

| arm | ‖Δθ‖ | relative to ‖θ₀‖ = 387.38 |
| --- | ---: | ---: |
| A | 1.752729 | 0.004525 |
| B | 1.714111 | 0.004425 |

**B travelled 97.8% as far as A, not 75.5%.** AdamW is approximately scale-insensitive
per coordinate — scaling a gradient by `c` leaves `m̂/√v̂` unchanged in the idealised
case — so a 24.5% reduction in measured gradient energy does not produce a comparable
reduction in parameter travel. The confound does not exist, and the 1e-5 null is cleaner
than it was claimed to be.

Do not upgrade this to exact scale invariance in general. Momentum history, `eps`,
gradient clipping, decoupled weight decay, and any change in the sparsity or support of
the gradient can all break the equivalence. What is claimed here is the measurement:
97.8% travel for a 24.5% energy reduction, in this window and setup.

The same approximate property produced two earlier findings in this project: `sharpness`
sitting bit-identical for 72 steps, and AdamW moving ~`lr` per element regardless of
gradient size. It should have been applied here the first time.

It also fixes the terminology. The 24.5% is **share-weighted gradient energy measured in
the trainable window**. It is not optimizer-update energy, and this arm is what
establishes the difference.

## Explicit reallocation via rate

With the step-magnitude route closed, the meaningful version of "explicitly reallocate the
removed magnitude to content" is **rate**: if B has less to fit, it may tolerate a rate at
which A overfits and reach content A cannot. The full grid, content NLL against the
untrained 0.48779:

| lr | A content | B content | B − A | ws select, A | ws select, B |
| --- | ---: | ---: | --- | ---: | ---: |
| 3e-6 | 0.46757 | 0.46758 | +0.000008 [−0.000335, +0.000383] | 0.08340 | 0.33786 |
| **1e-5** | **0.45918** | **0.45891** | −0.000276 [−0.000551, +0.000005] | 0.07350 | 0.34461 |
| 2e-5 | 0.47779 | 0.47773 | −0.000055 [−0.000642, +0.000467] | 0.08022 | 0.32221 |
| 3e-5 | 0.51361 | 0.51353 | −0.000080 [−0.000719, +0.000603] | 0.09136 | 0.29380 |

**Every comparison spans zero, and the two content curves are superimposed** — they agree
to within 0.0001 at every rate, peak at the same rate, and degrade at 3e-5 identically.
Best content: A 0.45918, B 0.45891, both at 1e-5, 0.00027 apart.

B does **not** tolerate a higher rate. Removing a quarter of the gradient energy did not
move the optimum, did not flatten the overfitting cliff, and did not open any regime A
could not already reach. The `ws select` columns confirm the intervention is live
throughout: B sits at 0.29–0.34 against A's 0.073–0.091 at every rate.

Scope this claim explicitly: it holds across the tested range 3e-6 to 3e-5 in this
training setup — this window, corpus, horizon, optimiser and denominator. It is not a
statement about all rates or all setups.

## Final verdict on responsibility transfer

Three distinct claims were tested and all three fail:

1. **Natural reallocation.** B ≈ A at matched rate, in a regime where A improves content
   by 0.0286 nats. −0.000276, ~1% of A's own gain, interval touching zero.
2. **Explicit reallocation via rate.** B's entire LR curve lies on A's. No rate becomes
   available to B that was not available to A.
3. **The step-magnitude confound** that would have muddied either. It does not exist:
   97.8%.

*Conditional whitespace selection is a large, nearly orthogonal gradient workload — 90.4%
of the whitespace gradient energy, about a quarter of everything the trainable window
spends. Removing it removes the workload and nothing else. The optimiser does not
redirect the freed magnitude toward content, and it cannot be persuaded to by raising the
rate. The removed magnitude simply disappears.*

Arm D is not built. There is no content benefit, natural or engineered, for a sidecar to
preserve.

Distillation replication remains held, for the reason it was held: the hidden-state term
can keep teaching whitespace-selection representations even with the token-level loss
masked, which would break the decomposition this design rests on. It would now be testing
whether a *different* objective shows a separation that plain CE does not, which is a new
question rather than a confirmation of this one.


---

# Branch closed: A/B/D responsibility transfer, negative

Marked complete. No further training runs on this branch unless a new mechanism predicts
a qualitatively different outcome.

## Final defensible claim

> Conditional whitespace selection is a large and nearly orthogonal gradient workload in
> the tested Qwen3.5 window. Removing it substantially changes whitespace learning but
> leaves content learning essentially unchanged across the tested CE learning-rate range.
> The optimizer does not naturally redirect the removed workload toward content, and
> increasing learning rate does not reveal a content advantage. This branch therefore
> provides no evidence that whitespace-selection offload frees useful content capacity in
> an already-trained single-stream backbone.

## The geometry

Observed behaviour is well described by

    g_A ≈ g_C + g_WS,    g_C ⊥ g_WS

Removing the whitespace term gives `g_B ≈ g_C`, but it does **not** give
`g_C → α·g_C` for any `α > 1`. Gradient energy is not a conserved budget the optimizer
redistributes. The removed whitespace-selection direction simply disappears.

## Wording that must not drift

* Never write that whitespace learning happens "at content's expense". A and B follow
  essentially the same content curve despite large differences in whitespace-selection
  learning.
* The 24.5% is **measured share-weighted gradient energy in the trainable window**, never
  optimizer-update energy.
* AdamW is **approximately** scale-insensitive per coordinate under common conditions.
  Momentum history, `eps`, clipping, weight decay and changing gradient support can break
  the equivalence.
* The LR result is scoped to the tested range and setup.

## Arm D

Not built, and not to be built on this evidence. D would restore the whitespace-selection
capability deliberately removed from B, but there is no B content advantage to preserve.
At best D recovers whitespace performance and returns to approximately A-like overall
behaviour, which does not justify another training branch.

## Distillation replication

Held. The `0.7` sparse top-k KL + `0.3` hidden-state cosine objective does not preserve
the CE decomposition this design rests on: the hidden-state term can still teach
representations associated with whitespace selection even when the token-level selection
loss is masked. Running it now would ask a different question — *does this particular
distillation objective create structural/content interference that plain CE does not?* —
which may be worth asking later but is not a confirmation of the offload hypothesis.

## Architectural implication

This weakens the **retrofit hypothesis**: that PLE frees an already-trained backbone by
taking over easy local/structural work. Within this window, setup and rate range, it does
not.

It does **not** rule out the stronger **donor-native hypothesis**: that PLE combined with
Gated Residual / HyperConnection structure influences how representational capacity is
allocated during joint pretraining. That remains unresolved — donor GR/HC routing could
not be inspected on the available hardware — and it is the live question this branch
leaves behind.
