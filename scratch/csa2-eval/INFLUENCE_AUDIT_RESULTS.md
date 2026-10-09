# Execution, targets and optimizer influence audit

Assisted-by: Codex

Started 2026-10-08; completed 2026-10-09. Diagnostics only. No checkpoint was
written, no training stage was launched, and the inherited CE/KL/preference
objective and approved replay masks were preserved.

## Decision

The measurements do not justify adding root loss, universal gradient
equalization, or gradient projection to the next recipe. Large source gradients
exist, but the sampled combined optimizer step improves its code reference.
Gradient magnitude alone would have identified the wrong parameter family as
the largest mover. Prioritize target quality and broader executed aggregation
trajectories, with disjoint outcome evaluation and code preservation gates.

One configuration issue needs an explicit next-phase decision: selection-only
routing receives no objective gradient in this stage, but zero-gradient indexer
tensors still undergo AdamW decay. Freeze the indexer for a stage intended to
preserve routing, or deliberately train it with its separate alignment objective.
This audit has not changed that inherited choice.

The current completion candidate remains experimental; u50 remains the reference.
The sampled losses below are training-data diagnostics, not a promotion result or
an explanation of the earlier held-out regression by themselves.

## Inputs and execution

Reused the existing Python environment, two RTX 3090s, native TP model loader,
Liger, selection replay cache, streaming head, full trainer backward, TP gradient
synchronization/clipping, Kahan AdamW8bit, SpillWatch and Docker verification
sandbox. No replacement trainer, binaries, image or dataset was installed.

The candidate configuration comes from
`scratch/dense_gr/completion-v2/completion/checkpoints/smoke-r1-1-gr-s25-csa2`.
Weights and optimizer moments come from that run's exact
`state-step-00000040/state.pt`; its TP parameter layout matched strictly.
The copied steps did not resume the production pair/scheduler cursor.

The [frozen plan](../dense_gr/influence-audit-20261008/plan.json) contains 21
replay documents spanning all 16 sources and seven stratified preference pairs.
Replay coefficients reproduce source shares in the historical prefix after the
four confirmed excluded visits are removed. Within a sampled source, weighting
uses its scored token counts. Raw code is capped at 8K; other records use up to
16K where needed to reach assistant targets. Seven pairs do not reconstruct
historical pair frequencies. Neither sample is a random estimate of a population
effect; there is no significance claim from these copied steps.

The code-preservation gradient is CE on the eight sampled raw/verified/short-code
documents, normalized to their sampled source shares. It is a diagnostic
reference only: raw code retains KL-only loss in the actual combined update.
Replicated TP parameters are counted once. Norms/dots use bounded float64 CPU
reductions over complete model gradients, including the tied embedding/head.

## Target and preference audit

The [target audit](../dense_gr/influence-audit-20261008/target-audit.json)
checked **1,425 document visits and 965,332 weighted targets** across the
historical prefix after four exclusions. This corrects the earlier progress
message's 1,420 visit count. Scored system, user and tool-result targets are all
zero. Assistant, thinking, tool-call, structural and raw continuation targets
remain scored. Active teacher IDs are in vocabulary, have no duplicate top-k
entries, and have finite values; weights are finite and nonnegative.

One numerical qualification: captured probability sums range from 0.163608 to
1.000921. Small overshoots are consistent with FP16 cache rounding, although the
original full-precision capture is unavailable to establish the cause. The
existing grouped-tail loss clamps the teacher tail and does not renormalize its
head. Consequently it is not an exactly normalized KL at an overshooting row.
Renormalizing the existing head-plus-tail target would rescale that row's
student gradient by its inverse total mass; the largest measured overshoot is
only 0.0921%. This is a numerical cleanup candidate, not evidence of a large
outlier mechanism or a reason to discard uncertain positions with substantial
tail mass. Document maxima do not give the affected-position frequency.

Teacher top-1 matches are not a target-quality certificate. The lowest agreement
is general raw continuation, where alternate wording can be appropriate. Current
teacher-code targets have high agreement and no actual-token omissions from the
captured top-k. Premature stop below means a teacher top-1 stop while the actual
next token is not a stop, weighted over active KL targets.

| Source | Replay target share | Teacher top-1 match | Actual token absent from top-k | Premature stop |
| --- | ---: | ---: | ---: | ---: |
| Raw code | 33.607% | 87.221% | 0.639% | 0.0003% |
| Teacher code | 5.311% | 96.803% | 0.000% | 0.0215% |
| QA answers | 3.290% | 78.792% | 0.277% | 0.2582% |
| General continuation | 6.761% | 64.085% | 4.183% | 0.0015% |
| Expanded code | 5.947% | 84.820% | 0.124% | 0.5452% |

The saved confident-disagreement examples cover only the first 80 collected
examples, not an unbiased sample of all sources. Disagreement often means a
plausible word or whitespace alternative. Masked tool-result predictions are not
part of the direct objective; the earlier teacher end-of-turn concern does not
establish corruption of this current masked objective.

All **222 chosen preference responses** match regenerated world prefixes and
expected rendered actions, with valid token IDs, assistant spans, finite
reference scores and no chosen execution errors. The generic student terminal
certifier does not recertify 52 negatives: 24 denials of verified completion,
12 retries without checking an uncertain outcome, ten false completions and six
unnecessary calls. These are factual truth, safe sequencing and efficiency
contrasts, not 52 demonstrated bad labels. Schema-valid execution cannot certify
a semantic preference. See the
[label review](../dense_gr/influence-audit-20261008/pair-label-review.json) for
the specific distinction; historical labels were not changed.

## Code target execution and test strength

The [execution audit](../dense_gr/influence-audit-20261008/code-execution-audit.json)
reexecuted 16 retained served final solutions: eight teacher-code and eight
short-code targets, including all five verified code documents in the GPU sample.
All 16 pass their inherited tests. Of 25 controlled return/comparison mutations,
23 are rejected (22 failures and one timeout); two survive. The timeout is a
pathological mutated loop and counts as rejection.

Manual review establishes one equivalent mutation: changing integer maximum
selection from `>` to `>=` produces the same value on equality. The other survivor
reveals a **confirmed nonfunctional email target**,
`onpolicy:kodcode:Filter_62706_I:s0`. Its inherited SMTP mock lacks
`set_debuglevel`; the solution catches that exception and silently returns False.
Its tests check neither return nor send effects. With a complete mock, it passes
a tuple to `sendmail` instead of a serialized string/bytes message. The
[stronger fake-SMTP check](../dense_gr/influence-audit-20261008/manual-quality-checks.json)
fails the original solution. No real mail or network action was performed.

This document was outside the recent training prefix and GPU sample, so it does
not explain their measured behavior. The
[quality proposal](../dense_gr/influence-audit-20261008/quality-quarantine-proposal.json)
flags it for requalification or omission from future replay. Historical targets
are unchanged. This is a data-quality finding separate from the approved policy
of excluding only confirmed matching benchmark tasks. A 16-target mutation
sample is not an exhaustive quality assessment of the retained pool.

## Gradient scale, direction and real updates

The [full GPU results](../dense_gr/influence-audit-20261008/gpu-audit.json) show
large variation at equal per-document normalization. Individual gradient norms
range from about 51 to 471. A curriculum document with 2.64% of historical replay
targets has norm 471; raw-code documents have norms 51-71. These are sampled
documents, not source-population mean norms. Weighted individual norms cannot be
added and interpreted as net influence, because vectors cancel.

Replay's aggregate norm is **38.169** and its cosine to code CE is **+0.3285**.
The stratified preference mean, already scaled by pair weight 0.1, has norm
**65.099** and code cosine **-0.0179**. Their mutual cosine is **-0.0074**.
The actual recomputed combined gradient has norm **75.310**, code cosine
**+0.1508**. Its norm is within 0.12% of the norm predicted from separately
accumulated replay/pair vectors; BF16 accumulation and recomputation need not
match an FP32 vector sum bitwise.

The QA and on-policy loop samples have code cosines -0.0816 and -0.1837.
Several preference examples also conflict weakly. That does not justify removing
their behaviors; their purpose may differ from code. Separate CE/KL gradients
are aligned on the sampled teacher-code and QA records (cosines +0.8622 and
+0.7681). This sample does not support a general teacher-versus-CE direction
failure. On the opening aggregation pair, DPO and chosen CE have cosine +0.1842;
their weighted norms are 23.945 and 4.111. Existing DPO uses summed response
log-probabilities, while its chosen CE is a token mean, so their relative
influence depends on response length and preference saturation.

MLA accounts for **99.43% of combined squared gradient norm**, but only **0.38%
of saved-moment compensated squared update norm**. Its rate is smaller and Adam
rescales coordinates. Thus family gradient norm alone would suggest the wrong
family to normalize. MLP gradient concentration is real: the largest per-shard
top-1%-of-rows energy is 94.3%, with median 19.2% for gate projection, 18.2% for up
projection and 4.3% for down projection. Full-forward MLP activation peaks reach
75.8 times aggregate RMS and max/median channel RMS reaches 40.5. These statistics
include input/context/padding activations and per-shard gradient rows. They do
not demonstrate that the large channels cause harmful updates. The stored
stride-sampled activation tail fractions may alias channels and should not be
used as precise tail estimates.

Two discarded optimizer steps use the same actual combined gradient, production
global clip at norm 1, and saved rates. The preclip norm is 75.344 (BF16
reconstruction), clip scale 0.013272. Fresh/saved moments differ deliberately.

| Copied step | Visible BF16 update norm | Compensated update norm | Code CE dot visible update | Measured code NLL delta |
| --- | ---: | ---: | ---: | ---: |
| Fresh moments | 0.006212 | 0.015259 | -0.001071 | **-0.001029** |
| Saved step-40 moments | 0.211396 | 0.011301 | -0.000385 | **-0.000517** |

Baseline sampled weighted code NLL is 0.836394. All eight documents improve in
the fresh step; seven improve with saved moments and one increases by 0.000023.
A negative dot predicts improvement locally. The compensated delta includes the
Kahan remainder change, while forward evaluation sees the rounded BF16 weights;
these are distinct measurements. Inherited compensation can make many weights
cross a BF16 rounding boundary in one step, increasing visible movement without
an equally large new compensated increment. No held-out or functional gain is
claimed from these training-data NLL changes.

The main diagnostic took 890 seconds, with peak allocations **9.52/9.69 GiB**.
SpillWatch had 93 valid process samples, shared-memory increase 0.086 GiB and no
spill. Its 1 GiB tolerance is the existing guard, not a claim of zero host-backed
GPU allocation. Diagnostics copy full gradients to CPU for offline analysis;
that is not a proposed per-step training implementation or throughput measure.

## Remaining interpretation and next-phase gates

Keep the existing assistant-only conversational objective and raw continuation
policy. Increase verified-document and complete-trajectory diversity before
increasing repeated raw-code weight. Requalify weak-test targets, distinguish
valid-but-unnecessary actions from invalid calls, and measure fresh disjoint
aggregation paths that require reads after search and continuation after a page.
Retain empty/direct-answer/recovery and code/math/knowledge gates.

Automatic magnitude balancing is unsupported by this small sample. If later
multi-step, multi-sample audits identify disproportionate harmful movement,
test a cap that leaves ordinary verified examples unchanged. If they identify
direction conflict, test preservation constraints separately. Equalizing every
example or removing hard examples because their loss is large could reduce the
rare behavior this next phase needs to learn.

The [focused audit](../dense_gr/influence-audit-20261008/focused-audit.json)
adds five existing training transitions, rather than fitting new examples to
evaluation failures:

| Existing transition | Weighted gradient norm | Cosine to code CE | Chosen response token NLL |
| --- | ---: | ---: | ---: |
| Aggregate 2, next page | 14.980 | -0.0301 | 0.00339 |
| Aggregate 2, following record read | 34.545 | -0.0108 | 0.04607 |
| Overlap 4, next page | 11.573 | -0.0601 | 0.00122 |
| Overlap 4, following record read | 25.291 | +0.0330 | 0.01817 |
| Aggregate 2, completed total | 456.130 | -0.0424 | 0.96718 |

For the first next-page pair, DPO and chosen CE have cosine +0.1242 and weighted
norms 11.504/8.219. Their separate code cosines are -0.0542/+0.0218. DPO is not
simply another scaled chosen-CE gradient. Low NLL on four trained action strings
and a harder final numeric response favor diverse executed arithmetic/trajectory
coverage over repeating the same calls. These means include protocol closing
tokens and do not establish unseen-world success. The reference gradient norm
changes by about 0.09% between independent backward passes; bitwise backward
repeatability is not claimed. The focused pass took 346 seconds and peaked at
7.10/6.37 GiB, without spill.

The [CPU summary](../dense_gr/influence-audit-20261008/audit-summary.json)
inspects the saved optimizer's deliberately wrapped uint8 moment state. All six
`index_q_proj.weight` tensors have exactly zero first/second moment scales at
step 40, and receive present-but-zero gradients in all five focused pairs. The
18 other indexer tensors have no gradients in those passes. Fresh/saved
compensated indexer movements match pure decay within 0.008%/0.346%, including
FP32 working and BF16 compensation rounding. Their rate is 0.000937, decay 0.1:
roughly 0.00937% shrink per such step at that rate. Only query projections move;
this is not uniform shrinkage of every router component. The main saved copied
step puts 90.92% of compensated squared movement and 99.84% of visible squared
movement into this untrained family. No routing-set or held-out harm is inferred
from those norm shares alone. The original focus receipt's top-level moment
max fields are null because the moments reside in the wrapper; the CPU summary
contains the verified zero scales.

Freezing selection-only router parameters uses the existing complete router
filter and prevents this drift. It can also avoid zero-gradient synchronization
and optimizer state/compensation for those query tensors. The measured query
weights contain 6.29M elements, so the memory opportunity is modest, not a major
new throughput claim. An explicit sparse alignment stage is a separate objective
and is incompatible with the current streaming-head path; it should be planned
and tested separately rather than slipped into this preservation stage.

## Reproduction and checks

Run from the DistillKit root using existing tooling:

```powershell
.venv/Scripts/python.exe -X utf8 scratch/dense_gr/influence_audit.py prepare
.venv/Scripts/python.exe -X utf8 scratch/dense_gr/influence_audit.py code
$env:CUDA_VISIBLE_DEVICES='0,1'
.venv/Scripts/python.exe -X utf8 scratch/dense_gr/influence_gpu.py
.venv/Scripts/python.exe -X utf8 scratch/dense_gr/influence_focus.py
.venv/Scripts/python.exe -X utf8 scratch/dense_gr/influence_summary.py
.venv/Scripts/python.exe -X utf8 -m pytest tests/test_dense_gr_training_step.py tests/test_completion_curriculum.py tests/test_completion_train.py tests/test_code_replay_review.py -q
```

Preparation/completed GPU audits refuse overwrite; preserve this receipt before
using a new output directory/date. `-X utf8` is needed for existing default-
encoding reads of code prompt files on this Windows installation. The initial
attempt without it failed before sandbox execution. The initial attempt to force
CPU testing with an empty CUDA device variable left CUDA visible to torch while
bitsandbytes selected its CPU library; two optimizer tests failed for that
environment mismatch. The corrected explicit two-GPU invocation passes **all
30 focused tests**, including quantized optimizer resume and TP equivalence.

The frozen plan validates run, exclusion, batch, pair, helper and cache manifest
hashes; selected token prefix hashes are checked before backward. Manifest hashes
are not full cache payload hashes. See the result source hashes and recorded
run [provenance](../dense_gr/influence-audit-20261008/provenance.json) when
reproducing exact diagnostics. The 12.57 GB saved state was streamed through
SHA-256 after the main audit; its original size/mtime and source/config hashes
are recorded. No source or state was changed while the GPU passes ran.
