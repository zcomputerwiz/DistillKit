# Targeted training: map what the student represents where, then train there

## Current decision (2026-10-05)

The completed [role-loss research review](ROLE_LOSS_REVIEW.md) recommends keeping
assistant-only supervision, with user/tool context visible. Tool-output NLL is
a diagnostic rather than an agent-success gate. The completed
[tool behavior screen](tool-behavior/RESULTS.md) compares base, round-5 step 100,
ramp, flat and context. All retain simple returned-value extraction; round-5
and context frequently omit calls. Explicit tool instructions recover part of
the gap. Flat is the preferred confirmation candidate, given tool behavior
comparable to ramp and better QA retention. No longer recipe or new preservation
objective has been adopted. The validated streaming/cache optimization is
available for the next run; further CCE work is optional.

October 5 execution diagnostics are recorded in [TRAINING_PROFILE_RESULTS.md](TRAINING_PROFILE_RESULTS.md).
The subsequent [streaming head experiment](STREAMING_HEAD_RESULTS.md) combines
immediate chunk backward and per-frame selection reuse for a measured 10.00%
whole-step gain on the frozen 32K pair. Under the user's practical numerical
criterion, post-update prediction changes are no more frequent than an ordinary
repeat on the tested prefix. Both are now optional trainer flags, with existing
defaults preserved; the larger-chunk test is separate from those gains.
The opt-in per-checkpoint selection cache improved matched 32K step throughput
by 4.96%, with unchanged allocated peaks. GPU backward repeatability remains a
validation limit even in unchanged single-GPU controls; bitwise equality is no
longer the acceptance gate after the user's clarification. CCE stable forward accuracy is verified; complete KL/UL backward
and any scheduling changes enabled by its memory savings remain unimplemented.

Goal: efficient distillation rounds that improve a target skill without regressions elsewhere,
and so without the post-hoc base blends (u50, ramp0-70) every round has needed so far.
Method: first build two maps of the 2B hybrid student, then use them to decide which
components each kind of data may change.

- **Atlas**: what each layer and component represents and computes, per domain.
- **Change map**: what a round changed where, and which changes produced its gains and its regressions.

## Revisions after the Codex review (capture-data/frontier/codex-review-targeted/REVIEW.md)

**First establish what regressed.**
- The long-context probe scores every token, and its agent documents are not what training scores:
  - **claude-code:** system prompt up to 8K, 85% tool output at 16-32K.
  - **codex:** system prompt to 4K, then a third tool output.
- Only 2 claude-code documents and 1 codex document reach 32K.
- Round 5's "+0.18 on agent traces" is therefore mostly tool-output prediction. Held-out loss on assistant turns, which is what training scores, improved (0.640 to 0.562).
- `atlas.py nll` is now the regression ledger:
  - held-out conversations from the agent captures' eval splits, by harness;
  - every token labelled by chat role;
  - a 95% bootstrap interval over documents.
- The screen's GSM8K non-thinking proxy allows 512 new tokens, against 2,048 in the benchmark. Round 5 learned the teacher's longer native answers, so its 5 → 55 "unboxed" answers there are probably truncation.

**Cheap causal controls before any map-guided intervention** (`control_arms.ps1`):
- Round 5's recipe stopped at step 100 of the same 10M schedule, compared with round 5's own step-100 state.
- The depth ramp and a flat 0.55× rate (the ramp's mean). The proposed no-new-data arm was disabled in the runner and has not been tested.

**Change map: a screening statistic only.**
- CSA2's discrete top-k makes the path non-smooth, and the HVP correction is unavailable: the hyper-connection kernels are `once_differentiable`.
- Exact reverts decide, starting with coherent families (`atlas.py revert`: embedding/head, norms, decay, write strength, DeltaNet, MLA, indexer, MLP, hyper-connections, depth thirds).
- The greedy result is a repair *candidate*, not a minimal set.

**Shields wait until the evidence shows separable damage.**
- Batches mix sources: the groups are keyed by width and objective, not capture.
- AdamW moves zero-gradient weights through momentum, weight decay and Kahan compensation.
- Hooks would have to sit on the parameters after tensor-parallel sharding.
- Better early levers: the depth ramp, a lower loss weight on the new sources, and cached KL-to-base on protected replay.

**Ledger statistics:** paired document bootstrap intervals, the signed contrast G_arm − 0.8·G_A, both 1→0 and 0→1 transitions, and a full-budget confirmation with more seeds.

## Phase 1 results: round 5 against its base (scratch/csa2-eval/atlas/long5)

**Ledger.** Round 5 unblended, loss change against long1-u50, by the role of the predicted token:

| domain | own turns | tool output | system | user / document |
|---|---|---|---|---|
| agent claude-code | -0.100 | +0.324 | +0.135 | +0.033 |
| agent codex | -0.111 | +0.307 | +0.162 | -0.021 |
| agent mini-swe | -0.085 | +0.211 | +0.035 | +0.175 |
| tools | -0.087 | -0.019 | -0.013 | -0.253 |
| teacher code | -0.026 | | | -0.031 |
| thinking math | -0.060 | | | +0.017 |
| non-thinking math | -0.051 | | | +0.038 |
| QA | -0.452 (answers) | | | +0.156 (the documents) |
| llama.cpp source (plain) | -0.019 | | | |

- Everything the model writes improved.
- What regressed is predicting other people's text inside chats. The assistant-only caches (agent, tools, QA) put no loss there.
- Round 4 u50 drifts the same way (tool output +0.06), so the problem predates round 5. This does not rule out additional damage from round 5's new data; the matched no-new-data control has not been run.
- The blends trade it off:
  - u50: tool output +0.06, own turns -0.06;
  - ramp0-70: no context regression, own turns -0.04.

**Family reverts** (round 5 with one family back at the base; change against round 5):
- **Layers 0-7** hold the whole context regression: tool output -0.33/-0.30, QA documents -0.19. Reverting them costs claude-code about a quarter of its own-turn gain (+0.024 of -0.100), and QA about half (+0.24 of -0.45).
- **Layers 8-15:** tool output -0.09/-0.11. **Layers 16-23:** none (+0.02).
- **By kind:**
  - MLPs: tool output -0.25, own turns +0.04;
  - DeltaNet: -0.15, +0.005;
  - MLA: -0.11, +0.015 (QA answers +0.085: long-range retrieval);
  - norms, decay, write strength, indexer, hyper-connections and the embedding/head: within ±0.006.

**Reading.** The separate depth and module reverts pointed to shallow MLPs. Their intersection was then tested directly, as documented below. The gains are spread across depth.

**Two levers were tested as step-100 arms against round 5's own step 100:**
- **The depth ramp** (`control_arms.ps1`: ramp, and flat 0.55× as its control).
- **Teacher KL alone on the context tokens** (`--context-kl 0.01 --context-every 8`, Codex-reviewed `CONTEXT_KL.md`; `context_arm.ps1`). It keeps the student's expectations there on the teacher's, with no cross entropy on text the teacher did not write.

## Follow-up diagnostics and step-100 controls (2026-10-05)

The missing intersection test is complete in `atlas/long5-mlp-intersection`:
round 5, its base, the shallow MLP revert, and the remaining MLP revert, all on
the same frozen domains. The base and round-5 results exactly reproduce all
84 comparable cells in the earlier family-revert ledger.

| MLPs reverted | Claude tool NLL change vs round 5 | Codex tool NLL change vs round 5 | Claude own-turn cost | Codex own-turn cost |
|---|---:|---:|---:|---:|
| Layers 0-7 | -0.234 | -0.221 | +0.009 | +0.009 |
| Layers 8-23 | -0.035 | -0.055 | +0.026 | +0.029 |

The shallow MLP revert retains 90.85%/91.52% of Claude/Codex own-turn gains,
94.55% of teacher-code own-turn gains, and 101.57% of plain llama.cpp code
gains. The paired signed 80%-retention contrasts are positive, including their
95% intervals, for those domains. QA answer retention is 88.10%, but only
three of sixteen QA documents contain scored answers (1,502 targets), so that
estimate has limited coverage. Claude/Codex tool-output regression remains
+0.090/+0.086 nats against the base. The shallow MLPs explain much of the
damage, but reverting them alone does not repair it completely. This is a
post-training causal revert, not evidence that freezing them during training
will produce the same outcome.

The existing whole-MLP and whole-shallow-third tests were already complete;
they were not repeated. Per-token NLL and top-1 observations are now retained
by the optional `atlas.py --save-token-evidence`, with document, model-source,
and checkpoint provenance. `atlas_compare.py` computes jointly paired
retention intervals and both 1-to-0 and 0-to-1 transitions from that evidence.

The two step-100 training controls finished before these diagnostics:

| Arm | Claude tool delta vs base | Codex tool delta vs base | Claude own-turn delta | Codex own-turn delta | QA answer delta |
|---|---:|---:|---:|---:|---:|
| Round 5 step 100 | +0.055 | +0.031 | -0.047 | -0.052 | -0.325 |
| Depth ramp 0.1 to 1.0 | -0.011 | -0.012 | -0.043 | -0.047 | -0.236 |
| Flat 0.55 rate | +0.045 | +0.029 | -0.045 | -0.048 | -0.294 |
| Context teacher KL W=0.01, K=8 | +1.499 | +1.488 | -0.046 | -0.051 | -0.329 |

The ramp reduces the measured tool regression more than the flat control,
while retaining most agent own-turn gains and less of the QA gain. This is
one seed and a short prefix of the 10M schedule, not a full-budget winner.
The context-KL arm completed 100 steps with exactly 2,255,244 original targets,
939.5 end-to-end tok/s, held-out loss 0.775074 and peak reserved memory 19.445 GiB.
It retained 97.7%/98.5% of Claude/Codex own-turn gains but strongly worsened tool
prediction. This fails the original context-prediction preservation criterion;
the user's later distinction between predicting and using tool results means
that this metric alone must not disqualify an arm on tool-use grounds. Do not
launch a longer recipe until the intended behavioral criterion is settled.
`control-arms/ledger-context/paired_comparison.json` uses jointly paired document
bootstrap samples for all four step-100 arms and the base. The ramp's Claude/Codex
own-turn signed 80% contrasts have positive intervals; its teacher-code contrast
crosses zero and its original QA contrast is negative. The original QA set has
only 3/16 answer-bearing documents; the completed 32K extension is reported below.

`control-arms/ledger-context/teacher_role_audit.json` examines the raw unsuppressed
cached teacher on those frozen token prefixes. The actual next tool token is
absent from top-64 in 46.0%/49.0% of Claude/Codex positions; teacher top-1 is only
26.5%/21.2%. Teacher KL alone can penalize the actual context tokens. This is a
plausible objective conflict, not proof of its entire causal contribution. A
frozen-u50 distribution anchor would measure preservation more directly, but is
a different objective requiring an explicit choice before another arm. Existing
tests establish shared/separate objective parity, not suitability of the targets.
The completed QA extension keeps the same 16 documents and exact original token
prefixes, with every answer-bearing document retained: 8,773 own-turn targets
instead of 1,502 in only three documents. All ten other domains are unchanged.
On this extension, own-turn deltas are ramp -0.23644, flat -0.28176, context
-0.32019. Retention of A's gain is 75.25%, 89.68%, 101.91%, respectively. The
ramp's signed 80% contrast is -0.01491 with paired 95% interval
[-0.02264, -0.00660]. Thus the broader answer set confirms the QA tradeoff.
Artifacts: `control-arms/ledger-context-qa32768/paired_comparison.json` and
`scratch/csa2-eval/atlas/domains-qa32768-20261005-checks.json`.

Teacher tool predictions are not predominantly turn endings: raw top-1
`<|im_end|>` rates are 2.63%/2.20% for Claude/Codex, and `<|endoftext|>` is never
top-1. Common predictions are commas, digits, spaces and newlines; some are
confident incorrect content guesses (e.g. 98.9% on `0` where the returned digit
is `1`). These numbers refer to tool-result content labels, excluding the
structure labels of actual turn-ending tokens. The audit JSON retains examples,
cached stop probability and top-token counts.

Tool results remain masked in the ordinary assistant-only objective; the
experimental context KL added a separate teacher loss on those positions.
Tool-call generation remains supervised separately. Tool-result prediction NLL
is a diagnostic, not a direct test of correctly reading returned results or
choosing tools. The user raised this distinction; do not infer degraded tool
use solely from the prediction regression, or silently change the preservation
objective to a frozen-student anchor. Result-conditioned behavior needs its own
evaluation before choosing a longer training recipe. All three arms improve
teacher-forced tool-call token NLL. Context deltas are -0.02477 (Claude),
-0.01940 (Codex), and -0.01321 (tools), with paired 95% intervals below zero.
Context tool-call top-1 rises from 92.47% to 92.83% (Claude), 91.19% to 91.53%
(Codex), and 95.28% to 95.35% (tools). This is evidence about predicted call
tokens, not executed-call correctness or result-conditioned behavior.

The short generation proxy reports context code NLL 0.8155, GSM8K 73.0% (24
unboxed), and MATH 41.0% (107 truncated); retain the existing 512-token cap when
comparing these proxies and do not mistake them for full benchmark scores.

These frozen agent domains are capped at 16K. The comparison cannot establish
the proposed 32K retention criterion without a separate long-context check.

Assisted-by: Codex

## What we already know (DISTILLATION.md)

- **Loops (rounds 8b/8c):** the loop fix lives in the deep layers. Splicing round 8b's deep layers under round 6's shallow ones keeps it (98 and 11 truncations); a uniform blend dilutes it.
- **Non-thinking code needs coordination across depth.** A shallow/deep splice did worse than either parent (HumanEval+ 38.4% against 40.9% and 44.5%), while the uniform blend restored it (43.3%).
- **`--lr-depth-ramp 0.1 1.0` during training forgot less.** Unblended, non-thinking GSM8K scored 69.1% against 62.1%. It was used in on-policy rounds 8c through 9b. The long rounds 1-5 don't use it, and no recorded reason explains why.
- **Round 5 (unblended) against round 4's u50 blend:**
  - Code NLL is better at every length (32K: 0.506 against 0.516; base 0.542).
  - Agent-trace NLL is worse at every length: +0.10 to +0.18 nats against the base, on both claude-code and codex.
  - This is the immediate test case.

## What the literature says works

**Localizing changes and functions**
- **Cross-model activation patching** (CMAP; Prakash et al., ICLR 2024): write one model's component outputs into the other's on the same inputs. Fine-tuning mostly *enhances existing circuits* rather than replacing them.
- **A four-stage funnel** (weight-delta prior, per-unit patch scoring, control-set check, greedy stacking) found 17 attention heads that repaired a bad 7B fine-tune: 27 safe answers against 30 for the clean model, ARC unchanged.
- **Attribution patching:** a first-order, gradient-times-difference estimate of every patch from one forward and one backward pass. A Hessian-vector-product correction (one more backward) removes its leading error (arXiv 2606.09899, 124M-9B). The recommended workflow is screen, flag, then fix with exact patching.
- **Forgetting is mostly misaligned activation, not erased knowledge:**
  - Function-vector analysis: forgetting comes from biased activation of functions, not overwritten functions (arXiv 2502.11019).
  - "Spurious forgetting" (ICLR 2025): lost task alignment early in training is recoverable, and freezing the bottom layers helps.
  - Mechanistic forgetting study (arXiv 2601.18699): disruption concentrates in lower-layer attention heads (15-23%) and in intermediate-layer CKA drops (0.32-0.47). Gradient alignment predicts forgetting severity (r = 0.87).
- **Activation differences** between base and fine-tune are clearly readable even on unrelated text (Minder et al., ICLR 2026, 1B-32B).
- **Crosscoders** recover features that tuning created or destroyed: the BatchTopK crosscoder (arXiv 2504.02922) and the Delta-Crosscoder for narrow fine-tunes (arXiv 2603.04426, 1B-9B).

**Training with fewer regressions**
- **Selective parameter updates do best.**
  - Source-Shielded Updates (SSU; ACL 2026, arXiv 2512.04844): score parameters by Wanda importance, |W| times the input activation's L2 norm, on about 500 source documents. Sum the scores by column and freeze the top 50% of columns.
  - Source degradation falls from 20.3% under full fine-tuning to 3.4-5.9%, with the target gain kept.
  - Column freezing beats row freezing, which beats element freezing. Data-informed scores beat random and magnitude-only choices.
- **Gradient routing** (Cloud et al., arXiv 2410.04332): data-dependent gradient masks, so chosen data updates only chosen parameters.
- **Anchoring toward the base:**
  - Fisher/EWC weighting and its layer- and element-wise variant (arXiv 2501.13669) are about 20x cheaper than classic EWC.
  - Null-space projection (AlphaEdit, ICLR 2025; CrispEdit) projects updates away from the covariance of preserved activations, leaving preserved outputs unchanged.
- **On-policy data forgets less.**
  - RL's Razor (arXiv 2509.04259): forgetting tracks the KL from the base on the *new* task.
  - "Retaining by Doing" (arXiv 2510.18874): on-policy data is the driver, and approximately on-policy data suffices.
  - A distillation-dynamics study (arXiv 2609.35259): the KL direction and the learning rate matter more than the rollout policy, and the learning rate governs forgetting.
- **Depth-scaled task vectors** (LiNeS, ICLR 2025): scale layer ℓ's update by γ + (1-γ)(ℓ-1)/(L-1). This is our ramp blend, applied after training. AdaMerging learns per-layer merge coefficients.
  - Caveat: a large-scale study finds merging "does not reliably mitigate forgetting" (arXiv 2510.17776).

## The student, as units

- 24 layers. Gated DeltaNet at 18 of them: 16 value heads of 128. MLA+CSA2 at layers 3, 7, 11, 15, 19 and 23: 8 heads of 256, a latent of 384, and the indexer.
- An MLP at every layer (6144 wide).
- A two-branch hyper-connection residual (`attn_residual` / `mlp_residual`: read, write, rank-64 mixing).

Units, coarse to fine:
1. **Sublayer (96):** 24 mixers, 24 MLPs, 48 residual read/write modules.
2. **Head (336):** 18×16 DeltaNet heads (the input of `linear_attn.out_proj`) and 6×8 MLA heads (the input of `self_attn.o_proj`).
3. **Column:** the input columns of every Linear. This is SSU's granularity, and what a training mask needs.

Hooks go on our own modules (`widened.py`). Off-the-shelf interpretability libraries don't support this architecture.

## Domains (fixed held-out sets, the same documents for every arm)

- **Agent traces:** claude-code and codex, from the long-context probe's SmolDataEnvs sources, at 2K and 32K.
- **Code:** llama.cpp source (the probe), plus the teacher-code eval split.
- **Thinking math:** the teacher-math-gen eval split.
- **Non-thinking math:** the teacher-nothink eval split.
- **Q&A answers:** the frontier-qa2 eval split, answer spans.
- **General text:** the general-pilot eval split.
- **Long-range retrieval:** pass-key at 4-32K.
- **Behaviour contrasts:** loop onset (loop-check documents, the first repeated n-gram against matched non-repeating positions); the thought-closing decision (`</think>` positions); tool-call formatting (JSON spans in agent traces).

## Phase 1: the maps, on the base (long1-u50) and round 5

**1a. Component importance per domain, base model (atlas).**
- Ablate each unit by resampling: replace its output with its output on a different document of the same domain. Record ΔNLL per domain.
- Screen all heads and sublayers with attribution patching, one forward and one backward per domain. Apply the HVP correction to the flagged units, and confirm the top 40 exactly.
- Output: a unit × domain importance matrix, which shows which components each skill depends on.

**1b. Where predictions form (atlas).**
- Logit lens per layer, through the final norm and the collapsed two-branch stream: the KL to the final distribution and top-1 agreement, per domain and per behaviour contrast.
- Compare base against tuned, to see whether a round moves the depth where a domain's predictions form.

**1c. What is linearly present where (atlas).**
- Logistic probes per layer, on the collapsed stream and on each branch: thought against answer, code against prose, inside a tool call, loop onset, thinking against non-thinking mode.
- This shows where information is available, which complements 1a's what is *used*.

**1d. Change map.**
- Relative ‖ΔW‖ and effective rank per unit.
- Cosine and CKA between base and tuned activations per layer and per domain.
- **Path-attributed loss change per column and unit:** ΔL_D(c) ≈ ½(g_base + g_tuned)·Δθ_c, from one backward pass on each model per domain.
  - This is the trapezoid rule along the straight line from base to tuned, exact for a quadratic loss.
  - Completeness check: the column scores must sum to the measured ΔL_D. If they miss by more than 10%, add a midpoint gradient (Simpson's rule).
- Output: for every unit, its share of the agent-trace regression and of the code gain.

**1e. Validation by exact revert.**
- Revert the top-K units' weights to the base and evaluate every domain.
- Then build a greedy minimal repair set, the 17-heads funnel in weight space: the fewest units whose revert removes the agent-trace regression while keeping at least 90% of the code gain.
- If no small set does it, the regression is diffuse, and Phase 2 uses anchoring instead of masks.

**1f. (Optional, after 1a-1e) Features.**
- BatchTopK or Delta-crosscoders, base against tuned, at the 2-3 layers 1a-1e flag. About 50-100M tokens of activations, a few hours per layer on one 3090.
- Features are labelled by their top activating contexts, with an LLM labeller.

**Compute:** 1a-1e take about 3-5 GPU hours on the two 3090s, forward-only except the gradient passes. 1f is optional.

## Phase 2: training from the maps, as short A/B runs

All arms use round 5's recipe from long1-u50, seed 25, at a 3M-token budget (about 55 minutes each), scored on the Phase 1 domains with the regression ledger below.

- **A. Baseline:** round 5's recipe as is.
- **B. Depth ramp** (`--lr-depth-ramp 0.1 1.0`): the existing, LiNeS-like lever.
- **C. SSU shield.**
  - Score columns by Wanda importance on base activations over the protected domains (agent traces plus replay).
  - Freeze the top p% of columns. Embeddings and the head stay trainable.
  - The shield is gradient-routed: the new-skill data (teacher code and non-thinking math) cannot move shielded columns, while replay batches update everything.
- **D. Map-guided shield:** like C, but the shielded columns are the ones Phase 1d blames for the regression, minus those carrying the gain. New-skill data is also steered toward the units 1a says the target skill uses.
- **E. If Phase 1 finds the regression diffuse:** soft anchoring instead of a hard mask, an importance-weighted L2 to the base on protected columns, or null-space projection of new-skill gradients against the protected activations' covariance.

**Implementation:** one gradient-mask hook per Linear (column masks, routed per batch source), plus an optional anchoring term. Masks are bits per column, so memory is negligible next to the 19.45 GiB peak, and so is speed.

## The regression ledger (success criteria)

Every arm is compared with the base on the same documents:
- **Per-domain NLL:** the agent-trace delta must be no worse than +0.01 nats at every length from 512 to 32K, and code must keep at least 80% of arm A's gain.
- **Sample-wise forgetting:** count the tokens and documents the base got right (top-1) and the arm gets wrong, the 1→0 transitions of arXiv 2510.17776.
- **Pass-key:** 100% at 4-32K.
- **Behaviour checks:** proxy GSM8K and MATH, and the fresh-bank loop gate.

**Target:** an unblended arm that meets the ledger without the base blend, at no more than 10% extra training cost.

## Order

1. Build the hooks, domain sets and Phase 1 tools now; test them on CPU and tiny models.
2. Run Phase 1 on the GPUs once round 5's evaluation is done (around 12:45).
3. Phase 2 arms A-D: about 4 hours of training, plus evaluation.
4. Round 6 uses the winning arm at full budget.

## Sources

[Surgical fine-tuning](https://arxiv.org/pdf/2210.11466) ·
[Localize-and-Stitch](https://arxiv.org/abs/2408.13656) ·
[AdaMerging](https://arxiv.org/pdf/2310.02575) ·
[LiNeS](https://arxiv.org/html/2410.17146v1) ·
[Crosscoder chat-tuning artifacts](https://arxiv.org/pdf/2504.02922) ·
[Delta-Crosscoder](https://arxiv.org/pdf/2603.04426) ·
[Narrow finetuning traces](https://arxiv.org/pdf/2510.13900) ·
[Attribution patching outperforms ACDC](https://aclanthology.org/2024.blackboxnlp-1.25.pdf) ·
[When attribution patching lies](https://arxiv.org/html/2606.09899) ·
[Fine-tuning enhances existing mechanisms](https://arxiv.org/pdf/2402.14811) ·
[17 heads repair a bad fine-tune](https://www.greaterwrong.com/posts/6iPEuEnguEtmtaqyJ/function-vectors-as-a-model-diffing-tool-17-heads-repair-a) ·
[Function vectors and forgetting](https://arxiv.org/pdf/2502.11019) ·
[Spurious forgetting](https://proceedings.iclr.cc/paper_files/paper/2025/file/a774503daed55eb53c634847ae071ec7-Paper-Conference.pdf) ·
[Mechanistic analysis of forgetting](https://arxiv.org/html/2601.18699v1) ·
[Source-Shielded Updates](https://arxiv.org/abs/2512.04844) ·
[Gradient routing](https://arxiv.org/abs/2410.04332) ·
[Layer/element-wise regularization](https://arxiv.org/pdf/2501.13669) ·
[AlphaEdit](https://arxiv.org/pdf/2410.02355) ·
[RL's Razor](https://arxiv.org/abs/2509.04259) ·
[Retaining by Doing](https://arxiv.org/pdf/2510.18874) ·
[On- vs off-policy distillation dynamics](https://arxiv.org/html/2609.35259v1) ·
[Post-training forgetting at scale](https://arxiv.org/pdf/2510.17776) ·
[Catastrophic forgetting is low-rank](https://arxiv.org/html/2606.18024)
