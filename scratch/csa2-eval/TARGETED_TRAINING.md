# Targeted training: map what the student represents where, then train there

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
- Three arms: no new data, the depth ramp, and a flat 0.55× rate (the ramp's mean).

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
