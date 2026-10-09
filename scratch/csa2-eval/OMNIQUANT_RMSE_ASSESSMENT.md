# Reconstruction loss, outliers and student update influence

Assisted-by: Codex

2026-10-08. Research and synthetic CPU checks only. No training objective,
checkpoint, model implementation or replay policy changed. The user clarified
that the interest is transferable concepts for handling outliers, rather than
directly adopting RMSE.

## Decision

Borrow the principle of measuring and limiting disproportionate influence at a
meaningful granularity. First distinguish data-quality problems, gradient scale
imbalance, gradient direction conflict and excessive parameter movement. They
need different remedies. A square root of the current CE/KL/preference objective
does not establish that distinction or fix all four problems.

The best first investigation is a small source/document gradient and actual-update
audit using the current masks, checkpoint and recipe. It should precede any
automatic balancing, clipping or projection rule. Preserve difficult verified
aggregation, code and action examples; rarity or high loss alone is not evidence
that a training example is harmful.

## Primary source and exact scope

[The Devil Is in the Reconstruction Loss Scale](https://arxiv.org/html/2610.00983v1)
is the October 1, 2026 paper applying RMSE to OmniQuant and other learning-based
post-training quantizers. It optimizes auxiliary quantization parameters against
full-precision block outputs, including Qwen3.5 experiments. Its constant-norm
result concerns reconstruction-output gradients, not a guarantee about optimizer
updates. Appendix A.2 groups errors within each sample:

| Variant | Elements under each root | Outer mean |
|---|---|---|
| Sample | Tokens and channels | Samples |
| Token | Channels of one token | Samples and tokens |
| Channel | Tokens of one channel | Samples and channels |
| Element | One scalar | All scalars; equivalent to MAE |

The [authors' repository](https://github.com/IntelChina-AI/RMSE) has not released
the implementation as of this assessment. The
[original OmniQuant code](https://github.com/OpenGVLab/OmniQuant/blob/main/quantize/omniquant.py)
uses MSE and AdamW to fit clipping/scaling parameters, sequentially by block.
Consequently there is no released implementation to check for epsilon handling
or directly install for our hybrid.

## What outlier handling actually changes

The following is our mathematical interpretation and CPU verification. For a
nonzero residual group e with n elements, its RMS has gradient
`e / (n * RMS(e))` and gradient norm `1 / sqrt(n)`. Squared-error gradients grow
with residual size. The root bounds the group's influence in residual space,
while preserving its internal error direction. It can also increase the relative
influence of small-error groups. It is not a cap-only procedure.

A root over one aggregate loss multiplies that entire gradient by one scalar. An
outlier inside that aggregate keeps its relative dominance. Separate roots for
tokens can contain an outlier token, but a large channel can still dominate that
token's direction. Separate roots for channels contain a high-error channel, but
an outlier position can still dominate within that channel. The grouping is a
decision about which units deserve bounded or equal influence, not just an
implementation detail. Equal influence is not necessarily the desired training
policy.

This also separates three meanings of outlier: large weights/activations that
are hard to quantize; large reconstruction residuals; and examples that cause
large or conflicting model updates. These can overlap without being identical.
Changing the influence of a residual does not itself remove an activation
outlier from the model.

For a linear layer, `dL/dW` is a sum of outer products of its input activation and
backpropagated error. A bounded CE gradient at logits can therefore still yield
large parameter gradients through unusual activations or network Jacobians.
This is a plausible route from the paper's idea to our MLPs, but has not been
demonstrated in this student. RMSNorm controls an aggregate input magnitude; it
does not guarantee identical channel contributions or bounded downstream MLP
outputs.

## Actual DistillKit objective and limits

Checked the saved `completion-v2/completion/train.json`, its 16 teacher cache
manifests, the pair file and the current training source:

- Replay uses weighted next-token CE and grouped-tail teacher KL, usually at
  teacher weight 0.5, with CE-only and KL-only record exceptions. Conversational
  context is masked under the approved policy; raw code/text keeps token loss.
- The completion arm adds 222 DPO-style preference pairs, pair weight 0.1,
  beta 0.1 and chosen-response CE weight 1.0. None of these rows is FTPO.
- All 16 cache manifests have `anchor_layers: []`. No hidden-state MSE
  reconstruction is active. The trainer also has dormant FTPO squared-error
  tethers; their existence does not make them part of this run.
- `training_step.py` weights each replay microbatch by its scored token weight,
  giving a global weighted token mean. Pair means have their own denominator.
  Neither is currently an equal-document mean.
- The shared streaming head accumulates chunk gradients before the final loss
  is known. The Kahan AdamW8bit update follows a global gradient clip at 1.0,
  with tensor-parallel replica handling and per-group learning rates.

These facts suggest several important limits:

1. Global clipping scales the combined gradient and leaves its direction intact.
   It can stabilize a step while allowing one document or source to dominate.
   Conversely, normalizing all documents would change the intentional replay mix
   and could amplify low-information or noisy documents.
2. CE has logit gradient `p - one_hot(target)`; full-distribution teacher KL has
   `p - q`. Loss magnitude is not a reliable proxy for their gradient norm.
   Taking their square roots preferentially attenuates high-loss examples and
   does not produce the reconstruction constant-norm property. This could weaken
   exactly the hard, correct examples we need to learn.
3. Adam's moments, weight decay, parameter-group rates and Kahan compensation
   determine actual movement. A fixed positive global gradient rescaling largely
   cancels in ideal Adam, whereas time-varying/group-dependent rescaling changes
   its dynamics. A gradient cap does not by itself guarantee a relative update
   cap. Zero current gradient does not imply zero update with existing momentum.
4. Equal output-gradient norms can become very unequal parameter-gradient norms
   after multiplication by a Jacobian. Magnitude control also cannot resolve
   conflicting directions between new skills and preservation replay.
5. Masked context remains available as causal input and can receive indirect
   gradients. The approved masking removes its direct prediction target; an
   influence experiment must not reintroduce those targets.

The [dataset assessment](CODE_DATASET_ASSESSMENT.md) found that raw code already
provides 33.56% of weighted replay targets through only 17 long documents.
Checking document concentration is therefore justified. It does not establish
that those documents dominate gradients or caused the observed code-NLL change.
The screened functional-code comparison remains statistically inconclusive.
Likewise, the earlier layer-revert evidence in
[TARGETED_TRAINING.md](TARGETED_TRAINING.md) localized context-prediction drift,
not proof that the current assistant-only run has the same damaging mechanism.

## Related concepts that fit different diagnoses

| Diagnosis | Research concept | Application hypothesis and limit |
|---|---|---|
| A few examples have disproportionate influence | Bounded per-document contribution, with quality checks | Cap exceptional contributions inside a source while preserving its aggregate budget. Do not discard rare verified examples based on loss. |
| CE, KL or preference terms dominate shared features | [GradNorm](https://arxiv.org/html/1711.02257v4) balances task gradient magnitudes and training rates | Measure the terms on shared parameters before altering fixed coefficients. Equal rates are not automatically the right goal for preservation versus acquisition. |
| Specific units move too strongly | [Adaptive Gradient Clipping](https://proceedings.mlr.press/v139/brock21a/brock21a.pdf) uses unit-wise gradient/parameter norm ratios | A cap-only mechanism is more conservative than boosting every weak group. Our Adam updates and near-zero residual/adapter parameters require separate measurement and sensible norm floors. The cited evidence is from vision, not this hybrid. |
| New-skill and preservation gradients oppose one another | [PCGrad](https://arxiv.org/abs/2001.06782) projects conflicting task gradients; [A-GEM](https://arxiv.org/html/1812.00420v2) constrains average replay interference | A preservation-oriented constraint may fit better than equal treatment of all tasks. Raw-gradient geometry is not an Adam update guarantee, nor does a local constraint ensure long-run retention. |
| High-loss examples are noisy or irrelevant rather than learnable | [RHO-LOSS](https://proceedings.mlr.press/v162/mindermann22a/mindermann22a.pdf) selects for reducible holdout loss | Borrow the distinction between hard-but-useful and irreducible error. Our cached teacher is not a validated irreducible-loss oracle; execution, factual provenance and mask audits remain stronger evidence here. |

These are primary-source mechanisms, not claims that the reported improvements
will transfer to our model. No combination is recommended before diagnosis.

## Bounded diagnostic plan

Reuse `training_step.py`, the current cached records and `streaming_head.py`'s
existing `update_check` setup. That diagnostic already restores initial weights,
constructs the actual Kahan optimizer and compares update effects. Its present
test is one raw-code prefix with a fresh optimizer; it does not measure per-source
influence, direction conflicts or resumed-state behavior.

1. Freeze a small representative training diagnostic set: raw-code documents,
   verified code answers, ordinary assistant replay, aggregation/action examples,
   and preference pairs. Track lengths, masks, target weights and provenance.
   Keep evaluation fixtures and confirmed task matches excluded from replay.
2. Start with loss/gradient distributions and concentration by document/source.
   Report source-weighted contribution, not just each source's standalone mean.
   Split CE, KL and chosen/rejected preference effects on a few diagnostic batches.
   Inspect early MLPs as well as later MLPs, DeltaNet, MLA, residuals and the head;
   do not assume early layers are the current problem.
3. Measure cross-objective gradient cosine and `g_replay dot actual_update`.
   A positive latter value predicts an increase in replay loss to first order.
   Report actual `norm(update) / max(norm(parameter), floor)` by coherent family,
   clipping frequency and moment effects. Near-zero parameters need separate
   absolute-update reporting. Check the local prediction by reevaluating replay
   after the controlled step; do not rely on a first-order proxy alone.
4. If concentration is established, first test a cap-only intervention inside
   the affected source or family, retaining the baseline aggregate replay budget.
   If direction conflict is the issue, test a preservation constraint instead.
   If it is poor target quality or insufficient world diversity, repair data.
   Compare one controlled step, then a bounded matched training arm only if the
   local evidence supports it. Use outcome/retention gates, not lower scalar loss.

This is a plan, not a completed model-gradient audit. It changes no approved
training policy. A resumed optimizer-state comparison is preferable where the
appropriate state is available; a fresh-state probe must label that limitation.

## Execution and memory opportunities

Document/source weights known before forward fit the existing streaming head and
weighted accumulation. Sparse diagnostic sampling and a small shared parameter
subset can keep balancing measurements out of most training steps. Parameter
norm reductions and cap scaling could be combined with existing gradient clipping
if profiling shows a benefit.

Full per-objective model-gradient copies would be expensive, particularly for the
large tied embedding/head, and could undo the memory improvements that prevented
spill. Start with selected parameter families or occasional isolated diagnostic
passes. A cheap output/hidden-gradient proxy may screen candidates, but cannot
establish full-body conflicts. A root whose scale depends on a whole step is
unknown when streaming chunk backward occurs; that would require changing
execution or retaining/recomputing contributions. Avoid doing so without evidence
of a quality benefit. No throughput gain is claimed.

Direct RMSE remains a candidate for later hidden-state or quantization calibration:
token groups can reduce independently inside our checkpointed hidden-state chunks;
sample/channel groups must accumulate squared errors and valid counts across
chunks before taking roots. FP32 residual/statistic arithmetic, empty-group
handling and a zero-error derivative convention need explicit validation. Unequal
valid lengths and fractional weights require a specified aggregation policy;
the simple equal-group `1 / sqrt(N)` identity does not automatically survive.
Current caches lack hidden-state targets, so this is not a free training toggle.

For quantization, compare the quantized student against its own full-precision
outputs using the intended GGUF representation. Generic OmniQuant packing is not
automatically our llama.cpp format. Validate end-to-end losses, functional tasks,
CSA2 selected positions and recurrent decode, rather than feature RMSE alone.
For `convert_full.py`'s closed-form least-squares/SVD initialization, a global square
root preserves the minimizer; grouped robust fitting would be a different
optimization problem, with additional fitting cost.

## CPU verification

Ran [reconstruction_influence_probe.py](../dense_gr/reconstruction_influence_probe.py)
with the existing DistillKit Python environment; all assertions passed. Results:
[reconstruction-influence-probe.json](reconstruction-influence-probe.json).
This is synthetic float64 CPU math, not a measured student improvement.

- On 70 nonzero residual elements scaled by 0.001, 1 and 1000, all five tested
  root/grouped-root variants have output-gradient norm 0.11952286. MSE scales with
  residual magnitude.
- One token's residual is 100 times larger. Its gradient-norm ratio to an ordinary
  token remains 100 under MSE and global RMSE; token RMSE makes the ratio 1.
- Global clipping of two orthogonal contributions of sizes 1 and 100 still has
  cosine 0.999950 to the larger contribution. Its dominance persists.
- As a wrong-class logit grows from 0 to 100, CE-gradient norm rises from 0.707107
  to 1.414214, while root-CE gradient norm falls from 0.424661 to 0.070711.
  Rooting CE does not reproduce the reconstruction property.
- Correct token/channel/sample aggregation has matching losses and gradients
  for one chunk versus uneven chunks of 2, 2 and 3 tokens, with masks and fractional
  weights. Taking roots separately over arbitrary chunks changes the objective.
- Equal unit output-gradient norms become parameter-gradient norms of 1 and 1000
  through a diagonal Jacobian. Output-space normalization is not an update bound.

Reproduce from the DistillKit root:

```powershell
.venv\Scripts\python.exe scratch/dense_gr/reconstruction_influence_probe.py --output scratch/csa2-eval/reconstruction-influence-probe.json
```
