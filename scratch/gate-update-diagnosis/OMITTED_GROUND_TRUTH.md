**Sparse top-k KL can oppose a correct memory prediction — 2026-09-10**

**Finding:** under this configuration's `missing_probability_handling: zero`, a ground-truth token omitted from the cached teacher top-64 gets zero target probability. Increasing that token's logit, holding the other logits fixed, increases the KL loss even though it decreases ground-truth cross entropy. This is an objective conflict established by the implemented loss. Its contribution to the observed assistant NLL regression has not been established.

This finding was included in [the research report](D:/DeepThought/Projects/HybridModel/DistillKit/scratch/gate-update-diagnosis/REPORT.md). During that report's preparation, commit `ef32abb` independently added a coverage probe and results to the repository. This note incorporates those recorded results; I did not run that coverage probe. The checkout was `02f4b32` for the three-step gate diagnosis and had advanced to `acfb7a2` by this follow-up. The KL implementation cited here was checked again at the latter revision.

**What the source does**

The cached top-k probabilities are renormalized to sum to one by the ZERO branch in [distillkit/lossfuncs/common.py:66](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/lossfuncs/common.py:66). The student's log probabilities retain the full-vocabulary normalization at [common.py:152](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/lossfuncs/common.py:152). [distillkit/lossfuncs/kl.py:49](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/lossfuncs/kl.py:49) computes the retained target contributions; missing target terms are omitted at [kl.py:74](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/lossfuncs/kl.py:74).

Let S be the cached top-k set, q the original teacher distribution, Q its mass on S, and p = softmax(z) the student distribution. At the configured temperature 1:

```text
q_tilde(j) = q(j)/Q  for j in S;  0 otherwise
L_KL = sum[j in S] q_tilde(j) * (log q_tilde(j) - log p(j))
dL_KL/dz(j) = p(j) - q_tilde(j)
```

For an omitted ground-truth token y, `dL_KL/dz(y) = p(y) > 0`. Gradient descent therefore lowers that coordinate. Ground-truth CE has derivative `p(y) - 1 < 0`, pointing the other way. With the configured KL weight, the KL contribution is `0.7 * p(y)` before token averaging. The hidden-state term adds a different gradient; this derivation does not determine the combined parameter update.

For example, if the retained target is `[0.9, 0.1, 0]`, the third token is correct, and the student puts probability 0.05 there, the KL derivative for its logit is +0.05 while the CE derivative is -0.95. The teacher's architectural absence of memory is irrelevant to this calculation. A memory-derived correction toward that third token encounters the same opposition as any other correction.

This is a coordinate-level statement. A real memory update changes many logits together and can still lower total KL by improving other predictions. It does not follow that every useful memory update increases KL or that the objective's global optimum has a silent sidecar.

**What the newly recorded coverage audit says**

The [saved coverage results](D:/DeepThought/Projects/HybridModel/DistillKit/scratch/gpu-checks/topk-coverage.json) cover the first 64 training-cache documents, 50,893 next-token positions, with top-k = 64:

| Slice | Positions | Ground truth outside top-64 | Mean original teacher mass retained |
| --- | ---: | ---: | ---: |
| All positions | 50,893 | 4.2206% | 99.1764% |
| Assistant | 33,539 | **0.2803%** | **99.9134%** |
| User | 8,778 | **16.3819%** | 98.4571% |
| System | 7,558 | 8.1106% | 96.6309% |
| Template | 1,018 | 0.2947% | 99.9957% |

The [probe](D:/DeepThought/Projects/HybridModel/DistillKit/scratch/topk_coverage_probe.py) compares cached distribution row t with the actual cached token at t+1, so the overall count uses cached IDs directly. Its role assignment decodes and retokenizes text without asserting that token IDs round-trip exactly. The script explicitly labels that breakdown indicative. This is a training-cache sample, not the 384-document held-out quality evaluation, and it provides no confidence interval over documents.

The low sampled assistant omission rate weakens the hypothesis that omitted ground-truth tokens are the principal assistant-token bottleneck. It does **not** mathematically rule it out: rare positions can have large NLL effects, and training-time changes can affect other positions. If the entire +0.027667 assistant NLL penalty were concentrated on a fraction 0.0028027 of assistant positions, their mean penalty would need to be approximately **9.87 nats each**. That is an illustrative scale calculation, not an attribution, because the coverage and quality evaluations use different samples.

The much larger user-token omission rate makes this mechanism plausible as a contributor to the prompt/assistant asymmetry. Coverage alone does not prove that it explains the user-turn damage. Also, teacher matching can oppose a correct-token improvement even when the token is inside top-k if its teacher target probability is too low; the omission audit does not test that broader conflict.

**What follows for the objective**

Ground-truth CE supplies a positive target for omitted correct tokens. For nonnegative weights a and b, `a CE(y,p) + b KL(q_tilde||p)` has unconstrained target distribution `(a one_hot(y) + b q_tilde)/(a+b)`. Unlike pure top-k imitation, this assigns the correct omitted token nonzero target mass. This is the standard combination of hard and soft targets described by Hinton, Vinyals and Dean in [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).

Retaining an aggregate teacher tail-mass term can avoid treating all missing teacher mass as zero, but it cannot identify which omitted token is correct. Increasing k or recapturing logits may reduce truncation error; the low sampled assistant omission rate is a reason to measure their likely value before paying that cost.

The next useful read-only audit would use exact cached token-role alignment and report enabled-minus-bypassed assistant NLL separately for targets inside and outside teacher top-k on the same held-out documents. Also compare CE and KL directional gradients for the same memory perturbation. That would test relevance to the regression rather than merely restating the derivative. None of those additional audits or training runs was launched for this note. Any future CE implementation should reuse the chunked head; simply adding the existing model-loss CE term can reintroduce the full-logit allocation problem.
