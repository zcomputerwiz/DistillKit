**Direction-gated PLE: update diagnosis and research reading list — 2026-09-10**

**The answer:** `sharpness` is stuck because AdamW updates its BF16 parameter directly, without an FP32 master copy. The gate direction is **not** stuck: the real trainer produces gradients and updates thousands of its coordinates. The checkpoint's gate norm is not evidence of weight decay; its saved optimizer group has **weight_decay = 0.0**. These findings invalidate the inference that the rerun's improvement was necessarily initialization alone. They do not establish how much learning contributed to that improvement.

Web access worked. Literature claims below link to original papers; project-specific suggestions are identified as suggestions, rather than attributed to the papers.

**Part A — evidence from the production path**

The diagnostic ran on `sidecar-distill`, HEAD `02f4b32`; source line references for Part A refer to that revision. Requested HEAD `4c5f3ee` is its immediate parent; the intervening commit changes only `PROGRESS.md` and the saved reply evaluation JSON. The relevant source is unchanged. I read the September 10 progress sections, including the corrections to earlier claims about assistant-only evaluation, early versus late injection, and whether a memory-free teacher mathematically implies a silent student sidecar.

Only one diagnostic reached model updates: three optimizer steps, batch 1, accumulation 1, complete cached training documents of lengths 145, 148, and 152. It used the real student, real IQ4_NL table, offline teacher cache, 0.7 sparse KL + 0.3 cosine at both configured anchors, two-device TP, non-reentrant gradient checkpointing, production freezing/grouping, and production clipping at 1.0. Learning rate was held at `1e-4`, with warmup disabled deliberately so a three-step diagnostic tests the configured maximum update size rather than spending its first step at zero LR. The original config uses 20 warmup steps. No held-out evaluation documents were used.

Elapsed time including imports, checkpoint inspection, model/table loading, and updates was **30.984 seconds**; the trainer's three-step loop took **4.601 seconds**. A 540-second watchdog bounded the process. Evaluation, integrations, checkpoint saving, and final model export were disabled. All new files and cache writes are under `scratch/`. Two earlier attempts stopped before any model update: a scratch config validation error and a datasets-cache permission error, both corrected in the probe. I did not change distillkit source, production configs, PROGRESS.md, or trained models.

Artifacts:

- [Probe script](D:/DeepThought/Projects/HybridModel/DistillKit/scratch/gate_update_diagnosis.py)
- [Raw measurements, including before/after synchronization, clipping, moments, and deltas](D:/DeepThought/Projects/HybridModel/DistillKit/scratch/gate-update-diagnosis/measurements.jsonl)
- [Console log](D:/DeepThought/Projects/HybridModel/DistillKit/scratch/gate-update-diagnosis-console.log)
- [Exact diagnostic config](D:/DeepThought/Projects/HybridModel/DistillKit/scratch/gate-update-diagnosis/trainer-output/distillkit_config.yaml)

Reproduce from the repository root against revision `02f4b32`. A later concurrent commit, `acfb7a2`, applies an FP32 gate fix; running against that revision tests the corrected behavior rather than reproducing the original failure:

```powershell
.\.venv\Scripts\python.exe scratch\gate_update_diagnosis.py
```

The probe overrides the trainer class only inside its process, observes the actual optimizer with pre/post hooks, and applies identical post-clipping gate gradients to separate FP32 AdamW parameters. Those shadow parameters never participate in the model forward. Thus this is an optimizer precision counterfactual, not a second trained model or an estimate of FP32 model quality.

**Measured gradients and updates**

All four parameters below were trainable BF16 tensors on `cuda:0`, assigned to AdamW, with zero weight decay. Names have the common prefix `model.layers.1.sidecar.ple.`. Gradient norms are immediately before the real optimizer step, after clipping; delta norms are computed from actual stored weights before and after the step, widened to FP32 for subtraction.

| Step | Parameter | Gradient L2 | Actual delta L2 | Maximum absolute delta | Coordinates changed |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | gate | 0 | 0 | 0 | 0 / 10,240 |
| 1 | sharpness | 0 | 0 | 0 | 0 / 4 |
| 1 | value_proj.weight | 0.510491 | 0.255997 | 1.01089e-4 | 6,553,600 / 6,553,600 |
| 1 | conv1d.weight | 0 | 0 | 0 | 0 / 20,480 |
| 2 | gate | 0.158433 | 0.00858376 | 1.22070e-4 | 9,024 / 10,240 |
| 2 | sharpness | 0.00324407 | **0** | **0** | **0 / 4** |
| 2 | value_proj.weight | 0.633933 | 0.196889 | 1.01089e-4 | 6,546,582 / 6,553,600 |
| 2 | conv1d.weight | 0.379765 | 0.0106482 | 7.48634e-5 | 20,480 / 20,480 |
| 3 | gate | 0.0907906 | 0.00756801 | 1.22070e-4 | 7,505 / 10,240 |
| 3 | sharpness | 0.00173964 | **0** | **0** | **0 / 4** |
| 3 | value_proj.weight | 0.451007 | 0.162571 | 1.02043e-4 | 6,537,077 / 6,553,600 |
| 3 | conv1d.weight | 0.362655 | 0.00936375 | 8.67844e-5 | 20,444 / 20,480 |

Step 1's zero gate, sharpness, and convolution gradients are expected: value projection and convolution weights start at zero. The value projection must move before there is anything to admit or convolve. At step 2, both gate parameters receive gradients.

Before synchronization/clipping, step 2 gradient norms were **0.989215** for gate and **0.0202721** for sharpness; step 3 had **0.399867** and **0.00767306**. Synchronization left all four observed gradients **bit-identical** on every step. Clipping reduced their magnitudes but did not eliminate them.

The decisive precision comparison:

| Step | BF16 sharpness maximum change | FP32 sharpness maximum change, identical gradients |
| ---: | ---: | ---: |
| 1 | 0 | 0 |
| 2 | **0** | **7.43866e-5** |
| 3 | **0** | **8.49962e-5** |

After step 3, BF16 sharpness remained `[1, 1, 1, 1]`. The FP32 copies were `[1.0001572371, 1.0001593828, 1.0001543760, 1.0001553297]`. Both optimizers advanced their state; the actual BF16 sharpness moments were nonzero. This demonstrates loss of the weight update at storage precision, rather than missing gradients.

The gate's norm changed only **2.02124548 → 2.02027202**, while **8,723** coordinates differed from initialization and its total displacement had L2 norm **0.0153006**. A nearly constant norm does not mean a direction stayed fixed. Its forward computation normalizes that direction anyway. This diagnostic initialization is not the original run's saved initial tensor; the raw log's `checkpoint_delta` for gate must not be interpreted as the production run's training displacement. The original initialization was not available as an exact tensor for comparison.

**The original 72-step checkpoint independently corroborates the diagnosis**

Read directly from `runs/widened-plegated-stage1-1m/checkpoint-72/optimizer.pt` and its safetensors, without loading the full model for inference:

| Parameter | Stored dtype | Adam step | Weight decay | First moment L2 | Second moment L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| gate | BF16 | **72** | **0.0** | **0.0692197** | **0.00187465** |
| sharpness | BF16 | **72** | **0.0** | **0.000802978** | **1.19034e-7** |

Both moments themselves are BF16, with every coordinate nonzero. Saved sharpness is exactly 1 in all four entries; gate norm is **2.0120914**, value norm **3.1604643**, and convolution norm **0.2782822**, reproducing the supplied checkpoint statistics. Step count alone would not prove useful updates, but the moments plus the live per-step measurements establish the gradient path.

**File:line causal chain**

| Evidence | Location |
| --- | --- |
| `bf16: true` causes pretrained model loading in BF16 | [distillkit/main.py:251](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/main.py:251) |
| Sharpness starts as a parameter filled with ones | [distillkit/ple_gated_sidecar.py:129](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/ple_gated_sidecar.py:129) |
| Gate and sharpness are widened during forward; this does not change parameter storage | [distillkit/ple_gated_sidecar.py:155](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/ple_gated_sidecar.py:155) |
| Auxiliary parameters remain trainable during stage-1 freezing | [distillkit/optimizers.py:302](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/optimizers.py:302) |
| Auxiliary ownership prevents Muon routing | [distillkit/optimizers.py:114](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/optimizers.py:114), [distillkit/optimizers.py:135](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/optimizers.py:135) |
| AdamW child is ordinary `torch.optim.AdamW(..., foreach=False)` | [distillkit/optimizers.py:207](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/optimizers.py:207) |
| The wrapper really steps its children | [distillkit/optimizers.py:240](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/optimizers.py:240) |
| Installed Adam creates moments with `zeros_like(p)` | [.venv/Lib/site-packages/torch/optim/adam.py:178](D:/DeepThought/Projects/HybridModel/DistillKit/.venv/Lib/site-packages/torch/optim/adam.py:178) |
| Installed Adam performs an in-place update on the original parameter | [.venv/Lib/site-packages/torch/optim/adam.py:547](D:/DeepThought/Projects/HybridModel/DistillKit/.venv/Lib/site-packages/torch/optim/adam.py:547) |
| Trainer passes its weight-decay setting explicitly, overriding wrapper defaults | [distillkit/trainer.py:388](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/trainer.py:388) |
| That setting defaults to zero in this installed version | [.venv/Lib/site-packages/transformers/training_args.py:814](D:/DeepThought/Projects/HybridModel/DistillKit/.venv/Lib/site-packages/transformers/training_args.py:814) |
| TP synchronization touches registered replicated groups; sidecar is on the home GPU | [distillkit/tp_gated_delta_module.py:218](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/tp_gated_delta_module.py:218), [distillkit/tp_model.py:9](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/tp_model.py:9) |
| TP clipping uses the ordinary norm clip, excluding duplicate replicas | [distillkit/tp_gated_delta_module.py:232](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/tp_gated_delta_module.py:232) |
| Value and convolution are zero-initialized; gate multiplies the value | [distillkit/ple_gated_sidecar.py:141](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/ple_gated_sidecar.py:141), [distillkit/ple_gated_sidecar.py:196](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/ple_gated_sidecar.py:196) |

Near 1, adjacent BF16 values are **0.99609375, 1.0, 1.0078125**. A downward update needs more than approximately **0.001953125**, or an upward update more than approximately **0.00390625**, to cross the rounding midpoint. An Adam update near `1e-4` does neither. Repeating 72 individually rounded updates does not accumulate a hidden `0.0072`: no master parameter retains the lost increments. This is not gradient underflow. The FP32 CPU experiment avoided precisely this failure.

At a direction coordinate near 0.02, BF16 spacing is **0.0001220703125**. Some gate updates therefore move a representable value, while smaller ones disappear. Near zero, where the value and convolution start, much smaller absolute increments are representable. This explains their different behavior without invoking different gradient connectivity.

Also, `72 × lr` is only a rough coherent-update scale, not a prediction of net drift: momentum, changing signs, bias correction, epsilon, warmup, and first-step zero gradients matter. The production schedule starts at zero LR. None of these facts rescues precision lost on assignment.

**Recommended correction, not applied by this diagnostic:** retain gate and sharpness as actual FP32 trainable parameters after loading and before optimizer construction, while keeping the large backbone BF16. This also makes their Adam moments FP32. For these 10,244 numbers the additional parameter-plus-two-moment storage is about **60 KiB** versus BF16. Preserve this policy on checkpoint reload. A generic FP32 master-weight optimizer is another solution; simply calling `.float()` in the forward or keeping only moments in FP32 is not sufficient. A zero-centered scale parameter such as `1 + delta` or `exp(log_scale)` is an alternative, but small updates still need to survive until the forward scale is formed, and FP32 is the straightforward fix. This is the master-weight rationale in Micikevicius et al., *Mixed Precision Training* (ICLR 2018, [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)).

This resolves the observed update puzzle. It does **not** show that FP32 sharpness will improve held-out NLL, that clipping is optimal, or that 72 steps suffice for the memory transport problem. No quality-training experiment was run.

**Part B — interpretation before the reading list**

Teacher architecture does not impose a prohibition on student memory. Output KL compares distributions, not implementations. A table can help a smaller student approximate a dense teacher. A learned projected hidden-state cosine term likewise does not require every extra student feature to vanish. Conversely, if a teacher distribution is wrong on a token, matching it can oppose a correct memory-derived prediction. These are different statements. The latter motivates ground-truth supervision; the former does not justify forcing every gate open.

There is a concrete issue in this repo beyond the generic distinction. With `missing_probability_handling: zero`, [common.py:66](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/lossfuncs/common.py:66) renormalizes the cached top-k teacher probabilities to sum to one. The student uses a full-vocabulary normalizer at [common.py:152](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/lossfuncs/common.py:152), and [kl.py:48](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/lossfuncs/kl.py:48) scores the retained teacher terms. At temperature 1, the per-token logit gradient is `p_student(j) - q_topk(j)`; for every omitted token it is **positive** `p_student(j)`. If ground truth is omitted from the teacher's top-k, this KL explicitly pushes its logit down. A coverage audit arrived in concurrent commit `ef32abb`: it reports 0.2803% assistant omissions in 64 training documents, with an approximate role partition. Its contribution to held-out NLL remains unmeasured. See [the standalone note](D:/DeepThought/Projects/HybridModel/DistillKit/scratch/gate-update-diagnosis/OMITTED_GROUND_TRUTH.md) for the data, derivation, and limitations. A tail-mass treatment reduces this particular approximation error but cannot replace ground-truth CE.

For the normalized learned direction, the useful scale prescription can be stated without treating it as attention. With unit-RMS stream `q` and raw `v_i ~ N(0, sigma²)`, `Var(q·v / sqrt(d)) ≈ sigma²`, under the usual independence/isotropy assumptions. At `sigma=0.02` that raw-score standard deviation is 0.02. Either use `q·v` with `sigma≈1/sqrt(d)`, or normalize `v` and use an explicit gain:

```text
u = v / ||v||₂
r = alpha * (q · u) + bias
```

Because `||q||₂≈sqrt(d)`, `Var(q·u)≈1` for an isotropic direction. Up to epsilon, your `RMS(v)` followed by `/sqrt(d)` is exactly this parameterization. If both q and u instead have unit L2 norm, the initial gain needs to be approximately `sqrt(d)` for comparable random-score variance. There is no universal recommended opening fraction for a signed-square-root sigmoid gate. Calibrate raw-score mean/variance and gate histograms on representative residuals, since RMS normalization does not whiten their covariance. Your signed square root also means a positive pre-root gain alpha changes the post-root logit magnitude as `sqrt(alpha)`, not alpha.

Finally, the apparent **0.0816–0.083 nat gap is a motivation, not a causal estimate of objective damage**. The linear readout was at the final hidden state; the real sidecar injects at layer 1 and must survive the frozen stack. The documented readout used a different held-out slice and training protocol. Placement, trainable widening, optimization budget, and objective differ. The corrected progress entry already acknowledges these confounds. A matched early-versus-late and CE-versus-KD comparison is needed before assigning a percentage of the gap to the objective.

**Reading list — initialization and scaling**

**1. Salimans & Kingma, “Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks,” NeurIPS 2016.** The paper explicitly separates a weight's direction from its length and, in Section 3, calibrates gain and bias using initial data so pre-activation statistics have a controlled scale. This is the closest conceptual match to a learned vector reading a normalized residual. [Original paper](https://arxiv.org/html/1602.07868).

*What this suggests for us:* the direction-plus-sharpness change has a standard foundation. The missing complement is calibration on actual layer-1 streams and reliable storage of the learned gain. Log direction change or angular change, not just direction norm. If a bias is introduced, use it to control admission prior separately from sharpness. Gain, bias, and branch-output size should be independently observable; a fixed unit RMS direction does not control variance along anisotropic pretrained features.

**2. Henry, Dachapally, Pawar & Chen, “Query-Key Normalization for Transformers,” Findings of EMNLP 2020.** QKNorm uses L2-normalized queries and keys with a learned scale instead of the ordinary attention division by square root of dimension. Its concern is controlling attention-score scale and saturation. [Original paper](https://aclanthology.org/2020.findings-emnlp.379/).

*What this suggests for us:* match the denominator to the actual operand norms. RMS/RMS plus division by sqrt(d), unit-RMS stream plus unit-L2 direction, and L2/L2 plus a dimension-dependent gain are different parameterizations of related geometry. Do not combine their scaling conventions accidentally. QKNorm is evidence for normalization plus explicit gain, not an empirical endorsement of the exact signed-root sigmoid or of a particular gate-open fraction for this sidecar.

**3. Srivastava, Greff & Schmidhuber, “Highway Networks,” 2015.** Highway networks deliberately bias transform gates toward carrying the input at initialization, using negative transform-gate biases. This separates a routing prior from weight-variance initialization. [Original paper](https://arxiv.org/abs/1505.00387).

*What this suggests for us:* “starts mostly closed” can be a deliberate stability strategy, whereas “unintentionally constant because the input scale is tiny” is a different failure. Your zero value projection already preserves the original function, so there is no need to obtain identity by making the admission computation insensitive. A modest explicit bias would allow mean admission to adapt without requiring direction rotation or changing its gain, but is a new architectural choice to test rather than a necessary fix.

**4. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” ICLR 2022.** LoRA initializes one factor randomly and the other to zero, preserving the original function; the residual update is scaled explicitly. It trains added capacity on the task objective while freezing original weights. [Original paper, Section 4.1](https://arxiv.org/html/2106.09685).

*What this suggests for us:* zero initialization makes first-step zero gradients for upstream factors normal. In `gate × value`, a zero value necessarily delays gate gradients. Avoid zeroing both multiplicative sides in an attempted stabilization. LoRA provides no general guarantee that an added feature source will be used, and an isolated CPU gradient of order 1e3 is not comparable to the normalized, clipped distillation gradient. Audit actual step magnitudes, as in Part A.

**5. Alayrac et al., “Flamingo: a Visual Language Model for Few-Shot Learning,” NeurIPS 2022.** Flamingo freezes pretrained language and vision models, inserts cross-attention/dense blocks, and multiplies their contributions by zero-initialized `tanh(alpha)`. Multimodal next-token training opens those gates. The authors explicitly caution that gate magnitudes alone are hard to interpret because the pre-gate activation scales vary. [Original paper, Sections 2.2 and A.1.2](https://arxiv.org/html/2204.14198).

*What this suggests for us:* useful new input can be learned through a frozen LM with an identity-preserving retrofit. The training target rewards information from that input. Preserve the initialization identity, but judge actual injected vectors and output effects alongside gate statistics. Do not copy Flamingo's zero gate on top of your already-zero value map; that can create a dead product. Its result supports task-supervised retrofitting, not forcing admission under a memory-free teacher.

**Reading list — making external memory useful**

**6. Khandelwal et al., “Generalization through Memorization: Nearest Neighbor Language Models,” ICLR 2020.** kNN-LM builds a datastore from pretrained representations and interpolates its next-token distribution with the LM distribution at inference. It does not rely on training the backbone to route information through an early adapter. [Original paper](https://arxiv.org/abs/1911.00172).

*What this suggests for us:* a fixed nonzero output mixture is a strong diagnostic baseline because the original network cannot erase memory downstream. Your final-hidden linear read has this same experimental advantage, although its dense feature correction is not a kNN token-distribution mixture. Its success demonstrates decodable information, not successful layer-1 transport. Compare deployment cost and gain of late integration before assuming the upstream insertion depth must be retained.

**7. Borgeaud et al., “Improving Language Models by Retrieving from Trillions of Tokens” (RETRO), ICML 2022.** In Section 4.2, RETRO-fitting freezes pretrained weights and trains the new neighbor encoder and chunked cross-attention on language modeling. Appendix D.3 reports that freezing helped preserve retrieval-off performance. Section 4.3 also leaves stronger reliance on retrieval as an open problem for QA; the paper does not claim universal prevention of memory neglect. [Original paper](https://proceedings.mlr.press/v162/borgeaud22a/borgeaud22a.pdf).

*What this suggests for us:* this is a direct precedent for the requested frozen-backbone retrofit, with ground-truth prediction as the incentive. A clean comparison would freeze widening as well as the original backbone, train only the reader on assistant CE, and preserve a fixed bypass baseline. The current stage 1 trains widening, so its bypass model can change. RETRO's trained reader and multiple integration layers also caution against assuming a single early linear read is equally easy to optimize.

**8. Zhong, Lei & Chen, “Training Language Models with Memory Augmentation” (TRIME), EMNLP 2022.** TRIME puts token embeddings and in-batch contextual memory into the training objective; memories with the correct next-token label are positives. This directly aligns representation learning with memory use. Its Section 3 footnote reports that an earlier learned-gating interpolation objective worked worse than its chosen objective. [Original paper](https://aclanthology.org/2022.emnlp-main.382.pdf).

*What this suggests for us:* this is the most relevant objective paper. Supervise something that can only improve by extracting predictive information from the table: a memory readout or next-token-conditioned contrastive term. Your frozen IQ4_NL rows are not TRIME's differentiable contextual key/token datastore, so the literal objective is not a drop-in replacement. Define correct-token positives from training data and exclude self/future leakage. A positive signal-to-label objective is more defensible than maximizing gate variance or arbitrary memory-on/off disagreement.

**9. Wu, Rabe, Hutchins & Szegedy, “Memorizing Transformers,” ICLR 2022.** The model mixes local and retrieved attention through a learned per-head sigmoid bias. The gate is not token-dependent, and the authors report most heads increasingly favor external memory. They also fine-tune a vanilla transformer into a memory model. Query/key normalization reduces representation-scale drift in stored memory. [Original paper, Sections 3.1, 3.2, and 4.5](https://arxiv.org/html/2203.08913).

*What this suggests for us:* a constant-across-tokens gate is not intrinsically a failed memory design. Useful memory, target supervision, and placement can matter more than gate selectivity. This paper typically inserts memory near the top of the stack, reinforcing the need for an insertion-depth control. Its existing-model fine-tuning experiment does not establish that all original backbone weights stayed frozen, and should not be described as equivalent to RETRO-fitting.

**10. Wang et al., “Augmenting Language Models with Long-Term Memory” (LongMem), NeurIPS 2023.** LongMem freezes the original LLM as a memory encoder and trains an adaptive residual side network as retriever and reader, avoiding memory staleness. Its adaptation uses memory-aware training, including a TRIME-based recipe. [Original paper](https://papers.neurips.cc/paper_files/paper/2023/file/ebd82705f44793b6f9ade5a669d0f0bf-Paper-Conference.pdf).

*What this suggests for us:* separate stable memory storage from a trainable reader. A frozen table is compatible with learning a useful interface; there is no requirement to update all 51.2B elements. However, LongMem's memory comes from a compatible frozen encoder, whereas your table comes from another model. Reader alignment and transport remain problems even after numerical gate fixes. An auxiliary table-to-token readout gives a more direct measure of alignment than teacher-anchor cosine alone.

**11. Guu et al., “REALM: Retrieval-Augmented Language Model Pre-Training,” ICML 2020.** REALM learns retrieval through prediction likelihood, emphasizes salient masked entities/dates to make knowledge useful, and includes a null document for cases that do not require retrieval. [Original paper](https://kentonl.com/pub/gltpc.2020.pdf).

*What this suggests for us:* construct or weight examples where the memory demonstrably reduces prediction error, rather than rewarding opening on every token. For an n-gram table these may be lexical completions, names, repeated forms, or rare tokens, not necessarily long-range factual QA. Estimate relevant slices using independent data and real-versus-shuffled features. Null retrieval is a reminder that choosing not to use memory can be correct; useful conditional reliance is the target.

**Reading list — distilling into different or more capable architectures**

**12. Hinton, Vinyals & Dean, “Distilling the Knowledge in a Neural Network,” 2015.** The standard recipe combines teacher soft-target matching with a separate correct-label objective when labels are available, including temperature compensation. The student need not share the teacher's implementation. [Original paper](https://arxiv.org/abs/1503.02531).

*What this suggests for us:* adding ground-truth CE is established practice, not a special workaround. With normalized teacher target q and nonnegative weights a,b, minimizing `a CE(y,p) + b KL(q||p)` targets `(a one_hot(y) + b q)/(a+b)` in the unconstrained per-example optimum. Thus ground truth can reward a correction the teacher misses. Pure teacher matching cannot reward arbitrary deviation from an already matched teacher distribution. It can still reward memory that helps an underpowered student get closer to the teacher.

**13. Touvron et al., “Training Data-Efficient Image Transformers & Distillation through Attention” (DeiT), ICML 2021.** A convolutional teacher trains a transformer student using a distinct distillation token/head alongside the label-supervised class token/head. Their predictions provide complementary information, and the class head is less closely tied to teacher predictions. [Original paper](https://arxiv.org/html/2012.12877).

*What this suggests for us:* this is a clear architecture-mismatch precedent and a useful loss-design analogy: reserve a path for teacher imitation and a path for the task. For this student, CE on the memory-enabled output plus KD on a bypassed or separate distillation output would remove the direct KL penalty on the memory correction. This is a proposed adaptation, not DeiT's algorithm for language memory. If every bypass-path parameter is frozen, bypass KD is constant and does no training work; if widening is trainable, it can train that path.

**14. Burns et al., “Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision,” 2023/ICML 2024.** A stronger pretrained student can surpass weak supervision, but naive imitation also copies weak-teacher errors. An auxiliary confidence objective can improve disagreement with incorrect weak labels; effects vary across tasks and model gaps. [Original paper, Section 4.3.2](https://arxiv.org/pdf/2312.09390).

*What this suggests for us:* this directly addresses capacity the supervisor cannot fully demonstrate. It supports allowing independent student evidence to compete with teacher targets. It does not imply a 4B student with a frozen table is globally stronger than a 27B teacher. Because your data already supplies ground-truth next tokens, CE is a better first intervention than unconditional confidence maximization, which could make a bad readout confidently wrong. The paper's classification gains are not an established recipe for this generation setup.

**15. Zhao et al., “Decoupled Knowledge Distillation,” CVPR 2022.** DKD separates target-class and non-target-class knowledge transfer, removing a coupling in conventional KD that suppresses some non-target information according to teacher target confidence. It exposes more control over which part of the teacher distribution is transferred. [Original paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.pdf).

*What this suggests for us:* if CE plus ordinary KD still conflicts on correct-token improvements, distinguish target-token supervision from matching relative alternatives. Full-vocabulary classifier DKD is not directly available from sparse top-k caches: a correct token may be absent and tail structure is unknown. First measure target coverage and tail mass; then consider decoupling that respects what the cache actually contains. DKD itself is not a memory-use regularizer and does not guarantee exploitation of new student capacity.

**Reading list — gates shutting and competition between paths**

**16. Shazeer et al., “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer,” ICLR 2017.** The paper describes self-reinforcing expert imbalance: selected experts train faster and become still more attractive. It adds importance/load balancing to discourage a small set from monopolizing routing. [Original paper, Section 4](https://arxiv.org/pdf/1701.06538).

*What this suggests for us:* early exclusion can starve a potentially useful branch, so temporary usage constraints have a real literature precedent. But balancing interchangeable experts is not equivalent to forcing a heterogeneous frozen memory to improve every token. An entropy or minimum-admission penalty can force nonzero gates while the value projection shrinks or downstream processing cancels it. If tried, make it temporary, monitor actual causal contribution, and select by held-out CE rather than compliance with a gate quota.

**17. Wang, Tran & Feiszli, “What Makes Training Multi-Modal Classification Networks Hard?,” CVPR 2020.** More input modalities can worsen performance despite greater available information. The paper identifies differing optimization/generalization behavior and proposes Gradient Blending using auxiliary modality losses. [Original paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.pdf).

*What this suggests for us:* the presence of useful table signal does not guarantee that joint optimization will extract it. A strong pretrained stream and a newly initialized table reader learn on different time scales. Measure gradients separately for CE, KL, each hidden-state anchor, and memory-specific supervision; their alignment on the reader is more informative than the weighted loss scalar. This is a relevant analogy, not proof that modality competition explains the measured 0.083-nat difference.

**18. Neverova, Wolf, Taylor & Nebout, “ModDrop: Adaptive Multi-Modal Gesture Recognition,” 2015 preprint / TPAMI 2016.** ModDrop combines initialization of individual modalities with gradual fusion and random dropping of modality channels, preserving useful representations and handling missing inputs. [Original paper](https://arxiv.org/abs/1501.00102).

*What this suggests for us:* modality dropout is a published mechanism for reducing reliance on a single source, but applying it to only the memory teaches the system to survive without memory. For this early residual retrofit, dropping the whole backbone stream would also create an unrealistic and damaging task. Prefer an auxiliary reader loss or a narrowly targeted, explicitly tested context-ablation curriculum over indiscriminate stream dropout. Neither method guarantees a benefit on ordinary deployment inputs.

**What a closing gate would actually establish**

For a local scalar admission a and value v in `h' = h + a v`, the partial derivative with respect to admission is `dL/da = <dL/dh', v>`. A positive derivative favors reducing admission for that batch and current value map. Through a sigmoid, saturation multiplies this by a small derivative. This can mean the objective dislikes the current injected value; it does not prove that the best possible reader is silent. Poor alignment, transient optimization, saturation, and numerical update loss can give similar observations.

In this exact implementation there are two additional traps. First, the actual direct-value multiplier is `2 * mean_k(gate_k)`, while the logged open/shut fractions describe individual directions. Second, [ple_gated_sidecar.py:196](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/ple_gated_sidecar.py:196) adds an **ungated convolution branch**. Closing direct admission would not silence the sidecar. Also, shrinking sharpness to zero makes individual sigmoid gates 0.5 and the aggregate multiplier 1; it removes selectivity rather than switching the value off. Thus “sharpness decreased,” “gate closed,” and “sidecar became silent” are not interchangeable claims.

To test objective conflict directly, introduce an analysis-only scalar rho multiplying the **entire** sidecar increment and measure separate derivatives of assistant CE, KL, and each anchor loss with respect to rho. Compare finite perturbations as a check. If KD prefers lower rho while CE prefers higher rho on the same held-out examples, that is local evidence of conflict. Also inspect gradients on value and direction, since one scalar does not characterize every useful update. This diagnostic was not run here.

**A concrete loss direction to review, not a training run launched**

The least speculative next objective is assistant-only ground-truth CE on the enabled model, with KL treated as an adjustable regularizer and the two anchor terms evaluated separately. Keep true-vs-shuffled features and enabled-vs-bypassed evaluation. Add CE through the shared chunked head calculation so it does not recreate the full-vocabulary allocation problem: the current [cross_entropy.py:36](D:/DeepThought/Projects/HybridModel/DistillKit/distillkit/lossfuncs/cross_entropy.py:36) consumes the model's own loss, so this should not be assumed to be a safe YAML-only change under `logits_to_keep=1`.

An explicit causal-benefit term, if ordinary CE is insufficient, could be:

```text
L = CE_enabled + lambda * KD
    + beta * mean(max(0, margin + NLL_enabled - stop_gradient(NLL_bypassed)))
```

This is a **proposed experiment**, not a standard remedy established by the papers above. With a fixed/detached bypass, it is chiefly a reweighting of enabled CE on tokens that have not met the desired improvement; it does not by itself prove the model uses the correct table rows. Do not penalize lack of benefit on all tokens indiscriminately, and do not let the loss improve by worsening the bypass. A matched shuffled-feature readout remains necessary. An unrestricted KL separation between enabled and bypassed predictions would reward arbitrary disagreement and has no correctness guarantee.

For the first comparison after correcting parameter precision, prioritize a matched placement/objective control over a gate-learning-rate sweep: same training/evaluation documents, same reader scale and budget, fixed widening, early versus late injection, CE versus the current KD objective. Reserve fresh evaluation data if reusing a readout trained on a slice of the manifest-defined unseen pool. These are recommended follow-up experiments; none was started for this research-and-diagnosis task.

