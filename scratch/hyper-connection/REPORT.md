**Faithful Flash-Next routing, with a measured path from identity — 2026-09-11**

The donor sublayer arithmetic is now available as an opt-in `routing: flash_next` module. The original `WidenedResidual` implementation is unchanged. Exact identity survives loading the real donor tensors when `blend=0`, including bit-identical real-student logits and all 33 returned hidden states. Three short optimizer steps and two batch-2 x 4096 steps completed with finite loss and stable streams.

**Correct arithmetic does not make full transplantation safe.** At `blend=1`, the collapsed stream stays bounded, but assistant NLL is **10.975871 versus 0.517634** for the pre-retrofit student on an eight-document diagnostic. A bounded RMS is necessary evidence about numerical scale, not sufficient evidence that the backbone still predicts correctly. I recommend starting at zero and warming only to **0.01**, then holding there for the next separately launched run. No full training run was launched; all changes remain uncommitted.

**What the donor computes, and where that answer came from**

The requested [HF revision](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) is accessible. Its API file listing contains no Python modeling files. Its configuration identifies `Qwen4ExpForConditionalGeneration`, four branches, hidden size 2560, rank 320, and RMS epsilon 1e-6. Modeling code was obtainable separately: [Transformers at commit 6b07e4510e3f9667f5256656118515bcde306fc4](https://github.com/huggingface/transformers/blob/6b07e4510e3f9667f5256656118515bcde306fc4/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1002), specifically `Qwen4ExpTextRMSNorm`, `Qwen4ExpTextGatedResidual`, and the decoder's injection operations. The installed Transformers lacks this model, so I downloaded source into scratch without installing or upgrading anything. This code is pinned independently of the weight revision; their joint historical provenance is not established.

For branch state R_i, n branches, and the extracted norm deviation delta_i, the sublayer equations are:

```text
Z_i = (R_i / sqrt(mean_channels(R_i^2) + eps)) * (1 + delta_i)
z   = concatenate_branches(Z)
G   = reshape_branches(sigmoid(W_up SiLU((W_down z) / n)))
x_D = mean_branches(G * Z)
s_D = 2 * sigmoid((W_write z) / n)
y   = attention_or_MLP(x_D)
R'_i = R_i + s_D[i] * y
```

Normalization uses FP32 arithmetic and casts back to the residual dtype. The norm gain is the donor's own gain, not its product with the student's layer-norm gain. The read has no unconditional branch anchor; its gates are per branch/channel. The write has one scalar per branch, ranges from zero to two, and is not normalized across branches. Both projections see all normalized branches. The original block pre-norm is replaced at the donor endpoint. These operations are directly visible in the [pinned implementation](https://github.com/huggingface/transformers/blob/6b07e4510e3f9667f5256656118515bcde306fc4/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L152).

The [Qwen technical report, Section 2.2, equations 30–34](https://arxiv.org/html/2608.30320v1#S2.SS2) independently gives the same arithmetic, including the easily missed mean on the read and factor two on the write. It calls the shipped variant **Gated Residual (GR)**, although checkpoint names retain hyper-connection terminology. It removes learned residual mixing and uses identity carry. The report explicitly says GR replaces pre-normalization. [NVIDIA NeMo's implementation](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/models/qwen3_8_flash_next/layers.py) supplies a third matching implementation; its downloaded bytes are hashed in `sources.json`, since that link is to a moving branch.

The original [Hyper-Connections paper, Sections 2.1–2.3](https://arxiv.org/html/2409.19606v3#S2) describes static and dynamic read/write/residual-mixing operators and initialization equivalent to a pre-norm network. That is useful background, but its tanh dynamic corrections and residual mixing are not the shipped donor equations. I did not infer sigmoid versus softmax or scaling from tensor shapes: the sources above settle them.

The task's prior diagnosis missed three distinctions beyond removing the anchor: division by n after gating the read, the factor two on the write, and replacing the student norm rather than multiplying its gain. Also, algebraically, `read_offset=-one_hot(read_index)` can cancel the old anchor and `lambda_read=1/n` supplies the mean. Thus “no offset can cancel the anchor” is too strong as a mathematical claim. A dedicated module avoids that cancellation and handles the independent donor norm explicitly. No legacy parameters or behavior were changed to apply these corrections retroactively.

**Implementation and identity bridge**

`distillkit/hyper_connection.py` implements `HyperConnection.read/write` using the existing decoder interface. It accepts the four extracted tensor names directly: `W_down.weight`, `W_up.weight`, `W_write.weight`, and `branch_gain_delta`. The existing loader copies those tensors but preserves the new module's requested blend; only legacy modules receive the old lambda/write-offset initialization. Attention and MLP each have their own route at all 32 layers. The existing proportional map sends student layer 24 to donor block 36 and layer 31 to block 47.

For a common scheduled scalar a:

```text
x_I = original_student_norm(R[read_index])
x_a = x_I + a * (x_D - x_I)
s_a = 1 + a * (s_D - 1)
R'_i = R_i + s_a[i] * F(x_a)
```

The two endpoints take explicit paths. At zero, donor arithmetic is skipped entirely and the original norm and residual addition run directly. Even nonfinite donor weights cannot introduce `0 * NaN`. Identically expanded branches therefore remain identical after each stock sublayer, and the existing centered collapse returns the original state exactly. At one, the module returns the pure donor read/write, with no student norm in its computation. Intermediate values interpolate block inputs and write coefficients; this is not an interpolation of two complete model outputs.

Blend is a persistent FP32 buffer, excluded from AdamW. Casting and checkpoint reload preserve even values such as 0.015 without BF16 rounding. `HyperConnectionWarmupCallback` advances it using completed optimizer steps and reconstructs it from `global_step` on resume. It changes only outside forward/backward, so non-reentrant checkpoint recomputation sees the same value. Step zero can be exactly identity while step one begins borrowing; the zero multiplier cannot silently persist when a nonzero warmup target is configured. With warmup disabled, blend stays fixed. Nondefault blend controls are rejected for legacy routing rather than silently ignored.

The new route belongs to the architecture's AdamW group, remains trainable under stage-1 freezing, emits blend/norm telemetry, and survives direct and TP checkpoint export. Saved routing type participates in the loader's architecture compatibility check; a donor-route checkpoint cannot silently load as the legacy route.

The donor's **final learned mixer is not transplanted**. The student's branch collapse, final norm, vocabulary head, attention and MLP weights remain its own. Therefore blend one is faithful *sublayer routing*, not a complete Flash-Next decoder. That boundary preserves the intended retrofit and is another reason not to equate matching routing equations with a successful cross-model transfer.

For memory, branch RMSNorm uses the existing saved-state custom backward. A new gated-mean autograd function processes each branch's gate/product separately, avoiding full widened sigmoid/product temporaries. It saves normalized states and logits, and recomputes gates in backward. Write products are also branch-sized and use separate multiply/add rounding like the reference. Normalized states, logits and returned widened streams still exist; the implementation does not claim to eliminate widened storage. No torch.compile or nested offload hooks were added. Backward uses FP32 intermediate arithmetic and supports first derivatives, as the existing norm does; it is not asserted bit-identical to BF16 eager backward.

**Verification**

Before the real-model probe, the expanded full suite passed **504 tests**. The final suite, including configuration and TP buffer coverage added during review, passes **506 tests**; see `full-tests.log`. `WidenedResidual` itself has no diff. Existing tests cover trained legacy routes, sidecars, identity, loading, checkpointing, and TP export. I did not rerun or rescore all historical 72-step arms.

New tests independently implement the equations on unequal synthetic branches and nonunit norm gains, check gradients and a double-precision gradcheck, assert bitwise identity with poisoned donor weights, and verify interpolation, optimizer ownership, checkpoint replay, dtype preservation, validation, and export. `source_oracle.py` additionally extracts only the two relevant classes from the downloaded Transformers source using AST, executes them without importing the whole new model, and compares identical synthetic weights. FP32 and BF16 CPU reads and writes are **bit-identical**, maximum read difference zero (`source-oracle.json`). GPU end-to-end identity also passes with the real student, separately loaded and sharded using the same TP kernels: logits and all 33 returned hidden states match bitwise.

The real sweep loads all 256 extracted tensors used by the 64 student sublayers. Its diagnostic consists of the eight shortest documents in the existing held-out reply screen: 200–259 tokens each, **730 assistant tokens** total. IDs and per-document scores are saved. Entire documents and existing role spans are retained. They are outside the training cache, but the small length-selected sample is not representative of the 384-document evaluation. Every NLL below is assistant-only, measured against a separately loaded pre-retrofit student on precisely the same targets.

| Blend, before training | Absolute assistant NLL | Change from student | Layer-31 collapsed RMS | Largest layer RMS ratio to student |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.517634 | 0.000000 | 1.256112 | 1.0000 |
| 0.01 | 0.515128 | -0.002506 | 1.249610 | 0.9964 |
| 0.05 | 0.506510 | -0.011124 | 1.224811 | 0.9841 |
| 0.10 | 0.503938 | -0.013696 | 1.195632 | 0.9706 |
| 0.25 | 0.540939 | +0.023305 | 1.121097 | 0.9366 |
| 0.50 | 1.195129 | +0.677495 | 1.129671 | 0.9121 |
| 1.00 | 10.975871 | +10.458237 | 0.669663 | 1.9508 |

All 32 per-layer collapsed RMS values are saved for every blend. The screen's predeclared scale guard is RMS <= 5 and layerwise ratio <= 4, with finiteness checked at every hooked forward. The document-mean profiles all pass, including the pure donor endpoint; the saved measurements were also asserted after the run. The baseline's final-layer value differs from the task's illustrative 1.105 because these are different documents. **The full donor fails NLL despite passing the scale guard.** This establishes a concrete counterexample to using bounded RMS alone to approve a transfer, without establishing which feature/basis mismatch causes the NLL failure.

The three-step production probe uses real resident IQ4_NL rows, the real 1M offline cache, complete training documents of lengths 145, 148, 152, the original 0.7 sparse KL + 0.3 two-anchor cosine objective, layer-24 sidecar, two-device TP, gradient checkpointing, stage-1 freezing, clipping at 1.0, and LR 1e-4. LR warmup is disabled to exercise a full optimizer update within the short budget. Blends during the steps are 0, 0.005, 0.01; the saved/evaluated endpoint is 0.015.

| Step | Training objective (finiteness only) | Largest collapsed layer RMS | L24 read-down coordinates updated | L24 read-up coordinates updated |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.567114 | 1.285581 | 0 | 0 |
| 2 | 4.152348 | 1.253497 | 2,030,481 | 1,590,612 |
| 3 | 2.732459 | 1.276997 | 1,752,385 | 1,366,276 |

Zero routing updates on the first identity step are intentional. At step two, down/up/write/norm receive gradients and update; the sidecar's zero-initialized value projection already updates at step one. The donor norm remains BF16 like the extracted-weight training path: only 465/10,240 observed L24 attention norm coordinates move on step two, while millions of projection coordinates move. FP32 norm storage could be tested separately; this work does not claim all small donor-weight updates survive BF16. Blend itself has no such storage problem. These objective values use different documents and are not a quality ranking.

After three steps, the small held-out screen gives:

| Mode | Absolute assistant NLL | Change from pre-retrofit student |
| --- | ---: | ---: |
| Enabled | 0.512473 | -0.005161 |
| Bypassed | 0.513987 | -0.003647 |

Enabled minus bypassed is **-0.001514** nats. A descriptive paired document bootstrap using `score_plegated.paired` gives [-0.002856, -0.000248] for that difference; corresponding enabled/bypassed versus student intervals are [-0.007237, -0.002890] and [-0.006091, -0.001291]. These eight short documents were inspected during the blend sweep, so the intervals are **not an independently confirmed improvement** or a substitute for the full 384-document decision. Absolute controls are shown to avoid the broken-backbone subtraction trap. Details are in `paired-screen.json`.

The allocator probe uses two real cached 4096-token documents, batch 2, accumulation 1, blend 0.01, and two optimizer steps. The second step includes resident Adam moments. Losses are finite at **8.724353 and 8.425455**; maximum collapsed layer RMS is **1.871658 and 1.873929**. Peak allocated memory is **12.443 GiB on card 0 / 7.162 GiB on card 1**; peak reserved memory is **13.086 / 7.928 GiB**. No OOM or checkpoint-recompute error occurred. These are maxima across recorded forward/backward and optimizer phases, not the end-of-run counters (the production telemetry resets peaks).

An initial one-step long-shape check preceded the two-step check; its evidence is retained under `initial-long-*`. Total optimizer steps actually executed are **six: three short, one initial long, two final long**. The final two-step loop took about 18 seconds including reporting; total process time was 48 seconds. The short process, including source-model comparison and the blend sweep, took 93.4 seconds. Every probe has a 540-second watchdog. Model export, checkpoint saving, and integrations were suppressed. The prepared next-run YAML was not executed.

**Recommendation and remaining uncertainty**

Use `next-run.yml` as the proposed next experiment: n=4, rank=320, donor proportional initialization, sidecar at layer 24, `blend=0`, `blend_target=0.01`, and `blend_warmup_steps=20`. It keeps the existing objective and starts from `student-hf`; batch 2 and accumulation 8 retain the original effective batch of 16 while using the measured per-microbatch shape. At completed optimizer step k, a(k)=0.01*min(k/20,1). A later increase should be conditional on the full assistant-NLL check, with bypassed absolute NLL reported alongside the sidecar difference. The 20-step bridge is a conservative recommendation, not a schedule proven optimal by the three-step probe. Do not automatically ramp to one merely because the implementation supports it.

Assumptions made: proportional 32-to-48 layer mapping is retained for continuity and is not a proven correspondence; the student's own final collapse/norm/head are retained; common scalar interpolation is an engineering bridge, not an upstream training method. The sources establish the shipped sublayer equations, not compatibility of donor-trained features with this student's backbone. Small-blend improvements in the diagnostic may reflect generic rescaling rather than useful donor knowledge; a matched zero/random-donor control would be needed to attribute them to borrowed information.

Not established: a benefit on the full 384 documents / 156,565 assistant tokens; successful recovery at blend one; the best blend, warmup length, or donor layer map; 72-step allocator behavior across varying document lengths and evaluation transitions; GPU bitwise equality to a full runnable Flash-Next model; or changes to the memory-free teacher objective's suitability. No full donor model was loaded and no attention/MLP/PLE weights were borrowed. The existing extraction was consumed, not changed or rerun.

Reproduce from the repository root with the specified interpreter:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scratch/hyper-connection/source_oracle.py
.venv\Scripts\python.exe -u scratch/hyper-connection/probe.py
.venv\Scripts\python.exe -u scratch/hyper-connection/probe.py --long
```

Full per-layer table and file inventory follow in `LAYER_RMS.md` and `FILES.md`. Those files list all retained task artifacts and every changed source file with its purpose. `PROGRESS.md`, the existing run configs/checkpoints, and the pre-existing untracked scratch work are unchanged.
