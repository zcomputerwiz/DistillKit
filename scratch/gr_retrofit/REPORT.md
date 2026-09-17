# Recipient-initialized four-stream GR — conversion verified for symmetric, refused for asymmetric

**Status: the conversion gate is complete and the retention pilot has not started.** No
final verdict is issued here, because no pilot tokens have been spent. Symmetric
conversion passed its predeclared gate exactly; asymmetric failed it and its arm is
stopped under the task's own rule.

## 1. The conversion, and the two traps in it

The recipient computes `h + F(RMSNorm(h; gamma, eps))`. With `W_up = 0` every read gate
is exactly `1/2`, with `W_write = 0` every write multiplier is exactly `2 sigmoid(0) = 1`,
and branch gains of `2 gamma` make the four-branch mean read equal the recipient's own
normalized input. Every branch then receives the recipient's own update, so branches that
start equal stay equal and the mean collapse returns `h`.

Two ways to get that wrong, both silent, both handled:

`Qwen3_5RMSNorm` stores a *deviation* and applies `1 + weight`; `HyperConnection` also
applies `1 + branch_gain_delta`, and as a *replacement* for the block norm rather than a
gain on top of it. The stored delta is therefore `1 + 2 * weight`. The natural-looking
`2 * weight` is wrong by exactly one on every branch.

`W_down` is seeded nonzero rather than zeroed alongside `W_up`. Zeroing both factors of
the read bottleneck strands it with no gradient path — the zero-multiplier trap that made
the first borrowed-routing transfer inert.

Conversion is an opt-in post-construction step,
`Qwen35WidenedForCausalLM.recipient_initialize(asymmetric=, seed=)`. Nothing calls it from
`__init__`, so loading a converted checkpoint restores trained gains rather than
overwriting them; reload restores only the mode, seed and perturbation recorded in the
config, and converting a model that already records a conversion is refused. `W_down` is
seeded once per sublayer, not once per model.

## 2. Defects found by review, and by the gate

Three, in the order they were caught.

**Device (static review).** The branch scales were built on the CPU while the recipient's
gains stayed on the norm's device. The scales are one-dimensional, so this is not the
scalar-promotion case: against a norm already on CUDA the multiply raises outright.
Conversion is now built on the norm's device and tested both before and after placement.

**Precision on cast (static review).** The initializer cast the new gains back to the
route's current dtype, so a recipient already in bf16 had `1 + 2 * weight` rounded on the
way in. bf16 spacing near the converted value of 3 is 1.6e-2 against an AdamW step near
the learning rate, so the gains would have arrived rounded and then sat bit-frozen, and a
numerical gate failing on that would have looked architectural. `_apply` now holds the
gains at fp32 or better the way it already held the blend, restoring the pre-cast values
rather than the post-cast ones — undoing a narrowing cast recovers the dtype but not the
digits.

**Precision on reload (found by this gate).** `from_pretrained(dtype=bfloat16)` casts
every loaded tensor on the way in, which undoes the fp32 storage `_apply` protects. The
checkpoint held the correct fp32 values and the reloaded model still disagreed with the
one that wrote it by up to 0.44 of a logit. Fixed with
`_keep_in_fp32_modules_strict = ["branch_gain_delta"]`; strict, because bf16 is the
deployed precision and the plain flag only fires for fp16.

A fourth problem was in the gate script rather than the model, and is worth recording
because it produced a plausible wrong answer. `distillkit.models` installs the
expanded-GQA attention dispatch at import time. Importing it between the baseline and the
comparison read back a 1.3e-3 nat aggregate change against *bitwise identical* logits:
the original model's own arithmetic had moved underneath its own baseline. Both models
are now measured under the deployed dispatch.

## 3. Real-recipient conversion gate

Recipient: `D:/DeepThought/Projects/HybridModel/student-2b-hf`, 24 layers, 48 sublayers,
four branches, lowrank 320, `model.safetensors` sha256 `c980c584e4450232…`, tokenizer
sha256 `5f9e4d4901a92b99…`. Execution bf16 on one RTX 3090;
`fused_kernel_supports_gqa` is false, so the expanded-GQA dispatch is active. 64 documents
from the `screen` split of `full-bundle-384.json`, 29,477 scored targets, 20,079 of them
content. The `confirmation` split was not opened. Predeclared threshold: absolute
token-weighted aggregate **and** content NLL change each at most 1e-4 nats.

Baseline nondeterminism was measured first: the original agrees with itself bitwise on
all 64 documents, so "bitwise identical" below is a claim the floor supports.

| | symmetric | asymmetric |
| --- | ---: | ---: |
| aggregate NLL delta | **0.0** | **3.727e-4** |
| content NLL delta | **0.0** | **7.465e-4** |
| worst document delta | 0.0 | 4.130e-3 |
| documents changed | 0 / 64 | 64 / 64 |
| max logit difference | 0.0 | 1.156 |
| RMS logit difference | 0.0 | 4.652e-2 |
| per-layer states, max | 0.0 (25 states) | 0.750 |
| argmax agreement | 1.000000 | 0.989144 |
| cached greedy decoding, 16 steps | identical | identical |
| padded batch vs rows alone | equals the original's own | larger than the original's |
| backbone tensors unchanged | yes | yes |
| save/reload gains bitwise | yes | yes |
| reloaded logits | bitwise | bitwise |
| trained round trip | bitwise | bitwise |
| initialization not reapplied on reload | yes | yes |
| **verdict** | **PASS** | **FAIL** |

`scratch/gr_retrofit/conversion-symmetric.json`,
`scratch/gr_retrofit/conversion-asymmetric.json`.

Symmetric is not "within tolerance": it is bitwise identical to the separately loaded
original at the logits, at all 25 returned hidden states, through 16 steps of cached
greedy decoding, and across a padded batch. The aggregate and content deltas are exactly
zero on 29,477 targets, and no document moved at all.

## 4. Why asymmetric fails, and why there is nothing to fix

`scratch/gr_retrofit/read_error.py` compares each sublayer's read against the norm it
replaced, on the real recipient, in the deployed precision, with every branch still
holding the same state:

| | symmetric | asymmetric |
| --- | ---: | ---: |
| sublayers whose read is exact | **48 / 48** | **0 / 48** |
| first sublayer read error | 0.0 | 3.125e-2 |
| max read error | 0.0 | 2.500e-1 |
| max write-multiplier error | 0.0 | 0.0 |
| max branch spread | 0.0 | 0.0 |

The write multipliers are exactly one and the branches stay exactly equal in both modes.
The entire failure is the read, and it is present at the first sublayer before anything
can accumulate.

That is the construction, not a defect. Symmetric is exact because `2 gamma` differs from
`gamma` by an exact power of two: the branch normalization rounds to the same significand
and the gate of exactly one half undoes the scale without touching it. Asymmetric scales
each branch by `1 + epsilon_i` with mean-zero `epsilon`, so the branches average to
`2 gamma` only *after* summation — and each branch is rounded to bf16 before that sum.
The module test measures the same thing in fp32, where the error is 2.384e-07, one ulp.
In bf16 it is one bf16 ulp, and 48 sublayers deep it reaches a quarter of a logit.

Under the task's rules this is not a repairable implementation or precision defect: the
fixes available are retuning the perturbation, which is forbidden, or changing the route's
bf16 arithmetic, which is a redesign rather than a repair. Symmetric passed, so the
asymmetric arm stops here, as specified.

## 5. Symmetry: the corrected account, retained

The earlier claim that the read has no branch-state gradient at zero `W_up` was too
strong. Only the gate *logits* lose their input derivative, since they are produced
through `W_up`. The normalized value path still differentiates with respect to every
branch state, with the gate pinned at one half. Gradient reaches the branch inputs from
the start; what is absent is any *difference* between them.

A single sublayer read out by a mean collapse cannot break symmetry in either mode, and
that is structural rather than a defect: the read depends on the gains only through their
mean, and a mean collapse sends identical gradient to every branch. Depth breaks it in
both modes.

`scratch/gr_retrofit/attribution.json` separates the three events. First step at which
each quantity becomes unequal:

| | symmetric | asymmetric |
| --- | ---: | ---: |
| gradient arriving at branch inputs | **1** | 0 |
| write rows / branch states | 2 | 1 |
| `W_up` rows / gate values | **3** | 1 |

`W_up`'s rows are still identical at steps 1 and 2 while the branch gradients already
differ, so the gates cannot be the source. `W_down`'s per-branch-slot columns are,
routing different gradient into each slot as soon as `W_up` is nonzero. The asymmetric
mode advances every onset by one to two steps and is then overtaken: by step 8 the
symmetric branch gap is 2.25e-1 against asymmetric's 1.99e-1. Bottleneck ordering is
unchanged: `W_up` receives gradient at step 1, `W_down` exactly zero at step 1 and
gradient at step 2.

This establishes that the tested multi-sublayer construction can break symmetry without
the gain perturbation. It does not demonstrate an advantage from the perturbation or any
quality benefit from distinct streams.

## 6. Not done

The resource check and the retention pilot have not been run. No pilot tokens have been
spent, so the full 1.05M scored-token budget and the four-arm limit remain intact; with
the asymmetric arm stopped, the eligible arms are the untouched original as retention
reference and symmetric GR.

## Reproduction

```powershell
python -m pytest tests/test_gr_reference.py tests/test_gr_recipient.py
python scratch/gr_retrofit/attribution.py
$env:CUDA_VISIBLE_DEVICES=0; python scratch/gr_retrofit/convert_gate.py --mode symmetric
$env:CUDA_VISIBLE_DEVICES=0; python scratch/gr_retrofit/convert_gate.py --mode asymmetric
$env:CUDA_VISIBLE_DEVICES=0; python scratch/gr_retrofit/read_error.py
```

Each gate run takes about 65 seconds and writes a full converted checkpoint plus a
perturbed copy under `scratch/gr_retrofit/checkpoint-<mode>*`, which are gitignored.
1043 tests pass.
