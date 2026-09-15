# Closing the code-domain post-hoc branch

All heldout controls are neutral or worse, the forced-open control is decisively worse, and
correct addressing is indistinguishable from wrong addressing. **The branch is closed.**
No further training was started.

| | |
| --- | --- |
| tests | 952 |
| injection layer | **12**, the canonical retrofit point — not swept |
| backbone | `B_code`, frozen; digest identical before and after |
| evaluation | 1,334 documents, 1,250,215 tokens, digest `0436eba10eeb2fdb`, 1,150,598 scored targets |
| parameters | 276,482 total; 137,218 gate-only; 137,216 trainable in the blind arm |
| training | 2,001,076 tokens per arm, AdamW, LR 3e-3 constant, seq 512 × micro-batch 4, 979 s / 984 s |
| aggregates | token-weighted throughout; the unweighted class-mean selector is not used |

## Heldout table

| class | `B_code` | blind | conditioned | conditioned_g1 | conditioned_wrong |
|---|---:|---:|---:|---:|---:|
| **aggregate** | **0.990868** | **+0.000161** | **+0.000490** | **+0.010977** | **+0.000494** |
| content | 1.3572 | +0.0002 | +0.0006 | +0.0158 | +0.0006 |
| keyword | 0.9674 | +0.0004 | +0.0012 | +0.0118 | +0.0007 |
| operator | 0.8908 | −0.0001 | +0.0001 | +0.0110 | +0.0002 |
| delimiter | 0.4351 | +0.0002 | +0.0003 | +0.0039 | +0.0003 |
| newline | 0.5536 | +0.0001 | +0.0005 | +0.0041 | +0.0006 |
| whitespace | 0.2892 | −0.0001 | −0.0001 | +0.0018 | +0.0001 |
| other punctuation | 0.6786 | +0.0001 | +0.0003 | +0.0067 | +0.0004 |
| control | 1.8782 | +0.0050 | −0.0014 | +0.0126 | −0.0062 |

Paired per-document deltas against `B_code`:

| arm | mean | stderr | t |
|---|---:|---:|---:|
| blind | +0.000222 | 0.000077 | +2.9 |
| conditioned | +0.000507 | 0.000100 | +5.0 |
| conditioned_g1 | +0.012874 | 0.000416 | +30.9 |
| conditioned_wrong | +0.000572 | 0.000101 | +5.7 |

## The two comparisons the task turns on

```
conditioned  -  conditioned_g1   -0.010487
correct      -  wrong addressing -0.000004
```

**The forced-open control is 22x worse than the gated arm, not better.** The reopen
condition — `conditioned_g1` materially beating `B_code` while `conditioned` does not — is
not merely unmet, it is contradicted in the opposite direction: opening the gate to `g = 1`
costs +0.0110 aggregate against `B_code`, with content +0.0158. This is not a
gate-starvation or training-dynamics problem. The gate is suppressing a genuinely harmful
branch, which is exactly what its monotone closing trajectory predicted.

**Correct addressing is worth four millionths of a nat.** At −0.000004 against the
wrong-address control, with paired t-statistics of +5.0 and +5.7 that overlap completely,
the learned branch retained **no context-specific signal at all**. Per §4 this is not
overinterpreted: the correctly-addressed arm is not itself useful, so the addressing
comparison is further evidence of nothing surviving rather than evidence about addressing.

## Learned admission trajectory

| tokens | gate mean | calibration content |
|---:|---:|---:|
| 400k | 0.3703 | 1.712085 |
| 800k | 0.2810 | 1.711083 |
| 1.2M | 0.2136 | 1.710855 |
| 1.6M | 0.1751 | 1.711080 |
| 2.0M | 0.1514 | 1.710979 |

Calibration content is flat and **non-monotonic** across the whole run, wandering inside a
band narrower than the noise floor measured on `G_code`. The blind arm behaves the same
way (1.711149 to 1.713640, best checkpoint at 1.6M). Meanwhile admission falls
monotonically and is still falling at 2M: given 137,218 parameters to decide when to admit
the correction, the optimizer spends them learning to admit less of it.

At the selected checkpoint the gate is live but useless — mean 0.2106, p10 0.1273,
p50 0.2033, p90 0.3061, range 0.091 to 0.396, so it has real variance and is nowhere near
saturated. The compatibility score is symmetric about zero (mean +0.00039, p10 −0.552,
p90 +0.549): query and key directions are essentially uncorrelated, which is what "the
state carries no admission signal for this branch" looks like mechanically. The injected
correction is not negligible in size — 4.8% of the residual norm on average, 7.8% at p90 —
so the branch is perturbing the stream meaningfully and the gate is scaling down damage
rather than trimming noise.

Because calibration was flat and the module was converging toward identity at 2M tokens,
the run was **not extended to 4M**, per the stop rule.

## Verdicts

```
G_code residual gate                        CLOSED
S_code post-hoc structural branch           CLOSED
POST-HOC CODE STRUCTURAL CORRECTION         CLOSED
```

After ~30.7M Python tokens of backbone specialization, the tested post-hoc hash-conditioned
structural correction provides no useful residual gain at this scale. Every arm is neutral
or worse; the state-conditioned gate does not beat its matched state-blind control
(+0.000329 in the wrong direction); forcing the gate open is decisively worse; and correct
addressing is indistinguishable from wrong addressing.

**The code-specialized backbone has already absorbed the structural opportunity available
to these post-hoc mechanisms.**

This says nothing about Flash-Next-style PLE/GR trained end-to-end from pretraining. It
says that grafting a post-hoc structural correction onto an already code-specialized 2B
backbone does not show enough residual signal to justify further optimization.

## Not run

No additional fits, no forced-open retraining, no larger gates or decoders, no different
hash widths, no learned tables, no GR/HyperConnections, no MBPP+ or HumanEval+, no backbone
training. The one result that would have justified reopening did not occur.
