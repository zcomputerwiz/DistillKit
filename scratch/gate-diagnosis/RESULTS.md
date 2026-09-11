# What the layer-1 gate should have done, measured on the frozen C1 checkpoint

No training. Four evaluation passes and two 150-step scalar fits.

```
python scratch/gate_diagnosis.py verify      --checkpoint ../runs/win-C1-L1-real-stage1-1m
python scratch/gate_diagnosis.py oracle      --checkpoint ... --grade content
python scratch/gate_diagnosis.py oracle      --checkpoint ... --grade all
python scratch/gate_diagnosis.py sensitivity --checkpoint ...
python scratch/gate_diagnosis.py logistic    --checkpoint ...
python scratch/gate_diagnosis.py calibrate   --checkpoint ... --grade content
python scratch/gate_diagnosis.py calibrate   --checkpoint ... --grade layout
```

`verify` asserts the instrumented forward is bit-identical to the module's at
α=β=t=1, b=0 (`max_forward_difference` and `max_gate_difference` both exactly 0.0),
which is what licenses reading anything off the reimplemented admission score.
Calibration fits on the bundle's `confirmation` split and reports on `screen`.

## The headline

| grade | tokens | α\* | β\* | checkpoint cost | calibrated cost |
|---|---|---|---|---|---|
| content | 147,918 | **0.000** | **0.000** | +0.003512 | **+0.000000** |
| layout | 8,647 | **0.000** | **6.048** | −0.103020 | **−0.128732** |

α is the gated value write — the path the gate exists to admit. **It fits to zero under
both grades.** On content the best available calibration of this module is to switch it
off, and the calibrated model reproduces bypassed exactly (0.469513 against 0.469513),
generalising from the split it was fitted on to the one it was not.

On layout the convolution wants to be **6× stronger** and was still climbing at step 150,
so 6.048 is a lower bound.

> **Do not read α\*=0 as "the value write carries nothing."** It was read that way here
> at first, and the follow-up below shows it is wrong: α and β are substitutes, because
> `c(v)` is a filter over the same `v`. Scored separately at the checkpoint's own scales,
> the gated value write is 81% of the layout gain and the convolution is 22%.

## The gate is not selecting, and cannot

`_admission` reads only the residual stream. Its operating point on 295,836 stream-token
verdicts: mean **0.9135**, std 0.1263, 1st–99th percentile 0.585–1.304 — sitting at its
neutral 1.0 with little spread.

Every per-token oracle AUC, where the oracle is q = ∂L_content/∂g from one backward pass
through an additive zero-initialised probe:

| predictor | AUC (all) | AUC (\|q\| above median) | held-out, 5-fold by document |
|---|---|---|---|
| z₀ | 0.4999 | 0.5026 | — |
| z₁ | 0.5007 | 0.4989 | — |
| z̄ (what the gate uses) | 0.5005 | 0.5004 | 0.4965 |
| gate | 0.5005 | 0.5006 | — |
| **cos(h, v)** | **0.5007** | 0.5006 | **0.4968** |
| log ‖v‖ | 0.4998 | 0.5016 | 0.4976 |
| z̄ + cos + log‖v‖ | — | — | 0.4973 |

Fitted logistic weights are +0.003 (z̄), +0.002 (cos), +0.000 (log‖v‖).

**The table-aware candidate buys nothing.** `cos(rms(h), rms(v))` is 0.5007 in sample and
0.4968 held out. It has almost no variance to offer either: mean **−0.0348**, std 0.0105.
The value write is close to orthogonal to the stream it is written into, and its size is
‖v‖/‖h‖ median **0.0084** — a sub-1% nearly-orthogonal perturbation.

### How blunt is this instrument?

An AUC near 0.5 only means "no signal" if the instrument could have shown one. The
layout/content split is a distinction already known to be large — −0.103 against +0.0035
nats, a 30× effect established without any gradient — and the oracle recovers it:

| class | mean q | want opening | median \|q\| |
|---|---|---|---|
| layout | **−8.36e-4** | 52.2% | 1.64e-3 |
| content | **+9.43e-5** | 49.0% | 1.43e-3 |

The sign flips between the classes, in the direction the NLL decomposition predicted.
But that known 30× effect registers at only **AUC 0.5181**. Read every number above
against 0.518, not against 1.0 — and note that 0.4965 held out still fails to reach even
a quarter of it.

## Averaging the two directions destroys the signal they have

The gate is `2·mean_k σ(z_k)`. The two directions within a stream correlate at
**−0.8066**. That is not a curiosity — it is the mechanism pinning the gate at neutral,
and it cancels real information. Asking each feature whether it can tell that the *next*
token is layout:

| predictor | AUC |
|---|---|
| ‖v‖/‖h‖ | **0.5362** |
| z₀ | 0.5206 |
| cos(h, v) | 0.5075 |
| gate | 0.5038 |
| **z̄ — what the module actually uses** | **0.4997** |
| z₁ | 0.4783 |

z₀ at 0.5206 and z₁ at 0.4783 carry opposite signal of nearly equal size. Their mean is
0.4997. The module computes a usable criterion and then averages it away.

`‖v‖/‖h‖` at 0.5362 is the best single feature available, and it beats the gradient
oracle's own 0.5181 on the same question.

## What the convolution's own gradient says

Summed over tokens, ∂L/∂β = **+41.42** on content (positive: β should fall) and **+30.59**
across all assistant tokens. 52.3% of content tokens want it attenuated. The layout fit
wants it 6× larger. A single β cannot serve both, and the checkpoint's β=1 is the
compromise — which is exactly the case for putting the convolution behind admission
control, `g·[v + c(v)]` rather than `g·v + c(v)`, where the RMS norm cannot cancel the
gate.

## Verdicts against the hypotheses

1. **"The gate is too permissive because its neutral is 1.0."** True as description
   (0.9135, std 0.126), false as diagnosis. Bias and temperature cannot help when the
   ranking is 0.50. In the content fit they drift to −1.040 and 0.622 only *after* α and
   β reach zero, at which point they are unidentifiable and mean nothing; in the layout
   fit the temperature goes to its 1e-3 floor, which is the fit flattening the gate into
   a constant because its ranking is worthless.
2. **"The convolution bypasses admission."** Confirmed as a fact about the code. Its
   weight is smaller than it looked from the fit: scored alone the convolution is
   −0.022354 on layout and +0.000749 on content, against the gated value write's
   −0.083207 and +0.002060. Gating it is worth at most 0.0007 on content.
3. **"A context-only gate is underinformed for semantics."** Confirmed, and adding the
   value read does not fix it. That hypothesis is not the one to build on.

The finding none of the three anticipated: **α = 0 under both grades** in the constrained
fit. That is a statement about where a constrained optimum sits, not about which path
carries the signal — see the correction in the follow-up. What it does establish, and
what survives everything below, is that on content the best available setting of this
module is off, and that no per-token feature tested can tell the two cases apart.

---

# Follow-up: signed scalars, the norm ratio, and the convolution taps

Three frozen-checkpoint measurements, no training.

## 1. The value write is not backwards — it is just small

Dropping the α, β ≥ 0 clamp, the content fit goes meaningfully negative:
α\* = **−0.51932**, β\* = **−0.72262**. Taken alone that says the learned direction is
usable but wrong-signed, which would be a parameterisation bug worth fixing rather than
a useless read.

The shuffled arm says otherwise. Applying C1's fitted scalars to each checkpoint:

| checkpoint | α | β | content cost | 95% |
|---|---|---|---|---|
| C1 real | −0.5193 | −0.7226 | **−0.000518** | [−0.000721, −0.000307] |
| **C2 shuffled** | −0.5193 | −0.7226 | **−0.000432** | [−0.000630, −0.000232] |

Overlapping intervals, 0.000086 apart. **Inverting the write helps the content-free
control just as much.** Subtracting half of a small near-orthogonal vector is a generic
regulariser, not recovered content. There is no sign bug; the direction simply carries
nothing usable, and −0.0005 is what noise-cancellation is worth.

## 2. ‖v‖/‖h‖ is an identity detector, not a confidence one

Sidecar cost by decile of the norm ratio:

| decile | all tokens | layout share | content only | layout only |
|---|---|---|---|---|
| 1 | +0.002390 | 2.9% | +0.002454 | −0.001309 |
| 5 | −0.003735 | 5.7% | +0.004194 | **−0.457970** |
| 6 | **−0.034843** | 7.7% | +0.002391 | −0.297577 |
| 10 | +0.005697 | 3.9% | +0.005746 | +0.003277 |

Within content the deciles run +0.001853 to +0.005746 — flat to mildly worsening, no
benefit at any ratio. The structure in the pooled column is composition: layout share
swings 2.9%→7.7% and decile 6's −0.0348 is those tokens. Its 0.5362 AUC on "the next
token is layout" was real and useless: it detects the token class, which is what we
already had. **Discard it as an admission variable.**

## 3. The temporal filter is not the mechanism — and the value write is

Cumulative tap ablation, kernel index 3 instantaneous (impulse-verified, not derived):

| kept taps | layout | content |
|---|---|---|
| conv off (α=1, β=0) | −0.083207 | +0.002060 |
| t | −0.096041 | +0.002789 |
| t, t−3 | −0.098330 | +0.003009 |
| t, t−3, t−6 | −0.100545 | +0.003302 |
| full | −0.103020 | +0.003512 |

And the two paths scored separately at the checkpoint's own scales:

| configuration | layout | content |
|---|---|---|
| (α=1, β=0) gated value only | **−0.083207** | +0.002060 |
| (α=0, β=1) convolution only | −0.022354 | +0.000749 |
| (α=1, β=1) the checkpoint | −0.103020 | +0.003512 |

Nearly additive. **The gated value write is 81% of the layout gain; the convolution is
22%; the three delayed taps together are 6.8%.** Codex's "tiny causal sequence model
over n-gram memory" is not what is happening — temporal mixing is a minor term.

### Correcting the earlier reading

The layout fit's α\*=0, β\*=6 was read here as "the whole useful output is
`short_conv(value)`". **That was wrong.** α and β are substitutes, because `c(v)` is a
filter over the same `v`: with β free to reach 6 the convolution can supply what the
value path supplied, slightly better (−0.1279 against −0.1030). It does not follow that
the value path contributes nothing at the scales it was trained at — scored alone it is
four times the convolution. A constrained optimum at a boundary does not identify which
path carries the signal when the paths are not independent.

## Where that leaves the redesign

Every path helps layout and hurts content, in rough proportion to its magnitude, and no
per-token feature separates the two cases (every AUC 0.4965–0.5026 against an instrument
ceiling of 0.5181). The only lever that worked is global scale, and on content its
optimum is zero. Retiring `value_proj` from the residual write is not supported — it is
the larger of the two paths. Gating the convolution is worth at most 0.000749 on content.

What survives untouched is the direction finding: corr(z₀, z₁) = −0.8066 with AUCs of
0.5206 and 0.4783 on the same question, cancelling to 0.4997. One direction instead of
two averaged ones is a change with a measured reason behind it, independent of
everything above.
