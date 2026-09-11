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
so 6.048 is a lower bound. The whole useful output of this module is `short_conv(value)`,
which is the one branch the gate never touches.

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
2. **"The convolution bypasses admission."** Confirmed, and it matters — but the branch
   it bypasses with is the only one that works.
3. **"A context-only gate is underinformed for semantics."** Confirmed, and adding the
   value read does not fix it. That hypothesis is not the one to build on.

The finding none of the three anticipated: **α = 0 under both grades.** The gated value
write — the entire thing the gate exists to admit — is worth nothing anywhere. The gate
is admission control over a path that should not exist, and the path that works goes
around it.
