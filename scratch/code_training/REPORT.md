# Python held-out baseline, B_code, and the unchanged-module post-evaluation

Complete. **No code-specific modules were fitted.** `G_code` and `S_code` do not exist.

| | |
| --- | --- |
| corpus | `code_corpus/v1`, dataset revision `611ef38fbfab`, split seed 20260914 |
| tokenizer | `Qwen2Tokenizer`, `tokenizer.json` sha256 `5f9e4d4901a9…` |
| evaluation subset | 1,334 documents, 1,250,215 tokens, digest `0436eba10eeb2fdb`, identical at every checkpoint |
| scored targets | 1,150,598 per arm |
| B_code | 30,741,846 tokens, 939 steps, 15,018 sequences, sha256 in `checkpoints/v1/final/milestone.json` |
| tests | 913 |

## Section 0 — vocabulary, resolved without a rebuild

Three correct numbers, not a discrepancy:

```
tokenizer.vocab_size   248,044   base BPE, excludes added tokens
len(tokenizer)         248,077   +33 added/special
config.vocab_size      248,320   padded to 1940 x 128
input embedding rows   248,320
lm_head rows           248,320   (tied)
max token id in corpus 248,046   train / calibration / heldout alike
```

`max(corpus id) < embedding rows` with 274 rows spare. Bit identity confirmed twice: all
42,229 files matched their stored `token_count` when the token store was built, and a
200-file-per-split sample confirmed the corpus pipeline equals a fresh
`add_special_tokens=False` call and that ids decode back to the stored source.
**0 pipeline mismatches, 0 round-trip failures.**

## Section 2 — token-density filter, sampled

20 rejects from 2,019 retrieved files, a 0.99% reject rate:

| category | n |
| --- | ---: |
| obfuscated / data-like | 7 |
| other (short, literal-heavy) | 8 |
| ordinary Python | 4 |
| minified | 1 |

Five of the seven "obfuscated" are **git-lfs pointer files** — `.py` paths whose entire
content is a spec URL and a sha256. Nothing else in the pipeline catches those. Genuine
Python loss is roughly 0.2–0.5% of the corpus; not material, threshold unchanged.

## Section 5–13 — pre-training baseline

| Arm | Aggregate | Content | Keyword | Operator | Delimiter | Newline | Whitespace | Other punct | Control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 1.0415 | 1.4073 | 1.0104 | 0.9256 | 0.4543 | 0.6598 | 0.2993 | 0.7041 | 7.0695 |
| B0+G′ | +0.0002 | +0.0015 | +0.0029 | +0.0012 | +0.0000 | **−0.0106** | +0.0008 | +0.0003 | +0.0607 |
| B0+S | +0.0327 | +0.0099 | +0.0150 | +0.0111 | +0.0170 | **+0.2052** | +0.0329 | +0.0326 | +0.0672 |
| B0+G′+S | +0.0327 | +0.0115 | +0.0181 | +0.0123 | +0.0170 | +0.1920 | +0.0337 | +0.0331 | +0.1339 |

Rows after the first are deltas against B0; negative is better.

**Neither module transfers.** G′ is a wash whose aggregate is *significantly worse*
(+0.00045, t = +2.1): it gains 0.0106 on newline (t = −7.4) and loses it back on keywords
(+0.0052, t = +7.0). S is straightforwardly harmful — aggregate +0.0400 (t = +45.3), with
only 63 of 1,334 documents improved, and newline +0.2052 where the same module achieved
**−0.4684 on general text**. The sign inverted on the one class it exists to correct.

### Why G′ does nothing: the policy survives, Python does not exercise it

| Familiarity bucket | Mean gate | Share of Python targets |
| --- | ---: | ---: |
| [0, 1) | 1.026 | **84.4%** |
| [1, 4) | 1.088 | 2.1% |
| [4, 100) | 0.988 | 10.3% |
| [100, 400) | 0.962 | 2.0% |
| [400, 800) | 0.932 | 0.7% |
| [800, 3000) | 0.912 | 0.4% |

Admission still falls monotonically with trigram familiarity — the exact policy learned on
general text, transferred intact. But the familiarity cache was built from general text, so
**84.4% of Python targets are trigrams it has never seen**, and across that mass the gate
emits a near-constant 1.026. The mechanism is not broken; there is nothing for it to act on.
Gate reach per layer is 0.80 / 1.00 / 1.00 / 0.96, so the gate is fully capable of moving.

### The harness was verified before these numbers were believed

A newline regression of +0.21 where the module scored −0.47 is either a finding or a
wiring bug, and both look identical in a table. Checked against the code that produced
every published structural-sidecar number:

```
sidecar vs gate_after_structure.corrected:  max |diff| 0.000e+00   MATCH
gate forced to identity vs stock:           max |diff| 0.000e+00   MATCH
sidecar changed 23,313,646 logits     gate changed 899,414,296 logits
PARITY OK -- the arms differ only by the modules
```

## Section 14–24 — code continued-pretraining

Plain causal CE over 100% of targets, no masking, **nothing attached**. AdamW8bit at 2e-5,
40-update linear warmup then cosine to 10%, packed sequence length 2048, micro-batch 2 ×
accumulate 8 = 32,752 targets per update, gradient clipping 1.0, bf16 weights with fp32
loss reduction, gradient checkpointing, seed 20260914.

8-bit AdamW is a memory decision recorded as one: fp32 AdamW state for 2B parameters is
16 GB, which with 4 GB of bf16 weights and 4 GB of gradients exceeds a 24 GB card before a
single activation. DDP across both cards is not an alternative here — NCCL is unavailable
on Windows and gloo would move 4 GB of gradients through the CPU per step.

| Python tokens | Aggregate | Δ | Content | Newline | Whitespace | Punct/Delim |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0415 | — | 1.3736 | 0.6598 | 0.2993 | 0.6118 |
| 5,011,056 | 1.0136 | −0.0279 | 1.3533 | 0.5622 | 0.3008 | 0.6025 |
| 10,022,112 | 1.0054 | −0.0082 | 1.3417 | 0.5615 | 0.2939 | 0.6011 |
| 20,011,472 | 0.9932 | −0.0122 | 1.3273 | 0.5527 | 0.2909 | 0.5895 |
| 30,741,846 | 0.9909 | **−0.0023** | 1.3240 | 0.5536 | 0.2892 | 0.5885 |

**Saturated.** The final interval's −0.0023 sits at the 0.002 early-stop floor declared
before the numbers were seen; the preceding interval was −0.0122 over half as many tokens.
30M tokens is enough for this backbone and this corpus, and more of the same would buy
little. Newline even ticks *up* slightly in the last interval (0.5527 → 0.5536).

### 12% of the headline gain is an EOS artifact, not Python learning

| class | tokens | B0 | B_code | Δ | share of aggregate gain |
|---|---:|---:|---:|---:|---:|
| content | 613,305 | 1.4073 | 1.3572 | −0.0501 | −0.02672 |
| newline | 99,889 | 0.6598 | 0.5536 | −0.1062 | −0.00922 |
| **control** | **1,318** | **7.0695** | **1.8782** | **−5.1913** | **−0.00595** |
| other_punct | 143,127 | 0.7041 | 0.6786 | −0.0254 | −0.00316 |
| keyword | 57,001 | 1.0104 | 0.9674 | −0.0430 | −0.00213 |
| delimiter | 122,183 | 0.4543 | 0.4351 | −0.0192 | −0.00204 |
| whitespace | 94,478 | 0.2993 | 0.2892 | −0.0101 | −0.00083 |
| operator | 19,297 | 0.9256 | 0.8908 | −0.0348 | −0.00058 |
| | | | | total | **−0.05062** |

1,318 control tokens — 0.115% of targets, one document-ending EOS each — carry 12% of the
aggregate improvement. B0 had never seen a file terminated by EOS in this format; B_code
saw it 15,018 times during packed training. That is the training format being learned, not
Python. **Excluding control, the genuine gain is −0.0447.** Stated because the headline
number would otherwise be overstated by an eighth.

## Section 25–28 — post-training evaluation, modules unchanged

| Arm | Aggregate | Content | Keyword | Operator | Delimiter | Newline | Whitespace | Other punct | Control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_code | 0.9909 | 1.3572 | 0.9674 | 0.8908 | 0.4351 | 0.5536 | 0.2892 | 0.6786 | 1.8782 |
| B_code+G′ | +0.0009 | +0.0012 | +0.0021 | +0.0009 | −0.0002 | **+0.0007** | +0.0006 | +0.0003 | +0.0170 |
| B_code+S | +0.0299 | +0.0095 | +0.0143 | +0.0093 | +0.0169 | **+0.1826** | +0.0294 | +0.0318 | −0.0715 |
| B_code+G′+S | +0.0309 | +0.0108 | +0.0165 | +0.0104 | +0.0167 | +0.1839 | +0.0302 | +0.0323 | −0.0541 |

### Before and after

| | on B0 | on B_code |
|---|---:|---:|
| G′ aggregate | +0.0002 | +0.0009 |
| **G′ newline** | **−0.0106** (t −7.4) | **+0.0007** (t +2.2) |
| S aggregate | +0.0327 | +0.0299 |
| S newline | +0.2052 | +0.1826 |
| G′+S aggregate | +0.0327 | +0.0309 |

The *only* thing either module did for Python — G′'s newline gain — **is gone.** Ordinary
continued pretraining took newline from 0.6598 to 0.5536 by itself, ten times the 0.0106
the gate was contributing, and left nothing for the gate to correct. S's harm is
essentially unchanged in magnitude, which is what one expects of a module whose bias is
computed from a frozen general-text hash table regardless of what the backbone does.

### Additivity

| | G′ alone | S alone | sum | observed both | additive? |
|---|---:|---:|---:|---:|---|
| on B0, aggregate | +0.0002 | +0.0327 | +0.0329 | +0.0327 | 99.4% |
| on B_code, aggregate | +0.0009 | +0.0299 | +0.0308 | +0.0309 | 100.3% |
| on B0, content | +0.0015 | +0.0099 | +0.0114 | +0.0115 | 101% |
| on B_code, content | +0.0012 | +0.0095 | +0.0107 | +0.0108 | 101% |
| on B_code, newline | +0.0007 | +0.1826 | +0.1833 | +0.1839 | 100% |

Cleanly additive on both backbones — in harm rather than in benefit, but additive.

## Section 33 — general-domain retention

| class | B0 | B_code | Δ | tokens |
|---|---:|---:|---:|---:|
| aggregate | 1.1291 | 1.1391 | +0.0100 | 298,744 |
| **content** | 1.4557 | 1.4304 | **−0.0253** | 194,882 |
| newline | 0.5028 | 0.6660 | +0.1632 | 20,177 |
| whitespace | 0.3235 | 0.4027 | +0.0792 | 14,100 |
| punctuation | 0.5253 | 0.5242 | −0.0011 | 66,827 |
| control | 1.3848 | 2.6870 | **+1.3023** | 2,758 |

Plain CE over all targets, so not comparable to the earlier assistant-masked general-domain
figures, but comparable between these two checkpoints.

**General content improved.** What regressed is layout and protocol: `control` — the chat
template tokens, absent from 30M tokens of raw Python — moves the aggregate by about +0.012
on its own, more than the entire aggregate regression. No catastrophic specialization; the
backbone got better at general words and worse at general formatting.

## Verdicts

**G′ — NO PYTHON TRANSFER.** It did not help held-out Python before code training
(aggregate +0.0002, significantly worse at t = +2.1) and does not after (+0.0009,
t = +6.4). The familiarity diagnostic explains it without ambiguity: 84.4% of Python
targets are unseen trigrams, so the learned policy has almost nothing to condition on. Its
one real effect, newline, is additionally **FULLY ABSORBED** — present at −0.0106 on B0
and gone at +0.0007 on B_code.

**S — NO PYTHON TRANSFER.** Harmful on both backbones by almost the same margin
(+0.0327 → +0.0299), driven by newline (+0.2052 → +0.1826), a class where it gained 0.4684
on general text. Not absorbed, not persistent — simply mis-specified for this domain, and
code training does not change that.

**Combined — ADDITIVE.** Within 1% on both backbones and in every class checked. The two
mechanisms do not interfere; they compose, which on Python means their errors compose.

### What the persistence question actually resolved to

Section 26 asks whether a gain is domain-mismatch correction or persistent architectural
residual error. Neither module had a gain on Python to persist, so that fork is not the one
this experiment landed on. What it produced instead is cleaner in one respect: with G′ at
±0.0009 and S at +0.03 on both backbones, no result here can be a surviving parameter fit,
because there is nothing fitted to Python to survive.

The one transfer effect that did exist — the gate's newline gain — was **absorbed
completely by 30M tokens of ordinary continued pretraining**, which is a direct §28-style
answer at the class level: on this class, stock pretraining can absorb the correction, and
does so with ten times the magnitude.

## Notes and defects

- **Machine crash at 27.5M tokens (89.4%).** Resumed from the 20M checkpoint by replaying
  and discarding the 9,776 sequences already consumed — the packed stream is a pure
  function of the corpus and the data seed, so no document was trained on twice or skipped,
  and the step counter and LR schedule continued at 611 / 7.3e-6, landing on the 2.0e-6
  cosine floor. **The AdamW moment estimates were not checkpointed and restarted from
  zero**, which is a genuine discontinuity in the low tail of the schedule; recorded in the
  provenance. Loss and gradient norms are continuous across the seam.
- **`training.json`'s `tokens_per_second: 5758` is wrong.** The resumed run divides
  cumulative tokens, including the 20M inherited from the checkpoint, by elapsed-since-
  restart. Real throughput was ~2,010 tok/s throughout, 18.0 GiB peak.
- The evaluation is a deterministic 1.25M-token subset rather than the full 5.01M heldout:
  paired stderr ~2×10⁻⁴, which resolves 10⁻³ at ~5σ, and twelve full passes were needed.
  The subset digest is stamped in every artifact.

## Stop point

Stopped as instructed. `B_code`, `B_code+G′`, `B_code+S`, `B_code+G′+S` are measured with
both modules bit-identical to their canonical checkpoints. No `G_code`, no `S_code`, no
matched architecture run, no MBPP+.
