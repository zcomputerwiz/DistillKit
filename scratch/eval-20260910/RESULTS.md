# Independent screen, 2026-09-10

Fresh inference-only runs: 128 documents / 59,794 causal targets, 128 MMLU and
128 ARC-Challenge questions. Text windows are at most 512 tokens. All 1,602
eligible source documents are absent from both cache manifests by ID.
The separately hashed confirmation split (128 items per task) remains unscored.

Values are estimate [95% paired percentile bootstrap CI], 10,000 draws.
NLL is nats/token; accuracy is percent and accuracy deltas are percentage points.
Positive NLL deltas hurt; positive accuracy deltas help. MMLU normalization
does not change predictions on this sample; ARC's three variants are retained.

## Absolute arms

| Checkpoint | Comparison | NLL | MMLU | ARC raw | ARC token norm | ARC char norm |
|---|---|---|---|---|---|---|
| student-hf | enabled | 1.1859 [1.0718, 1.3084] | 71.1 [63.3, 78.9] | 52.3 [43.8, 60.9] | 46.9 [38.3, 55.5] | 50.8 [42.2, 59.4] |
| student-hf | bypassed | 1.1859 [1.0718, 1.3084] | 71.1 [63.3, 78.9] | 52.3 [43.8, 60.9] | 46.9 [38.3, 55.5] | 50.8 [42.2, 59.4] |
| student-hf | pre_retrofit | 1.1859 [1.0718, 1.3084] | 71.1 [63.3, 78.9] | 52.3 [43.8, 60.9] | 46.9 [38.3, 55.5] | 50.8 [42.2, 59.4] |
| gr-stage1-1m | enabled | 1.7219 [1.6065, 1.8420] | 71.1 [63.3, 78.9] | 55.5 [46.9, 64.1] | 49.2 [40.6, 57.8] | 50.8 [42.2, 59.4] |
| gr-stage1-1m | bypassed | 1.5758 [1.4638, 1.6933] | 71.1 [63.3, 78.9] | 52.3 [43.8, 60.9] | 46.9 [38.3, 55.5] | 50.8 [42.2, 59.4] |
| gr-stage1-1m | pre_retrofit | 1.1859 [1.0718, 1.3084] | 71.1 [63.3, 78.9] | 52.3 [43.8, 60.9] | 46.9 [38.3, 55.5] | 50.8 [42.2, 59.4] |
| gr-stage1-1m | full_bypass | 1.1859 [1.0718, 1.3084] | 71.1 [63.3, 78.9] | 52.3 [43.8, 60.9] | 46.9 [38.3, 55.5] | 50.8 [42.2, 59.4] |
| ple-stage1-1m | enabled | 1.6458 [1.5223, 1.7764] | 70.3 [62.5, 78.1] | 52.3 [43.8, 60.9] | 48.4 [39.8, 57.0] | 49.2 [40.6, 57.8] |
| ple-stage1-1m | bypassed | 1.1859 [1.0718, 1.3084] | 71.1 [63.3, 78.9] | 52.3 [43.8, 60.9] | 46.9 [38.3, 55.5] | 50.8 [42.2, 59.4] |
| ple-stage1-1m | pre_retrofit | 1.1859 [1.0718, 1.3084] | 71.1 [63.3, 78.9] | 52.3 [43.8, 60.9] | 46.9 [38.3, 55.5] | 50.8 [42.2, 59.4] |
| lr-sweep-1e3 | enabled | 1.8639 [1.7510, 1.9820] | 68.8 [60.2, 76.6] | 52.3 [43.8, 60.9] | 51.6 [43.0, 60.2] | 55.5 [46.9, 64.1] |
| lr-sweep-1e3 | bypassed | 1.0918 [0.9904, 1.2011] | 71.9 [64.1, 79.7] | 52.3 [43.8, 60.9] | 49.2 [40.6, 57.8] | 51.6 [43.0, 60.2] |
| lr-sweep-1e3 | pre_retrofit | 1.1859 [1.0718, 1.3084] | 71.1 [63.3, 78.9] | 52.3 [43.8, 60.9] | 46.9 [38.3, 55.5] | 50.8 [42.2, 59.4] |

## Paired differences

| Checkpoint | Comparison | NLL | MMLU | ARC raw | ARC token norm | ARC char norm |
|---|---|---|---|---|---|---|
| student-hf | enabled - bypassed | 0.0000 [0.0000, 0.0000] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| student-hf | enabled - pre_retrofit | 0.0000 [0.0000, 0.0000] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| student-hf | bypassed - pre_retrofit | 0.0000 [0.0000, 0.0000] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| gr-stage1-1m | enabled - bypassed | 0.1461 [0.1265, 0.1678] | 0.0 [-2.3, 2.3] | 3.1 [0.8, 6.2] | 2.3 [0.0, 5.5] | 0.0 [-3.1, 3.1] |
| gr-stage1-1m | enabled - pre_retrofit | 0.5360 [0.4761, 0.5990] | 0.0 [-3.9, 4.7] | 3.1 [-0.8, 7.8] | 2.3 [-1.6, 6.2] | 0.0 [-3.9, 3.9] |
| gr-stage1-1m | bypassed - pre_retrofit | 0.3899 [0.3468, 0.4348] | 0.0 [-3.9, 4.7] | 0.0 [-3.9, 3.9] | 0.0 [-3.1, 3.1] | 0.0 [-3.1, 3.1] |
| ple-stage1-1m | enabled - bypassed | 0.4600 [0.4145, 0.5076] | -0.8 [-4.7, 3.1] | 0.0 [-3.1, 3.1] | 1.6 [-1.6, 4.7] | -1.6 [-4.7, 1.6] |
| ple-stage1-1m | enabled - pre_retrofit | 0.4600 [0.4145, 0.5076] | -0.8 [-4.7, 3.1] | 0.0 [-3.1, 3.1] | 1.6 [-1.6, 4.7] | -1.6 [-4.7, 1.6] |
| ple-stage1-1m | bypassed - pre_retrofit | 0.0000 [0.0000, 0.0000] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| lr-sweep-1e3 | enabled - bypassed | 0.7721 [0.7101, 0.8365] | -3.1 [-10.2, 3.9] | 0.0 [-4.7, 4.7] | 2.3 [-3.1, 7.8] | 3.9 [-0.8, 9.4] |
| lr-sweep-1e3 | enabled - pre_retrofit | 0.6780 [0.6072, 0.7513] | -2.3 [-9.4, 4.7] | 0.0 [-5.5, 5.5] | 4.7 [-0.8, 10.2] | 4.7 [-0.8, 10.2] |
| lr-sweep-1e3 | bypassed - pre_retrofit | -0.0941 [-0.1104, -0.0790] | 0.8 [-2.3, 4.7] | 0.0 [-3.9, 3.9] | 2.3 [0.0, 5.5] | 0.8 [0.0, 2.3] |

## Screen verdicts

- student-hf: indistinguishable by construction; it has no sidecar.
- gr-stage1-1m: hurting text NLL; raw ARC improves against flag bypass only,
  with no consistent gain across normalizations or against the stock student.
- ple-stage1-1m: hurting text NLL; benchmark accuracy is indistinguishable.
- lr-sweep-1e3: hurting text NLL; benchmark accuracy is indistinguishable.

The whole-window NLL damage is dominated by prompt/template prediction.
Assistant-span enabled-minus-bypassed NLL is also worse: GR +0.0224
[0.0199, 0.0250], PLE +0.0306 [0.0223, 0.0383], LR +0.0705
[0.0621, 0.0788]. These span checks use 121 documents with assistant tokens;
seven windows never reach an assistant span. They are a diagnostic on the
same forwards, not a separate confirmation test.

## Stage-1 parity checks

```json
[
  {
    "checkpoint": "gr-stage1-1m",
    "mode": "full_bypass",
    "documents": 128,
    "max_document_sum_nll_difference": 0.0,
    "exact_match": true
  },
  {
    "checkpoint": "ple-stage1-1m",
    "mode": "bypassed",
    "documents": 128,
    "max_document_sum_nll_difference": 0.0,
    "exact_match": true
  }
]
```

GR's False flag removes the table projection but retains the trained gated-residual
branches. Its complete-adapter bypass and PLE's False flag are the correct
frozen-backbone checks against student-hf. Training semantics were not changed.

## GR +0.0000 root cause

The old sketch loaded examples/_lr_sweep_base.yml, which forced variant=ple.
load_student_model replaced the saved gated_residual variant. Reproducing that
load discards all seven GR adapter tensors and initializes six PLE tensors; the
new PLE value projection and convolution have zero norm, making the adapter inert.
Thus +0.0000 measured a newly initialized PLE, not the trained GR. The fresh tiny
correct-load probe verifies all seven tensors exactly and changes a logit by
3.3046875. Current W_side_proj norm is 2.100469 (the task cited 2.077). See
legacy-loading-audit.json and tiny-gr.json.

## Scope and validation

This measures next-token prediction on cache-ID-excluded conversations and zero-shot
multiple-choice likelihood. It does not use teacher logits, projection losses,
generation, an optimizer, or training. Whole-document NLL includes template/prompt
tokens; the JSON report also breaks it down by role. It is not out-of-domain
Uncheatable PPL, a full subject-macro MMLU score, proof of pretraining or benchmark
decontamination, or a measurement of generated reasoning. The small screen's CIs
do not account for repeated LR selection or checkpoint seed variation. A zero
accuracy-delta interval from no observed flips does not establish equivalence.

The tokenizer emits its existing regex warning. Saved tokenization is preserved
for training comparability; no tokenizer or training-path fix was made.

451 tests passed with both GPUs visible. Tiny plumbing passed before full runs.
No training implementation was modified. The package evaluator is maintained here
because safe checkpoint selection needs a tested loading and scoring contract.

Rerun from the repository root:

```powershell
powershell -File scratch/run_independent_screen.ps1 -OutputDirectory scratch/eval-screen
```

Completed job durations:

- tiny-gr: 13.9 seconds on cuda:0.
- student-hf: 56.3 seconds on cuda:0.
- gr-stage1-1m: 176.8 seconds on cuda:1.
- ple-stage1-1m: 105.2 seconds on cuda:0.
- lr-sweep-1e3: 101.7 seconds on cuda:1.

Normalization disagreement counts:

```json
[
  {
    "checkpoint": "student-hf",
    "task": "mmlu",
    "mode": "enabled",
    "questions": 128,
    "count": 0
  },
  {
    "checkpoint": "student-hf",
    "task": "mmlu",
    "mode": "bypassed",
    "questions": 128,
    "count": 0
  },
  {
    "checkpoint": "student-hf",
    "task": "arc",
    "mode": "enabled",
    "questions": 128,
    "count": 58
  },
  {
    "checkpoint": "student-hf",
    "task": "arc",
    "mode": "bypassed",
    "questions": 128,
    "count": 58
  },
  {
    "checkpoint": "gr-stage1-1m",
    "task": "mmlu",
    "mode": "enabled",
    "questions": 128,
    "count": 0
  },
  {
    "checkpoint": "gr-stage1-1m",
    "task": "mmlu",
    "mode": "bypassed",
    "questions": 128,
    "count": 0
  },
  {
    "checkpoint": "gr-stage1-1m",
    "task": "mmlu",
    "mode": "full_bypass",
    "questions": 128,
    "count": 0
  },
  {
    "checkpoint": "gr-stage1-1m",
    "task": "arc",
    "mode": "enabled",
    "questions": 128,
    "count": 56
  },
  {
    "checkpoint": "gr-stage1-1m",
    "task": "arc",
    "mode": "bypassed",
    "questions": 128,
    "count": 56
  },
  {
    "checkpoint": "gr-stage1-1m",
    "task": "arc",
    "mode": "full_bypass",
    "questions": 128,
    "count": 58
  },
  {
    "checkpoint": "ple-stage1-1m",
    "task": "mmlu",
    "mode": "enabled",
    "questions": 128,
    "count": 0
  },
  {
    "checkpoint": "ple-stage1-1m",
    "task": "mmlu",
    "mode": "bypassed",
    "questions": 128,
    "count": 0
  },
  {
    "checkpoint": "ple-stage1-1m",
    "task": "arc",
    "mode": "enabled",
    "questions": 128,
    "count": 63
  },
  {
    "checkpoint": "ple-stage1-1m",
    "task": "arc",
    "mode": "bypassed",
    "questions": 128,
    "count": 58
  },
  {
    "checkpoint": "lr-sweep-1e3",
    "task": "mmlu",
    "mode": "enabled",
    "questions": 128,
    "count": 0
  },
  {
    "checkpoint": "lr-sweep-1e3",
    "task": "mmlu",
    "mode": "bypassed",
    "questions": 128,
    "count": 0
  },
  {
    "checkpoint": "lr-sweep-1e3",
    "task": "arc",
    "mode": "enabled",
    "questions": 128,
    "count": 57
  },
  {
    "checkpoint": "lr-sweep-1e3",
    "task": "arc",
    "mode": "bypassed",
    "questions": 128,
    "count": 58
  }
]
```
