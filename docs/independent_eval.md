# Independent checkpoint evaluation

The probe expects one sidecar call per residual branch, so a widened checkpoint
reports `branch_calls_per_forward` alongside `sidecar_calls`; `load_checkpoint`
selects the widened model class from the checkpoint's own `residual_stream_enabled`
and verifies its routing tensors exactly, the same way it verifies a sidecar's.

`distillkit.independent_eval` belongs in the package because checkpoint selection
needs a maintained loading contract and regression tests. It has no trainer import,
optimizer, generation loop, teacher logits, or hidden-state projections. No training
implementation was changed. `scratch/heldout_ce.py` is superseded; do not use its
numbers to compare GR and PLE checkpoints.

## Protocol

* Text: causal cross-entropy in nats, with the first token unscored and padding
  excluded. Sum document NLLs and divide by the total scored tokens. Documents
  are kept separate, with no carry-over state or packing. The bounded screen uses
  32 documents, up to 512 tokens each, minimum original length 128.
* Exclude IDs appearing in either supplied teacher-cache manifest. The supplied
  heldout file contains 1,602 surviving unique documents. Also deduplicate exact
  text within the heldout source before assigning splits. This verifies cache-ID
  exclusion, **not** absence of paraphrases, source-corpus duplicates under other
  IDs, benchmark contamination, or exposure during original pretraining.
* This is independent of the project's teacher-loss objective. It is **not** a
  reproduction of Qwen's out-of-domain Uncheatable PPL: this heldout file is from
  the same distillation corpus and contains rendered conversations. Score all
  tokens, including boilerplate and role markers; do not describe it as
  assistant-only loss or natural-prose perplexity.
* MMLU: zero-shot multiple choice with the question, A-D choices, and `Answer:`
  in context; continuations are the letters A-D. ARC-Challenge: `Question: ...`
  followed by `Answer:`, scoring the actual answer texts. No chat template,
  demonstrations, generation, or chain of thought. These follow the prompt and
  target conventions in the [MMLU template](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml)
  and [ARC template](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/arc/arc_easy.yaml).
  This small pooled question sample is not a full subject-macro MMLU benchmark.
* Jointly tokenize prompt plus space plus continuation. Require an exact token
  boundary, and score only continuation tokens. Refuse excessive question
  lengths rather than silently dropping question text or answer tokens.
  Always retain unnormalized accuracy, token-normalized accuracy, and
  character-normalized accuracy (the conventional harness `acc_norm` for ARC).
  Raw per-choice scores and all predictions remain in the result, including
  every normalization disagreement. Accuracy ties select the first choice.
* Bootstrap 10,000 paired resamples (fixed seed 20260909), with documents as
  resampling units for token-weighted NLL and questions for accuracy. JSON and
  report output contain each arm's absolute metric and all important deltas.
  A positive NLL delta hurts; a positive accuracy delta helps. Intervals quantify
  sampling variation on this selected source, not checkpoint-seed variability,
  repeated-selection bias, or numerical error. Discrete zero-width accuracy
  delta intervals do not establish population equivalence.

## Screening and confirmation

SHA256 of a versioned document/question ID assigns permanent 50/50 membership.
A second fixed hash orders records within each split. Increasing the sample size
or shuffling source data preserves both membership and each selected prefix.
Preparation stores both sets of exact token sequences and manifest/tokenizer
hashes. Evaluation defaults to `--split screen`. Confirmation is only read when
explicitly requested, and report rejects split, tokenizer, item, or task-hash
mismatches. Reserve confirmation for a fixed decision after screen-based LR
selection. Do not inspect confirmation results after each screening iteration;
an already-used confirmation set needs replacement for future independent claims.

The shipped text runs use screening only. Confirmation membership was prepared
but its model scores were not evaluated. The tiny plumbing run used two screening
documents, so it did not consume the confirmation split.

## Loading and ablations

Load the saved text configuration directly, never a training YAML. Require all
backbone and adapter weights to load; the only allowed unused keys are
`distillation_projections.*`. Compare every adapter tensor exactly to safetensors
after dtype conversion. Record adapter norms, tensor count, and a runtime probe
that observes both enabled/disabled calls and their residual/logit differences.
The probe fails if the evaluator drops the adapter call or forwarding flag.
A zero output difference alone is not an error: an authentic dormant adapter can
be inert, so output sensitivity is diagnostic alongside exact weight verification.

All table gathering goes through `SidecarDataCollator.__call__`, including its
masked-position-to-EOS conversion before hashing. The table remains memory
mapped. Use the training hasher defaults, whose EOS is 248044; do not substitute
the checkpoint's generation EOS. The tokenizer emits an existing regex warning
on this repository's tokenizer. The evaluator preserves its saved behavior for
comparability; it does not silently rewrite the tokenizer or training pipeline.

There are three primary arms: `enabled`, `bypassed` (the actual forward kwarg
`sidecar_enabled=False`), and `pre_retrofit` (stock `student-hf`). For GR only,
there is also `full_bypass`: temporarily detach the layer's entire adapter in a
try/finally context, restoring it immediately after forward. This touches only
the evaluation model instance, not code or saved weights.

The distinction is necessary because GR's False flag bypasses the n-gram
projection **but still applies trained gated-residual branches**. PLE's False flag
is already an identity. Therefore GR's flag-only stage-1 score need not equal the
student. The meaningful frozen-backbone parity check is GR `full_bypass` and
PLE `bypassed` versus `student-hf`. Report the flag-only GR comparison as well;
never label it a complete sidecar removal.

## Reproduce (PowerShell, repository root)

Use `.venv/Scripts/python.exe` throughout. To prepare full benchmark input when
Hugging Face downloads are available:

```powershell
$env:HF_DATASETS_CACHE = "$PWD/scratch/eval-datasets-cache"
.venv/Scripts/python.exe -m distillkit.independent_eval prepare --tokenizer ../student-hf --documents ../capture-data/heldout.jsonl --manifests ../teacher-cache-1m/manifest.json ../teacher-cache-5m/manifest.json --docs 32 --questions 32 --document-tokens 512 --output scratch/independent-eval/bundle.json
```

Alternatively pass `--mmlu-json <file> --arc-json <file>` with JSON arrays of
native dataset records (or dataset-viewer `rows` objects). `--text-only` prepares
local NLL input without network access. Check the source files are the requested
test datasets; the evaluator does not invent benchmark questions when downloads
are unavailable.

```powershell
$evalTable = (Get-Content examples/_lr_sweep_base.yml | Select-String '^  table_path: ').Line.Substring(14).Trim()
.venv/Scripts/python.exe -m distillkit.independent_eval evaluate --bundle scratch/independent-eval/bundle.json --checkpoint ../runs/gr-stage1-1m --table $evalTable --limit 2 --output scratch/independent-eval/tiny.json
.venv/Scripts/python.exe -m distillkit.independent_eval evaluate --bundle scratch/independent-eval/bundle.json --checkpoint ../student-hf --output scratch/independent-eval/student-hf.json
foreach ($evalName in @('gr-stage1-1m', 'ple-stage1-1m', 'lr-sweep-1e3')) {
    .venv/Scripts/python.exe -m distillkit.independent_eval evaluate --bundle scratch/independent-eval/bundle.json --checkpoint "../runs/$evalName" --table $evalTable --output "scratch/independent-eval/$evalName.json"
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $evalName" }
}
.venv/Scripts/python.exe -m distillkit.independent_eval report --reference scratch/independent-eval/student-hf.json --results scratch/independent-eval/student-hf.json scratch/independent-eval/gr-stage1-1m.json scratch/independent-eval/ple-stage1-1m.json scratch/independent-eval/lr-sweep-1e3.json --output scratch/independent-eval/report.json
```

For the already prepared local text bundle substitute `text-bundle.json`, add
`--tasks nll`, and use the `.text.json` output names. Each evaluation process has
a 540-second watchdog; `--max-seconds` cannot exceed 570. Timeout exits 124 and
leaves only incomplete `.partial` diagnostics, which reporting refuses. If a
sample would exceed the limit, prepare fewer records. Never pool partial results
from different selections to conceal a timeout. Different checkpoints can use
`--device cuda:0` and `cuda:1` without hiding either GPU.

Run the full suite with writable temporary/cache directories:

```powershell
$env:HF_DATASETS_CACHE = "$PWD/scratch/eval-datasets-cache"
.venv/Scripts/python.exe -m pytest tests -q --basetemp scratch/eval-suite-temp-fresh -o cache_dir=scratch/eval-pytest-cache
```

Use a new `--basetemp` path when an existing directory is owned by a different
execution identity. No CUDA tests are skipped deliberately. The tests cover
correct causal masks, actual collation, saved variants, missing/mismatched
adapters, swallowed forwarding flags, GR bypass semantics, stock parity,
normalization disagreement, stable disjoint splits, and paired reporting.
