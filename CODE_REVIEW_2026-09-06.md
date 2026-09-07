# Code review — outstanding integration issues

Reviewed: 2026-09-06. Repository: `D:\DeepThought\Projects\HybridModel\DistillKit`.

This reviews the combined uncommitted working tree, without attributing changes
to individual contributors. No implementation changes were made during review.
All four findings below are P2 (medium priority). Line numbers refer to the
reviewed snapshot; function names identify the relevant code if lines move.

The implementation hints and regression checks are recommendations for the
current coder. They have not been applied. Address findings 1 and 2 first because
they prevent affected training configurations from starting.

## 1. Preserve the cached vocabulary independently of sidecar configuration

**Location:** `distillkit/main.py:227–234`, `load_student_model`; downstream
rejection at `main.py:347–349`, `do_distill`.

**Trigger:** Use `teacher.cache_path` with a padded-vocabulary teacher cache and
omit the `sidecar` configuration entirely.

**Problem:** The loader preserves the original padded head only when
`config.sidecar is not None`. Otherwise, it resizes the student to the tokenizer
vocabulary size. In this project that reduces 248,320 entries to 248,077. The cache
contains top-k signals normalized over the original 248,320-entry head, so
`do_distill` subsequently rejects the now-smaller student vocabulary.

An explicitly configured `sidecar: {enabled: false}` already takes the preserving
branch; this finding concerns the valid standalone cached-distillation path with
no sidecar configuration.

**Implementation hints:**

- Determine the required signal vocabulary before loading/resizing the student.
  `do_distill` already obtains `signal_source.vocab_size` before calling
  `load_student_model`, so it can pass that requirement into the loader.
- For cached signals, preserve an existing compatible padded head and validate
  that tokenizer IDs are covered. Do not shrink it just because the tokenizer
  has fewer real entries.
- Reject genuinely incompatible model/cache vocabularies with a clear error.
  Do not repair this by dropping cached IDs: that would change the captured
  distribution and its normalization.
- Preserve existing tokenizer-resize behavior for unrelated workflows where it
  remains intentional.

**Suggested regression check:** Use a tiny model with a 64-entry head and a
60-entry tokenizer, a cache whose vocabulary is 64, and no sidecar configuration.
Verify that loading preserves 64 entries and a cached loss computation accepts
IDs 60–63. Also cover an actually undersized head and the explicit disabled-sidecar
control configuration.

## 2. Handle empty cache splits before constructing a generator dataset

**Location:** `distillkit/offline_cache.py:396–402`,
`OfflineTeacherCache.to_dataset`; caller at `distillkit/main.py:335–341`.

**Trigger:** A valid cache has training documents but no evaluation documents,
for example when every input record explicitly specifies `split: train`.

**Problem:** `Dataset.from_generator` raises on an empty generator even when
features are supplied. The observed error is
`ValueError: Instruction "train" corresponds to no data!`.
Consequently, the intended `if not len(ds_eval): ds_eval = None` fallback is never
reached. The word `train` in that error comes from the dataset builder and does
not mean the caller selected the wrong cache split.

**Implementation hints:**

- Check `self.document_ids(split)` before calling `Dataset.from_generator`.
- For an empty split, return `Dataset.from_dict` with empty lists for `doc_id`,
  `input_ids`, and `attention_mask`, using the same explicit `Features` schema.
  This makes `to_dataset` consistent for both empty and populated splits.
- Alternatively, check for evaluation documents in the caller and skip dataset
  construction, but consider other callers of `to_dataset` too.
- Keep the separate error for an empty training split. If evaluation is explicitly
  requested in training arguments, report the missing eval data clearly.

**Suggested regression check:** Build a minimal train-only cache. Confirm that
`to_dataset("eval")` returns an empty dataset with the expected schema and that
training setup without evaluation proceeds. Also verify that an eval-only cache
produces the intended "no training documents" error.

## 3. Add architecture metrics before reporting integrations consume logs

**Location:** `distillkit/optimizers.py:308–320`,
`ArchitectureMetricsCallback.on_log`; callback registration in
`distillkit/main.py:433–446`.

**Trigger:** Enable the architecture metrics callback and a reporting integration
such as W&B or TensorBoard.

**Problem:** The installed Transformers Trainer registers standard reporting
callbacks before user-supplied callbacks. Those integrations consume the log
dictionary before `ArchitectureMetricsCallback` adds the gate/projection metrics.
Updating `state.log_history` preserves the metrics locally but does not deliver
them to the reporting integrations. This undermines the planned dashboard checks
for gates or projections that never leave their initial state.

**Implementation hints:**

- Prefer enriching the log dictionary before calling the inherited trainer
  `log` method, so all reporting callbacks receive the same complete payload.
  Preserve the existing cadence and avoid logging recursively.
- If retaining the callback implementation, explicitly place the metrics callback
  before reporting integrations after trainer construction. Merely passing it
  first in the user callbacks list does not put it before default integrations.
- Preserve the existing local-history behavior and avoid duplicate emission for
  repeated logs at the same global step.

**Suggested regression check:** Use a small fake reporting callback that copies
the received dictionary immediately, with the same ordering as Trainer. Verify
that its copy contains gate and projection metrics at the requested cadence,
alongside ordinary loss metrics. Verify that local trainer history receives them
as well. No external logging account is needed.

## 4. Separate SSM value conversion from optional head reordering

**Location:** `distillkit/convert_gguf_student.py:416–432`,
`convert_gguf_to_hf`; helper `_invert_ssm_a` at line 329.

**Trigger:** Convert a supported linear-attention model whose key-head count
equals its value-head count.

**Problem:** The converter applies `_invert_ssm_a` only inside the unequal-head
reordering branch. GGUF stores this tensor as `-exp(A_log)`, and recovering
`A_log` is necessary regardless of whether the heads need reordering. Equal-head
models therefore retain the wrong values while satisfying shape and key checks.
For example, original `A_log = [1, 2]` becomes approximately
`[-2.71828, -7.38906]` in the converted checkpoint instead of `[1, 2]`.

The current 4B artifact has unequal head counts (16 key heads, 32 value heads), so
this finding does not explain a failure specific to that artifact. It affects
other configurations accepted by the converter.

**Implementation hints:**

- Keep V-head reordering conditional on unequal head counts.
- For `ssm_a`, perform any required reordering first, then call `_invert_ssm_a`
  unconditionally for the linear-attention tensor.
- Ensure the unequal-head path applies the inverse exactly once after refactoring.
- Retain the helper's validation that the GGUF values are strictly negative.

**Suggested regression check:** Convert synthetic equal-head and unequal-head
GGUF fixtures with known `A_log` values encoded as `-exp(A_log)`. Check the final
exported `A_log` values, including the expected head ordering, rather than only
testing the helper in isolation.

## Scope and validation notes

- During the earlier review phase, the CPU suite passed 115 tests and a tiny
  integrated training step succeeded. Those results do not cover the four
  failure cases above. After the user requested code inspection only, no further
  tests were run.
- Inherited autoregressive generation does not currently refresh `ngram_raw`
  between decoding steps. That is a separate implementation limitation, not a
  blocker for the current teacher-forced training scope; it should be addressed
  before presenting the custom model as ready for normal generation.
- This review does not establish full-size parity, successful teacher capture on
  the selected corpus, or a validated Muon plus ZeRO-2 CPU-offload training path.
