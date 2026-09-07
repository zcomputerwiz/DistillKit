# HybridModel / DistillKit fork — status & handoff (updated 2026-09-07)

This is the continuation/handoff record for the DistillKit n-gram-sidecar fork in
`D:\DeepThought\Projects\HybridModel\DistillKit`. It began as a log of Claude's
unfinished GGUF provider + verification work and has grown into the authoritative
"where things stand" document. Read **Where things stand** first; the sections below
are the chronological evidence trail.

## Where things stand (as of 2026-09-07)

**Built and verified (all CPU-testable parts done):**
- GGUF IQ4_NL table provider (`ngram_table.py`): hash ported from Flash-Next modeling
  code, row geometry + byte-block validation, Q4_0 rejection, no-codec raw top-k path.
  Reference check vs the official BF16 revision passes (gate 2, sampled rows).
- Custom student `Qwen35SidecarForCausalLM` (`models/qwen35_sidecar.py`): dormant IQ4_NL
  sidecar at layer 1 (zero-init `W_side_proj` + identity-init gated residual), stock
  checkpoint names preserved, hidden-state outputs intact, checkpoint round-trip clean.
- `SidecarDataCollator` (`sidecar_collator.py`): CPU hash + GGUF row gather feeding a
  per-batch `ngram_raw` tensor.
- Student GGUF→HF converter (`convert_gguf_student.py`) → `student-hf/` (426 tensors,
  7.83 GiB), parity-validated three ways (bit-parity, row-norm 1.0000, differential loss).
- Single-pass teacher capture + `OfflineHiddenStateSignalSource` with token-alignment
  validation; end-to-end round-trip covered by `tests/test_signal_alignment.py`
  (gate 4, synthetic form).
- Trainer/optimizer plumbing: optimizer groups, stage-1 backbone freeze + unfreeze
  callback, architecture/gate metrics logging to W&B/TensorBoard.
- All four P2 code-review findings and all five Codex working-tree findings fixed, each
  with a regression test.

**Test suite:** green — **154 passed** in the CUDA-enabled dev environment (≈20 s).
Under strict CPU-only forcing (`torch.cuda.is_available = False`) it is 127 passed + 1
bf16-trainer test that needs `use_cpu` (an environmental artifact of CPU forcing, not a
regression), and two CUDA-parametrized cases are skipped.

**Verification gates:**
- Gate 1 (imports/tests): **pass** on installed Transformers 5.16.1.
- Gate 2 (hash reconstruction vs reference): **pass** for sampled rows — mean cosine
  0.9971, min 0.9962; wrong-row control max |cos| 0.2319. Sampled agreement up to
  quantization, not an exhaustive 320M-row comparison.
- Gate 3 (forward parity, dormant sidecar): **pass** bit-identically on the converted
  `student-hf` weights (stock vs sidecar), **and now with the real 28.8 GB table wired
  end-to-end** — see "Real-table forward" below.
- Gate 4 (signal alignment): **pass** in synthetic form (`test_signal_alignment.py`);
  the real-corpus version needs the capture run (remaining work #1).
- Gate 5 (1M-token smoke): **pass** in stage-1 form on one RTX 3090 - see
  "Stage-1 GPU smoke" below. Not with ZeRO-2 offload (impossible here), which
  stage 1 does not need.
- Gate 6 (5M pilot with control arm): **open** - about 1 h of GPU time, ready to run.

**What remains:**
1. Real-corpus teacher capture (needs the 27B teacher resident).
2. 5M-token controlled pilot with the sidecar-disabled control arm (gate 6). ~1 h GPU.
3. ZeRO-2 is **not needed for stages 1 and 6** and is unavailable on this OS anyway
   (see "Environment blockers"). It only returns as a question for full-backbone
   stage-2 training.

## Environment blockers found 2026-09-07 (verified in the venv)

These contradict spec §0/§4 and change the plan, so they are recorded here rather
than discovered at run time:

- **`deepspeed` is not installed, and `torch.distributed.is_nccl_available()` is
  False** on this Windows torch 2.11.0+cu128 build (gloo only). DeepSpeed's Windows
  build is single-GPU because multi-GPU ZeRO relies on NCCL. Spec §0's "optimizer
  offload (DeepSpeed ZeRO-2) is the chosen memory strategy" therefore cannot run
  two-rank on Windows — it needs WSL2/Linux, or the run goes single-process.
- **ZeRO-2 and Muon are mutually exclusive as specified.** ZeRO-1/2 flattens each
  param group into one contiguous fp32 partition and hands the optimizer 1-D
  tensors; `torch.optim.Muon` raises on non-2D gradients. Separately, with
  `offload_optimizer.device=cpu`, accelerate replaces any optimizer with
  `DeepSpeedCPUAdam` unless `zero_force_ds_cpu_optimizer: false` is set. Muon is
  usable in the plain single-process/DDP path only.
- **`flash_attn` is not installed** (no Windows wheels) while
  `use_flash_attention` defaults to true. `load_student_model` now fails fast with
  an actionable message instead of dying inside `from_pretrained` after the dataset
  and cache are built — and the message notes that the same flag is what selects
  bfloat16, so turning it off requires setting `training_args.bf16` explicitly to
  keep the fp32 distillation projections working under autocast.
  Regression: `test_main_vocab.py::test_missing_flash_attn_fails_before_loading_with_actionable_message`.
- **`bitsandbytes` is not installed**, so spec §1's "load the 27B text decoder in
  int8" has no backend here. Resolve before the capture run.

## Cross-entropy memory, and not computing it at all (2026-09-07)

Follow-up to the launch-bound work. With the delta rule fixed, the remaining
inefficiency was the 248,320-wide head. Transformers' `ForCausalLMLoss` opens with
`logits = logits.float()`, a full fp32 copy of `[batch * seq, 248320]` kept alive for
backward -- 3.05 GB at batch 3 x 1024, measured at ~54% of all activation memory.

Two changes, both landed.

**1. Withhold labels when nothing reads the model's loss (free).** DistillKit computes
its own losses; the model's cross-entropy is only consumed by the `cross_entropy` loss
function, via `student_outputs.loss`. For any config without it -- a KL-only or
KL + hidden-state pilot -- the model was computing a full-vocabulary cross-entropy and
discarding it. `LossFunctionBase.requires_model_loss()` now declares the dependency and
the trainer forwards `labels` only when some loss needs them. `CrossEntropyLoss` raises
a clear error if it is ever called without them rather than returning `None`.

**2. Chunked cross-entropy when it *is* needed (a real trade).** `distillkit/chunked_ce.py`
computes the same loss in token chunks, each wrapped in `torch.utils.checkpoint` so its
fp32 upcast is freed immediately and recomputed in backward. Installed via
`model.loss_function`, which transformers looks up per call, so no loss class or
signature changes. Opt out with `chunked_cross_entropy: false`.

Measured at batch 3 x seq 1024, backbone frozen, real weights:

| variant | step | throughput | peak VRAM |
| --- | ---: | ---: | ---: |
| cross-entropy, stock loss | 2149 ms | 1430 tok/s | 18.76 GB |
| cross-entropy, chunked | 2312 ms | 1329 tok/s | **13.15 GB** |
| no cross-entropy (labels withheld) | 2113 ms | **1454 tok/s** | **11.70 GB** |

So chunking costs about 7% throughput for 30% less memory, and skipping the loss
entirely is strictly better on both axes. Which applies depends only on whether
`cross_entropy` is in `loss_functions`.

**What the memory buys.** Shapes that previously hit OOM now fit:

| shape | tokens/step | before | after |
| --- | ---: | --- | ---: |
| b3 x 1024 | 3,072 | 18.76 GB | 13.15 GB |
| b4 x 1024 | 4,096 | OOM | 14.73 GB |
| b6 x 1024 | 6,144 | OOM | 17.91 GB |
| b8 x 1024 | 8,192 | OOM | 21.09 GB |
| b1 x 4096 | 4,096 | OOM | 15.19 GB |
| b2 x 2048 | 4,096 | OOM | 14.73 GB |

Per-token throughput is now roughly flat across shapes (1232-1454 tok/s), which is the
signature of a compute-bound step -- the launch-bound behaviour is gone. b3 x 1024 is
the throughput sweet spot; larger shapes exist for when the distillation losses need
the batch, not because they are faster.

Note this headroom is not spare: the real run adds cached teacher signals, hidden-state
anchors and the top-k KL machinery on top of what the smoke measures.

**Equivalence.** `tests/test_chunked_ce.py` compares the chunked loss against
`ForCausalLMLoss` on value and on `dL/dlogits` (the checkpointed recompute being the
part most likely to be wrong), across chunk sizes, ignore-index fractions, explicit
`shift_labels`, `num_items_in_batch`, and bf16 inputs, plus an all-padding batch that
would divide by zero in the naive form.

Two bugs found while building it, both caught by tests rather than review:

* The first default was a fixed `chunk_tokens=4096`. Memory scales with
  `tokens x vocab`, so at a 1024-token batch that is exactly one chunk and saves
  nothing -- the memory test measured 2.84 GB either way. The budget is now in bytes
  (`DEFAULT_CHUNK_BYTES`, 128 MB of fp32 logits), with a regression test asserting a
  248k vocabulary is actually split.
* The first version of the install-gate test reimplemented the trainer's condition
  instead of calling it, so it could have passed while the trainer silently gave back
  5.6 GB. The gate is now `maybe_install_chunked_loss()` and the test calls it,
  including the DDP/DeepSpeed wrapper path.

## Launch-bound step, diagnosed and fixed (2026-09-07)

The first GPU run measured 303 tok/s at batch 1 x 1024 and, tellingly, step time
barely moved from batch 1 (3.15 s) to batch 2 (3.46 s). That is the signature of a
launch-bound step: thousands of tiny kernels with the GPU idle between them, not a
compute bottleneck. Bigger batches were not the fix.

**Cause.** 24 of Qwen3.5-4B's 32 layers are `linear_attention`. Their chunked delta
rule, `transformers...torch_chunk_gated_delta_rule`, is a pure-PyTorch fallback
containing two Python loops - 63 iterations for the intra-chunk triangular solve, plus
one per 64-token chunk. A CPU-side profile of one step showed **28,420 `aten::copy_`,
15,809 `aten::mul`, 7,937 `aten::bmm` and 73,496 `aten::as_strided` calls**. Isolated,
one call cost **47.7 ms**; across 24 layers that is ~1146 ms per forward, doubled again
by gradient-checkpoint recompute - essentially the whole step.

The function is decorated
`@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")`, so it uses
a fused Triton kernel when one is available. None of `fla`, `kernels`, `triton` or
`causal_conv1d` was installed, so every run took the slowest path.

**Fix.** `triton-windows` 3.8.0 compiles and runs here, and `flash-linear-attention`
0.5.2 installs additively (3 packages, no torch/transformers change). Declared as the
`fused` extra in `pyproject.toml`.

| | before | after | |
| --- | ---: | ---: | ---: |
| one delta-rule call (b1 x 1024) | 47.7 ms | 1.3 ms | **37x** |
| step, b1 x 1024 | 3147 ms | 780 ms | 4.0x |
| step, b3 x 1024 | 4349 ms | 2151 ms | 2.0x |
| throughput, b3 x 1024 | 706 tok/s | **1428 tok/s** | 2.0x |
| 5M-token pilot | 1.97 h | **0.97 h** | |

Losses are unchanged to four decimal places across the swap (13.8114 -> 13.5438 in
both), which is the correctness evidence for the fused path.

**One regression this introduced, and its fix.** transformers binds fla at *import*
time with no device check, and fla's kernels are Triton, hence CUDA-only. Once
installed, every CPU forward of any Qwen3.5 raised
`ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)` - 12 tests
failed, and the CPU-only verification scripts this box depends on would have broken
too. `distillkit/linear_attention_dispatch.py` restores a per-call device check,
recovering the original torch implementation from the wrapper's `__wrapped__` rather
than reimplementing it. Installed from the sidecar model, the capture path and the CLI.

Note the test suite hid this at first: `test_signal_alignment.py` passed only because
it imports `test_sidecar_model`, which installs the patch as a side effect. The
regression test therefore runs a *stock* model in a subprocess
(`test_sidecar_model.py::test_stock_qwen35_runs_on_cpu_after_importing_capture_path`)
so import order cannot mask it again.

## Stage-1 GPU smoke - gate 5 (2026-09-07)

`scratch/gpu_stage1_smoke.py`, one RTX 3090, real 28.8 GB IQ4_NL table resident,
real `student-hf` weights, backbone frozen.

Spec 5.5 assumes ZeRO-2 with optimizer offload. Stage 1 does not need it: with the
backbone frozen only **65.5M of 4.27B parameters (1.535%)** train, so grads and
optimizer state stay under a gigabyte and the step fits on one card unaided.

| Measurement | Value |
| --- | --- |
| Config | batch 3 x seq 1024, gradient checkpointing, bf16, sdpa attention |
| Peak VRAM | 18.76 GB allocated / 19.35 GB reserved, of 24.0 GB |
| Throughput | **1428 tok/s** (median step 2151 ms) |
| Table gather | 7.7 ms = **0.4% of a step** |
| Losses | finite, 13.5438-13.8114, monotonically decreasing |
| `W_side_proj` grad norm | 0.157-0.200 every step, never zero |
| Extrapolated | 1M tokens 0.19 h, 5M tokens 0.97 h |

**`training_overlap_verified` is now settled.** With the table resident the gather is
0.4% of a step even measured *serially*, so no overlap is required at all. Cold mmap is
the opposite: the same gather took **2826 ms**, 80% of the step (5,790 rows/s, matching
the known cold-fault rate). Residency is mandatory and is the only thing that matters
here - `--resident` costs 8-21 s at startup and 28.8 GB of RAM.

**Two spec corrections.**

*Batch 8 x 4096 does not fit.* Measured ceiling on a 24 GB card is ~3-4k tokens per
step (b3 x 1024 = 18.76 GB; b4 x 1024, b2 x 2048 and b1 x 4096 all OOM). The spec's
5.5 shape is 32,768 tokens, roughly 8x over budget. Use gradient accumulation for the
effective batch.

*The memory wall is the vocabulary head, not the backbone.* With 248,320 logits per
token, one logits tensor at b1 x 1024 is 0.47 GB bf16 and 0.95 GB upcast to fp32.
Isolated (`scratch/logit_memory.py`): activations are 1.60 GB without a loss and
**3.50 GB with cross-entropy** - so the head and its loss are ~54% of activation
memory, scaling linearly with tokens. A chunked or fused cross-entropy is the lever if
larger batches are wanted; the backbone is not the problem.

## Real-table forward — gate 3 closed (2026-09-07)

`scratch/real_table_forward.py` drives the actual 28.8 GB IQ4_NL table through
`SidecarDataCollator` into `Qwen35SidecarForCausalLM` loaded from `student-hf`.
CPU-only (`torch.cuda.is_available` stubbed in-process); the GPUs were at
21.6/24.6 GB with the user's own model resident and were not touched.

The point of the script is that dormancy alone cannot prove the table path is
wired: with a zero-init projection, a working feature tensor and a silently
dropped one produce identical logits. So it perturbs the projection and compares
against rows for *different* tokens.

| Check | Result |
| --- | --- |
| Real rows dequantize finite, non-degenerate | l2 mean 0.4252 (0.3753–0.6089), matches the table's per-element std ~0.0076 over 2560 dims |
| Features vary with tokens | 100% distinct feature vectors across positions |
| Dormant sidecar vs `sidecar_enabled=False` | **bit-identical** with real features flowing |
| Perturbed projection moves logits | max abs delta 0.61182 |
| These rows vs other-token rows | max abs delta 0.79142 — token-dependent, so the gather reaches the residual |
| Stage-1 freeze + backward | 7 trainable tensors (all sidecar), loss 6.2596, `W_side_proj` grad norm 1.9617 |

An earlier revision compared against *zeroed* rows; that is not a real control,
because zero bytes dequantize to zero features and `W @ 0` simply reproduces the
dormant logits (both deltas came out identically 0.61182). Rows for different
tokens are what distinguish "these specific rows arrived" from "some nonzero
tensor arrived" — a permuted or off-by-one gather passes the zero-row version.

Still not covered: that the lookup *overlaps* the training step under optimizer-offload
memory pressure (`training_overlap_verified` remains false). The isolated benchmark
shows ~70 ms at batch 8 × 4096, but overlap is a property of the real run.

**Hard constraints for this machine (do not violate):**
- **Do not load models on the GPUs** — the user's own model occupies most VRAM. Verify
  CPU-only via in-process `torch.cuda.is_available = lambda: False`
  (`CUDA_VISIBLE_DEVICES=-1` segfaults this Windows build).
- **Never run llama.cpp model binaries here** — they touched GPU/RAM despite `-ngl 0`.
- No stock Qwen 3.5 downloads; the student base is `empero-ai/Qwen3.8-4B-Distill-GGUF:BF16`
  (already converted to `student-hf/`).
- **RAM is the binding constraint** for the real sidecar run: even the quantized IQ4_NL
  table (28.8 GB) plus ZeRO-2 offloaded AdamW states (~32–48 GB) exceeds 32 GB RAM, so an
  8 GB / 16 GB VRAM + 32 GB RAM box cannot run the sidecar; at best a 16 GB VRAM box runs a
  no-sidecar (control-arm) smoke test. The dev box (128 GB RAM, 2× RTX 3090) is required.
- Apply the spec's 250 W power limits to both GPUs before any long job.

## Results

- Restored the missing Python 3.12.14 runtime referenced by the repository's
  `.venv`. The parent project's separate Python 3.13 environment is not the test
  environment. Verified direct execution with `.venv\Scripts\python.exe`.
- Environment: PyTorch 2.11.0+cu128, Transformers 5.16.1, TRL 0.25.1,
  Accelerate 1.11.0, gguf 0.19.0. These are the installed versions, superseding
  the older Transformers 5.8 target in the spec. Trainer and CLI imports pass.
- Test suite: **74 passed in 14.64 seconds**, including CPU/CUDA dequantization,
  real local GGUF checks, malformed-artifact rejection, and raw top-k signals
  preserving token IDs above 65535.
- Repaired the reference check: `HfFileSystem` with `block_size=0` is streaming
  and cannot seek. It now uses seekable uncached range reads, a pinned official
  revision, repeated-range caching, and an optional JSON report.
- Official reference revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540` in
  [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540).
  Checkpoint offsets, head vocabulary sizes, and multipliers match exactly.
- **80 GGUF rows compared with BF16 reference rows**, five token triples across
  all 16 heads: mean cosine **0.9971**, minimum **0.9962**; mean relative L2
  **0.076**, maximum **0.087**. Wrong-row controls: maximum absolute cosine
  **0.2319**. Criterion 2 passed for these sampled rows. This establishes sampled
  agreement up to quantization, not an exhaustive comparison of all 320M rows.
- Confirmed local table: IQ4_NL, 51,200,245,760 elements, 28,800,138,240 bytes.
  The provider now explicitly rejects Q4_0 even though it has the same byte
  size; validates geometry and byte blocks; rejects invalid row indices; and
  preserves the compressor's no-codec raw top-k path with clear errors if
  compression is attempted without configuration.
- Declared the optional `ngram` dependency group for gguf. Ignored only
  `/scratch/*.u8` so the existing 16 GB synthetic benchmark data cannot be
  accidentally added with the implementation.

## Actual table benchmark

Batch 8, sequence length 4096, 16 rows per token (524,288 row fetches).
The real 28.8 GB table was loaded into anonymous process memory with 116.1 GB
available RAM beforehand. Loading took 22.15 seconds. Process memory was released
when the benchmark exited.

| Measurement | Iteration 1 | Iteration 2 | Iteration 3 |
| --- | ---: | ---: | ---: |
| Hashing | 5.56 ms | 2.78 ms | 2.14 ms |
| Raw gather | 89.66 ms | 72.08 ms | 94.54 ms |
| Gather + CPU dequant | 576.52 ms | 507.10 ms | 494.21 ms |
| Gather + pin + H2D + GPU dequant | 443.47 ms | 70.01 ms | 70.30 ms |

The first GPU measurement includes initialization overhead. These measurements
support using the local IQ4_NL artifact. Acceptance criterion 3 remains provisional
until lookup overlaps successfully with the actual training step under optimizer
offload memory pressure; this isolated benchmark does not establish that overlap.
Windows worker processes must not each make a 28.8 GB resident copy.

## Parity status

Claude's existing tiny random Qwen3.5 probe passed on CUDA in BF16: bit-identical
logits and all nine hidden states; nonzero zero-initialized projection gradient
(norm 0.14859); forward/backward with gradient checkpointing succeeded.

This probe wraps a layer with a zero projection only. It does **not** integrate
the real table or gated residual, and does **not** close the required stock
Qwen3.5-4B versus final custom student parity gate. Only tokenizer/config files
for that 4B model are currently cached locally.

## Student GGUF → HF conversion (2026-09-06)

Converted the actual training-base GGUF to an HF checkpoint so the sidecar
student can be built and parity-validated on real weights, not random probes.

- New `distillkit/convert_gguf_student.py` (click CLI): reads the qwen35 GGUF
  with gguf 0.19.0 (data views are already HF `[out, in]` — no transposes),
  derives the full `Qwen3_5TextConfig` from `qwen35.*` KV metadata, and applies
  the exact inverses of llama.cpp's qwen35 conversion conventions: V-head
  grouped→tiled reorder (in_proj_qkv V rows, in_proj_z, in_proj_a/b, conv1d V
  channels, out_proj columns, A_log/dt_bias elements), `A_log = log(-stored)`,
  and `w + 1` → `w` for every RMSNorm weight except `linear_attn.norm`. MTP
  block (blk.32.*) excluded; F32 kept only for `A_log` and `linear_attn.norm`;
  eos_token_id 248044 (family constant — the GGUF's 248046 is llama.cpp-specific).
- Output: `D:\DeepThought\Projects\HybridModel\student-hf` — single
  model.safetensors, **426 tensors, 7.83 GiB** (exactly the stock
  language_model geometry: 2 top-level + 24 linear layers × 14 + 8 full layers
  × 11), plus config.json and tokenizer files.
- `scratch/verify_converted_student.py` — three checks, all **PASS**:
  1. Gate-3 bit parity on the converted weights: stock `Qwen3_5ForCausalLM` vs
     `Qwen35SidecarForCausalLM` give BIT-IDENTICAL logits (1, 127, 248320) bf16
     and all 33 hidden states; load report shows exactly the 7 sidecar keys
     missing, 0 unexpected.
  2. Row-norm correlation vs `student-stock` = **1.0000** on embed_tokens,
     in_proj_qkv, in_proj_a/b, q_proj, gate_proj (catches transposes and
     same-shape swaps that parity alone cannot).
  3. Differential LM loss on the same 127 real tokens: converted fine-tune
     **2.2109** vs stock base **2.2634** — sane ordering, no garbage.
- Coherence generation from `student-hf` (CPU-only, `scratch/hf_greedy_ref.py` +
  `scratch/hf_sample_ref.py`): pure greedy (temp 0) degenerated into a repetition
  loop ("the same way of" ×N) — a known temp-0 artifact on a small fine-tune, not a
  weight defect. Under sampling (temp 0.7, top-p 0.9, rep-penalty 1.15) the same
  prompt produced coherent prose: it continued the lighthouse-keeper narrative into
  a well-formed math word problem and solved it correctly (183 − 1 = 182). Confirms
  the converted head is semantically sound and usable for generation.
- Test suite now **115 passed in 29.74 s** (adds 16 converter unit tests:
  config derivation, name mapping, key-set geometry, bf16/f32 decode vs torch
  native, V-head reorder round-trip against an independent forward copy,
  A_log inverse + guards).
- The planned llama.cpp cross-check (greedy token diff) was **abandoned**:
  `llama-cli.exe` from the CUDA build initialized a CUDA context and consumed
  ~60 GB RAM plus 2.5 GB VRAM despite `-ngl 0` / `LLAMA_CUDA=0`; the user ended
  it. No llama.cpp model binary will be run on this machine again — the HF-side
  verification above is sufficient evidence of conversion correctness.

## Code-review fixes (2026-09-06)

Addressed all four P2 findings in `CODE_REVIEW_2026-09-06.md`, each with a
regression test. The two that blocked affected training configurations from
starting (1 and 2) are fixed first, per the review's ordering.

1. **Cached vocabulary preserved independently of sidecar config**
   (`main.py::load_student_model`). A teacher cache stores top-k signals
   normalized over the teacher's full padded head (e.g. 248,320 for a 248,077
   tokenizer). The loader now takes `signal_vocab_size` and, when it exceeds the
   tokenizer size and the student head already covers it, preserves the padded
   head instead of resizing down to the tokenizer (which would fabricate rows
   for IDs the cache already normalizes over, changing the captured
   distribution). An undersized head is rejected with a clear error. The plain
   resize-to-tokenizer path is unchanged when there is no cached-signal
   requirement. Regression: `tests/test_main_vocab.py` (4 tests — padded head
   preserved, undersized rejected, disabled-sidecar control preserves, plain
   resize unchanged).
2. **Empty cache splits handled before `Dataset.from_generator`**
   (`offline_cache.py::to_dataset`). A split with no documents now returns an
   explicitly-typed empty dataset (correct `doc_id`/`input_ids`/`attention_mask`
   schema) instead of raising from `from_generator`, so callers can test split
   emptiness with `len()`. Regression: `tests/test_offline_cache.py` (2 tests —
   train-only cache yields empty eval with full schema; eval-only cache yields
   empty train, the exact condition `do_distill`'s "no training documents" fires on).
3. **Architecture metrics reach reporting integrations before callback copy**
   (`optimizers.py::ArchitectureMetricsCallback`, `trainer.py::HybridDistillationTrainer`).
   `Trainer.log` registers W&B/TensorBoard ahead of user callbacks and they
   consume the dict inside `super().log`, so the trainer now enriches `logs` with
   gate/projection metrics before calling it (the callback gained an idempotent
   `enrich_logs`; its `on_log` updates local history in place). Regression:
   `tests/test_optimizers.py::test_hybrid_trainer_delivers_metrics_to_integrations_before_callback_copy`
   (a fake integration that copies the dict immediately still receives gate and
   projection metrics; cadence respected — a non-multiple step adds nothing new).
4. **SSM value conversion separated from optional head reordering**
   (`convert_gguf_student.py::convert_gguf_to_hf`). The `A_log = log(-stored)`
   inverse now applies to `ssm_a` regardless of whether the V-head grouped→tiled
   reorder runs, so the two independent conventions can't be skipped or applied
   out of order. Regression: `tests/test_convert_gguf.py::test_synthetic_gguf_round_trip_recovers_a_log`
   (parametrized equal/unequal head counts; writes a minimal qwen35 GGUF with
   known distinct A_log values and asserts the exported f32 recovers them).

- Test suite now **124 passed in 18.78 s** (was 115 before these fixes; +9 new
  regression tests across the four files above).

## Capture → signal alignment gate (2026-09-07)

Closed the one open item from "remaining work" #2 that had no test: nothing drove
the single-pass capture and the offline signal source *together*. The rest of
milestone 2 was already built (`sample_transformers.capture_teacher` writes both
streams + manifest + eval holdout; `OfflineHiddenStateSignalSource.get_signal`
reads them back with token-alignment validation), but there was no end-to-end proof
that a freshly captured cache round-trips correctly.

- New `tests/test_signal_alignment.py` (4 tests, CPU-only tiny teacher):
  1. **Manifest + split policy** — explicit per-doc split wins over the
     `eval_every` fallback; a doc longer than `sequence_length` is truncated to
     exactly that window; manifest records vocab/hidden/anchors/top_k/seq_len.
  2. **Round-trip alignment** (parametrized left/right padding) — for each doc the
     signal source's top-k IDs match an *independent* recompute from the same
     teacher exactly, values match to fp16 precision, and each anchor state matches
     the fp8→bf16 cast of that layer's hidden state bit-for-bit; padded positions
     keep the zero/-1e4 sentinels; and the source's rows equal the raw cache read.
  3. **Misalignment rejected** — a batch whose tokens differ from what capture
     stored raises "token mismatch" instead of feeding shifted signals (gate 4's
     failure mode).
- Test suite now **128 passed in 18.74 s** (+4 over the post-review 124).

This is gate 4 ("signal alignment") in synthetic form on a tiny teacher; the real
corpus version still needs an actual capture run (blocked on GPU/teacher, see
remaining work #4).

## Codex working-tree review + fixes (2026-09-07)

Ran a Codex (`codex-cli 0.153.4`) review over the uncommitted DistillKit working tree at
the milestone boundary. It returned **1 P1 + 4 P2**; all five were verified against the
code and fixed, each with a regression test where the path is CPU-testable.

- **[P1] Ordinary runs must not force a custom collator** (`main.py`). TRL 0.25.1 sets
  `padding_free=True` for BFD packing and raises if a custom collator is passed, so the
  unconditional `DataCollatorForLanguageModeling` broke `packing=True` configs such as
  `examples/afm_test.yml`. Ordinary (non-cache, non-prepacked) runs now pass `collator=None`
  and let SFTTrainer build its own; the sidecar wrap builds a concrete base collator only
  when it needs one (packing is forced off for sidecar runs).
- **[P2] No-growth vocab check was too broad** (`main.py::load_student_model`). The
  "smaller than the cached signal vocabulary" guard fired for online teachers, where
  `signal_vocab_size == tokenizer_vocab_size`, blocking legitimate embedding growth. It now
  only applies when the cached head is genuinely larger than the tokenizer. Regression:
  `test_main_vocab.py::test_online_teacher_signal_reaches_resize_not_cache_error`.
- **[P2] Control arm left a trainable unused parameter** (`qwen35_sidecar.py`, `main.py`).
  With `sidecar.enabled=False` the trainer forwards `sidecar_enabled=False`, bypassing
  `W_side_proj` while it stayed trainable — a DDP `find_unused_parameters=False` failure.
  New `Qwen35SidecarForCausalLM.disable_sidecar_projection()` freezes just that weight (the
  gated residual stays trainable); `do_distill` calls it for the control arm before
  optimizer/distributed setup. Regression:
  `test_sidecar_model.py::test_disable_sidecar_projection_freezes_only_bypassed_weight`.
- **[P2] Greedy reference dropped context each step** (`scratch/hf_greedy_ref.py`). The
  manual loop captured `past_key_values` but never fed it back, so every prediction saw only
  the latest token — the cause of the earlier degenerate repetition loop. Now uses
  `model.generate(do_sample=False)`, which threads cache/position correctly.
- **[P2] Converter test imported an optional dep unconditionally** (`test_convert_gguf.py`).
  `gguf` ships only in the optional `ngram` extra, so a base/dev install failed at collection.
  Now uses `pytest.importorskip("gguf")`, matching the sibling GGUF tests.

- Test suite now **130 passed in 18.91 s** (+2 regression tests over the 128 after the
  alignment gate). The P1 collator fix and the DDP control-arm freeze are not covered by a
  dedicated CPU test (they need a full TRL/`packing=True` build or multi-GPU DDP); both were
  verified against TRL's source and the trainer's `sidecar_enabled` flow respectively.

## Remaining work, in order

Completed items are struck through for continuity with the original plan; what
actually remains is #1–#4 below.

1. ~~Build the custom student and collator~~ — **done.** `Qwen35SidecarForCausalLM` +
   `SidecarDataCollator` are built, tested, and parity-verified (dormant sidecar).
2. ~~Implement single-pass teacher logit/anchor capture, cache manifest, eval holdout,
   and `OfflineHiddenStateSignalSource` with doc/token alignment tests~~ — **CPU parts done.**
   Covered end-to-end by `tests/test_signal_alignment.py`. What remains is the real-corpus
   capture run (#2 below). Keep the official BF16 reference distinct from the fp8
   hidden-state cache: they are different artifacts.
3. ~~Wire optimizer groups, stage freezing/unfreezing, gate logging~~ — **done**
   (`optimizers.py`: Muon/AdamW groups, `UnfreezeBackboneCallback`,
   `ArchitectureMetricsCallback` → W&B/TensorBoard). Packing stays disabled.

What remains, in order:

1. ~~**Integrate the real table end-to-end.**~~ — **done** on CPU, 2026-09-07; see
   "Real-table forward" above. Gate 3 is closed on the final custom student. What is
   still open from that item is only `training_overlap_verified`: the isolated benchmark
   shows gather+pin+H2D+GPU-dequant ≈ 70 ms at batch 8 × 4096, but whether the lookup
   overlaps the training step under optimizer-offload memory pressure is a property of
   the real run and is folded into item 3 below.
2. **Real-corpus teacher capture.** Run `capture_teacher` with the 27B text decoder
   resident to produce the actual fp8 hidden-state + top-k logit cache (needs GPU/teacher).
3. **ZeRO-2 offload verification on Windows.** Confirm optimizer offload coexists with the
   table in page cache (spec §4) — the offloaded states and the 28.8 GB table compete for
   the same RAM.
4. **1M-token smoke run, then 5M-token controlled pilot** (backbone frozen, control arm).
   Gates 5–6. Apply the spec's 250 W power limits to both GPUs first; observed limits were
   still 333 W during the short verification work.

No teacher/student weight download, training run, or commit was made. Existing
uncommitted work was retained. Gate status is summarized in **Where things stand** above:
gates 1–2 pass, gate 3 passes bit-identically on the converted (dormant) weights with the
final custom-student check remaining as #1 here, gate 4 passes in synthetic form, gates
5–6 open.

## Reproduction (PowerShell from the DistillKit directory)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -u scratch/ngram_reference_check.py --report reference-check.json
.\.venv\Scripts\python.exe -u scratch/parity_probe.py
# Use the local second GGUF shard as the --gguf argument:
.\.venv\Scripts\python.exe -u scratch/ngram_table_benchmark.py --gguf <shard-path> --resident --report benchmark.json
# Student GGUF -> HF conversion (output dir must be empty/new) + verification:
.\.venv\Scripts\python.exe -m distillkit.convert_gguf_student --gguf ..\StudentSourceModel\Qwen3.8-4B-BF16.gguf --output ..\student-hf --tokenizer-source ..\student-stock
.\.venv\Scripts\python.exe -u scratch/verify_converted_student.py
```

Detailed results are in `ngram-reference-check.json` and
`ngram-table-benchmark.json` alongside this report. Copies are also in the
repository's `verification/` directory so the next session can find the evidence.
