# Handoff (2026-09-08)

What remains to be done on the sidecar-distillation fork, in priority order, with the
state each item is in and the trap that comes with it. `PROGRESS.md` is the full record
and the place to read *why*; this is the short list of *what next*. Paths are for this
machine (`D:/DeepThought/Projects/HybridModel`); the user has said they are not
sensitive.

## State at handoff

- Branch `sidecar-distill` at `github.com/zcomputerwiz/DistillKit`, tree clean, all
  commits pushed. **323 tests pass** (`.venv/Scripts/python.exe -m pytest tests -q`,
  about 47 s on the CUDA machine; two-GPU tests skip elsewhere).
- Stage-2 full-backbone distillation runs three ways, all verified end to end:
  layer split (`examples/qwen35_sidecar_stage2_sharded.yml`, 1263 s, eval_loss 0.5330),
  chained from stage 1 (`..._stage2_chained.yml`, 0.5262), and **tensor parallel**
  (`..._stage2_tp.yml`, **1193 s, eval_loss 0.5329**). The tensor-parallel arm's
  weights move identically to the layer split's -- same 86 of 426 tensors unchanged,
  same 5.96e-08 nudge to `A_log` -- and its export loads as an ordinary one-card
  checkpoint.
- Tensor parallelism shards 98.5% of the 4.271B parameters across both cards with no
  NCCL: 2.598 s per 4096-token microbatch against the layer split's 3.99 s, steady-state
  peaks 13.25 / 12.62 GiB against a 22.8 GiB cap. Code: `distillkit/tensor_parallel.py`
  (collectives), `tp_linear.py`, `tp_blocks.py` (MLP, attention), `tp_gated_delta*.py`
  (GatedDeltaNet by head), `tp_vocab.py` (tied embedding/head by vocabulary row),
  `tp_model.py` (`shard_model`), `tp_checkpoint.py` (stock-named export, layout marker).
- Sidecar vs control at 1M tokens: eval_loss 0.5807 vs 0.6255, the whole gap in the KL
  term; single-arm run-to-run spread 0.017, about 38% of the effect.

Always run with `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8`
(`expandable_segments` is unsupported on this platform). Runs and logs live in
`D:/DeepThought/Projects/HybridModel/runs/`; the teacher cache in `teacher-cache-1m`; the
converted student in `student-hf`.

## What remains

1. **The 5M-token pilot, at least two seeds per arm.** This is the research question
   everything above was built to answer, and it has not been run. The 1M result
   justifies it; the spread says one seed per arm cannot settle it. Steps: capture 5M
   tokens of teacher logits and anchor hidden states (`sample_transformers.py`; the 1M
   capture ran at ~550 tokens/s, so budget about 2.5 hours and ~5x the 1M cache's disk),
   then stage 1 sidecar and control from `examples/qwen35_sidecar_1m.yml` and
   `..._1m_control.yml` with the new cache path and two `seed` values each, then stage 2
   chained from the better arm. Stage 1 freezes the backbone and is not the bottleneck;
   if it is worth speeding up, `tensor_parallel: true` applies to it as well
   (`shard_model` preserves `requires_grad`), but then `chunked_head: true` is mandatory.

2. **Resume a tensor-parallel run from a real `checkpoint-*` directory.** Export is
   verified (keys, shapes, dtypes, one-card load, finite logits -- see PROGRESS,
   "Export verified against a stock load"). *Resume* is tested on a tiny CPU model
   only: `runs/sidecar-stage2-tp/checkpoint-72` plus its `distillkit_tp.json` layout
   marker is the first real one, and the marker refuses a mismatched shard, dtype or
   frozen-parameter layout rather than silently loading. Worth one run before trusting
   a long job to it. The *exported* model is what to evaluate or generate with: the
   sharded model is training-only and refuses a KV cache.

3. **Flash-Attention once the Windows build lands.** This PyTorch wheel has no flash
   SDPA kernel, so the 8 full-attention layers run the math kernel with grouped-query
   expansion (`gqa_dispatch.py`) -- per card per layer, `[1, 8, 4096, 4096]` fp32 scores
   during recompute, about 0.5 GiB plus its softmax. Two places to wire a `flash_attn`
   wheel in: the stock path through transformers' `attn_implementation`
   (`use_flash_attention` in the config, currently `false`), and
   `TensorParallelAttention` in `tp_blocks.py`, which calls
   `scaled_dot_product_attention` directly at line ~137 and would call
   `flash_attn_func` instead. The 24 GatedDeltaNet layers are unaffected (FLA kernels).
   Expect memory first, speed second: attention is 5.4% of parameters.

4. **Batching -- mostly solved; one decision left before the pilot.** Throughput is set
   by *tokens per microbatch*, not batch size (744 tok/s at 512 tokens, plateau near
   1700 from 2048 up), and this corpus is median 547 tokens, so at batch 1 most
   microbatches run the GPU at under half its rate. Batch 4 is ~1.5-1.6x.
   `train_sampling_strategy: group_by_length` costs 0.0176 of eval_loss for it;
   `sortish_batching: true` (`examples/qwen35_sidecar_stage2_sortish.yml`) costs 0.0084
   at the same speed. Order-seed variance is 0.0005 (baseline reshuffled, seed 42 vs 43:
   0.5329 vs 0.5324), so the residual is real, not noise. Either run the pilot at batch 1
   -- directly comparable with the 1M results, no argument needed -- or run every arm
   with identical sampling and rely on so reproducible an offset cancelling in the
   sidecar-minus-control difference. **Do not mix samplers across arms.** PROGRESS,
   "Batching: throughput is set by tokens per microbatch", has the full table and the two
   untested candidates for the residual 0.005.

5. **MTP head.** Conventions resolved from llama.cpp; implementation
   pending ("MTP: reference resolved, implementation pending" in PROGRESS). Under tensor
   parallelism it needs its own head treatment: either share
   the vocab-parallel shards or add a second `VocabParallelHead`.

6. **Sidecar parameter group at a higher learning rate in stage 2**, if the sidecar
   should keep adapting rather than freezing at stage 1's value (PROGRESS item 8).

7. **Loss generality of the vocabulary split.** Only the sparse KL composes its
   log-sum-exp from per-card pieces (`get_logprobs` in `lossfuncs/common.py`, via
   `VocabShardedLogits.sparse_logprobs`). A dense-signal KL or a cross-entropy over the
   student's own logits would go through `VocabParallelHead.forward`, which gathers the
   full row onto card 0 -- correct, at full-logits memory. If either ever needs the folded
   head, add a `VocabShardedLogits` branch next to the existing one; the distributed
   pieces needed (per-rank log-sum-exp, masked gather) are already there.

8. **Small cleanups.** `scratch/tp_real_probe.py` prints a stale layer-split footer;
   `examples/qwen35_sidecar_stage2_tp.yml` still carries layer-split commentary that no
   longer applies to it; the `RemoteEmbedding` commit (7f93c9d) is superseded and only
   worth knowing about because it is the measured "whole on card 1" row in PROGRESS.

## Traps, each of which cost at least one run

- **Duplicate YAML keys keep the last value.** `chunked_head: true` followed by a stale
  `chunked_head: false` trained with the head unfolded and OOM'd at step 4 "in backward"
  -- that was the 1.89 GiB logits gradient, not a bug. `grep -c` the knob; do not read
  the file.
- **`chunked_head` is only correct relative to a placement.** Off is 4.4% faster under
  the layer split; under tensor parallelism it is required.
- **Preflight peaks understate the trainer.** The first step has no optimizer state and
  reads about 4 GiB low; the trainer also adds clipping temporaries and the collator's
  rows. `scratch/tp_optimizer_probe.py` measures the second step with AdamW8bit built.
- **0-dim tensors on two CUDA devices do not mix.** Loss terms come off different cards
  and are reduced onto one in `total_distillation_loss`; anything new that combines
  scalars must do the same.
- **`torch.cuda.memory._record_memory_history` breaks checkpoint recompute** on this
  platform (`SystemError: ... returned NULL without setting an exception` inside a
  nested `Function.apply`). Measure peaks with `max_memory_allocated` instead.
- **Non-reentrant checkpointing does not serialize recomputation across device
  workers.** Every collective that forks in backward (`AllReduce`, `Reduce`, `Collect`)
  saves an empty sentinel and unpacks it first. A new collective must do the same or
  fail intermittently with "recomputed values have different values".
- **`pytest | tail` masks the exit code.** Run pytest bare, or check `PIPESTATUS`.
- **A test fixture whose size divides evenly by the batch size hides tail-batch bugs.**
  That is how a 1.4% -> 23.8% padding regression shipped in the first sortish sampler.
  Parametrize dataset sizes that do *not* divide.
- **Comparing runs across different data orders needs a reshuffle control**, not an
  execution-reproducibility one. Two different pipelines running the same order agreeing
  to four decimals says nothing about how much the order itself is worth (0.0005, as it
  turns out).
- **Never run llama.cpp model binaries on this machine** -- they touched the GPU and
  RAM despite `-ngl 0`. Reading GGUF files with Python is fine.
- The dataset's license restricts use to controlled, noncommercial research; the user
  has confirmed this is that.

## How to run

```
# stage 1, sidecar arm and control arm
python -m distillkit.main examples/qwen35_sidecar_1m.yml -v
python -m distillkit.main examples/qwen35_sidecar_1m_control.yml -v

# stage 2, tensor parallel (fastest)
python -m distillkit.main examples/qwen35_sidecar_stage2_tp.yml -v

# memory and timing probes for the tensor-parallel student
python scratch/tp_optimizer_probe.py 4096
python scratch/tp_real_probe.py 4096
```
