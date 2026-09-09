# HybridModel / DistillKit fork — status & handoff (updated 2026-09-07)

This is the continuation/handoff record for the DistillKit n-gram-sidecar fork in
`D:\DeepThought\Projects\HybridModel\DistillKit`. It began as a log of Claude's
unfinished GGUF provider + verification work and has grown into the authoritative
"where things stand" document. Read **Where things stand** first; the sections below
are the chronological evidence trail.

## Where things stand (as of 2026-09-08)

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
- Memory work that made stage 2 comfortable: anchor taps instead of all 33 hidden
  states, the folded output head, grouped-query KV expansion (the single largest win --
  4328 -> 249 MiB per attention call), and a split rebalanced from measurement.
- **Tensor parallelism** (`tp_*.py`): 83.6% of parameters sharded across both cards,
  1.41x faster than the layer split and smaller on both. No NCCL and none needed.

**Test suite:** green — **306 passed** in the CUDA-enabled dev environment (≈47 s).
One case (`test_sharded_step_matches_single_device_step`) skips unless two CUDA devices
are visible.
Under strict CPU-only forcing (`torch.cuda.is_available = False`) it is 127 passed + 1
bf16-trainer test that needs `use_cpu` (an environmental artifact of CPU forcing, not a
regression), and two CUDA-parametrized cases are skipped.

**How to run the thing (current configuration):**

```
# Stage 1, frozen backbone, sidecar vs control -- both cards, no CUDA_VISIBLE_DEVICES pin
python -m distillkit.main examples/qwen35_sidecar_1m.yml -v
python -m distillkit.main examples/qwen35_sidecar_1m_control.yml -v

# Stage 2, full backbone, layer split at boundary 13
python -m distillkit.main examples/qwen35_sidecar_stage2_sharded.yml -v

# Stage 2 continuing from stage 1 rather than the stock student
python -m distillkit.main examples/qwen35_sidecar_stage2_chained.yml -v

# Stage 2, tensor parallel across both cards -- fastest; see "Hybrid tensor parallelism"
python -m distillkit.main examples/qwen35_sidecar_stage2_tp.yml -v
```

Set `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8`. `expandable_segments`
is unsupported on this platform, so that is the only defence against the fragmentation
that killed several runs.

Knobs whose right value depends on what is binding, not on taste:
- `chunked_head`: off at batch 1 (costs 4.4%), **required** at batch 4 or for a
  data-parallel rank at full sequence length, where the logits it removes are 7.6 GiB
  plus the same again in gradient.
- `per_device_train_batch_size`: **4 with `train_sampling_strategy: group_by_length`**
  under tensor parallelism. Throughput is set by tokens per microbatch, not batch size,
  and this corpus is median 547 tokens -- at batch 1 most microbatches run at under half
  the GPU's rate. See "Batching: throughput is set by tokens per microbatch".
- `tensor_parallel: true` needs `chunked_head: true`, `gradient_checkpointing_kwargs:
  {use_reentrant: false}` and no `model_kwargs.device_map`;
  `examples/qwen35_sidecar_stage2_tp.yml` has all three. Under tensor parallelism the
  folded head is *required*, not the 4.4% option it is under the layer split, and a
  duplicated YAML key silently keeps the last value -- `grep -c chunked_head` the file.

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
- Gate 5 (1M-token smoke): **pass**. Real 1M-token capture + a full stage-1 training
  run in 998 s; losses finite and falling, eval 0.6569 -> 0.5853, and the sidecar moved
  off its zero init (see "Teacher capture and the 1M stage-1 run").
- Gate 6 (paired arms): **pass at 1M**. Both arms completed sharded; the sidecar arm
  reaches eval_loss 0.5807 against the control's 0.6255, with the whole gap in the KL
  term. See "Gate 6, 1M paired arms". The 5M version is a scale-up of the same two
  configs and should carry at least two seeds per arm.

**What remains:**
1. ~~Real-corpus teacher capture~~ - **done**: 1,014,574 tokens cached.
2. ~~Compare the 1M sidecar and control arms~~ - **done**: -0.0448 eval_loss for the
   sidecar, concentrated entirely in the KL term.
3. ~~5M-token pilot~~ - **done, and the effect largely washed out**: -0.0027 mean
   against a 0.0032 seed spread, where 1M measured -0.0448. Sidecar ahead in 14 of 14
   paired evaluations, so the sign is consistent and the magnitude is not established.
   See "The 5M pilot".
4b. ~~Batching~~ - **measured, and mostly fixed**: batch 4 is ~1.5-1.6x, and
   `sortish_batching` cuts its loss cost from 0.0176 to 0.0084 at the same speed.
   Order-seed variance is 0.0005, so the residual is real. Decide per-arm before the
   pilot, not during it. See "Batching: throughput is set by tokens per microbatch".
4a. ~~The gate~~ - **both candidates done and both were right.** Flash-Next's
   integration is ported (`ple_sidecar.py`) and its computed gate moves where the
   learned one never did; and stage 2's single learning rate was keeping the sidecar
   frozen, which `optimizer.sidecar_lr` fixes for -0.07 of eval_loss so far. See "The
   PLE port". What remains is the matched control at the chosen rate.
4. ~~Stage 2 sharding integration~~ - **done, and run**: 4.298B trainable parameters
   across both cards in 1269 s, eval_loss 0.5347, checkpoint verified. See "Stage 2
   runs". The thin margins recorded there (22.12 / 21.45 GiB against 22.80 allowed) are
   historical: the grouped-query fix and the rebalance to boundary 13 brought the layer
   split to 13.68 / 12.92 GiB, and tensor parallelism to 9.03 / 8.53.
5. ~~Port `_AnchorTap` into the trainer~~ - **done**: `distillkit/anchor_tap.py`.
6. ~~Run the staged curriculum~~ - **done**: chaining wins, 0.5262 against 0.5347, but
   inside the run-to-run spread. See "The staged curriculum beats training jointly".
7. ~~Threaded microbatch overlap~~ - **done and measured: 8.7% slower**, left opt-in and
   off. See "Threaded microbatch overlap". A real gain needs the explicit stage
   schedule, whose ceiling is bounded by the reachable split rather than the balanced
   one.
8. ~~Give the sidecar its own parameter group at a higher learning rate in stage 2~~ -
   **done, and it was not optional**: at the backbone's 1e-5 the sidecar does not move
   at all. `optimizer.sidecar_lr`; see "Stage 2 had never trained a sidecar at all".
9. MTP head - conventions now resolved from llama.cpp; implementation pending.
10. ~~Hybrid tensor parallelism~~ - **done and through a full epoch**: 98.5% of
    parameters sharded, 1193 s against the layer split's 1263 s, eval_loss 0.5329
    against 0.5330, export verified as a stock checkpoint. 1.54x per microbatch but
    1.06x per epoch -- see "Hybrid tensor parallelism" for why, and what it buys.

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

## Teacher capture: corpus chosen and pipeline proven (2026-09-07)

### Corpus

`r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation` -- 57,937 traces distilled from
Qwen3.8-Max-Preview (48,283), GLM-5.2 (5,307) and Kimi Code K3 (4,347). Public,
ungated.

Sizing from the dataset's own `token_stats` config (per-row Qwen3 token counts):

| split | rows | tokens | mean |
| --- | ---: | ---: | ---: |
| train | 98,455 | 92,026,135 | 934 |
| validation | 2,872 | 2,966,098 | 1,032 |
| test | 2,860 | 2,982,956 | 1,042 |
| **total** | **104,187** | **97,975,189** | **940** |

97% of rows fit in 4096 tokens. Domain mix: code 33%, agent/tool 19%, math 12%,
grounded long-context 11%, reasoning 9%, instruction 8%.

That covers the 1M smoke and 5M pilot with enormous margin, and even the spec's
100M-token ambition almost exactly -- though 98M tokens is ~1 TB of cache at two
anchors, so the practical ceiling here is disk, not data.

### Does the teacher find it in-distribution?

Measured from the capture itself rather than argued: the cache stores the teacher's
top-64 logprobs, so its surprise on this corpus falls out directly. Over 28,141
captured tokens:

* true next token inside the teacher's **top-64: 90.62%**
* mean logprob of the true next token when present: **-2.93** (perplexity 18.66)
* **median top-1 probability: 0.979**

A wrong chat template or alien formatting would show up as collapsed top-64 coverage
and a diffuse top-1; neither is present. The corpus is in-distribution for this teacher.

Note this measures the *rendering*, not the data's provenance. The traces were written
by other frontier models, which is fine here: we teacher-force our own 27B over that
text and record *its* distribution, so the other models' identities never enter the
targets. It does mean the student learns the 27B's conditional distribution over text
the 27B did not itself write, which is the normal and intended setup.

### Rendering

`distillkit/prepare_corpus.py` renders the `sft_balanced` messages through **the
teacher's own chat template**. The dataset also ships `prompt_completion_text`
(generic `<|system|>` markers) and `glm47_native` (pre-tokenized with GLM's
tokenizer); both would put the 27B off-distribution and were rejected for that reason.
`reasoning_content` is empty in this release -- the `<think>` blocks already live
inside assistant `content` -- so nothing needs merging, though the renderer folds it
back in if a future release splits it out.

One trap found the hard way: shard filenames sort `test-*` before `train-*`, so the
first `--limit` run drew all 40 documents from the held-out split. `--split train` now
also reorders shards, so a truncated run is representative rather than silently
entirely eval.

### Measured capture

40 documents / 28,141 tokens, teacher int8 across both 3090s, anchors 8 and 64:

| | |
| --- | --- |
| wall clock | 96 s including a ~45 s model load |
| throughput | **~550 tokens/s** |
| output | 286 MB (10.4 KB/token) |
| 1M tokens | ~0.5 h, 10.6 GB |
| 5M tokens | ~2.5 h, 53 GB |

### Two capture-blocking fixes

**Anchor capture now uses hooks, not `output_hidden_states=True`.** The teacher at
int8 leaves only ~2 GiB free per card, and requesting all 65 hidden states costs
2.7 GB at 4096 tokens to extract two anchors worth 84 MB. Capture OOM'd on the first
real document. `_AnchorTap` registers a forward-pre-hook on each anchor layer instead.

Index semantics are preserved exactly, and the subtlety is load-bearing:
`hidden_states[i]` is the *input* to layer `i`, but the final entry
`hidden_states[num_layers]` is the last layer's output **after `model.norm`**. Hooking
the last decoder layer there returns the pre-norm state -- a different tensor, and a
silently wrong target for the deepest anchor. `scratch/hook_index_check.py` asserts all
`num_layers + 1` indices match the reference tuple; the first version failed only at
the last index, which is exactly how that bug would have shipped.

**fp8 anchor range constrains which layers are usable** (see the previous section):
layers 55-59 overflow, and 16-48 have only 2.0-2.8x headroom. Anchors 8 and 64
(5.1x and 6.7x) were used for the smoke and are the safe default.

### Licensing, flagged not decided

The dataset is `license: other` and its LICENSE restricts use to "controlled,
noncommercial research", noting Alibaba Cloud Model Studio terms that bar using Model
Studio outputs to train products competing with Alibaba or its affiliates. The mixture
also carries CC BY-NC-SA / CC BY-NC / CC BY-SA / Apache / MIT / ODC-BY material with
per-record provenance in the `canonical` view.

Separately, the upstream README states the validation and test splits are
benchmark-derived and contaminated. Usable as a held-out loss signal; **not** usable as
a capability claim.

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

## Teacher capture and the 1M stage-1 run (2026-09-07)

### Capture

1,213 documents / **1,014,574 tokens**, 11 GB, anchors 8 and 64, ~220 tok/s on the
27B teacher in int8 across both cards. Split 1,152 train / 61 eval, with the eval
holdout carved out in the same pass.

Corpus is `r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation`, rendered through the
*teacher's* chat template by `distillkit/prepare_corpus.py`.

### Sidecar arm result

`examples/qwen35_sidecar_1m.yml`, backbone frozen, one RTX 3090:

| | |
| --- | --- |
| runtime | **998 s (16.6 min)**, 72 optimizer steps |
| train loss | 1.049 (final step 0.5677) |
| eval loss | 0.6569 -> **0.5853** |

**The architecture is not dead weight.** Spec section 3b asks whether the gates and
projection ever move off identity; if they do not, the gated residual should be dropped
rather than carried into the long run. Measured over the run:

| metric | start | end |
| --- | ---: | ---: |
| `W_side_proj` weight norm | 0.0000 | **1.7834** |
| gated-residual branch 1/2/3 norms | 0.0000 | **1.788** |
| `W_x` bias absmax | 0.0000 | 0.0036 |
| gate means | 0.5004 | 0.5010 |
| gate saturation | 0.0508 | 0.0547 |

The zero-init projection and all three zero-init branches moved off zero, and the gates
stayed unsaturated at ~0.501 (saturation 0.055, where 0.5 would mean fully saturated).
That is exactly the intended behaviour: dormant at init, trainable thereafter.

The control arm (identical but `sidecar.enabled: false`) is what makes the loss numbers
interpretable, and is running now.

### Five failures worth recording

Each cost a model load, and all but the first are real bugs rather than config typos.

1. **`group_by_length` is not an SFTConfig field** in trl 0.25.1 (only
   `length_column_name` survives). Removed; both arms eat the same padding waste so the
   comparison is unaffected.
2. **Stale `Accelerator` at two sites.** `trl.SFTConfig(...)` can call
   `AcceleratorState._reset_state()`, after which the `Accelerator` built earlier in
   `do_distill` raises on any state access. Fixed at both sites by reading
   `training_arguments.world_size`, plus a source-level guard test so a third site
   cannot appear.
3. **DataParallel returns one loss per replica.** With both cards visible and no
   distributed launcher, HF wraps the model in `nn.DataParallel` and
   `student_outputs.loss` arrives shaped `[n_gpu]`. Everything downstream wants a
   scalar. Invisible at batch 1, because DataParallel cannot split a single example.
   Now mean-reduced in `compute_loss`.
4. **DataParallel then OOM'd anyway**, gathering both replicas' `[B, T, 248320]` logits
   onto GPU 0 at 22.96 GB. A 248k-wide head makes the gather the bottleneck. Runs now
   use `CUDA_VISIBLE_DEVICES=0`; real multi-GPU needs torchrun + DDP, which NCCL blocks
   (see below).
5. **Hidden-state projections depended on ambient autocast.** `HiddenStateMapping`
   builds them in fp32 after the student loads in bf16, so the matmul only worked if
   autocast happened to be active at that call site. It was not, and the failure was a
   bare "expected mat1 and mat2 to have the same dtype" naming neither side. Now aligned
   explicitly, with a parametrized test over three dtype combinations.

### The VRAM spillover trap (the important one)

At batch 2 the run *appeared* to train: 100% GPU utilisation, steps advancing, losses
falling. It was 7x too slow (200 tok/s against 1428 in the stage-1 smoke) and the
profile was wrong in a specific way:

| signal | value | what it means |
| --- | --- | --- |
| `utilization.gpu` | 100% | a kernel is resident -- not that it is doing work |
| `utilization.memory` | **0-2%** | the device memory controller is idle |
| power | **133 W of 333 W**, flat | stalled, not computing |
| process CPU | 0.45 cores | not CPU-bound either |

Nothing was saturated. The cause, found by checking the Windows performance counters
rather than nvidia-smi:

    GPU Process Memory / Dedicated Usage : 24,294 MB   (card full)
    GPU Process Memory / Shared Usage    : 19,682 MB   <-- spilled to host RAM

**Windows WDDM does not OOM when VRAM is exhausted. It silently spills to shared system
memory and services those pages over PCIe.** Nearly 20 GB of the working set was living
in host RAM. That also explains the saturated PCIe bus observed at the time, which had
been attributed to teacher-signal streaming -- the sidecar is only ~6% of host-to-device
traffic (2.6 MB per microbatch against the teacher cache's 36.9 MB).

On Linux this is a clean OOM. On Windows it is a silent 7x slowdown that looks like a
healthy run, which makes it exactly the kind of thing to guard against rather than
notice by luck.

Two changes:

* **`max_vram_fraction`** (new config field, set to 0.92 in both arms) caps PyTorch's
  allocator so an over-budget run raises `torch.OutOfMemoryError` instead of degrading
  silently. It worked immediately: the next attempt failed loudly with "22.08 GiB
  allowed" rather than spilling.
* **batch 1 x accumulation 16** (same effective batch) to fit.

The loud OOM then pointed at a **3.79 GiB allocation in backward** -- the fp32 gradient
of the full `[1, 4096, 248320]` logits. Chunking the forward does not shrink it: every
chunk backpropagates into one full-size gradient buffer. Dropping `cross_entropy` from
the loss list removed it entirely, because that is the only loss reading
`student_outputs.loss`, so `requires_model_loss` goes false and the model skips its
248k-wide cross-entropy in both directions. KL is the teacher-alignment term regardless;
ground-truth CE can return in stage 2.

Final loss mix: `kl` 0.7 (`sparse_chunk_length: 256`) + `hs_cosine` 0.3.

## Multi-GPU and offload: what is actually available here

Measured, not assumed: `torch.distributed.is_nccl_available()` is **False** on this
Windows torch 2.11.0+cu128 build; gloo only.

**Update 2026-09-07 (later): NCCL itself is not the blocker; the PyTorch wheel is.**
NCCL 2.31.2 builds on Windows and runs here. A native all-reduce probe selected
`P2P/direct pointer` in both directions and all nine reduction checks passed, reaching
**37.5 GB/s at 64 MiB** (32.6-33.0 GB/s at 8 MiB, 1.6-1.7 GB/s at 64 KiB, fp32/fp16/bf16
alike). One real Windows bug had to be fixed to get there: `src/debug.cc` called
`setvbuf(file, NULL, _IOLBF, 0)`, and the Microsoft CRT treats `_IOLBF` as full buffering
and rejects a zero size, so any run with `NCCL_DEBUG_FILE` set aborted in `ucrtbase.dll`
with `0xc0000409`. `_IONBF` accepts the zero size and gives the intended immediate
logging. Full details, artifacts and reproduction:
`NVLINK_PARALLELISM_RESEARCH_2026-09-07.md`.

What that does *not* change: `torch 2.11.0+cu128` gates `USE_NCCL` on UNIX at build time
and imports `ProcessGroupNCCL` from compiled extension code. Dropping a DLL beside the
wheel cannot add the backend. So:

* **DDP, FSDP and DeepSpeed multi-GPU ZeRO stay blocked** until PyTorch itself is
  rebuilt -- not for want of a working NCCL, but for want of a backend that can call it.
* **Single-GPU ZeRO-2 with CPU offload is *not* blocked** -- `world_size=1` runs no
  collectives, so NCCL is irrelevant. An earlier note in this document said ZeRO-2 needs
  WSL2/Linux; that over-generalised from the multi-GPU case and is corrected here.
  The blocker for that path is toolchain, not NCCL: pre-built DeepSpeed wheels target
  CUDA 12.1/12.4, torch here is cu128, and the installed standalone toolkit is v13.3
  with `CUDA_HOME` unset. It would need a source build against a CUDA 12.8 toolkit.
* ZeRO-1/2 also flattens each param group into a 1-D partition, and `torch.optim.Muon`
  raises on non-2D gradients -- so ZeRO costs Muon either way.

**NVLink is present and healthy**: 4 links x 14.062 GB/s = **56 GB/s per direction**,
`can_device_access_peer` true both ways, about 3.5x PCIe 4.0 x8.

What it does and does not buy:

* It is **GPU-to-GPU only**. The largest bandwidth consumer here is *host-to-device*
  (the teacher cache), which still crosses PCIe. NVLink does nothing for it.
* Plain CUDA P2P needs no collectives, so **`device_map` model parallelism works**
  without NCCL -- the same mechanism already used for the 27B teacher.
* For stage 2 that is decisive: full-backbone AdamW fp32 is 34.2 GB of optimizer state
  (51.2 GB total with weights and grads) and does not fit one card. With `AdamW8bit`
  the total is 25.5 GB -- still over 24 GB, but comfortable **across two cards over
  NVLink, with no CPU offload at all**. That sidesteps the DeepSpeed question entirely.
* Caveat: naive `device_map` pipeline parallelism serialises, so it buys capacity, not
  throughput. Real 2x would need microbatch interleaving (GPipe/1F1B).

## Gate 6, 1M paired arms: the table earns its place (2026-09-07)

Both arms completed sharded across the two cards, exit 0. Identical data, order, seed,
schedule and trainable-parameter policy; the only difference is `sidecar.enabled`.

| | sidecar | control | delta |
| --- | ---: | ---: | ---: |
| eval_loss @ epoch 0.694 | 0.6607 | 0.6976 | **-0.0369** |
| eval_loss @ epoch 1.0 | **0.5807** | 0.6255 | **-0.0448** (-7.2%) |
| train_loss | 1.047 | 1.089 | -0.042 |
| KL term, mean of last 50 logs | 0.5177 | 0.5803 | -0.0626 |
| hs_cosine term, mean of last 50 | 0.7404 | 0.7494 | -0.0090 |
| train_runtime | 1026 s | 1020 s | +6 s |

The weighted terms reconstruct the gap: `0.7 x -0.0626 + 0.3 x -0.0090 = -0.0465`
against an observed -0.042. **Essentially all of the improvement is in the KL term** --
the student's agreement with the teacher's distribution -- and almost none in the
hidden-state term. That is the expected shape: the sidecar injects at decoder layer 1
while the anchors sit at layer 4 and post-norm, so the n-gram features reach the head
much more directly than they reach either anchor.

Architecture diagnostics, first -> last logged value:

| metric | sidecar | control |
| --- | --- | --- |
| `W_side_proj/weight_norm` | 0.0000 -> 1.8000 | 0.0000 -> **0.0000** |
| `branch_1/2/3_weight_norm` | 0.0000 -> 1.817 | 0.0000 -> 1.889 |
| `W_x_norm` | 88.66 -> 88.68 | 88.69 -> 88.71 |
| `W_x_bias_absmax` | 0.0000 -> 0.0031 | 0.0000 -> 0.0034 |
| `gate_1_mean` | 0.5001 -> 0.5008 | 0.5006 -> 0.5015 |
| `gate_saturation` | 0.0664 -> 0.0664 | 0.0644 -> 0.0664 |

Two things worth reading off that table. `W_side_proj` stays at exactly 0.0000 in the
control for the whole run, which is the check that `disable_sidecar_projection()` really
does bypass the table rather than merely zeroing its output. And the gated residual moves
*slightly further* in the control (1.889 vs 1.817), so the control is not handicapped on
optimisation -- it got the same budget and used it. The sidecar arm wins with a smaller
residual because it has better features to route.

The gates themselves barely moved: means still ~0.5, saturation flat at 0.066. Over 1M
tokens the gain is coming from `W_side_proj` and the zero-initialised branches, not from
learned gating. That is consistent with the design note that gate gradients only start
flowing once the branch weights leave zero.

**How much to trust -0.0448.** The arms are paired tightly, which controls for data order
and for the gated residual's initialisation. What there is no estimate of is seed
variance -- one run per arm. The nearest thing available is the same sidecar arm run three
times under different memory configurations: eval_loss 0.5853, 0.5973 and 0.5807, a spread
of 0.017. That is not a seed replicate (the configurations differed), but it bounds
run-to-run wobble at roughly 38% of the effect. The effect is real but a 5M pilot should
carry at least two seeds per arm before anything is concluded about its size.

## Threaded microbatch overlap: measured, slower, left off by default (2026-09-07)

`distillkit/concurrent_training.py` runs two microbatches of an accumulation window in
worker threads, overlapping one forward with another's head and backward. It is correct
-- it reproduces serial losses and gradients -- and it is **8.7% slower**. It stays
opt-in behind `concurrent_microbatches: 2`, which defaults to 1.

Steady-state steps of the real cache at boundary 10, four accumulated microbatches per
step (`scratch/concurrent_training_probe.py`):

| step | serial | threaded |
| --- | ---: | ---: |
| 1 (threaded warms up serially) | 4.85 s | 5.15 s |
| 2 | **3.33 s** | **3.63 s** |
| 3 | **3.32 s** | **3.61 s** |

It also costs memory: card 0 reserved 19.79 GiB against serial's 18.80, card 1 16.54
against 14.37, for the second in-flight microbatch.

### What the measurement cost to get right

Three separate mistakes, each caught by the next measurement rather than by reasoning:

* **Boundary 18 does not fit.** The stage timings put the compute-balanced split at 18
  layers on card 0 (2296 ms vs 2342 ms, a 1.98x pipelining ceiling) and the preflight
  reported a 18.23 GiB *peak* against a 22.80 GiB cap. But the cap applies to
  *reserved*, and both modes OOM'd there at 18.34 GiB allocated plus 4.40 GiB stranded.
  At boundary 7 fragmentation had 7 GiB to hide in; at 18 it has none. The balanced
  split is unreachable, so the 1.98x ceiling is theoretical.
* **"Launch-bound" was wrong.** CPU issue time is 90% of wall time at every sequence
  length, which looks conclusive. It is not: holding kernel count fixed and scaling the
  batch 1 -> 2 -> 4 gave 1020 -> 1941 -> 3684 ms, near-linear in work. A launch
  bottleneck would have been flat. The 90% is the CUDA launch queue filling and blocking
  the CPU inside `cudaLaunchKernel` -- a symptom of a GPU-bound step, not its cause.
* **The first four comparisons measured the wrong window.** Both modes OOM'd at step 2,
  so every number came from step 1 -- the one window where `warmup_first` runs the first
  microbatch alone and pays one-time setup. The probe selected the eight *longest*
  documents, pinning every microbatch at the 4096 cap. Median-length documents (closer
  to the corpus mean of ~870 tokens anyway) reach step 3, and the steady-state result is
  the same as the biased one, so the bias was not what made threading lose.

### Two fixes that stand regardless

* `_OrderedGate` admits microbatches to the head in submission order. The original
  semaphore admitted whichever worker arrived first, so gradients accumulated into the
  shared `.grad` buffers in a nondeterministic order; bf16 addition is not associative
  and the equivalence test failed on a different element each run. Backward is
  serialized under that gate anyway, so fixing *which* order costs no overlap and buys
  reproducibility.
* The per-worker post-backward sync is scoped to that worker's own streams.
  `torch.cuda.synchronize(device)` is a whole-device barrier that also waits on the
  other worker's in-flight forward -- blocking on precisely the work being overlapped.
  Fixing it changed the timing by 0.03 s, so it was not the bottleneck, but a
  whole-device barrier inside a pipeline is wrong on its own terms.

### Why it does not pay, and what would

The gate holds from `lm_head` entry through backward, and backward is roughly two thirds
of a microbatch, so at best one third can overlap. The step is GPU-bound, so two threads
contending for the same saturated devices add scheduling overhead without creating
capacity.

A real gain needs the schedule to keep *different devices* busy on *different*
microbatches, rather than letting two threads race for both. That is the explicit stage
schedule: split front/middle/head into separately callable stages with detached
boundaries, and drive them from one thread in 1F1B order. It is a larger change --
it reimplements `Qwen3_5TextModel.forward`'s layer loop, mask construction and rotary
setup, which then has to be kept in step with the library -- and the honest ceiling for
it here is bounded by the reachable split, not the balanced one.

## The staged curriculum beats training jointly (2026-09-07)

`examples/qwen35_sidecar_stage2_chained.yml` is byte-identical to the from-scratch
stage-2 config except that `model:` is stage 1's output, so the sidecar starts from an
adapter already trained to read the n-gram table rather than from zero.

| run | eval @0.694 | eval @1.0 | runtime | final `W_side_proj` |
| --- | ---: | ---: | ---: | ---: |
| stage 1, frozen backbone | 0.6607 | 0.5807 | 1026 s | 2.1055 |
| stage 2, from the stock student | 0.5709 | 0.5347 | 1269 s | 0.1652 |
| **stage 2, chained from stage 1** | **0.5468** | **0.5262** | 1291 s | 2.1077 |

Chaining wins, but the margin (0.0085) is smaller than the 0.017 run-to-run spread
measured on repeated sidecar arms, so treat the ordering as suggestive rather than
established. `train_loss` is not comparable across these rows: it is the epoch mean, and
the chained run starts from a much better model (0.5623 against 1.025).

**The finding worth keeping is the weight norm, not the loss.** The chained sidecar moved
from 2.1055 to 2.1077 across an entire stage-2 epoch -- it carried over and then sat
still. The from-scratch run reached only 0.1652 in the same budget. So:

* Stage 1 is where the sidecar is actually learned. Its `lr 1e-4` against a frozen
  backbone moves `W_side_proj` two orders of magnitude further than stage 2's `lr 1e-5`
  against a free backbone does.
* Stage 2 cannot substitute for stage 1. Given a free backbone and a low learning rate,
  the optimiser improves the model through the 4.27B backbone parameters and leaves the
  65.5M sidecar path where it found it.
* If the sidecar should keep adapting during stage 2, it needs its own parameter group
  at a higher learning rate. That is untested.

Note the chaining is partial in both arms: `distillation_projections.*` are written into
the checkpoint but reload as UNEXPECTED (they belong to the trainer, not the
architecture), so the hidden-state projections restart from fresh xavier init either
way. Equal treatment, so the comparison holds, but the sidecar is the only thing that
actually carries over.

## Anchor taps: stop gathering 33 hidden states for 2 (2026-09-07)

`output_hidden_states=True` retains every state and, on a device-mapped model,
accelerate's output hook then copies *all of them* to the input device: 33 tensors of
`[1, 4096, 2560]` bf16, about 0.7 GiB retained plus the same again copied plus gradients
for the copies, to serve the two anchors the loss reads. `distillkit/anchor_tap.py`
hooks just those two modules instead.

Two consequences beyond the memory:

* The captured state stays on the card that produced it, so a projection built there
  consumes it in place and nothing crosses the bus. That removed the need for the
  `anchor_device` / `returns_outputs_on_input_device` pair, which existed only to
  predict where accelerate would gather a state to; both are deleted.
* It is the mechanism 1F1B needs anyway, since a pipelined step has to collect anchors
  from two different stages.

`tests/test_anchor_tap.py` asserts the tapped states equal `output_hidden_states=True`
bit for bit at every anchor position, that the off-by-one is right (`hidden_states[i]` is
layer *i-1*'s output and the last entry is post-norm), that gradients flow through the
captured tensor, and that gradient checkpointing's recompute does not replace it.

The two-GPU equivalence test caught a real bug in the process: `compute_hs_loss` read
`hidden_states[0]` unconditionally, only to pick a device for its accumulator. With a
tapped forward that index is absent -- the real run's anchors are 4 and 32 -- so it now
takes its reference from the first mapped anchor. `CapturedStates` raises on an untapped
index rather than returning a neighbouring state, which is what surfaced it.

## Why 1F1B needs the tap first

1F1B keeps at least two microbatches in flight, so the question is what a second one
costs. Measured at the 7/25 split:

| | card 0 | card 1 |
| --- | ---: | ---: |
| persistent (weights + grads + AdamW8bit) | 8.0 GiB | 15.87 GiB |
| measured peak | 15.62 GiB | 16.71 GiB |
| activations | **7.6 GiB** | 0.84 GiB |

Card 0's activation cost is almost entirely the head -- logits 1.89 GiB, their gradient
1.89 GiB, the chunked-KL temporaries, and the gathered hidden states. With gradient
checkpointing the *front stage* retains only 7 layers x `[1, 4096, 2560]` bf16, about
150 MB. So a second in-flight microbatch costs roughly 200 MB on card 0 and 525 MB on
card 1, provided only one microbatch is in the head at a time.

Card 1 has 6.1 GiB of margin for that. Card 0 had 0.7 GiB against its *reservation*,
which is why the tap comes first.

## Hybrid tensor parallelism: through a full epoch, 1.06x, balanced (2026-09-08)

### The epoch-level result

| | layer split (boundary 13) | tensor parallel |
| --- | ---: | ---: |
| `train_runtime` | 1263 s | **1193 s (1.06x)** |
| `eval_loss` | 0.5330 | **0.5329** |
| card 0 peak / reserved | 13.68 GiB | **14.98 / 15.99 GiB** |
| card 1 peak / reserved | 12.92 GiB | **13.71 / 14.81 GiB** |

72 steps, 4.271B parameters, `runs/sidecar-stage2-tp`. **The loss matches the layer
split to the fourth decimal**, which is the trajectory-level check a microbatch
measurement cannot give: a gradient scaled by a constant, a shard drifting from its
partner or the replicated-norm reduction firing intermittently would all show here.
Stronger still, the exported weights move *identically*: the same 86 of 426 tensors are
unchanged from the starting checkpoint under both arms -- the norms and `dt_bias`, whose
bf16 updates at lr 1e-5 round away -- and even `A_log`'s single 5.96e-08 nudge is the
same in both.

**1.06x, not 1.54x.** The microbatch measurement is real and so is this one; they differ
because a step is not only forward and backward. Each of the 72 steps carries 16
microbatches plus clipping over 795 parameter tensors, the AdamW8bit update, the
collator and the evaluation passes, none of which the probe timed and none of which the
split makes faster. That is the honest speedup for this configuration: the microbatch
work shrank by a third and it moved the epoch by 5.5%. The memory is the real prize --
and it is what buys batching, which is where the microbatch speedup would actually
convert (see "What remains").

Trainer peaks run about 1.7 GiB above `scratch/tp_optimizer_probe.py` (14.98 against
13.25): clipping's `foreach_norm` temporaries and the collator's on-GPU rows, the same
gap that made three earlier preflights understate their runs.

### Export verified against a stock load

`runs/sidecar-stage2-tp/model.safetensors` reconstructs stock names and shapes from the
shards: no key missing against `student-hf`, no shape mismatch, and no dtype mismatch
against the layer-split export. It loads on **one** card without any TP machinery
(8.05 GiB) and produces finite logits. The nine extra keys are the sidecar and the
distillation projections, as in every other export.

One pre-existing wrinkle, not a TP one: `A_log` and the gated `norm.weight` are fp32 in
`student-hf` and bf16 in *every* export this project has produced -- stage 1, both layer
splits, the chained run and this one. It is the bf16 training load, and it is why the
comparison above is against a layer-split export rather than the starting weights.


### The training-run OOM was a duplicate key, not a bug

Three attempts failed before an epoch ran. The first two were my own config errors, and
the third turned out to be the first one again: `examples/qwen35_sidecar_stage2_tp.yml`
carried `chunked_head: true` and, eight lines later, the stale `chunked_head: false` it
had inherited from the layer-split config. YAML keeps the last duplicate. The run trained
with the head unfolded and died at step 4 asking for the 1.89 GiB logits gradient with
card 0 at 20.52 GiB -- "during backward, not forward", which read like a bug and was the
unfolded head's gradient. The tell I missed: a probe of the same step *with* the real
AdamW8bit state fit at 14.30 GiB, so the arithmetic left no room for a 6 GiB mystery.

**The knob is only correct relative to a placement**, and that stays true: the layer
split gives card 0 thirteen layers so weights bind and the head's 4.4% recompute is not
worth paying; tensor parallelism halves every layer, so the head binds instead. Under
tensor parallelism the folded head is required. And count the key rather than reading
the file.

### Where the tied embedding lives, measured three ways

With the head folded the run fits, but card 0 carried the tied embedding on top of its
half of every layer: 0.636B parameters, 14.9% of the model, 3.5 GiB with gradient and
8-bit moments. Steady-state peaks at sequence 4096 with the optimizer state present
(`scratchpad/probe_tp_optimizer.py`; the first step has no optimizer state yet and
understates every later one by about 4 GiB, so it measures the second step too):

| tied embedding | card 0 | card 1 | 4096-token microbatch |
| --- | ---: | ---: | ---: |
| whole, on the home card | ~19.0 GiB | ~13.6 GiB | 2.838 s |
| whole, moved to card 1 (commit 7f93c9d) | 11.42 GiB | 15.02 GiB | -- |
| **split by vocabulary rows (`tp_vocab.py`)** | **13.25 GiB** | **12.62 GiB** | **2.598 s** |

Moving the parameter whole is the simple shift and it overshoots by 3.6 GiB: card 1
holds no norms and then held the head's entire working set too. Splitting by rows
balances the cards to within 0.6 GiB and, because the head's ~20 TFLOP per microbatch
(projection, recompute, two backward products) splits with it, is 8.5% faster as well.
98.5% of parameters are now sharded; what stays whole is the norms, the sidecar and the
distillation projections.

The split is Megatron's `VocabParallelEmbedding` in single-process form. The lookup
embeds each rank's range and sums the rows onto home through the existing `Reduce`. The
head keeps its per-rank logits apart: the sparse KL needs only a row's log-sum-exp and
its values at the teacher's top-k ids, and both compose from per-rank pieces
(`VocabShardedLogits.sparse_logprobs`), so the 248,320-wide chunk row never exists on
any card and what crosses NVLink per chunk is `[batch, chunk, 1]` and
`[batch, chunk, k]`. `Collect`, the many-output sibling of `Reduce`, carries the same
recompute barrier. Checkpoints still export the stock `model.embed_tokens.weight` /
`lm_head.weight` pair. Verified against the dense head at rtol 1e-5 on CPU and across
two cards, gradients included, through the checkpointed chunk loop.

Two latent bugs surfaced on the way, both pinned by tests now: `KLDLoss` counted
`num_items_in_batch` from the mask on the batch's card and divided a result on the
head's card -- 0-dim tensors on two CUDA devices do not mix, and the layer split never
hit it because its head was on card 0 -- and three callers read
`get_input_embeddings().weight`, which a sharded embedding does not have
(`sharding.embedding_device`).

### Measured against the layer split

Single-process tensor parallelism across both cards, sharding **98.5% of the 4.271B
parameters**. Measured on the real student with the same losses the layer split was
measured with -- chunked KL over the 248,320-wide vocabulary plus hidden-state cosine,
forward and backward, no optimizer state:

| | layer split (boundary 13) | tensor parallel |
| --- | ---: | ---: |
| 4096-token microbatch | 3.99 s | **2.598 s (1.54x)** |
| 1024-token microbatch | 1.049 s | 0.834 s (1.26x; before the vocabulary split) |
| card 0 peak | 13.68 GiB | **9.03 GiB** |
| card 1 peak | 12.92 GiB | **8.53 GiB** |

Faster *and* smaller on both cards. It is the first thing in this project to actually
convert the second GPU into throughput: the layer split is capacity-only by
construction, and threaded overlap measured 8.7% slower and did not recover at a
balanced split.

**Beware the number that is not this one.** An earlier run reported 1.80x, comparing a
tensor-parallel step whose loss was `logits.pow(2).mean()` against a layer-split
baseline carrying the real chunked-KL and hidden-state losses. Matching the losses cost
28% of the apparent speedup; 1.41x was the comparable figure with the embedding whole
on card 0, and 1.54x is it with the embedding split.

### No NCCL, and none needed

An all-reduce between two peer-accessible devices in one process is a peer copy and an
add, which autograd differentiates without help. Peer copies measured 38-48 GB/s at
64 MiB against NCCL's 37.5 GB/s all-reduce, so a collective library would add `Work`
objects, process-group lifecycle and `wait()` ordering for no bandwidth. That deletes
the ZeRO-2 backend project rather than deferring it: tensor parallelism already gives
each rank half the weights, gradients *and* optimizer state, which is what ZeRO-2 was
wanted for.

llama.cpp was considered as a donor and rejected -- its `allreduce.cu` stages through
pinned host memory *because* it targets machines without NVLink, and being inference-only
it lacks backward through the collective. What was borrowed is transformers'
`base_model_tp_plan` as the sharding specification.

### What is sharded, and what is deliberately not

MLPs (58.5%), full attention (5.4%), the 24 GatedDeltaNet layers by head (19.7%) and
the tied embedding/head by vocabulary row (14.9%). Upstream's plan marks every
`linear_attn` projection `colwise_gather_output`, replicating the recurrence on both
ranks and capping coverage at 63.9%; sharding by head lifts it to 83.6%, and the
vocabulary split to 98.5%.

Replicated on purpose: the norms, the n-gram sidecar and the distillation projections --
reunifying any of them costs more than the flops saved. The residual stream stays on the
home card, so the outer model is untouched and gradient checkpointing, the anchor tap and
the folded head keep working.

### Three silent-failure traps, all now pinned by tests

**`q_proj` packs an output gate**, emitting `num_heads * head_dim * 2`. The layout is
head-major, so a contiguous column split is valid -- but only by luck of the packing. Had
it been `[all queries | all gates]`, the same split would have given one rank every query
and the other every gate, run without error, and trained nonsense. `_assert_head_major`
fails loudly if that changes.

**GatedDeltaNet's conv channels are packed `[all Q | all K | all V]`**, not per-head. My
reasoning -- "conv1d is depthwise, so channels are independent, so split them" -- was true
at every step and produced the wrong implementation, because depthwise independence says
nothing about packing order. Found by adversarial review, along with: `RMSNormGated`'s
weight is one shared parameter whose per-rank gradient is a *partial* needing
`sync_replicated_gradients`; the recurrent cache is keyed by `layer_idx` so both ranks
would alias one slot (cache is refused -- training only); and after `repeat_interleave`
each shard hands FLA 16 Q/K/V heads, not 8 and 16.

**Non-reentrant checkpointing does not serialize recomputation across device workers.**
A reduction with no saved tensor lets both upstream shard branches unpack concurrently and
recompute the same frame, corrupting its saved-tensor counter -- surfacing variously as
`target_frame.early_stop is set`, "a different number of tensors was saved", or
"recomputed values have different values". `AllReduce`/`Reduce` now save an empty
sentinel, so unpacking it in the reduction's single backward node finishes recomputation
before either branch is scheduled. Zero storage, zero numerical change.

I bisected this to `TensorParallelMLP` and falsified four hypotheses (input-aliasing
outputs, per-projection replication, `non_blocking` copies, the discarded device-1
output) before handing it over. The bisection was misleading: it was never MLP-specific,
the MLP's reduction just forks straight into two branches where attention and gated-delta
have work in between, so only the MLP lost the race reliably. The same fix also cleared
the FLA autotuner returning `None` at production shapes -- both were two device workers
racing on shared state.

## Attention was on the math kernel (2026-09-07) — largest single win

`sdpa_attention_forward` was allocating **4328 MiB per call** at sequence 4096, eight
times per forward. The cause is a platform-dependent dispatch, not the model:

transformers hands Qwen3.5's 16:4 query/KV head ratio to SDPA as `enable_gqa=True`
instead of repeating the heads. `use_gqa_in_sdpa` gates that on "no attention mask and
head_dim <= 256", and its comment says the point is to keep SDPA off the math kernel.
That holds where a fused kernel implements the broadcast. This build has none -- it
reports *"Torch was not compiled with flash attention"*, and the memory-efficient kernel
refuses unequal head counts (*"both fused kernels require query, key and value to have
the same num_heads"*). So the flag meant to avoid the math kernel is what selects it.

| per attention call, sequence 4096 | forward | with backward |
| --- | ---: | ---: |
| `enable_gqa=True` (math kernel) | 2728 MiB | 4328 MiB |
| KV expanded to 16 heads | 96 MiB | 249 MiB |

`distillkit/gqa_dispatch.py` patches `use_gqa_in_sdpa` to return False, sending
transformers down its own `repeat_kv` path. It **probes** whether any fused backend
accepts a broadcast GQA call and patches only when none does, so a build that gains
flash attention keeps upstream behaviour.

On the real student at boundary 13, sequence 4096: card 0 **16.21 -> 13.68 GiB**, card 1
**15.99 -> 12.92 GiB**. Memory became near-flat in sequence length (1024 costs
13.50/12.13 against 4096's 13.68/12.92) because attention no longer materializes scores.

## Where the memory goes now, and the split that follows

Measured at sequence 4096, batch 1, full backbone trainable, boundary 7:

| | card 0 | card 1 |
| --- | ---: | ---: |
| weights / gradients / AdamW8bit | 2.82 / 2.82 / 2.86 GiB | 5.19 / 5.19 / 5.27 GiB |
| persistent total | 8.50 | 15.65 |
| activations | 3.96 | 4.47 |

Persistent state is 24.15 GiB across both cards against ~4 GiB of activations per card,
so recompute tricks have little left to attack; the remaining levers are structural. It
also showed the split was memory-*imbalanced*: card 1 sat 2.7 GiB from the cap while card
0 had 10 GiB idle. Rebalanced to **boundary 13** (16.21/15.99 before the GQA fix,
13.68/12.92 after), which leaves both cards ~9 GiB clear.

Ruled out by measurement: bitsandbytes 0.50.2 has no 4-bit AdamW (8 and 32 only);
optimizer-in-backward would free the 8 GiB gradient buffer but is incompatible with
gradient accumulation; freezing the tied embeddings saves 2.55 GiB on the card that is no
longer binding.

## The folded head is conditional, not a win

`chunked_head` projects `lm_head` inside the loss chunk loop instead of materializing
`[batch, seq, 248320]` logits and their gradient. It is a memory-for-compute trade -- the
head is recomputed during backward -- and which way it pays depends on what is binding:

| | throughput | card 0 |
| --- | ---: | ---: |
| `chunked_head: true`, batch 1 | 955 tok/s | 13.50 GiB |
| `chunked_head: false`, batch 1 | 997 tok/s | 15.16 GiB |

4.4% for 1.66 GiB at batch 1. But what it eliminates scales with `batch x sequence`: at
`[4, 4096, 248320]` the logits are 7.6 GiB and their gradient another 7.6 GiB, so at
batch 4 it is not optional. **Off at batch 1, required at batch 4.**

## The objective this project tunes against does not measure the thing it wants (2026-09-09)

The sidecar learning-rate sweep produced a monotonic `eval_loss` improvement, 0.5445 at
the backbone's rate down to 0.3435 at 1e-3, better than anything this project had
produced. It does not survive contact with any other measurement.

### The confound I built

`optimizer.sidecar_lr` raises the rate for everything `_auxiliary_parameter_ids` matches,
and that set includes **`distillation_projections`** -- free linear maps whose only
purpose is to compute the hidden-state term of the loss. Their norms moved in step with
the rate:

| `sidecar_lr` | projection norms |
| --- | --- |
| 1e-5 | 58.426 / 58.421 |
| 1e-4 | 58.541 / 58.422 |
| 5e-4 | 53.477 / 59.621 |
| 1e-3 | 48.241 / 63.330 |

Static at the base rate, ten units apart at 1e-3. Raising the rate let the loss-shaping
layer shape the loss. The fix is to scope `sidecar_lr` to the architecture -- sidecar,
PLE, gated residual -- and hold the projections fixed across any sweep.

**A correction to how that was first reported.** I decomposed the improvement using the
`distillation_loss/1_kl` and `2_hs_cosine` values from the logs and concluded the KL was
flat while the hidden-state term collapsed. Those are *training-step* values and do not
reconstruct the evaluation total: 0.7 x 0.4345 + 0.3 x 0.9844 = 0.5995 against an
`eval_loss` of 0.5445, and 0.3906 against 0.3435 at 1e-3. The projection-norm evidence
above stands on its own, but the component split was not measured on the evaluation pass
and the trainer does not currently log one. Adversarial review caught the arithmetic.

### What upstream actually validates against

From the Flash-Next tech report, which was read rather than assumed:

* **§2.3.2, p.15**: "The out-of-domain uncheatable PPL changes little across budgets, and
  downstream benchmarks show no clear improvement over the MoE-only baseline." Table 8
  puts numbers on it -- loss reaches its optimum at 10x vocabulary (1.197 against 1.202
  at none) while uncheatable PPL sits flat at 5.54-5.59 across every budget.
* **§2.3.2, p.15**: "Loss decreases monotonically as the N-gram vocabulary grows, while
  downstream performance does not follow the same trend."
* **Table 7, p.14**: no n-gram gives loss 1.585 and benchmark average 45.44; layer 2 gives
  1.541 and 47.94; layers 2+25 give **1.540 and 47.75** -- the lowest loss is not the best
  average.

So loss and quality diverge in their own measurements, repeatedly, and they never tune on
loss alone. Note the report does **not** define or reproduce the uncheatable corpus and
explicitly calls it out-of-domain; arbitrary uncached text is in the same spirit but is
not the same thing.

**There is no gate statistic anywhere in the report.** No gate mean, variance, entropy,
open-fraction or selectivity threshold for the PLE. A regime table circulated in this
session citing the report for `gate_std` bands is unsupported; `gate_std` is this fork's
own telemetry, and its ~0.20 initial value is derivable arithmetic (0.2033 analytically,
scale-invariant from hidden 512 to 5120) rather than anything upstream published.

### Upstream does split the optimizer, differently than I guessed

**§3.1, p.16**: "We apply Muon to the two-dimensional weights that genuinely act as linear
maps: ... and the key/value projection in N-gram embedding layers." And: "Finally, the
n-gram embedding table runs on Adam with weight decay disabled." The GR low-rank
projections get AdamW, attributed to "their very elongated shape".

| parameter | upstream treatment |
| --- | --- |
| n-gram lookup table | Adam, weight decay disabled |
| n-gram key/value projections | Muon |
| GR's two low-rank projections | AdamW |

So separate treatment for these parameters is upstream practice, not an invention -- but
by *parameter kind*, not by the blanket "auxiliary" grouping this fork uses.

### First independent measurement, and it is not encouraging

Cross-entropy on 64 documents (45,069 tokens) that appear in no cache manifest, with the
sidecar enabled and then bypassed:

| checkpoint | sidecar on | sidecar bypassed | sidecar costs |
| --- | ---: | ---: | ---: |
| `student-hf` | 1.0205 | 1.0205 | +0.0000 |
| `gr-stage1-1m` | 1.0205 | 1.0205 | +0.0000 |
| `ple-stage1-1m` | 1.3231 | 1.0205 | **+0.3026** |
| `lr-sweep-base` | 1.5543 | 0.9628 | +0.5915 |
| `lr-sweep-1e3` | 1.5696 | 0.9476 | **+0.6220** |

The bypassed column is identical across the three stage-1 rows, which is the harness
self-checking: stage 1 freezes the backbone, so those checkpoints share `student-hf`'s.
Beyond that the reading is bad -- the PLE sidecar costs 0.30 nats at stage 1 and 0.59 at
stage 2, and the damage grows with the learning rate the distillation objective preferred.

Two reasons not to bank it yet. `gr-stage1-1m` reports +0.0000 for a sidecar whose
`W_side_proj` demonstrably trained to norm 2.077, which is not credible and means the
harness may not be exercising that path. And this evaluator gathers n-gram rows directly
rather than through `SidecarDataCollator`, so evaluation and training can differ. A
proper suite is being built; these numbers are a warning, not a verdict.

## The PLE port, and the sidecar learning rate stage 2 never had (2026-09-09)

The 5M pilot's diagnosis was that the sidecar's projection trained while its gate did
not. Flash-Next answers that question in its own model, so the answer was transcribed
rather than invented: `distillkit/ple_sidecar.py` ports
`Qwen4ExpTextPLELayer`. The configurations already lined up exactly -- their
`ple_layer_ids: [2]` one-indexed is our `sidecar_layer_index: 1`, and `ngram_size: 3`
with `heads_per_ngram: 8` gives 16 heads of 160 dimensions, which is precisely the
`sidecar_num_heads` x `sidecar_head_dim` table this fork already reads. Our dequantized
feature vector *is* their `ple_embedding` output.

The substantive difference is that the gate is not a parameter. It is a dot product
between a query read off the residual stream and a key read off the n-gram embedding,
signed-sqrt compressed through a sigmoid, followed by a depthwise convolution dilated by
`ngram_size`. Selectivity exists from the first step rather than having to be discovered.

### Four bugs, none of which the happy path could show

Recorded because the pattern is the point: every one was invisible under fp32, batch 1, a
constructor, or a fresh model, and the identity-at-init test passed throughout because a
zero `value_proj` masks everything downstream.

* **`nn.RMSNorm` cannot learn here.** Upstream stores a zero-initialised weight and scales
  by `(1 + w)`; torch stores the scale directly. Identical at initialisation, which is
  what made the substitution look free. But **bf16 spacing near 1.0 is 0.0078**, so a
  scale stored directly cannot represent a deviation below ~0.004, and at this fork's
  1e-4 the norms would have sat frozen at exactly 1.0. The run later measured
  `norm_conv_deviation` at 0.0061 -- inside the range that would have been lost.
* **Autocast promotes the gate.** `sum` is on autocast's fp32 list, so the reduction came
  back fp32 and the bf16 value multiply promoted with it. That escaped the module and
  killed the first run inside `lm_head`, far from the cause.
* **The identity was a knife-edge.** Zeroing `value_proj` alone is not enough, because
  `norm_conv` renormalises to unit RMS and the convolution branch then fires at full scale
  the instant the value moves -- 0.64 into a stream of RMS 1.01 at std 1e-4. Upstream can
  afford that because it trains jointly; a frozen retrofit cannot. The convolution is
  zero-initialised too.
* **`from_pretrained` re-initialises missing norms to ones**, which under `(1 + w)` is a
  scale of 2.0. Only a real checkpoint load shows it; the constructor zeroes them.

### The retrofit that could not work, and why

Bolting the port onto the best existing backbone (`sidecar-stage2-chained`, 0.5262) gave
2.184. Not the port's fault: with `learning_rate: 0`, so the module stayed exactly the
identity, it still gave 2.184. The `sidecar` attribute holds **two** things -- the n-gram
projection and a trained `GatedResidual` whose `W_x`/`W_h` norms are 88.6 -- and swapping
variants deletes the latter. Disabling the sidecar only stops the n-gram input, which is
why that probe gave 0.53 while the variant swap gave 2.18. A fresh backbone was needed.

### Stage 1: the gate moves, and the loss is worse

Both designs, identical curriculum, fresh student, frozen backbone, 1M cache:

| | `train_runtime` | final `eval_loss` |
| --- | ---: | ---: |
| gated_residual | 747.9 s | **0.5921** |
| ple | 746.4 s | 0.7739 |

The control reproduces the historical 1M stage-1 arm (0.5853), so tensor parallelism,
batch 4 and sortish were roughly neutral and the deficit is the design. But the telemetry
says the mechanism works, over the same run:

| | control | ple |
| --- | --- | --- |
| projection | `W_side_proj` 0 -> **2.077** | `value_norm` 0 -> **3.147** |
| gate | `gate_1_mean` 0.5001 -> **0.5007** | `gate_mean` 0.516 -> **0.711** |
| selectivity | `gate_saturation` 0.0586 -> **0.0586** | `gate_std` **0.163**, 83% open / 10% shut |

The control's gate is inert to four decimal places for the second time; PLE's moves
decisively and still discriminates between tokens. **The mechanism works and the stage-1
loss is worse** -- PLE starts as an exact identity with *both* branches at zero, where
`W_side_proj` injects features immediately.

### Stage 2 had never trained a sidecar at all

Watching stage 2 with the new telemetry showed `value_norm`, `conv_norm` and all three
norm deviations identical **to five significant figures** across fifty logged steps,
while the model around them trained normally. PROGRESS already recorded the same thing
for the old design -- the chained sidecar went 2.1055 -> 2.1077 across an entire epoch --
but as a curiosity rather than a defect. It is a defect: stage 2's 1e-5 does not move
these parameters, so every stage-2 comparison this project has run was really *which
frozen sidecar a backbone can adapt around best*.

`optimizer.sidecar_lr` fixes it, and the sweep says the effect is large. Stage 2 from the
PLE stage-1 checkpoint, 1M cache, backbone at 1e-5 throughout:

| `sidecar_lr` | `eval_loss` | `value_norm` | `gate_std` | vs base |
| --- | ---: | --- | ---: | ---: |
| 1e-5 (backbone rate) | 0.5445 | 3.176 -> 3.176 (**does not move**) | 0.1811 | -- |
| 5e-5 | 0.5191 | 3.176 -> 3.223 | 0.1797 | -0.0254 |
| 1e-4 | 0.4947 | 3.176 -> 3.342 | 0.1788 | -0.0498 |
| 5e-4 | **0.4720** | 3.176 -> 5.020 | 0.1502 | **-0.0725** |
| 1e-3 | *running* | | | |

Monotonic so far, `gate_std` holding, and the base row is the diagnosis restated: at the
backbone's rate the sidecar is a fixed function. For scale, the old design chained into
stage 2 at 1M reached 0.5262, so 1e-4 and 5e-4 are both well past it -- though that
number came from a different configuration and the matched control has not been run yet.

## The 5M pilot: the sidecar helps, and the 1M number overstated it 8x (2026-09-08)

The question the whole project was built to answer, run at last: four stage-1 arms,
sidecar and control at two seeds each, on a 4.75M-token cache with a 295-document
holdout. Every arm ran the 1M recipe unchanged except for batching.

| arm | `train_runtime` | `eval_loss` |
| --- | ---: | ---: |
| sidecar, seed 42 | 4915 s | **0.2924** |
| control, seed 42 | 4781 s | 0.2967 |
| sidecar, seed 43 | 4853 s | **0.2898** |
| control, seed 43 | 4750 s | 0.2909 |

Paired differences **-0.0043** and **-0.0011**: mean **-0.0027**, seed spread **0.0032**.

### It did not replicate at scale

| | effect | as a share of control |
| --- | ---: | ---: |
| 1M | -0.0448 | 7.2% |
| **5M** | **-0.0027** | **0.9%** |

Eight times smaller in relative terms, and **the spread between the two seeds exceeds the
mean effect**. By the reading written into `scratch/run_5m_pilot.py` before any result was
seen -- read the spread before the mean -- the endpoint measurement does not establish the
effect. The 1M comparison, one seed per arm, was overconfident, exactly as the note
recording it warned it might be.

### The sign, however, is perfectly consistent

The sidecar is ahead in **14 of 14 paired evaluations**, both seeds, every checkpoint:

| eval | seed 42 | seed 43 |
| ---: | ---: | ---: |
| 1 | -0.0156 | -0.0251 |
| 2 | -0.0192 | -0.0197 |
| 3 | -0.0015 | -0.0214 |
| 4 | -0.0064 | -0.0166 |
| 5 | -0.0020 | -0.0100 |
| 6 | -0.0020 | -0.0073 |
| 7 | -0.0043 | -0.0011 |

Fourteen of fourteen is not what noise looks like, though the evaluations within a run
are not independent so this is not a clean sign test. The shape is unambiguous: the
sidecar's advantage is **front-loaded and erodes monotonically**, worth about 0.02 early
and 0.002 by the end. It behaves like a warm start that more data eventually supplies by
itself.

### Two trivial explanations ruled out, and one real finding

**The sidecar trained.** `W_side_proj` weight norm moved 0 -> 4.007 from its zero
initialisation, and the sidecar arms ran ~130 s slower than their controls. It is not
bypassed.

**The gate did not.** `gate_1_mean` went 0.5001 -> 0.5013, `W_x_norm` 88.68 -> 88.74,
saturation ~0.07. The gated residual is sitting where it was initialised, passing a fixed
half of the sidecar's contribution rather than learning *when* to use it. That is the one
component demonstrably not earning its keep, and it is the most actionable thing the
pilot produced.

### What this does and does not license

It does not license "the sidecar works" at the strength the 1M number suggested. It does
not license abandoning it either: a consistent front-loaded gain with an inert gate is a
description of an architecture that is half-wired, not one that does nothing. The
defensible next steps are the gate's learning rate (item 8) and a faithful re-reading of
how Flash-Next integrates its per-layer table, rather than more seeds of the same
configuration.

## The teacher capture cannot be batched, and does not need to be (2026-09-08)

Batching was worth 1.5-1.6x on the student, so the same question was put to the capture.
The answer is no, for a reason worth recording, and the practical news is better anyway.

### Where capture time goes

Measured with `scratch/capture_profile.py` on the real teacher and corpus:

| phase | share |
| --- | ---: |
| forward | **97.5%** |
| top-k over the 248,320-wide vocabulary | 2.2% |
| fp8 anchor conversion | 0.4% |

The cache writer runs at 47,000 tok/s equivalent (497 MB/s), nowhere near binding. Two
plausible optimizations died here: raising `logit_chunk_tokens` above its default of 64
makes the top-k *slower* (0.093 -> 0.140 s at 1024), and there is nothing outside the
forward worth touching.

The starvation is real -- batch 1 at this corpus's median 537 tokens runs 719 tok/s
against a ~1350 tok/s plateau reached from about 2048 tokens up, the same curve the
student showed. It just cannot be collected.

### Batching changes what the teacher says

| | max abs logit delta | top-1 agreement |
| --- | ---: | ---: |
| teacher 27B, int8, batch 1 vs 4 | 12-24 | 0.88-0.94 |
| student 4B, bf16, batch 1 vs 4 | 0.32 | 0.98-0.99 |

Against a median top1-top2 gap of 3.4, moving a logit by 24 is not rounding. Three
controls localize it. It is not padding: batching documents of *equal* length with no
padding at all is just as bad. It is not nondeterminism: the same document duplicated
within one batch gives bit-identical rows. And it is 40-70x worse in int8 than bf16.

The mechanism is LLM.int8()'s outlier decomposition. Columns whose activations exceed
`threshold` (6.0 here) are computed in fp16 and the rest in int8, and that column set is
chosen per *call*, so the other rows of a batch change which columns a given row takes.
Setting `threshold=0` makes batched and unbatched output **bit-identical** -- 0.0000
logit difference, 1.0000 agreement -- which confirms it exactly.

### Selective upcasting: a cascade, not a hotspot

Divergence compounds through 64 layers, so the useful measurement is where it is
*injected*: modules whose input is still bit-identical but whose output is not.
`scratch/int8_batch_sensitivity.py` finds four, all in layer 0's `linear_attn`
(`in_proj_a/b/qkv/z`). Neutralizing those does not fix it -- it moves the onset to layer
2's `mlp.down_proj`, then 5, 6, 7, then layer 8's `linear_attn`, with top-1 agreement
flat across the whole iteration (0.9219 -> 0.9336 after twelve modules). Every downstream
int8 layer with outlier-prone activations keeps injecting more; "first onset" only finds
where it starts. Fixing it properly means neutralizing all 496 modules, which is the
global switch.

And the global switch is too expensive. At batch 1, `threshold=0` against the reference:

| | |
| --- | --- |
| top-1 agreement | 0.85-0.89 |
| **top-64 set overlap** | **0.77-0.80** |

The top-64 set *is* the cached signal, so that is a 21-23% change to the training data,
to gain 1.9x on a job now measured in hours. A per-batch correction term cannot rescue
it either: the correction depends on which documents share the batch, so computing it
requires the batch-1 forward it would replace.

**Capture stays at batch 1 with outlier handling on.** The existing 1M cache is
consistent with that, and so is the 5M one.

### The recorded throughput was stale

The real win needed no code. PROGRESS recorded the 1M capture at ~220 tok/s, which would
put 5M at 6.3 hours. The same code measured now does 673 tok/s on median-length documents
and projects ~900 tok/s integrated over the real length distribution -- about **1.5
hours** for 5M. The likeliest explanation is that the 1M capture predates the
grouped-query attention fix, which was the largest single win in this project and is
installed at import in `sample_transformers`.

## Muon, tensor parallelism, and where each optimizer earns its place (2026-09-08)

Stage 1 could not use tensor parallelism: configuration refused `tensor_parallel`
alongside `optimizer.strategy=hybrid`. The cost was measured, not assumed -- stage 1 runs
the layer split, `nvidia-smi` shows exactly one card busy in 21 of 30 samples, and the
batching worth 1.5-1.6x under tensor parallelism is worth **1.12x** there (13.86 -> 12.42
s/it at a constant 16 sequences per optimizer step).

### The guard was refusing something equivalent to what it allowed

Census on the real model. Unfrozen, routing sends 3569.1M parameters to Muon and 702.2M
to AdamW. After `freeze_backbone_for_stage1`, the Muon group holds **0.0M trainable**
against AdamW's 65.5M, because everything stage 1 trains -- sidecar, gated residual,
distillation projections -- is auxiliary and routed to AdamW. **Muon has never trained
anything in this project**: stage 1 gives it nothing, and stage 2 uses `adamw`.

So `hybrid` under stage-1 freezing *is* AdamW-on-auxiliary, and tensor parallelism is now
permitted exactly when `freeze_backbone` is set and `unfreeze_at_step` is None.

### Two corrections from adversarial review, both of which I had wrong

**Tensor parallelism does not simply fall back to AdamW.** I claimed sharded matrices all
miss Muon's `isinstance(module, nn.Linear)` test. Column- and row-parallel weights do --
they are bare `nn.Parameter` in a `ParameterList` -- but the sharded GatedDeltaNet
projections are built by `_slice_linear` and *are* real `nn.Linear`, so they still route
to Muon. Measured on a sharded 4-layer model: 16 parameters remain in the Muon group. The
combination gives an inconsistent mixture rather than a clean fallback, which makes the
refusal more necessary, not less.

**Muon is not more expensive here.** I measured 4.00 bytes per parameter against
AdamW8bit's 2.05 and concluded Muon would double optimizer memory. That measurement used
fp32 parameters. `torch.optim.Muon` allocates its momentum with `zeros_like(p.grad)`, and
these runs load bf16, so the real figure is **2.00 bytes per parameter -- a wash with
AdamW8bit**, not double:

| optimizer | bytes/param | 3.57B backbone |
| --- | ---: | ---: |
| Muon, bf16 parameters (what a run would see) | 2.00 | 7.1 GB |
| bnb AdamW8bit (stage 2 today) | 2.05 | 7.3 GB |
| Muon, fp32 parameters (the misleading measurement) | 4.00 | 14.3 GB |
| torch AdamW fp32 | 8.00 | 28.6 GB |

Muon still costs compute the table does not show -- five Newton-Schulz iterations are
about fifteen matmuls per 2D parameter per optimizer step, amortized over the accumulation
window.

### 8-bit Muon exists, and is a memory option we do not currently need

Checked because it would change the arithmetic above if true. It is largely true.

`Effective Quantization of Muon Optimizer States` (arXiv 2509.23106, Gupta et al.,
Nubank/LinkedIn) is real: blockwise quantization of Muon's momentum, linear and dynamic
schemes, parity with full Muon on validation loss and downstream benchmarks up to 2.7B
pretraining plus instruction fine-tuning, up to 62% off the optimizer state.
bitsandbytes issue #1973 requesting `bnb.optim.Muon8bit` is open, with the submitter
reporting 8-bit matching 32-bit at 43% less allocated and 26% less peak memory.
`YupengSu/MuonQ` is real but primarily a **4-bit** framework (arXiv 2605.11396), not the
8-bit one. `junaidaliop/zij` **has no implementation** -- it lists the paper in a
reference table with dashes where the code column would be.

No Muon ships in bitsandbytes 0.50.2, so this would be built, not configured. The
`bitsandbytes.functional` route is viable here: `quantize_blockwise` and
`dequantize_blockwise` exist with compatible signatures, and measured on a
`[2560, 9728]` bf16 momentum buffer:

| blocksize | bytes/param | mean relative error | cosine |
| ---: | ---: | ---: | ---: |
| 256 | 1.016 | 0.0106 | 0.999940 |
| 2048 | 1.002 | 0.0121 | 0.999921 |
| 4096 | 1.001 | 0.0125 | 0.999915 |

For this student's 3.569B Muon-eligible parameters that is 7.14 GB of resident momentum
down to 3.58 GB. **The saving is on the resident buffer, not the peak**: Newton-Schulz
needs a bf16 copy of each matrix it touches, which is why the issue's own numbers are 43%
allocated against only 26% peak. Anyone quoting the headline reduction as a peak saving
is quoting the wrong number.

Not needed now. Stage 1 peaks 8.4 / 10.3 GiB and stage 2 peaks 13.3 / 12.6 GiB against a
22.80 GiB cap, so nothing is VRAM-bound. It is also a memory optimization layered on a
decision not yet made -- whether Muon beats AdamW8bit on convergence here at all -- and
this student is 83.6% Muon-eligible with a 248,320-token vocabulary, which is the
embedding-dominated case where the benefit is diluted.

### Where each optimizer earns its place

* **Stage 1** -- AdamW, whatever the config says. Only 65.5M auxiliary parameters train
  and none of them is a hidden matrix Muon would want. `hybrid` here is a label.
* **Stage 2** -- an open question, and now a fair one. Muon would cover the 3569.1M
  backbone matrices at memory parity with AdamW8bit, so the choice is about convergence
  rather than capacity. Nothing in this project has tested it. Settling it needs a
  matched pair on the same data, seed and hardware with its own learning-rate sweep,
  since Muon's scale is not AdamW's.
* **Never** -- Muon on a tensor-parallel shard. Newton-Schulz orthogonalization does not
  commute with slicing, and the shape-dependent learning-rate scaling would use the
  shard's dimensions. Configuration refuses the combination, and
  `build_mixed_optimizer` now refuses it again against live parameters, because
  configuration only governs runs driven by `main.py` while `unfreeze_backbone()` is
  reachable programmatically.

## Batching: throughput is set by tokens per microbatch, not by batch size (2026-09-08)

The single most mispriced knob in this project. Batching was written off twice on a
throughput model that assumed a **fixed cost per token**, so the only thing batching
could change was padding waste. Measured on the tensor-parallel student, the per-token
cost is not fixed at all:

| tokens per microbatch | shape | tok/s |
| ---: | --- | ---: |
| 512 | 1 x 512 | **744** |
| 1024 | 1 x 1024 | 1340 |
| 2048 | 4 x 512 | 1636 |
| 4096 | 1 x 4096 | 1592 |
| 4096 | 4 x 1024 | 1724 |
| 4096 | 8 x 512 | **1756** |
| 16384 | 4 x 4096 | 1698 |

Throughput plateaus near **1700 tok/s from about 2048 tokens up** and falls off a cliff
below it: a 512-token microbatch runs at 43% of the plateau. The GPU is starved, not
busy. Three readings make the shape unmistakable -- the same 4096 tokens cost the same
whether they arrive as one long sequence or eight short ones (1592 vs 1756 tok/s, the
*batched* form slightly ahead), and 4 x 4096 = 16384 tokens still runs at the plateau,
so nothing degrades at the top end either.

**This corpus is median 547 tokens, 76% under 1024.** At batch 1 every document is its
own microbatch, so most of the epoch runs in the starved regime. That is the finding:
batching here is not about padding, it is about giving the GPU enough work to fill.

Simulated over the real length distribution with HF's own longest-first sampler:

| batch (grouped) | projected microbatch compute | speedup | padding waste |
| --- | ---: | ---: | ---: |
| 1 | 951 s | -- | 0% |
| 2 | 672 s | 1.42x | 0.2% |
| 4 | **592 s** | **1.61x** | 0.6% |
| 8 | 580 s | 1.64x | 1.5% |

Batch 4 takes nearly all of the available gain; batch 8 adds 2% and does not fit the
full-length groups. Note how small the padding waste is once grouped -- the thing the
old model spent all its attention on is worth well under a percent, while the thing it
did not model is worth 61%.

### Why the earlier verdict was wrong, and what to take from it

The note this replaces concluded "the predicted gain was only ~1% anyway -- the padding
saving (+6%) is nearly cancelled by the `chunked_head` recompute (-4.4%)". Both of those
numbers were real measurements. The error was the frame around them: they were measured
**at fixed sequence length**, where the GPU is already saturated and batching genuinely
buys only the padding back. Generalizing that to a corpus of 547-token documents assumed
the very thing that is false.

The lesson worth keeping is not about batching. It is that a benchmark measured at one
shape does not license a conclusion at another, and that "cost per token" is a modelling
assumption to be checked rather than a unit. Everything else in that note was correct and
survives: `group_by_length` was **renamed**, not dropped (`train_sampling_strategy=
"group_by_length"`, still reading `length_column_name`); the cache exposes a `length`
column so the sampler does not materialize every `input_ids` row; the sampler orders
longest-first, so the most expensive groups arrive at step 0 and `chunked_head` must be
on; and `sparse_chunk_length` counts positions while a chunk's logits are
`[batch, positions, vocab]`, so it is read as a row budget at batch 1 and divided by the
batch.

### The memory that makes it possible

Batch scaling at full sequence length, steady state with AdamW8bit present
(`scratch/tp_optimizer_probe.py 4096 <batch>`), against the 22.80 GiB cap:

| batch x 4096 | backward peak | reserved |
| --- | ---: | ---: |
| 1 | 13.25 / 12.62 | 14.63 / 13.57 |
| 2 | 14.11 / 12.87 | 15.74 / 14.27 |
| 4 | 16.15 / 13.87 | 17.94 / 15.34 |
| 8 | OOM | -- |

Batch 8 fails on **fragmentation, not capacity**: 18.18 GiB allocated with 4.19 GiB
reserved but unallocated, and Windows has no `expandable_segments` to recover. Batch 4's
worst case is a group of four full-length documents, exactly the 16384-token shape
measured above, and it is the largest group the sampler can build from this corpus.

This is what the tensor-parallel memory work actually bought. The layer split's earlier
attempt at grouped batching OOM'd at step 18 with 6.12 GiB reserved but unallocated;
tensor parallelism leaves about 5 GiB more headroom, and longest-first ordering allocates
the largest blocks first, which is the friendly direction for a fragmenting allocator.
`examples/qwen35_sidecar_stage2_batch4.yml` is batch 4 grouped with
`gradient_accumulation_steps` cut 16 -> 4, holding the effective batch at 16 so the loss
stays comparable with the batch-1 run's 0.5329.

### The run: 1.58x, and a loss cost that is not noise

| | batch 1 | grouped batch 4 |
| --- | ---: | ---: |
| `train_runtime` | 1193 s | **756.6 s (1.58x)** |
| `eval_loss` | **0.5329** | 0.5505 (+0.0176) |
| card 0 peak / reserved | 15.00 / 15.99 GiB | 18.44 / 19.99 GiB |
| card 1 peak / reserved | 13.77 / 14.81 GiB | 14.52 / 16.63 GiB |

The speed prediction held: 1.58x measured against 1.61x projected on microbatch compute,
and the epoch beat the projection's implied 1.43x because cutting
`gradient_accumulation_steps` 16 -> 4 also removed three quarters of the per-microbatch
overhead. Memory landed where the probe said plus the trainer's usual ~1.7 GiB
(19.99 GiB reserved against the 22.80 cap), no OOM, and it went straight past step 18,
where the layer split's attempt at grouping died of fragmentation.

**The loss cost is real.** 0.0176 would be inside the 0.017 run-to-run spread recorded
for stage-1 arms, but that spread is the wrong yardstick here: it was measured across
*different memory configurations*, and this project has a much tighter control available.
The layer split and tensor parallelism -- different execution strategies, identical
mathematics -- produced **0.5330 and 0.5329**. When the math is held fixed this pipeline
reproduces `eval_loss` to four decimals, so a sampler change that moves it by 0.0176 is a
trajectory change, not wobble. For scale, that is 39% of the 0.0448 sidecar effect the
whole project exists to measure.

`train_loss` looks *better* under grouping (0.4819 against 0.5167 at the end) and that
comparison is worthless: HF's sampler sorts by length **within** megabatches
(`mega_batch_mult * batch_size`, here 50 x 4 = 200, so about six sawtooth passes over the
epoch), so late steps in every pass are short documents and the running training loss is
confounded with document length. Only `eval_loss`, measured on a fixed set, compares.

What the sampler actually does, since the cost has to come from one of these: it permutes
randomly, cuts megabatches, sorts each descending, and swaps the single longest element
to position 0 "so that an OOM happens sooner rather than later". So grouping changes both
(a) the order examples arrive in and (b) the composition of each optimizer step -- 16
sequences of *similar* length rather than 16 random ones. The first logged gradient norm
is **464 against the baseline's 76.5**, clipped to `max_grad_norm: 1.0`, which is the
densest-possible first batch doing almost nothing useful.

### Calibrating against a reshuffle, which is the control that was missing

Every comparison in this section is between different data orders, so the yardstick has
to be how much `eval_loss` moves when the order changes *for no other reason*. The
0.5330 / 0.5329 pair from the layer split and tensor parallelism does not measure that --
it holds the order fixed and measures execution reproducibility. The right control is the
baseline config with `training_args.seed` 42 -> 43 and `dataset.seed` untouched, so the
train/eval split is identical and only the shuffle differs:

| | `train_runtime` | `eval_loss` | vs baseline |
| --- | ---: | ---: | ---: |
| batch 1, no grouping, seed 42 | 1193 s | 0.5329 | -- |
| batch 1, no grouping, **seed 43** | 1128 s | **0.5324** | **0.0005** |
| batch 1, `group_by_length` | 1131 s | 0.5474 | +0.0145 |
| batch 4, `group_by_length`, accum 4 | 756.6 s | 0.5505 | +0.0176 |
| batch 4, **sortish**, accum 4 | **752.1 s** | **0.5413** | **+0.0084** |

**Order-seed variance is 0.0005.** Every sampler effect above is 17x to 35x that, so all
of them are real. Note also that `train_runtime` is the noisier quantity here -- 1193 s
and 1128 s are the same configuration -- so treat the speedups as ~1.5-1.6x rather than
1.58x to three figures.

### Sortish recovers half the loss at full speed

`distillkit/sortish_sampler.py` groups at the microbatch and shuffles the batch order
(see that module for why HF's sampler does neither, and the two bugs review found in the
first version). It keeps all of the throughput -- 752.1 s against grouped's 756.6 s --
and takes the regression from 0.0176 to **0.0084**, so the descending-length sawtooth was
about half the problem.

Half, not all. Accounting for the rest: batching itself is 0.0031 (batch-1-grouped 0.5474
to batch-4-grouped 0.5505), which leaves roughly 0.005 of ordering effect that shuffling
the batch order did not remove. Two candidates, neither tested: each microbatch is still
length-homogeneous, so sortish restored diversity *between* microbatches within an
optimizer step but not *inside* one; and the deliberately-first longest batch still
arrives at step 0 with an elevated gradient norm (124.5, against the baseline's 76.5 and
grouped's 462). The first would be tested by grouping more loosely and paying padding for
it; the second by dropping the longest-first placement, which costs the OOM-fail-fast
property that placement exists for.

### What this means for the pilot

Sortish is 1.5-1.6x for 0.0084 of `eval_loss`, which is 19% of the 0.0448 sidecar effect
-- better than `group_by_length`'s 39%, still not free. The reshuffle control improves
the case for spending it: at 0.0005 of order-seed variance, a sampler offset is highly
reproducible, so running *every arm* with identical sampling should cancel it in the
sidecar-minus-control difference far more cleanly than a noisy offset would. Running the
pilot at batch 1 remains the option that needs no such argument and stays directly
comparable with the 1M results already recorded.


### What this means for the pilot

Grouped batch 4 is a 1.58x throughput win that costs 0.0176 of `eval_loss`. For
*infrastructure* work that is an easy trade. For the 5M pilot it is not, because the
quantity being measured is 0.0448 and an arm-independent offset of 39% of it eats the
margin that two seeds were supposed to establish. Two defensible ways to spend it:

* **Run the pilot at batch 1** and pay 1.58x in wall time for arms directly comparable
  with the 1M results already recorded.
* **Run every arm grouped**, identically, and treat the offset as a constant that cancels
  in the sidecar-minus-control difference -- which it should, but "should" is exactly the
  kind of assumption the fixed-cost-per-token model above already got wrong once.

The second is cheaper and probably fine; the first is what the existing numbers can be
compared against. Not a decision to make silently either way.


## What the memory work bought, end to end

| | runtime | eval_loss |
| --- | ---: | ---: |
| v1: boundary 7, no folded head, no GQA fix | 1269 s | 0.5347 |
| v2: tap + folded head + GQA fix + boundary 13 | **1263 s** | **0.5330** |

**Throughput-neutral.** The stack bought headroom, not speed, and the isolated 10%
microbatch gain from the GQA fix does not survive a corpus whose median document is 545
tokens rather than 4096. Worth stating plainly because it was twice implied otherwise
during the work.

## Threading, re-tested at the balanced split

The bounded threaded overlap was measured at 8.7% slower, but always at memory-forced
splits where card 1 did 60-73% of the work. With attention fixed, boundary 18 became
reachable in a preflight and the compute balance re-measured at 2047 vs 2015 ms, a 1.98x
ceiling. Threaded there: **5.41 s against serial's 5.20 s** -- still slower. Balance
raises what is available to overlap; it does not change what a design that serializes
backward can take.

Boundary 18 remains impractical anyway: it fits a preflight at 16.82 GiB and the real
trainer peaks at 18.95 and OOMs at step 2 (21.39 GiB allocated, 871 MiB stranded). That
is the third time a preflight has understated the real trainer by 2-4 GiB -- it omits
gradient clipping's `foreach_norm` temporaries, the collator's on-GPU sidecar rows, and
HF's own buffers. **Preflight numbers are not feasibility; only a real run is.**

## NVLink: the driver routes it, and it is not a lever here

Verified rather than assumed. `can_device_access_peer` is true both ways, and NVLink byte
counters rise in step with a cross-device `.to()`, so the layer split has been using the
bridge all along. The application never names an interconnect -- it asks for a peer copy
and the driver routes it over whatever link exists. That is also why llama.cpp shows
NVLink traffic in tensor-split mode: `GGML_CUDA_P2P` enables peer access and its tensor
copies ride the bridge. Its `allreduce.cu` staging through pinned host memory is not a
contradiction; that path is the fallback for machines *without* NVLink.

Scale, though: the boundary payload is one 20 MB activation, 1.49 ms over NVLink against
3.32 ms host-staged -- **0.05% of a ~4000 ms microbatch**. NVLink is decisive for ZeRO-2
or tensor parallelism, which move gigabytes per step, and irrelevant to the layer split.

## Stage 2 runs: 4.3B trainable across two cards (2026-09-07)

**It works.** `examples/qwen35_sidecar_stage2_sharded.yml` completed in 1269 s with the
entire backbone trainable, split across both cards, no NCCL and no process group.

| | value |
| --- | ---: |
| eval_loss @ epoch 0.694 | 0.5709 |
| eval_loss @ epoch 1.0 | **0.5347** |
| train_loss | 1.025 |
| train_runtime | 1269 s (vs 1026 s for the frozen-backbone arm) |
| trainable | 4.298B parameters, 24.15 GiB of weights + grads + AdamW8bit state |

The saved checkpoint was checked rather than assumed: 435 tensors, key set **identical**
to the stage-1 checkpoint's, `tie_word_embeddings` preserved with no duplicated
`lm_head.weight`, and it reloads and produces finite logits. The device map does not
leak into the saved model.

**Caveat on what this run is.** `model:` points at the stock `student-hf`, not at
`runs/sidecar-1m`, so this trained everything jointly from a zero-initialised sidecar
instead of continuing stage 1 -- visible in the final `W_side_proj` norm of 0.165
against stage 1's 1.80. As a baseline it beats the stage-1 arm (0.5347 vs 0.5807), but
the staged curriculum has not actually been run. The config header now says so.

### Four attempts, and what each one taught

It took four tries, and the first three failed at step 4 of 72 for three different
reasons. Worth recording because two of them were diagnosed wrongly first.

**1. accelerate upcasts the entire head to fp32.** `Accelerator.prepare_model` wraps a
mixed-precision forward in `convert_outputs_to_fp32`, which calls `.float()` on every
bf16 tensor the model returns -- 3.79 GiB for `[1, 4096, 248320]` logits, again for the
gradient, and ~1.4 GB more for the 33 hidden states. This is the allocation named in the
1M control arm's traceback too, so it had been the real cause there all along. It also
defeats `sparse_chunk_length` outright: the KL loss chunks precisely so that no
full-vocabulary fp32 tensor exists, and this materialises one before the loss runs.

Removing it (`chunked_ce.keep_bf16_forward_outputs`) was *not* free, and the
equivalence test caught what inspection missed: the sparse divergences took `out_dtype`
from the logits handed to them, so bf16 logits meant differencing 248k-vocabulary
log-probabilities in bf16. `lossfuncs.common.divergence_dtype` restores fp32 inside the
losses, on chunk-sized slices. Because widening bf16 to fp32 is exact, the result is
bit-identical to the blanket upcast -- asserted at rtol=atol=0.

**2. Guessing at memory from a traceback does not work.** Two diagnoses in a row were
wrong (fp32 weights, then absent gradient checkpointing -- the latter inferred from a
missing `use_cache=True is incompatible with gradient checkpointing` warning, which
never fires when the config already has `use_cache` off). A standalone probe peaked at
14.80 GiB where the real run reached 18.17, because a probe does not build what the
Trainer builds.

`optimizers.memory_metrics` now reports per-device peak and reserved bytes plus the
model's actual `is_gradient_checkpointing` through the existing metrics callback, into
the logs and TensorBoard on every run. It immediately showed checkpointing *was* on and
that the problem was never the peak:

| split | card 0 peak | card 0 reserved | card 1 peak | card 1 reserved | died |
| --- | ---: | ---: | ---: | ---: | --- |
| 9 / 23 | 16.86 | **20.75** | 15.74 | 16.29 | card 0, wanting 1024 MiB |
| 6 / 26 | 14.99 | **20.36** | 17.12 | 18.23 | card 1 at 20.37 allocated |
| 7 / 25 | 15.62 | 22.12 | 16.71 | 21.45 | completed |

**3. Reducing live memory does not reclaim a stranded reservation.** Moving three layers
off card 0 dropped its peak by 1.9 GiB and left its reservation within 0.4 GiB of where
it was; the OOM simply moved to card 1. Windows cannot defragment --
`expandable_segments not supported on this platform`, allocator stays `native` -- but
`garbage_collection_threshold` *is* plain native-allocator behaviour and is not blocked.
Setting it to 0.8, raising `max_vram_fraction` from 0.92 to 0.95 (the cap was holding
1.9 GiB idle while runs failed by ~1 GiB), and settling on 7 layers for card 0 is what
got past step 4.

### Still on the edge

Final margins are thin: card 0 reserved 22.12 GiB and card 1 21.45 GiB against 22.80
allowed. The reservation still sits well above the live peak, so it is the garbage
collector holding this together rather than genuine headroom.

The largest remaining waste is known and unaddressed: `output_hidden_states=True`
materialises all 33 student states and accelerate's output hook copies every one to
card 0 -- roughly 1.4 GiB with gradients -- for the **two** anchors that are read. That
is more than the entire current margin. The capture script already solved this shape
with `_AnchorTap`; porting it into the trainer is the next memory item.

## Tensor sharding: the training integration (2026-09-07)

Stage 2 is now wired. `examples/qwen35_sidecar_stage2_sharded.yml` splits the student
across both cards in **one process, with no process group and no NCCL**. That is not a
compromise: `Tensor.to(device)` is differentiable, so autograd already moves activations
forward and gradients back by itself, over NVLink where peer access exists. A collective
library is only needed when the *same* parameter lives on several ranks, which a layer
split never does.

HF Trainer needed no changes at all. It reads `model.hf_device_map`, and on more than one
device sets `is_model_parallel`, `place_model_on_device = False`, and `_n_gpu = 1` -- the
last of which is what keeps `nn.DataParallel` (and its 22.96 GB logits gather) out of the
way. The map is passed straight through `model_kwargs.device_map`.

What did need work is everything that assumed one device:

* **`distillkit/sharding.py`** (new) answers "where did this end up" from the device map.
  `hidden_state_device` is the subtle one: `hidden_states[i]` is decoder layer *i-1*'s
  output, and the final entry is taken after `model.norm`, not from the last layer. An
  off-by-one there builds a projection on the wrong card.
* **The device map says where a tensor was *produced*, not where it is *observed*.**
  `dispatch_model` puts an `AlignDevicesHook(io_same_device=True)` on the root, which
  sends everything the forward returns -- the logits *and the whole hidden-state tuple*
  -- back to the input device. The first version of this work placed projections by the
  map and was caught by the sidecar test: the post-norm anchor is computed on card 1 and
  handed back on card 0. `anchor_device` reports the observed device and is what
  placement uses; `hidden_state_device` remains the map answer.
* **`hsd_mapping.py`** now builds each distillation projection on its anchor's observed
  device rather than all of them on the embedding's by assumption. A projection cannot be
  relocated later without orphaning its optimizer state, so this has to be right at
  construction.
* **`lossfuncs/kl.py`** pulls the sparse signal and mask to the *head's* device. Direction
  matters: the sparse tensors are `[batch, seq, 64]`, the logits are 248,320 wide, so the
  small side crosses.
* **`lossfuncs/hidden_state.py`** aligns each anchor's teacher state and mask to that
  anchor's card, and moves only the resulting scalar between cards.
* **`trainer.py`** reduces the per-loss scalars onto one device before weighting.
* **Tied embeddings are checked, not assumed.** `tie_word_embeddings` is true, so
  `embed_tokens` and `lm_head` are one parameter; a map that separates them would
  fabricate a second copy that trains apart and reconstructs into a checkpoint matching
  neither. `check_tied_embeddings_colocated` reads the *map* (so it fires before anything
  is materialised) and refuses.

The split in the config balances at 6 bytes per trainable parameter (bf16 weight + bf16
grad + AdamW8bit m/v) plus ~5.5 GB of head working set charged to card 0: **9 layers plus
the embedding/head on card 0 (~15.4 GB), 23 layers on card 1 (~15.7 GB)**, both under the
22.08 GB that `max_vram_fraction` allows.

`tests/test_sharding.py` is the gate. Placement is unit-tested without a GPU; the real
check runs a tiny student twice -- once on one card, once split -- and requires the loss
*and every gradient*, backbone and projections alike, to match. A wrong seam here does
not raise; it quietly trains against a misplaced anchor.

Still serial: one microbatch crosses card 0 then card 1, so this buys capacity, not
speed. 1F1B interleaving is a separate change to the training loop.

Known cost, not yet addressed: `output_hidden_states=True` materialises all 33 student
states and the output hook then copies every one of them to card 0 -- about 693 MB at
sequence 4096, plus the matching gradient traffic, for two anchors that are actually
read. The capture script already solved this shape with `_AnchorTap` (a forward hook on
just the anchor modules); porting that into the trainer would remove both the copies and
the retained states.

## Why the control arm OOM'd: fragmentation at the cap, on a platform that cannot defragment

Measured, after one wrong diagnosis recorded below.

A single 4096-token step of the control arm on one card peaks at **15.16 GiB** of live
tensors, against the 22.08 GiB that `max_vram_fraction: 0.92` allows. The two largest
allocations are both the logits: 1.89 GiB for `[1, 4096, 248320]` bf16 out of `lm_head`,
and 1.89 GiB again for its gradient. Nothing in the step is close to the size that
failed.

What failed:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.79 GiB.
GPU 0 has a total capacity of 24.00 GiB of which 1.51 GiB is free. 22.08 GiB allowed;
Of the allocated memory 17.16 GiB is allocated by PyTorch, and 4.04 GiB is reserved
by PyTorch but unallocated.
```

17.16 GiB live plus **4.04 GiB reserved but unallocated** is 21.2 GiB of reservation
against a 22.08 GiB cap, so a 3.79 GiB contiguous request had nowhere to go. The error's
own suggested remedy is not available here:

```
UserWarning: expandable_segments not supported on this platform
```

That is torch 2.11.0+cu128 on Windows, verified directly -- `get_allocator_backend()`
stays `native` and every segment reports `is_expandable: False`. The launcher had been
setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and it was being ignored.

So the run was not too big; it was running at the edge of a cap it could not defragment,
and step 42 lost the coin flip. The sidecar arm, with the same peak, won it.

**The fix is to stop running at the edge: both 1M arms now use the device map.** Split
across the two cards the same step peaks at **10.29 GiB on card 0 and 9.58 GiB on card
1**, and card 0's weights drop from 8.10 GiB to 3.23. See "Tensor sharding".

### A wrong diagnosis, corrected

The first reading of that traceback was that `17.16 GiB` was a 4.27B-parameter model in
fp32 and `3.79 GiB` was `4096 x 248320 x 4` bytes of fp32 logits gradient. The arithmetic
works, and `load_student_model` really did set a dtype only on its flash-attention
branch, so `use_flash_attention: false` looked like it must be loading fp32.

It was not. `student-hf/config.json` carries `dtype: bfloat16`, and transformers 5.x
honours the checkpoint's own dtype by default. Measured: the student loads at **7.96 GiB,
bfloat16 throughout**, before and after the change. The 1M arms were re-run for nothing,
and the byte-identical repeat of the OOM message was the clue -- a real 9 GiB swing in
weights cannot leave the numbers unchanged.

The dtype change was kept anyway, because it is right for a checkpoint whose config does
*not* name a dtype: `load_student_model` now honours `training_args.bf16` / `fp16` when
nothing else has set one. It fixed nothing here.

## Fixed: cached anchors crossed PCIe at double width

`signals.py` used to upcast the cached anchors fp8 -> bf16 **on the CPU, before** the
host-to-device copy, so they crossed PCIe at 2 bytes per dimension instead of 1: 36.9 MB
per microbatch rather than 18.4 MB. The assembly buffer is now `float8_e4m3fn` and the
widening happens on the device -- the same trick the sidecar already uses by shipping raw
IQ4_NL rows and dequantising on the GPU. The finiteness check moved with it:
`float8_e4m3fn` has NaN but no infinities, and NaN survives the widening, so it still
catches the same corruption.

## MTP: reference resolved, implementation pending

The student GGUF carries a complete trained MTP block (`blk.32`, 15 tensors, 241 MB)
that the converter currently discards, because transformers has no MTP implementation
for `qwen3_5` -- its only two mentions are `_keys_to_ignore_on_load_unexpected`.

`llama.cpp/src/models/qwen35.cpp` settles the two conventions that shapes alone cannot,
and both would have been guessed wrong:

* **Concat order** is `ggml_concat(e_norm, h_norm, dim=0)`, i.e. torch
  `cat([enorm(embed), hnorm(hidden)], dim=-1)`.
* **Which hidden state**: the main graph exports `h_nextn` *after* `model.output_norm`
  (qwen35.cpp:206-209), so MTP consumes the **post-norm** trunk output. DeepSeek uses
  the pre-norm state, which is what a reasonable guess would have assumed.

The rest is a stock `Qwen3_5DecoderLayer` (full attention) applied to `eh_proj(concat)`,
then `shared_head_norm`, then the shared `lm_head`. No dedicated MTP embeddings in this
checkpoint, so the main embedding table is reused.

Why it is worth doing: the student has no draft model, so MTP is the missing piece for
self-speculative decoding. And the cache already supports training it without recapture
-- the MTP head at position *t* predicts token *t+2*, which is exactly the teacher's
next-token distribution at *t+1*, one index over.

Measured sidecar cost under speculative decoding, since it was raised as a concern:

| K (draft tokens) | rows/step | cold | warm |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 3.82 ms | 0.002 ms |
| 8 | 128 | 22.88 ms | 0.004 ms |
| 32 | 512 | 96.73 ms | 0.008 ms |

Warm, that is 0.0003-0.002 ms per output token against a ~20-40 ms decode step. Larger
draft chunks make it *cheaper* per token, not more expensive: total rows for N output
tokens is 16N regardless of K, and batching amortises. The risk is residency, not chunk
size -- cold is 1,900x slower.

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
