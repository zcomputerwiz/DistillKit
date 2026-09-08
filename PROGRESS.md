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

**Test suite:** green — **233 passed** in the CUDA-enabled dev environment (≈40 s).
One case (`test_sharded_step_matches_single_device_step`) skips unless two CUDA devices
are visible.
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
3. 5M-token pilot -- the 1M comparison justifies it. Run at least two seeds per arm:
   the single-arm run-to-run spread is 0.017, about 38% of the measured effect.
4. ~~Stage 2 sharding integration~~ - **done, and run**: 4.298B trainable parameters
   across both cards in 1269 s, eval_loss 0.5347, checkpoint verified. See "Stage 2
   runs". Margins are thin (22.12 / 21.45 GiB reserved against 22.80 allowed).
5. ~~Port `_AnchorTap` into the trainer~~ - **done**: `distillkit/anchor_tap.py`.
6. ~~Run the staged curriculum~~ - **done**: chaining wins, 0.5262 against 0.5347, but
   inside the run-to-run spread. See "The staged curriculum beats training jointly".
7. ~~Threaded microbatch overlap~~ - **done and measured: 8.7% slower**, left opt-in and
   off. See "Threaded microbatch overlap". A real gain needs the explicit stage
   schedule, whose ceiling is bounded by the reachable split rather than the balanced
   one.
8. Give the sidecar its own parameter group at a higher learning rate in stage 2, if
   it should keep adapting rather than freezing at stage 1's value.
9. MTP head - conventions now resolved from llama.cpp; implementation pending.
10. Hybrid tensor parallelism - in progress, see "Hybrid tensor parallelism".

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

## Batching needs length grouping, which already existed

Batch size looked like a free throughput lever (997 -> 1092 tok/s at batch 4, fixed
length) and is a large net loss on this corpus: a batch pads to its longest member, and
documents are median 545 / max 4096 tokens. Simulated over the real length distribution
with HF's own sampler:

| batch | random | grouped |
| --- | --- | --- |
| 2 | 28.8% waste, 747 tok/s | 2.5%, 1024 tok/s |
| 4 | 49.7% waste, 549 tok/s | 3.5%, 1053 tok/s |

An earlier note here concluded trl 0.25.1 had dropped `group_by_length`. It had not --
it was **renamed**. The Trainer still builds `LengthGroupedSampler`, now selected by
`train_sampling_strategy="group_by_length"`, and still reads `length_column_name`. The
offline cache dataset now exposes a `length` column so the sampler does not reconstruct
it by materializing every `input_ids` row.

Two bugs surfaced enabling it, both fixed and worth keeping. The grouped sampler orders
**longest-first**, so a batch of four ~4096-token documents arrives at step 0 rather than
rarely -- useful (it fails fast), but it means `chunked_head` must be on. And
`sparse_chunk_length` counts *positions* while a chunk's logits are
`[batch, positions, vocab]`, so the same setting allocated 970 MiB at batch 4 against
242 MiB at batch 1; it is now read as a row budget at batch 1 and divided by the batch --
the same lesson `chunked_ce` already recorded in the opposite direction.

**Grouped batching is nevertheless off, blocked by the allocator.** Three attempts: OOM
at step 0 (no `chunked_head`), step 2 (position-counted chunk), and step 18 with 1.19 GiB
requested against **6.12 GiB reserved but unallocated**. That last is fragmentation, not
exhaustion: sorting by length gives every step a different shape and Windows has no
`expandable_segments` to recover. The predicted gain was only ~1% anyway -- the padding
saving (+6%) is nearly cancelled by the `chunked_head` recompute (-4.4%) that batching
then forces -- so it is not worth fighting the allocator. The `length` column and the row
budget stay; revisit if something else forces a larger effective batch, where the padding
saving would no longer have to pay for the head.

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
