# Two-GPU training review — 2026-09-20

Reviewed HEAD `a3315065437d8f2ebcdc30c08a2e64068fbdf90a`, including the recent norm compilation, narrow-projection packing, autotuner warmup, and nonblocking-copy commits. Existing CSA2/conversion edits were preserved. No production implementation, checkpoints, or remote repository was changed. Disposable random models and process-local substitutions were used.

## Main result

Regional Inductor compilation of the active gated routing helper is the best measured lead. The default `compiled` norm does not compile `_GatedProjectMean` or the complete route. Compiling that helper improved synchronized training-step throughput substantially, without requiring a full-model graph-break cleanup or a handwritten communication kernel.

Two RTX 3090 24GB GPUs, NV4, bidirectional peer access; Windows, PyTorch 2.11.0+cu128 and triton-windows 3.8.0.post28. Models: 10 layers, attention ratio 3:1, four residual branches, blend=1, vocabulary 16,384, BF16, length 1,024. Each step includes forward, CCE loss, backward, replicated-gradient synchronization, clipping and AdamW8bit. Fixed seed 20260920 and identical random token input per configuration. One optimizer update per measured microbatch, not the trainer's default two-microbatch accumulation. All runs use compiled norms. Initial compilation/autotuning and three additional warmup steps precede eight timed steps; both devices are synchronized. Throughput counts input tokens (B×L), not shifted CE targets (B×(L−1)). These are short performance diagnostics, not trained-quality comparisons.

| Configuration | Step median ms | Input tokens/s | Peak allocated GiB, GPU 0 / 1 |
|---|---:|---:|---:|
| H512 B32, one GPU | 553.92 | 59,156 | 12.21 / 0 |
| H512 B32, current two-GPU, repeat | 617.69 | 53,049 | 9.62 / 3.43 |
| H512 B32, compile entire gated helper | 447.97 | 73,148 | 12.02 / 3.43 |
| H512 B32, separately compiled forward/backward | 420.98 | 77,837 | 9.57 / 3.43 |
| H1024 B8 checkpointed, one GPU | 468.38 | 17,490 | 1.97 / 0 |
| H1024 B8 checkpointed, current two-GPU | 506.23 | 16,182 | 1.57 / 0.79 |
| H1024 B8 checkpointed, compile entire helper | 400.80 | 20,439 | 1.48 / 0.79 |

At H512, whole-helper compilation improves throughput 37.9%; split compilation improves it 46.7%, relative to the repeated two-GPU baseline. At checkpointed H1024, whole-helper compilation improves throughput 26.3%. Baseline H512 measurements drifted (repeat range 555–658 ms; initial median 637 ms). Do not treat differences of a few percent as established. The single-GPU comparisons above use the original helper, so they do not establish that optimized tensor parallelism beats an equally optimized single-GPU implementation.

The isolated nonzero-weight routing check (`gated-project-mean-split.json`) measures 9.60 ms eager, 2.15 ms whole-helper compiled, and 2.35 ms split compiled, including backward. Whole-helper compilation saves 260.25 MiB versus the original 132.25 MiB; split compilation retains 132.25 MiB. This explains the whole-helper activation-memory penalty without checkpointing. Ignore isolated peak-allocation comparisons across these arms: retained result snapshots contaminate them; use the saved-tensor inventory and separate full-model processes instead.

Neither compiled candidate is bit-identical: relative L2 differences are 0.00321 for outputs, 0.00277 for input gradients, 0.00390 for code gradients and 0.00394 for weight gradients. Production adoption needs explicit BF16 tolerances, autocast/FP32-parameter tests, checkpoint recomputation tests, and a bounded paired quality check. The benchmark is not authorization to relax the exact-conversion path.

## Actionable review findings

1. **Communication accounting is fourfold understated.** `distillkit/parallel/collectives.py:30` and `PROGRESS.md:5464` report 640 MiB at H512 B32 L1024. Instrumentation records 20 transfers of 32 MiB in EACH of Replicate.forward, Reduce.forward, Reduce.backward and Replicate.backward: **2,560 MiB total**. Checkpointed H1024 B8 records **1,920 MiB** body traffic because forwards are recomputed. Checkpoint-boundary offload is additional, not included in these counters. Consequently the documented 4.1%/2.6% collective-cost estimates are not established ceilings. Summing payload/link bandwidth is not a measured critical-path fraction either.
2. **The costly routing helper remains eager.** `distillkit/experimental/hyper_connection.py:94` contains four per-branch projection/gate paths and recomputes them in backward. In a separate instrumented H512 step its forward/backward GPU0 stream intervals sum to 41/145 ms. These include dependencies, are not exclusive kernel time, and overlap enclosing route intervals. The end-to-end substitution above is the stronger evidence. Preserve the manual save/recompute contract and compile its two pure arithmetic functions separately as the first implementation candidate.
3. **Compile regions, not the entire Python shell first.** A small four-layer two-GPU forward produced 15 graphs and 14 graph breaks (`dynamo-tp.json`). Sources include `float(self.blend)` at hyper_connection.py:349, the context manager at widened.py:230, and FLA's compiler-disabled SwiGLU boundary reached through parallel/blocks.py:78. FLA SwiGLU is already fused; its graph break is not evidence it needs replacement. Resolve schedule values and hardware probes outside compiled regions; retain explicit boundaries around custom kernels until properly registered. `fullgraph=True` is useful for enforcing each selected region, not as a promise the full trainer can compile unchanged. See [PyTorch's graph-break guidance](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html) (current docs; local measurements used 2.11).
4. **Avoid repeated GPU-to-Python mask decisions.** `parallel/blocks.py:177` and `:214` both evaluate all-ones masks in Python. For genuinely unpadded packed windows, evaluate using the no-mask path or resolve padding metadata once before layer execution, while preserving causal semantics. Speedup remains unmeasured.
5. **Checkpoint offload still uses blocking copies.** `experimental/widened_residual.py:44` and `:49` lack the flags added elsewhere by e4da768. Examine these with producer/consumer stream ordering and buffer lifetime tests. Do not remove the empty saved-tensor recompute barrier at collectives.py:87: it prevents concurrent checkpoint-recompute corruption.
6. **Warmup is shape-specific, not a general autotuner lock.** tp_train.py:95 serializes one configuration's cold backward, consistent with the installed Triton autotuner's mutable `self.nargs`. New shapes, layouts or dtypes can cause fresh tuning. Restore the previous autograd-threading setting rather than unconditionally enabling it if this helper becomes reusable.
7. **Older conclusions use a different workload.** docs/dense_gr.md:223–236's 4.7% routing cost and 17% compile slowdown describe the older route/blend setup, not the active routing path benchmarked here. profile_step.py accepts accumulation without implementing it, defaults blend=0, lacks the compiled norm choice, and does not shard the model. It should not serve as the current TP training benchmark unchanged.

## Custom Triton and P2P

A direct peer-read reduction is technically possible. It removes the intermediate copy-destination write AND the following staged read, not just one local read as the current cost argument states. It cannot remove the NVLink payload and uses SMs instead of only the copy engine. A Python rank loop also does not prove serialized GPU execution: device streams can overlap. Producer events, cross-device waits and source-allocation lifetime must remain correct. See [NVIDIA's multi-GPU guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html).

The isolated Triton peer-sum prototype is present in the script but **was not benchmarked or validated**. No custom-P2P speedup is claimed. The device-kernel profiler could not initialize CUPTI, so GPU traces are empty; CPU traces and CUDA-event intervals cannot establish the communication/computation overlap timeline. Also, comparing raw peer-copy bandwidth to NCCL all-reduce bandwidth does not establish that a collective library has no scheduling benefit. There is no reason to undertake a distributed-backend rewrite on this evidence.

## Exclusions and checks

- `tp-h512-b32-split-gpm.json` is explicitly invalidated: another conversion job occupied GPU0. It was repeated after that job ended as `tp-h512-b32-split-gpm-idle.json`.
- The later H1024 split-compilation attempt was stopped before completion when other jobs were detected. No result is claimed. Those jobs were not stopped.
- Whole-helper H512 first-use compilation overlapped briefly with a second review process, which was stopped before steady measurements. Its cold-start duration is not a clean compile-cost measurement.
- Focused CPU suite: **40 passed, 14 GPU-dependent tests skipped**. Ruff passed for the isolated script. No fresh GPU correctness-suite result is claimed.

## Reproduction

From the repository root, with both GPUs idle, run each command sequentially:

```powershell
.venv/Scripts/python.exe scratch/tp_review_20260920.py --hidden 512 --batch 32 --profile --output scratch/tp-review-20260920/repro-baseline.json
.venv/Scripts/python.exe scratch/tp_review_20260920.py --hidden 512 --batch 32 --compile-gated-mean --output scratch/tp-review-20260920/repro-compiled.json
.venv/Scripts/python.exe scratch/tp_review_20260920.py --hidden 512 --batch 32 --split-compile-gated-mean --output scratch/tp-review-20260920/repro-split.json
.venv/Scripts/python.exe scratch/tp_review_20260920.py --mode route --output scratch/tp-review-20260920/repro-route.json
```

Use `--hidden 1024 --batch 8 --checkpointing` for the larger checkpointed configuration; use `--cards 1` for an unsharded comparison. The direct-peer prototype is available via `--mode peer`, pending validation. Individual JSON files retain all timing samples and measured allocation data.
