# NVLink GPU parallelism: findings and implementation handoff

Date: 2026-09-07. Target: DistillKit stage 2, two RTX 3090 GPUs, native Windows, GPU-resident trainable parameters, gradients and optimizer state. The desired outcome includes concurrent GPU computation. A serial layer split alone does not satisfy that outcome.

## Decision

**CUDA P2P over NVLink works on this machine, including from the installed PyTorch. NCCL 2.31.2 builds on Windows and completes single-process all-reduces through P2P/direct pointers. After fixing a Windows logging bug, all nine Python/ctypes reduction checks passed, reaching approximately 37.5 GB/s for 64 MiB reductions.** There is no need to forward-port the entire linked 2.29.7 patch merely to obtain Windows build support: the current release already contains Windows implementations.

NCCL transport availability and PyTorch distributed integration are separate milestones. The existing `torch 2.11.0+cu128` still reports `torch.distributed.is_nccl_available() == False`. Adding this DLL does not add `ProcessGroupNCCL` to that wheel. The recommended first training implementation is a **single-process, two-GPU scheduled pipeline using native CUDA peer copies**, with AdamW8bit state on each parameter's owning GPU. Single-process tensor parallelism is another viable route, with more model-specific implementation work. Both avoid a PyTorch rebuild and CPU optimizer offload.

## Sources and versions inspected

- [NVIDIA NCCL 2.31.2-1 release](https://github.com/NVIDIA/nccl/releases/tag/v2.31.2-1), commit `7b83616df3ae082a1f32bb74c27458bfe8153a13`. Retrieved from NVIDIA's Git repository and cross-checked against its raw release source. Its [CMake configuration](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/CMakeLists.txt) and `src/os/` already contain Windows support.
- [User-linked Windows port](https://github.com/NVIDIA/nccl/commit/c5a10bce7fcf833d454b95486effee142d5b48c3), titled NCCL Windows v2.29.7-1. The [SystemPanic Windows branch](https://github.com/SystemPanic/nccl-windows/tree/nccl-windows) was cloned at `f22ac6ea89e5c5d38f2b127e9df0760422fe216b`; its code matches the linked port apart from later documentation/metadata changes. This is older than the current NVIDIA release.
- User ZIP: `D:\DeepThought\Projects\HybridModel\llama.cpp-master.zip`.
- User checkout: `D:\DeepThought\Projects\Claude\llama.cpp`, HEAD `6a1a922` (`metal : fix memory leak in early return (#28399)`). Working tree was clean. The inspected CUDA all-reduce and build files match the ZIP after newline normalization; `ggml-cuda.cu` differs only by one blank line in HIP device-name parsing.
- Installed PyTorch, Transformers 5.16.1, Accelerate and DistillKit source, read without changing the training environment.

## What was verified locally

### Native PyTorch P2P, autograd and GPU optimizer state

Standalone script: `p2p_training_probe.py`; results: `p2p_training_probe.json`; complete output: `p2p-probe.log`.

| Check | Result |
| --- | --- |
| CUDA peer accessibility | True in both directions |
| NVIDIA topology | `NV4` between GPU 0 and GPU 1 |
| Link status | Four active links per GPU, each reporting 14.062 GB/s |
| 64 MiB peer-copy loop | Latest run: 38.04 GB/s 0-to-1; 48.18 GB/s 1-to-0 |
| Actual NVLink traffic | GPU 0 transmit and receive counters each increased by 6,881,280 KiB, exactly the 105 copies of 64 MiB per direction, including warmup |
| Tensor-sharded FP32 MLP forward | Matched unsharded reference within tolerance; maximum absolute difference 0.00006103515625 |
| Input and both weight gradients | Matched reference; maximum absolute differences approximately 4.29e-6, 7.63e-6 and 3.81e-6 |
| AdamW8bit | One update completed; every tensor in optimizer state was on its parameter's GPU |
| Independent FP16 matrix multiplications | Median sequential time 0.3447 s versus concurrently queued 0.1736 s, about 1.99x |

These are feasibility measurements, not a full-model throughput forecast. Copy timings include Python submission and synchronization overhead and vary with clocks/warmup. The 56.25 GB/s sum of advertised link rates is not a measured sustained rate. The matrix multiplication comparison demonstrates available compute concurrency; it does not establish twofold end-to-end training speed.

PyTorch emitted an `AccumulateGrad` stream-mismatch warning during the small cross-device backward. Numerical comparisons passed, but stream ownership and CUDA graph behavior need explicit validation in a production implementation. This probe does not establish CUDA graph capture support or long-run optimizer/checkpoint correctness.

### Current NCCL build and native runtime

Built with VS 2022 Build Tools / MSVC 19.44, CUDA Toolkit 13.3.73, CMake and Ninja, targeting `sm_86`. No package was installed into DistillKit's venv. The CUDA 13.3 runtime DLL resides under the toolkit's **`bin\x64`**, which must be included in the probe's DLL search path.

The completed diagnostic build uses:

```text
-DCMAKE_CUDA_ARCHITECTURES=86
-DONLY_FUNCS=AllReduce Sum (f32|f16|bf16) (RING|TREE) *
```

This is a restricted diagnostic library, not a general-purpose NCCL installation. Use a fresh build directory without `ONLY_FUNCS` for production and validate the additional collectives needed by the chosen training strategy. Earlier broad builds were stopped to focus the runtime investigation; they were not reported as successful full builds.

The unmodified release's native CUDA executable completed `1 + 2 = 3` on both GPUs and exited successfully. Its logs selected `via P2P/direct pointer` in both directions with `NCCL_P2P_LEVEL=NVL`. Bootstrap/control sockets still appear in the log; that does not mean tensor payloads used sockets. Ordinary NVLink P2P does not require NVLink Switch/NVLS support.

`cudaIpcGetMemHandle` also succeeded in a separate local CUDA 13.3 probe. This establishes handle creation only; cross-process import and training remain unverified. A blanket claim that all CUDA IPC is unavailable on Windows would be inaccurate for this machine.

### A real Windows NCCL bug found and fixed

With `NCCL_DEBUG_FILE` set, the unmodified release aborted during communicator initialization with Windows exit code `0xc0000409` in `ucrtbase.dll`. Reproduced in both a Python/ctypes caller and an independent native CUDA executable. The same native executable passed without `NCCL_DEBUG_FILE`.

Root cause in `src/debug.cc`: the Windows branch called `setvbuf(file, NULL, _IOLBF, 0)`. Microsoft's CRT treats `_IOLBF` as full buffering and rejects a zero buffer size. Changed that branch to `_IONBF`, which accepts and ignores the zero size and provides the intended immediate logging. [Microsoft CRT specification](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/setvbuf?view=msvc-170).

Patch: `nccl-2.31.2-windows-debug-file.patch`. Only this one source line was changed in the isolated current-release checkout. No port-wide patch replay, global installation, driver change, or PyTorch rebuild was performed.

After rebuilding, the original Python/ctypes caller completed with `NCCL_DEBUG_FILE` enabled, exited 0, and produced a populated NCCL log. All nine sum reductions passed:

| Payload per GPU | FP32 GB/s | FP16 GB/s | BF16 GB/s |
| --- | ---: | ---: | ---: |
| 64 KiB | 1.69 | 1.60 | 1.59 |
| 8 MiB | 32.59 | 32.90 | 32.99 |
| 64 MiB | 37.35 | 37.59 | 37.54 |

Rates are payload bytes per GPU divided by elapsed time for the grouped collective (including Python submission), not aggregate link traffic. Each case used five warmup and twenty timed out-of-place reductions. All elements on both GPUs equaled the expected sum. Logs selected `P2P/direct pointer` in both directions, and NVLink counters increased during the run. This also verifies that this CUDA 13.3-built DLL can operate on the installed CUDA 12.8 PyTorch wheel's buffers/streams in this narrow test. It does not establish general mixed-runtime compatibility, PyTorch distributed-backend integration, NCCL autograd support, or multiprocess correctness.

Evidence: `nccl-patched-probe.json`, `nccl-patched-probe.log`, `nccl-patched-probe.nccl.log`. `nccl_probe.py` calls the C API directly; it is not an installed training backend.

## What the supplied llama.cpp answers

The relevant paths are present in **both** supplied sources:

- `ggml/src/ggml-cuda/ggml-cuda.cu`: `GGML_CUDA_P2P` enables `cudaDeviceEnablePeerAccess`; copies can use `cudaMemcpyPeerAsync`. The VMM allocator also grants access to peer GPUs when required.
- The same file initializes NCCL with `ncclCommInitAll`, one communicator per GPU in the same process, and groups its all-reduce calls. This is a concrete example of using NCCL without PyTorch's distributed backend. NVIDIA documents that [single-process communicator model](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html).
- `ggml/src/ggml-cuda/allreduce.cu`: the built-in two-GPU reduction exchanges data through **pinned host buffers**, using either kernel accesses or D2H/H2D copy engines. Enabling `GGML_CUDA_P2P` does not rewrite this internal algorithm into an NVLink reduction.
- Windows defaults `GGML_CUDA_ALLREDUCE` to the internal implementation. `GGML_CUDA_ALLREDUCE=none` delegates to the meta-backend fallback; `nccl` requests NCCL if compiled in. The actual route must be checked in logs/traffic counters.
- The second checkout's `build/CMakeCache.txt` had `GGML_CUDA_NCCL=ON`, but `NCCL_INCLUDE_DIR` and `NCCL_LIBRARY` were both `NOTFOUND`. That cache does not establish a working NCCL-linked binary.

Thus the user is right that CUDA P2P can work on these Windows GPUs. Layer transfers and tensor-reduction transport must nevertheless be identified separately. The second checkout does not contain a distinct NVLink all-reduce modification in the inspected code.

## Training implementation choices

| Route | GPU concurrency and capacity | Remaining work |
| --- | --- | --- |
| Single-process scheduled pipeline + native CUDA P2P | Shards layers and state; overlaps different microbatches on both GPUs | Recommended first: explicit pipeline schedule, stage-local losses, correct activation/gradient transfers |
| Single-process tensor parallelism + native P2P reductions or NCCL DLL | Both GPUs execute shards of the same layers; shards associated optimizer state | Model-specific sharding, autograd collectives, replicated-gradient rules, checkpoint reconstruction |
| Windows PyTorch NCCL backend + FSDP/ZeRO/TP | Potential standard distributed stack; GPU-only sharding can address capacity | Separate backend integration/build; validate all-gather, reduce-scatter, send/recv, error handling and actual P2P across processes |
| Native Linux + standard CUDA/NCCL PyTorch | Established distributed route, GPU-only sharding available | Separate OS/environment and project compatibility validation; not installed or switched here |
| Plain DDP | Concurrent replicas, but each GPU stores its own parameters/gradients/optimizer state | Does not solve the stated per-GPU memory problem by itself |
| Plain inference `device_map` | Adds capacity; ordinary sequential stage execution | Insufficient evidence of training correctness or the requested concurrent compute |

PyTorch 2.11's [build configuration](https://github.com/pytorch/pytorch/blob/v2.11.0/CMakeLists.txt#L259) still gates `USE_NCCL` on UNIX. The installed distributed module imports `ProcessGroupNCCL` from compiled extension code and marks NCCL unavailable when that import fails. Environment variables or copying an NCCL DLL beside the wheel cannot supply the missing backend. The installed `torch.distributed.pipelining` stage implementation also uses distributed send/receive operations; it is not automatically a single-process peer-copy scheduler.

## Implementation hints for the current coder

1. **Use an explicit training schedule.** For a two-stage pipeline, overlap GPU 0 processing microbatch k+1 with GPU 1 processing microbatch k. Start with a correctness-first microbatch schedule, then implement 1F1B to bound stored activations. CUDA streams/events should order copies and compute; CPU threads/queues can coordinate metadata without staging tensor payloads on the host. Keep sends and their source buffers alive until consumers finish. A layer split plus ordinary serial forward is not this schedule.
2. **Split by measured memory and time.** Parameters, gradients and optimizer state are only part of the budget. Include saved activations, loss temporaries, logit gradients, copy buffers and allocator reserve. The approximately 25.5 GB total estimate is a persistent-state estimate, not a guaranteed fit or a balanced stage assignment. Keep the VRAM guard and check each GPU independently for WDDM spill.
3. **Preserve tied embeddings.** `student-hf/config.json` has `tie_word_embeddings=true`. Placing input embeddings and `lm_head` on different GPUs cannot silently turn one shared parameter into two independently trained parameters. Choose shared ownership with appropriate activation transfer, or explicitly synchronized replicas and checkpoint reconstruction. Account for the head's loss memory when balancing stages.
4. **Place sidecar and hidden-state losses explicitly.** The sidecar belongs with its consuming layer. In `distillkit/lossfuncs/hidden_state.py`, projection casting, teacher casting, the mask and accumulated loss currently assume co-location. Put each projection/cache anchor/mask with its student anchor, compute that loss locally, and move only scalar loss contributions when possible. Do not gather every hidden state or full logits to the input GPU merely to retain the old trainer interface.
5. **Keep chunked CE/distillation losses.** The large head/logit-gradient allocation remains relevant with two cards. Verify chunked-loss behavior at the selected head device. Avoid inference dispatch hooks that unexpectedly move the complete output back to the input device.
6. **Select the actual GPU optimizer deliberately.** For the AdamW8bit route, start a separate full-backbone stage with `freeze_backbone=false`, no mid-run unfreeze callback, and an optimizer configuration that constructs bitsandbytes AdamW8bit. The current `hybrid` strategy constructs the composite Muon/AdamW optimizer, and the dynamic unfreeze path constructs ordinary torch AdamW. Do not assume setting a generic HF optimizer field overrides those paths. Verify every state tensor's device after the first update and after checkpoint resume. Avoid paged/offloaded optimizer variants for this requirement.
7. **For tensor parallelism, shard the actual architecture.** MLP gate/up projections can split output channels and down projections can split input channels, reducing partial outputs over NVLink. Attention needs consistent Q/K/V/head partitioning. Qwen3.5's GatedDeltaNet has coupled Q/K/V projections, head counts, convolution state and gating. Do not treat it as an ordinary attention block. The installed Transformers TP plan gathers several linear-attention projection outputs, so it does not automatically divide all of that block's state or computation by two.
8. **Define autograd collective semantics.** A DLL call over `tensor.data_ptr()` is invisible to autograd. Supply correct backward rules, device/stream ordering, buffer lifetime and communicator cleanup. For a mathematical pair of outputs `y0=y1=x0+x1`, both input gradients are `g0+g1`; replicated logical losses in a TP architecture can require different placement/reduction conventions to avoid double counting. Establish the convention before implementing wrappers.
9. **Validate the real training step before a long run.** Compare logits, combined losses, gradients and one optimizer update against a small unsharded reference; include the sidecar and anchor losses. Then verify checkpoint resume, parameter ties, GPU-resident optimizer state and actual overlapping GPU work. Use NVLink counter deltas or a profiler alongside NCCL transport logs. GPU utilization alone does not establish useful parallelism or absence of host staging.

The CPU remains responsible for data preparation and the existing frozen table/cache storage. The no-offload requirement here concerns trainable model/gradient/optimizer state and GPU-to-GPU payload movement; it does not require moving the entire existing host-side dataset and table into VRAM.

## Artifacts and reproduction

The complete isolated source/build work is under:

`C:\Users\Owner\Documents\Codex\2026-09-06\hi-chatgpt-claude-left-off-doing\work\nvlink-research`

A handoff copy of the report, source patch, scripts, measured results, logs and diagnostic DLL is under:

`C:\Users\Owner\Documents\Codex\2026-09-06\hi-chatgpt-claude-left-off-doing\outputs\nvlink-research`

From that output directory, using the existing project venv:

```powershell
& 'D:\DeepThought\Projects\HybridModel\DistillKit\.venv\Scripts\python.exe' p2p_training_probe.py
& 'D:\DeepThought\Projects\HybridModel\DistillKit\.venv\Scripts\python.exe' nccl_probe.py diagnostic-nccl\nccl.dll rerun-nccl
```

The NCCL probe sets environment variables only inside its own process. For diagnostic reruns, use an external timeout as done in this investigation; a library/driver hang can otherwise block CUDA synchronization indefinitely. The scripts use synthetic tensors and do not load model checkpoints. They should be run when the GPUs are free of training workloads.

To build a **complete** library, start a VS x64 developer command prompt, create a fresh checkout/build directory, apply the saved patch, and use the following. This recipe is provided for the next integration step; the completed library in the handoff is the restricted diagnostic build described above.

```bat
git clone --depth 1 --branch v2.31.2-1 https://github.com/NVIDIA/nccl.git nccl-2.31.2
git -C nccl-2.31.2 apply <absolute-path-to-nccl-2.31.2-windows-debug-file.patch>
cmake -S nccl-2.31.2 -B build-full -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_INSTALL_PREFIX=install-full -DPython3_EXECUTABLE=D:/DeepThought/Projects/HybridModel/DistillKit/.venv/Scripts/python.exe
cmake --build build-full --parallel 4 --target install
```

Do not copy the restricted diagnostic DLL into a production package or expect unsupported collectives to work. For llama.cpp, a future isolated build can point `NCCL_INCLUDE_DIR` at the full install's `include` and `NCCL_LIBRARY` at its `lib/nccl.lib`, ensure `nccl.dll` and CUDA runtime dependencies are discoverable, and request `GGML_CUDA_ALLREDUCE=nccl`. Verify logs and counters for the exact executable/configuration. Neither supplied llama.cpp checkout was rebuilt here.

## Scope of this work

No DistillKit or llama.cpp implementation files were modified. No training run, model download, dependency replacement, driver update, global DLL installation, commit or push was performed. Short synthetic GPU probes were run after checking that no project training Python process was active. The isolated NCCL checkout, patch, diagnostic library, scripts and logs are retained for reproduction.
