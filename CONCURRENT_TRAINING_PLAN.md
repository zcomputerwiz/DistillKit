# Concurrent two-GPU training integration

2026-09-07. Continues Claude's stage-2 sharding and anchor-tap implementation.

## Existing baseline

The serial sharded student, tied embeddings, stage-local projections, chunked loss,
and chained curriculum have already run. Preserve those paths. Claude measured a
near-balanced boundary at layer 18, with reported peaks of 18.23/10.57 GiB. Those
serial measurements guide the new experiment; concurrent peak memory must still be
measured. Individual GPU utilization percentages do not have to sum to 100%.

## First implementation

1. Finish anchor isolation: each tap captures only its entering thread's forward.
   Hooks belonging to another in-flight forward must not retain this graph.
2. Add an opt-in two-worker runner using the existing model forward, one CUDA stream
   per device per worker, and CUDA P2P through the existing device map.
3. Bound the wide head with a semaphore held from head entry through backward and
   device completion. Serialize backward initially so shared `.grad` accumulation
   cannot race between worker streams. Other microbatch forwards can overlap it.
   This is bounded threaded overlap, not a claimed strict 1F1B scheduler.
4. Integrate at accumulation-window boundaries in HF Trainer. Normalize by the
   actual number of microbatches, including an incomplete final window. Join all
   workers before clipping, optimizer update, evaluation or checkpointing. Keep
   callbacks and logging on the main thread. Propagate worker errors and clear a
   partially accumulated window instead of stepping on incomplete gradients.
5. Limit the first path to native, single-process, GPU-only Qwen3.5 with an offline
   teacher and AdamW. Reject distributed wrappers, CPU/disk offload, fp16 scaling,
   stochastic dropout, dynamic rotary updates and checkpoint RNG preservation.
   Non-reentrant deterministic checkpoints must not restore shared RNG state from
   concurrently executing threads.

## Validation and rollout

- Compare serial and concurrent losses, every gradient and one optimizer update on
  a small sharded sidecar model with anchor losses and checkpoint recomputation.
- Exercise uneven accumulation windows, worker exceptions and tap cleanup.
- Keep existing serial tests green; the feature defaults off.
- Provide an isolated stage-2 configuration at boundary 18. Run a short real-cache
  benchmark with identical documents in serial and threaded modes, recording wall
  time and per-device peak/reserved memory. Do not overwrite existing runs.
- Only claim acceleration after this measurement. If backward serialization limits
  gains, use these results to design an explicit stage schedule or gradient-stream
  protocol. NCCL/custom c10d backend work is a separate later tier.

## Outcome (2026-09-07)

Measured on the real cache at boundary 10: threaded steady-state steps take 3.63/3.61 s
against serial's 3.33/3.32 s -- **8.7% slower** -- and reserve about 1 GiB more on card 0
and 2 GiB more on card 1. Correctness held throughout: serial losses and gradients are
reproduced, with bf16 weights differing by a couple of ulp.

This is the branch the plan named: backward serialization limits gains. The feature
stays opt-in and off. See PROGRESS.md, "Threaded microbatch overlap", for the three
measurement mistakes made along the way and for what an explicit stage schedule would
have to do differently.
