"""Bounded forward overlap on a single, GPU-sharded Qwen3.5 student.

Backward and the wide head are serialized. This protects shared gradient buffers
and bounds logits memory while another worker runs the backbone forward.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import threading

import torch


def validate_concurrent_model(model):
    devices = sorted({p.device for p in model.parameters()}, key=str)
    if len(devices) != 2 or any(d.type != "cuda" for d in devices):
        raise ValueError("Concurrent training requires parameters resident on exactly two CUDA devices")
    if getattr(model.config, "model_type", None) != "qwen3_5_text":
        raise ValueError("Concurrent training currently supports qwen3_5_text only")
    if getattr(model.config, "use_cache", False):
        raise ValueError("Concurrent training requires model.config.use_cache=False")
    rope = getattr(model.config, "rope_parameters", {}) or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError("Concurrent training requires static default rotary embeddings")
    if getattr(model.config, "attention_dropout", 0):
        raise ValueError("Concurrent training requires zero attention dropout")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p:
            raise ValueError("Concurrent training requires zero dropout")
        hook = getattr(module, "_hf_hook", None)
        hooks = getattr(hook, "hooks", (hook,))
        if any(getattr(h, "offload", False) for h in hooks):
            raise ValueError("Concurrent training does not support CPU/disk offload hooks")
        if getattr(module, "gradient_checkpointing", False):
            options = getattr(getattr(module, "_gradient_checkpointing_func", None), "keywords", {})
            if options.get("use_reentrant", True) or options.get("preserve_rng_state", True):
                raise ValueError("Concurrent checkpoints require use_reentrant=False and preserve_rng_state=False")
    for a, b in ((devices[0], devices[1]), (devices[1], devices[0])):
        if not torch.cuda.can_device_access_peer(a.index, b.index):
            raise ValueError(f"CUDA peer access is unavailable from {a} to {b}")
    return devices


class _OrderedGate:
    """Admits microbatches to the head, and thus to backward, in submission order.

    Backward is serialized here anyway, so constraining *which* order costs no
    overlap -- forwards still run concurrently ahead of the gate. What it buys is
    reproducibility: gradients accumulate into the shared ``.grad`` buffers in
    microbatch order, exactly as the serial path does. Without it the order is
    whichever worker arrives first, and bf16 addition is not associative, so the
    same window produces weights that differ by a few ulp from run to run.
    """

    def __init__(self):
        self._next = 0
        self._condition = threading.Condition()
        self._failed = False

    def acquire(self, index: int) -> None:
        with self._condition:
            while self._next != index and not self._failed:
                self._condition.wait()
            if self._failed:
                raise RuntimeError("Concurrent accumulation window was cancelled")

    def release(self, index: int) -> None:
        with self._condition:
            self._next = max(self._next, index + 1)
            self._condition.notify_all()

    def abort(self) -> None:
        with self._condition:
            self._failed = True
            self._condition.notify_all()


class ConcurrentMicrobatches:
    """Run one complete accumulation window; the caller owns optimizer updates.

    The callback must return a scalar loss and detached logging data, and must not
    update parameters, invoke trainer callbacks, or use an online teacher. Inputs
    and parameters are made ready before worker streams begin. A successful return
    means every backward and CUDA operation has completed on both devices.
    """

    def __init__(self, model):
        self.model = model
        self.devices = validate_concurrent_model(model)
        # Native CUDA allocator blocks belong to streams. Reuse the same two
        # stream sets across windows instead of stranding pools on retired streams.
        self._worker_streams = [
            {d: torch.cuda.Stream(device=d) for d in self.devices} for _ in range(2)
        ]

    def run(self, batches, forward_loss, *, warmup_first=False):
        if not batches:
            raise ValueError("An accumulation window cannot be empty")
        self._sync()
        local = threading.local()
        stream_sets = iter(self._worker_streams)
        stream_lock = threading.Lock()
        head = _OrderedGate()
        aborted = threading.Event()

        def enter_head(_module, _args):
            if not getattr(local, "active", False) or local.held:
                return
            head.acquire(local.index)
            local.held = True
            if aborted.is_set():
                raise RuntimeError("Concurrent accumulation window was cancelled")

        def worker(index, batch):
            if aborted.is_set():
                raise RuntimeError("Concurrent accumulation window was cancelled")
            if not hasattr(local, "streams"):
                with stream_lock:
                    local.streams = next(stream_sets)
            local.active, local.held, local.index = True, False, index
            try:
                with ExitStack() as stack:
                    for d in self.devices:
                        stack.enter_context(torch.cuda.stream(local.streams[d]))
                    stack.enter_context(torch.cuda.device(self.model.get_input_embeddings().weight.device))
                    # The caller owns precision contexts. HF already autocasts
                    # model.forward; autocasting the loss too changes its numerics.
                    loss, logs = forward_loss(batch)
                    # Also protect callbacks that do not execute the output head.
                    enter_head(None, None)
                    scaled = loss / len(batches)
                    scaled.backward()
                    value = scaled.detach().float().item()
                    del loss, scaled
                    # AccumulateGrad can use a stream inherited from an older graph.
                    # Complete all writers before another backward or optimizer step.
                    self._sync()
                    return value, logs
            except BaseException:
                aborted.set()
                head.abort()
                raise
            finally:
                local.active = False
                if local.held:
                    local.held = False
                    head.release(index)

        handle = self.model.get_output_embeddings().register_forward_pre_hook(enter_head)
        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="distill-microbatch") as pool:
                # Prime model-wide lazy initialization before concurrent entry.
                results = [pool.submit(worker, 0, batches[0]).result()] if warmup_first else []
                futures = [
                    pool.submit(worker, index, batch)
                    for index, batch in enumerate(batches[len(results):], start=len(results))
                ]
                results.extend(future.result() for future in futures)
            self._sync()
            return sum(value for value, _ in results), [logs for _, logs in results]
        except BaseException:
            # Executor shutdown has joined every worker before discarding partial grads.
            self._sync()
            self.model.zero_grad(set_to_none=True)
            raise
        finally:
            handle.remove()

    def _sync(self):
        for device in self.devices:
            torch.cuda.synchronize(device)
