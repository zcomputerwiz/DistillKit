# Assisted-by: Codex
"""Opt-in diagnostics for the actual trainer; never keep extra tensor storage alive."""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import threading
import time
import weakref
from collections import Counter
from pathlib import Path

import torch


class FrozenBatches:
    """Replay the same complete accumulation cycle, with its original loss weights."""

    def __init__(self, records):
        if not records:
            raise ValueError("fixed-record benchmark needs nonempty records")
        self.records, self.cursor = records, 0
        self.counts = []
        for record in records:
            ids = record["input_ids"]
            weight = record.get("weight", record.get("supervised"))
            count = float(weight[:, :-1].sum()) if weight is not None else ids.numel() - ids.shape[0]
            if count <= 0:
                raise ValueError("fixed-record benchmark needs positive target weights")
            self.counts.append(count)

    def next_targets(self):
        return self.counts[self.cursor % len(self.records)]

    def take(self, remaining):
        if self.next_targets() > remaining:
            return None
        record = self.records[self.cursor % len(self.records)]
        self.cursor += 1
        return record


def record_manifest(records, *, teacher_weight=0.0, unlikelihood_weight=1.0):
    """Hash every input/target/mask once, outside measured steps."""
    rows = []
    for record in records:
        tensors, digest = {}, hashlib.sha256()
        for key, value in sorted(record.items()):
            if isinstance(value, torch.Tensor):
                meta = {"shape": list(value.shape), "dtype": str(value.dtype)}
                digest.update(json.dumps([key, meta], sort_keys=True).encode())
                digest.update(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
                tensors[key] = meta
            else:
                digest.update(json.dumps([key, value], sort_keys=True, default=str).encode())
        ids = record["input_ids"]
        weight = record.get("weight", record.get("supervised"))
        active = torch.ones_like(ids, dtype=torch.bool) if weight is None else weight > 0
        active = active.clone()
        active[:, -1] = False
        extra = record.get("context_kl")
        kl_only, ce_only = bool(record.get("kl_only", False)), bool(record.get("ce_only", False))
        distil = bool(teacher_weight or kl_only) and not ce_only
        negative = record.get("negative")
        context = torch.zeros_like(active) if extra is None else extra > 0
        context = context.clone()
        context[:, -1] = False
        if not distil:
            context.zero_()
        kl = (active if negative is None else active & ~negative) | context
        ce_active = not kl_only and (not distil or teacher_weight < 1)
        negative_rows = 0 if negative is None or not distil else int((negative & active).sum())
        rows.append({"sha256": digest.hexdigest(), "doc_ids": record.get("doc_ids"),
                     "ce_only": ce_only, "kl_only": kl_only, "tensors": tensors,
                     "targets": float(weight[:, :-1].sum()) if weight is not None else int(active.sum()),
                     "original_supervised_rows": int(active.sum()),
                     "ce_rows": int(active.sum()) if ce_active else 0,
                     "kl_rows": int(kl.sum()) if distil else 0,
                     "context_rows": int(context.sum()), "negative_rows": negative_rows,
                     "unlikelihood_objective_rows": negative_rows if kl_only and unlikelihood_weight else 0,
                     "shared_head_rows": int((active | context).sum())})
    return rows


class _Saved:
    __slots__ = ("payload", "account", "keys")

    def __init__(self, payload, account, keys):
        self.payload, self.account, self.keys = payload, account, keys

    def __del__(self):
        self.account.release(self.keys)


class SavedTensorAccount:
    """Logical save traffic and live unique storage in transparent hook payloads.

    Opaque checkpoint holders are counted, not treated as retained activations.
    Envelopes hold exactly the original payload; accounting stores only metadata.
    The storage peak includes parameters saved for backward and is not a device
    allocator peak. Separate parameter/activation totals avoid charging aliases twice.
    """

    def __init__(self, model, *, scope=None):
        self.lock = threading.RLock()
        self.scope = scope or torch.profiler.record_function
        self.parameters = {self.storage(p)[0]: weakref.ref(p)
                           for p in model.parameters() if p.numel()}
        self.live, self.current, self.peak, self.groups = {}, Counter(), Counter(), {}

    @staticmethod
    def storage(tensor):
        storage = tensor.untyped_storage()
        return (str(tensor.device), storage._cdata), storage.nbytes()

    def tensors(self, payload):
        if isinstance(payload, torch.Tensor):
            yield payload
        elif isinstance(payload, (tuple, list)):
            for value in payload:
                yield from self.tensors(value)
        elif isinstance(payload, dict):
            for value in payload.values():
                yield from self.tensors(value)

    def save(self, tensor, payload, hook):
        logical = tensor.numel() * tensor.element_size()
        group = (hook, str(tensor.device), tuple(tensor.shape), str(tensor.dtype))
        retained = {self.storage(t)[0]: self.storage(t)[1]
                    for t in self.tensors(payload) if t.numel()}
        with self.lock:
            row = self.groups.setdefault(group, Counter())
            row["calls"] += 1
            row["input_logical_bytes"] += logical
            row["retained_payload_storage_bytes_total"] += sum(retained.values())
            row["opaque_payloads"] += not isinstance(payload, (torch.Tensor, tuple, list, dict))
            for key, size in retained.items():
                if key not in self.live:
                    reference = self.parameters.get(key)
                    category = "parameter" if reference is not None and reference() is not None else "other"
                    self.live[key] = [0, size, category]
                    self.current[(key[0], category)] += size
                    self.current[(key[0], "all")] += size
                    for kind in (category, "all"):
                        counter = (key[0], kind)
                        self.peak[counter] = max(self.peak[counter], self.current[counter])
                self.live[key][0] += 1
        return tuple(retained)

    def release(self, keys):
        with self.lock:
            for key in keys:
                self.live[key][0] -= 1
                if self.live[key][0] == 0:
                    _, size, category = self.live.pop(key)
                    self.current[(key[0], category)] -= size
                    self.current[(key[0], "all")] -= size

    @contextlib.contextmanager
    def capture(self):
        # Wrap constructors rather than replacing inner checkpoint/offload hooks.
        # Every original pack/unpack sees its original payload unchanged.
        cls, original = torch.autograd.graph.saved_tensors_hooks, torch.autograd.graph.saved_tensors_hooks.__init__
        account = self

        def init(instance, pack_hook, unpack_hook):
            pack, unpack = pack_hook, unpack_hook
            hook = getattr(pack, "__qualname__", type(pack).__name__)
            is_offload = "offload_stream_boundaries" in hook

            def packed(tensor):
                scope = account.scope("saved/peer_pack") if is_offload else contextlib.nullcontext()
                with scope:
                    payload = pack(tensor)
                return _Saved(payload, account, account.save(tensor, payload, hook))

            def unpacked(saved):
                scope = account.scope("saved/peer_unpack") if is_offload else contextlib.nullcontext()
                with scope:
                    return unpack(saved.payload)

            original(instance, packed, unpacked)

        cls.__init__ = init
        try:
            with cls(lambda tensor: tensor.detach(), lambda tensor: tensor):
                yield
        finally:
            cls.__init__ = original

    def report(self):
        rows = [{"hook": key[0], "device": key[1], "shape": list(key[2]),
                 "dtype": key[3], **value} for key, value in self.groups.items()]
        return {"peak_live_unique_storage_bytes": {
                    device: {kind: self.peak[(device, kind)] for kind in ("all", "parameter", "other")}
                    for device in sorted({d for d, _ in self.peak})},
                "remaining_live_unique_storage_bytes": {
                    device: self.current[(device, "all")] for device in sorted({d for d, _ in self.current})},
                "groups": sorted(rows, key=lambda r: -r["input_logical_bytes"]),
                "scope": "Transparent Python saved-hook payloads; opaque holders excluded. Peaks include parameter aliases once; other includes activations and copied parameters. Logical totals are traffic, not peak memory."}


class TrainingProfile:
    """One measured optimizer step per trace/accounting file; no warmup tracing."""

    def __init__(self, directory, model, *, grad_streams=False, backend="kineto"):
        if backend not in ("kineto", "nsys"):
            raise ValueError("profile backend must be kineto or nsys")
        self.directory, self.model = Path(directory), model
        self.directory.mkdir(parents=True, exist_ok=False)
        self.steps, self.grad_streams, self.backend = [], grad_streams, backend
        self.capture_active = False

    def scope(self, label):
        return (torch.cuda.nvtx.range(label) if self.backend == "nsys"
                else torch.profiler.record_function(label))

    def close(self):
        """Stop an external capture after synchronized measured work or on failure."""
        if self.capture_active:
            try:
                torch.cuda.profiler.stop()
            finally:
                self.capture_active = False

    @contextlib.contextmanager
    def capture_range(self, *, last=False):
        if self.backend != "nsys":
            yield
            return
        if not self.capture_active:
            torch.cuda.profiler.start()
            self.capture_active = True
        try:
            yield
        except BaseException:
            self.close()
            raise
        else:
            if last:
                self.close()

    @contextlib.contextmanager
    def grad_metadata(self):
        events, handles, lock = [], [], threading.Lock()
        if self.grad_streams:
            # This API stores hooks on leaf tensors, read lazily by AccumulateGrad.
            # Do not request gradient edges or construct accumulator nodes ourselves.
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue

                def accumulated(param, name=name):
                    with self.scope("grad/accumulate/" + name):
                        event = {"parameter": name, "device": str(param.device),
                                 "thread_id": threading.get_ident(), "time_ns": time.time_ns(),
                                 "cuda_stream": torch.cuda.current_stream(param.device).cuda_stream if param.is_cuda else None}
                        with lock:
                            events.append(event)

                handles.append(parameter.register_post_accumulate_grad_hook(accumulated))
        try:
            yield events
        finally:
            for handle in handles:
                handle.remove()

    @contextlib.contextmanager
    def annotations(self):
        import shared_head
        import training_step

        patches = []

        def wrap(owner, attr, label):
            fn = getattr(owner, attr)

            @functools.wraps(fn)
            def timed(*args, **kwargs):
                with self.scope(label):
                    return fn(*args, **kwargs)

            patches.append((owner, attr, fn))
            setattr(owner, attr, timed)

        wrap(self.model.model, "forward", "train/body")
        for i, layer in enumerate(self.model.model.layers):
            wrap(layer, "forward", "train/layer/%02d" % i)
            for attr in ("self_attn", "linear_attn", "mlp"):
                child = getattr(layer, attr, None)
                if child is not None:
                    wrap(child, "forward", "train/layer/%02d/%s" % (i, attr))
                    if attr == "self_attn":
                        for method in ("_project", "_select_positions", "_attend_chunk"):
                            if hasattr(child, method):
                                wrap(child, method, "train/layer/%02d/%s" % (i, method))
        wrap(training_step, "head_losses", "train/head")
        wrap(shared_head, "_chunk_losses", "train/head/chunk")
        wrap(torch.autograd, "backward", "train/backward")
        try:
            yield
        finally:
            for owner, attr, fn in reversed(patches):
                setattr(owner, attr, fn)

    @contextlib.contextmanager
    def step(self, index, optimizer, *, last=False):
        account = SavedTensorAccount(self.model, scope=self.scope)
        traced = contextlib.nullcontext()
        if self.backend == "kineto":
            from torch.profiler import ProfilerActivity, profile

            if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
                raise RuntimeError("CPU/CUDA timeline requested, but this PyTorch profiler lacks CUDA activity support")
            traced = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=False, profile_memory=True, with_stack=False)
        devices = {p.device for p in self.model.parameters() if p.is_cuda}
        before = {str(d): {"allocated_bytes": torch.cuda.memory_allocated(d),
                           "reserved_bytes": torch.cuda.memory_reserved(d)} for d in devices}
        with self.capture_range(last=last), self.grad_metadata() as gradient_events, \
                self.annotations(), account.capture(), traced as prof:
            original = optimizer.step
            try:
                def update(*args, **kwargs):
                    with self.scope("train/optimizer"):
                        return original(*args, **kwargs)
                optimizer.step = update
                with self.scope("train/step/%d" % index):
                    yield
                for device in devices:
                    torch.cuda.synchronize(device)
            finally:
                optimizer.step = original
        report = account.report()
        report["timeline_backend"] = self.backend
        report["gradient_accumulation_events"] = gradient_events
        report["gradient_stream_scope"] = "Current stream inside the leaf post-accumulation callback, not the producer's stream; correlate CPU ranges and CUDA trace flows/waits. Hooks do not inspect/create gradient edges."
        report["allocator_by_device"] = {
            str(d): {"before": before[str(d)], "after_allocated_bytes": torch.cuda.memory_allocated(d),
                     "after_reserved_bytes": torch.cuda.memory_reserved(d),
                     "peak_allocated_bytes_since_warmup": torch.cuda.max_memory_allocated(d),
                     "peak_reserved_bytes_since_warmup": torch.cuda.max_memory_reserved(d)} for d in devices}
        self.pending = prof, report, index

    def finish(self):
        """Export outside the trainer's synchronized optimizer-step timer."""
        prof, report, index = self.pending
        trace = None
        if self.backend == "kineto":
            trace = str(self.directory / ("step-%04d.trace.json" % index))
            prof.export_chrome_trace(trace)
            events = json.loads(Path(trace).read_text(encoding="utf-8"))["traceEvents"]
            report["cuda_device_events"] = sum(e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") for e in events)
            report["cuda_timeline_complete"] = report["cuda_device_events"] > 0
        else:
            # Nsight owns CUPTI exclusively; validate its report after the process exits.
            report["cuda_timeline_complete"] = None
            report["external_timeline"] = "Nsight Systems report; no simultaneous Kineto capture"
        report["step"], report["trace"] = index, trace
        path = self.directory / ("step-%04d.saved.json" % index)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.steps.append({"step": index, "trace": trace, "saved_tensors": str(path)})
        del self.pending
