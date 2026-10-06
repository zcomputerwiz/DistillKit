# Assisted-by: Codex
"""Opt-in: retain nondifferentiable CSA2 selection per checkpoint frame.

No global bus lookup: two live forwards can replay in either order. Inner attention
checkpoints keep their original behavior. Model files and training defaults are unchanged.
"""
import contextlib
import contextvars
import functools

_frame = contextvars.ContextVar("selection_replay_frame", default=None)


@contextlib.contextmanager
def frame_scope(cache, replay):
    token = _frame.set((cache, replay))
    try:
        yield
    finally:
        _frame.reset(token)


class SelectionReplayCache:
    def __init__(self, model):
        self.model = model
        self.computed = self.reused = 0
        self.patches = []

    def select(self, owner, original, *args):
        active = _frame.get()
        if active is None:
            return original(*args)
        cache, replay = active
        key = id(owner)
        signature = tuple((tuple(t.shape), str(t.dtype), str(t.device)) for t in args)
        if replay:
            if key not in cache or cache[key][0] != signature:
                raise RuntimeError("checkpoint selection replay has no matching forward frame")
            self.reused += 1
            return cache[key][1]
        if key in cache:
            raise RuntimeError("selector ran twice in one forward checkpoint frame")
        result = original(*args)
        if any(t.requires_grad for t in result):
            raise RuntimeError("selection cache requires nondifferentiable outputs")
        cache[key] = signature, result
        self.computed += 1
        return result

    def __enter__(self):
        from distillkit.models.qwen35 import widened
        from distillkit.models.qwen35.csa2 import Qwen35SparseLatentAttention

        if not self.model.model.gradient_checkpointing or not self.model.training:
            raise ValueError("selection replay needs checkpointed training")
        modules = [m for m in self.model.modules() if isinstance(m, Qwen35SparseLatentAttention)]
        if not modules or any(m.mode != "full" for m in modules):
            raise ValueError("selection replay supports Full layers only")
        original_checkpoint = widened.checkpoint

        def checkpoint(function, *args, **kwargs):
            if kwargs.get("use_reentrant") is not False or "context_fn" in kwargs:
                raise ValueError("selection replay needs the existing non-reentrant outer checkpoint")
            cache = {}
            kwargs["context_fn"] = lambda: (frame_scope(cache, False), frame_scope(cache, True))
            return original_checkpoint(function, *args, **kwargs)

        self.patches.append((widened, "checkpoint", original_checkpoint))
        widened.checkpoint = checkpoint
        for module in modules:
            original = module._select_positions

            @functools.wraps(original)
            def select(*args, owner=module, original=original):
                return self.select(owner, original, *args)

            self.patches.append((module, "_select_positions", original))
            module._select_positions = select
        return self

    def __exit__(self, *exc):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()

    def report(self):
        return {"computed": self.computed, "reused": self.reused,
                "ownership": "outer non-reentrant checkpoint frame; no global bus lookup"}


def gradient_check(model, args):
    """Actual u50 BF16 body/head gradients on one frozen raw-code teacher prefix."""
    import json
    from pathlib import Path
    import torch
    from benchmark import apply_liger
    from teacher_kl import CachedTeacher
    from training_step import backward_step, synchronize

    torch.set_num_threads(4)
    model.train()
    model.model.gradient_checkpointing = True
    swapped = apply_liger(model, model.config)
    if args.selection_tp:
        from distillkit.parallel.model import shard_model
        shard_model(model, ["cuda:0", "cuda:1"], shard_embeddings=False, embedding_device="cuda:1")
    path = Path(__file__).resolve().parents[3] / "teacher-cache-frontier-code-raw"
    teacher = CachedTeacher(path, "train", device="cuda:0", max_length=512, kl_only=[path])
    teacher.pad_blocks = True
    group, width = max(teacher._groups(1, 128, 512), key=lambda row: row[1])
    batch = teacher.read_batch(group, width)

    streaming = getattr(args, "streaming_head_check", False)
    predictions = []
    def record_predictions(module, inputs, output):
        with torch.no_grad():
            hidden = output.last_hidden_state.to(model.lm_head.weight.device)
            predictions.append(torch.cat([(h @ model.lm_head.weight.T).argmax(-1).cpu()
                for h in hidden.reshape(-1, hidden.shape[-1]).split(64)]))

    def run(cache=False, stream=False):
        model.zero_grad(set_to_none=True)
        diagnostic = SelectionReplayCache(model) if cache else contextlib.nullcontext()
        hook = model.model.register_forward_hook(record_predictions)
        try:
            with torch.autograd.set_multithreading_enabled(False), diagnostic:
                metrics = backward_step(model, [batch], teacher_weight=0.5, shared_head=True,
                                        head_chunk=64, streaming_head=stream)
        finally:
            hook.remove()
        if args.selection_tp:
            from distillkit.parallel.sync import sync_replicated_gradients
            sync_replicated_gradients(model)
        synchronize(model)
        selections = {str(i): a.self_attn.last_selection[0].masked_fill(
            ~a.self_attn.last_selection[1], -1).sort(-1).values.cpu()
            for i, a in enumerate(model.model.layers) if hasattr(getattr(a, "self_attn", None), "last_selection")}
        return metrics, selections, diagnostic.report() if cache else None

    baseline, expected_sets, _ = run()
    expected = {n: None if p.grad is None else p.grad.detach().cpu() for n, p in model.named_parameters()}
    repeat_control = getattr(args, "selection_repeat_control", False)
    actual, actual_sets, stats = run(not repeat_control and not streaming, streaming and not repeat_control)
    errors = {}
    for name, parameter in model.named_parameters():
        wanted = expected.pop(name)
        if wanted is None or parameter.grad is None:
            if (wanted is None) != (parameter.grad is None):
                raise RuntimeError("gradient presence changed: " + name)
            errors[name] = {"gradient": None}
            continue
        got = parameter.grad.detach().cpu().reshape(-1)
        wanted = wanted.reshape(-1)
        aa = bb = ab = dd = 0.
        maximum = 0.
        exact = True
        flips = elements = strong_flips = strong_elements = 0
        flip_energy = 0.
        threshold = wanted.float().square().mean().sqrt().item() * 1e-3
        for start in range(0, got.numel(), 1 << 20):
            a, b = got[start:start + (1 << 20)].float(), wanted[start:start + (1 << 20)].float()
            delta = a - b
            aa += float(a.square().sum(dtype=torch.float64))
            bb += float(b.square().sum(dtype=torch.float64))
            ab += float((a * b).sum(dtype=torch.float64))
            dd += float(delta.square().sum(dtype=torch.float64))
            maximum = max(maximum, float(delta.abs().max()))
            exact &= bool(torch.equal(a, b))
            eligible = (a != 0) & (b != 0)
            changed = eligible & ((a > 0) != (b > 0))
            strong = eligible & (b.abs() > threshold)
            flips += int(changed.sum()); elements += int(eligible.sum())
            strong_flips += int((changed & strong).sum()); strong_elements += int(strong.sum())
            flip_energy += float(b[changed].square().sum(dtype=torch.float64))
        errors[name] = {"bitwise_equal": exact, "max_abs": maximum,
                        "relative_l2": (dd / max(bb, 1e-30)) ** 0.5,
                        "cosine": ab / (aa * bb) ** 0.5 if aa and bb else (1. if aa == bb else 0.),
                        "sign_flips": flips, "nonzero_pairs": elements,
                        "strong_sign_flips": strong_flips, "strong_nonzero_pairs": strong_elements,
                        "strong_threshold": threshold, "flipped_reference_energy_fraction": flip_energy / max(bb, 1e-30)}
    teacher.close()
    payload = {"checkpoint": str(args.checkpoint), "tensor_parallel": args.selection_tp,
               "repeat_control": repeat_control,
               "streaming_head": streaming,
               "liger": swapped, "doc_ids": batch["doc_ids"], "width": width,
               "baseline": baseline, "cached": actual, "cache": stats, "gradients": errors,
               "selected_sets_equal": {i: torch.equal(s, actual_sets[i]) for i, s in expected_sets.items()},
               "all_gradients_bitwise_equal": all(e.get("bitwise_equal", True) for e in errors.values())}
    payload["prediction_flips"] = int((predictions[0] != predictions[1]).sum())
    payload["prediction_positions"] = predictions[0].numel()
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print({k: v for k, v in payload.items() if k != "gradients"}, flush=True)
