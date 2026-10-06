# Assisted-by: Codex
"""Opt-in eager head backward; the body graph is traversed once later."""
import torch
from shared_head import grouped_tail_kl_rows


def backward_head(hidden, head, ids, *, weight=None, topk_ids=None,
                  topk_logprobs=None, kl_weight=None, negative=None, chunk=512,
                  kl_beyond=False, coefficients=(1., 0., 0.)):
    rows, length = ids.shape
    w = torch.zeros(rows, length, device=hidden.device, dtype=torch.float32)
    w[:, :-1] = 1. if weight is None else weight[:, :-1].float()
    if kl_weight is not None:
        kl_weight = kl_weight.clone()
        kl_weight[:, -1] = 0
        if not kl_beyond and bool(((kl_weight > 0) & (w <= 0)).any()):
            raise ValueError("kl_weight must be zero wherever weight is")
    active = w > 0
    if kl_beyond and kl_weight is not None:
        active = active | (kl_weight > 0)
    at = active.nonzero()
    r, t = at[:, 0], at[:, 1]
    h = hidden.detach()[r, t]
    targets, wr = ids[r, t + 1], w[r, t]
    kid = None if topk_ids is None else topk_ids[r, t]
    kval = None if topk_logprobs is None else topk_logprobs[r, t].float()
    kw = None if kl_weight is None else kl_weight[r, t].float()
    neg = None if negative is None else negative[r, t]
    # One host transfer prepares every chunk's membership, instead of scalar
    # GPU-to-host tests in both forward and checkpoint replay for each chunk.
    membership = torch.stack([torch.zeros_like(wr, dtype=torch.bool) if kw is None else kw > 0,
                              torch.zeros_like(wr, dtype=torch.bool) if neg is None else neg]).cpu()
    plans = [(membership[0, b:b + chunk].nonzero().flatten().to(hidden.device),
              membership[1, b:b + chunk].nonzero().flatten().to(hidden.device))
             for b in range(0, len(h), chunk)]
    total = hidden.new_zeros(3, dtype=torch.float32)
    dh = torch.zeros_like(hidden)
    for index, b in enumerate(range(0, len(h), chunk)):
        s = slice(b, b + chunk)
        leaf = h[s].detach().requires_grad_(True)
        logits = (leaf @ head.T).float()
        lse = torch.logsumexp(logits, -1)
        nll = lse - logits.gather(-1, targets[s, None]).squeeze(-1)
        ce = (nll * wr[s]).sum()
        kl = ul = logits.new_zeros(())
        keep, bad = plans[index]
        if keep.numel():
            student = logits.gather(-1, kid[s])[keep] - lse[keep, None]
            kl = (grouped_tail_kl_rows(student, kval[s][keep]) * kw[s][keep]).sum()
        if bad.numel():
            probability = (-nll[bad]).exp()
            ul = (-torch.log1p(-probability.clamp(max=1 - 1e-6)) * wr[s][bad]).sum()
        terms = torch.stack([ce, kl, ul])
        total += terms.detach()
        objective = sum(value * coefficient for value, coefficient in zip(terms, coefficients))
        objective.backward()
        dh[r[s], t[s]] = leaf.grad
        # Release the vocabulary-wide intermediates before the next projection.
        del objective, terms, logits, lse, nll, ce, kl, ul, leaf
    return {"nll": total[0], "weight": wr.sum(), "kl": total[1],
            "unlikelihood": total[2]}, dh


def update_check(model, args):
    """One fresh Kahan optimizer step, identical recipe rates, three controlled arms.

    No checkpoint writes. This checks local update effects, not long-run training.
    """
    import gc
    import contextlib
    import json
    import re
    from pathlib import Path
    from benchmark import apply_liger
    from teacher_kl import CachedTeacher
    from training_step import KahanAdamW8bit, optimizer_step, synchronize
    from distillkit.models.qwen35.csa2 import router_parameters

    torch.set_num_threads(4)
    model.train()
    model.model.gradient_checkpointing = True
    apply_liger(model, model.config)
    if args.selection_tp:
        from distillkit.parallel.model import shard_model
        shard_model(model, ["cuda:0", "cuda:1"], shard_embeddings=False, embedding_device="cuda:1")
    recipe_path = Path(__file__).parent / "train-ctl-context.json"
    recipe = json.loads(recipe_path.read_text())["run_args"]
    if recipe["no_kahan"] or recipe["lr_depth_ramp"] is not None:
        raise ValueError("update diagnostic expects the existing flat Kahan recipe")
    path = Path(__file__).resolve().parents[3] / "teacher-cache-frontier-code-raw"
    teacher = CachedTeacher(path, "train", device="cuda:0", max_length=512, kl_only=[path])
    teacher.pad_blocks = True
    group, width = max(teacher._groups(1, 128, 512), key=lambda row: row[1])
    batch = teacher.read_batch(group, width)
    parameters = dict(model.named_parameters())
    initial = {n: p.detach().cpu() for n, p in parameters.items()}
    router = {id(p) for _, p in router_parameters(model)}
    scales = [(re.compile(spec.rsplit("=", 1)[0]), float(spec.rsplit("=", 1)[1]))
              for spec in recipe["lr_scale"] or []]
    grouped = {}
    for name, p in parameters.items():
        lr = recipe["lr"]
        if id(p) in router:
            lr = recipe["router_lr"] or lr
        elif "_residual." in name:
            lr = recipe["adapter_lr"] or lr
        else:
            for pattern, factor in scales:
                if pattern.search(name):
                    lr *= factor
        grouped.setdefault(lr, []).append(p)

    def run(stream):
        with torch.no_grad():
            for n, p in parameters.items():
                p.copy_(initial[n])
        model.train()
        optimizer = KahanAdamW8bit([{"params": ps, "lr": lr} for lr, ps in grouped.items()],
                                 lr=recipe["lr"], betas=(.9, .95), weight_decay=.1)
        from selection_replay import SelectionReplayCache
        cache = SelectionReplayCache(model) if stream and args.update_replay_cache else contextlib.nullcontext()
        with torch.autograd.set_multithreading_enabled(False), cache:
            metrics = optimizer_step(model, optimizer, [batch], tensor_parallel=args.selection_tp,
                teacher_weight=.5, shared_head=True, head_chunk=64, streaming_head=stream)
        synchronize(model)
        model.eval()
        with torch.no_grad():
            hidden = model.model(input_ids=batch["input_ids"], attention_mask=torch.ones_like(batch["input_ids"]),
                                 use_cache=False).last_hidden_state.to(model.lm_head.weight.device)
            logits = torch.cat([(h @ model.lm_head.weight.T).float().cpu()
                for h in hidden.reshape(-1, hidden.shape[-1]).split(64)])
        updated = {n: p.detach().cpu() for n, p in parameters.items()}
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        gc.collect()
        synchronize(model)
        return metrics, logits, updated

    baseline_metrics, baseline_logits, baseline_weights = run(False)
    results = {}
    for name, stream in [("ordinary_repeat", False), ("streaming", True)]:
        metrics, logits, updated = run(stream)
        delta = (logits - baseline_logits).abs()
        different = (logits.argmax(-1) != baseline_logits.argmax(-1))
        max_weight = 0.
        changed_weights = weight_elements = 0
        for n, w in updated.items():
            max_weight = max(max_weight, float((w.float() - baseline_weights[n].float()).abs().max()))
            changed_weights += int((w != baseline_weights[n]).sum())
            weight_elements += w.numel()
        results[name] = {"training_metrics": metrics, "prediction_flips": int(different.sum()),
                         "prediction_positions": logits.shape[0], "max_abs_logit_difference": float(delta.max()),
                         "mean_abs_logit_difference": float(delta.mean()),
                         "different_weight_elements": changed_weights, "weight_elements": weight_elements,
                         "max_abs_weight_difference": max_weight}
        del updated, logits, delta
    teacher.close()
    payload = {"checkpoint": str(args.checkpoint), "recipe": str(recipe_path),
               "optimizer": "fresh KahanAdamW8bit, betas=(.9,.95), weight_decay=.1, global clip=1",
               "rates": list(grouped), "tensor_parallel": args.selection_tp, "width": width,
               "combined_selection_cache": args.update_replay_cache,
               "doc_ids": batch["doc_ids"], "baseline": baseline_metrics, "comparisons": results,
               "limitation": "one fresh optimizer step on one raw-code prefix, not resumed-state or long-run equivalence"}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(payload, flush=True)
