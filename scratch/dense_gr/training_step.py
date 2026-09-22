"""The optimizer step shared by the dense-GR trainer, benchmark and tests."""
from __future__ import annotations

import contextlib

import torch

from teacher_kl import accumulation_shares, grouped_tail_kl, scored_mask


def check_optimizer(model, optimizer):
    current = {id(p) for p in model.parameters()}
    held = [p for group in optimizer.param_groups for p in group["params"]]
    if len({id(p) for p in held}) != len(held):
        raise ValueError("optimizer contains duplicate parameters")
    if any(id(p) not in current for p in held):
        raise ValueError("optimizer contains stale parameters; construct it after sharding")
    held_ids = {id(p) for p in held}
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and id(p) not in held_ids]
    if missing:
        raise ValueError("trainable parameters missing from optimizer: " + ", ".join(missing[:6]))


def causal_ce(model, hidden, ids):
    from cut_cross_entropy import linear_cross_entropy

    device = model.lm_head.weight.device
    return linear_cross_entropy(hidden.to(device), model.lm_head.weight,
                                ids.to(device), shift=1, reduction="mean").to(hidden.device)


def backward_step(model, records, *, teacher_weight=0.0, indexer_weight=1.0,
                  sparse_stage=None, kl_chunk=256, ce=causal_ce):
    """Accumulate means over the identical B*(L-1) positions for all three terms.

    Does not clear gradients or update weights, so warm-up exercises this exact path.
    The injected CE callable is only for CPU correctness tests; production uses CCE.
    """
    from distillkit.models.qwen35.csa2 import isolated_indexer, recorded_attention

    if not 0 <= teacher_weight <= 1 or indexer_weight < 0:
        raise ValueError("invalid objective weights")
    counts, total = accumulation_shares([r["input_ids"] for r in records])
    if not records or any(n <= 0 for n in counts):
        raise ValueError("an optimizer step needs nonempty causal targets")
    result = dict(loss=0.0, teacher_kl=0.0, indexer=0.0, objective=0.0, targets=sum(counts))
    for record, count in zip(records, counts):
        ids = record["input_ids"]
        mask = scored_mask(ids.shape[1], ids.device, ids.shape[0])
        handles = []
        context = contextlib.nullcontext()
        if sparse_stage is not None:
            from indexer_kl import indexer_loss, watch

            seen, handles = watch(model, sparse_stage)
            context = contextlib.ExitStack()
        try:
            with context:
                if sparse_stage is not None:
                    context.enter_context(recorded_attention(model))
                    context.enter_context(isolated_indexer(model))
                hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     use_cache=False).last_hidden_state
                for handle in handles:
                    handle.remove()
                language = ce(model, hidden, ids)
                objective = language
                carried = language.new_zeros(())
                aligned = language.new_zeros(())
                if teacher_weight:
                    if "topk_ids" not in record:
                        raise ValueError("teacher weight requires cached targets")
                    where = model.lm_head.weight.device
                    carried = grouped_tail_kl(
                        hidden.to(where), model.lm_head, record["topk_ids"].to(where),
                        record["topk_logprobs"].to(where), mask.to(where),
                        chunk_length=kl_chunk).to(hidden.device) / count
                    objective = (1 - teacher_weight) * language + teacher_weight * carried
                if sparse_stage is not None:
                    targets = {i: a.last_attention for i, a in sparse_stage}
                    chosen = {i: a.last_allowed for i, a in sparse_stage}
                    borrowed = {i: a.bus.require_latent(a.latent_donor, i)
                                for i, a in sparse_stage}
                    aligned = indexer_loss(model, sparse_stage, seen, targets, borrowed,
                                           selected=chosen, query_mask=mask)
                    objective = objective + indexer_weight * aligned
            share = count / total
            (objective * share).backward()
            for name, value in (("loss", language), ("teacher_kl", carried),
                                ("indexer", aligned), ("objective", objective)):
                result[name] += float(value.detach()) * share
        finally:
            for handle in handles:
                handle.remove()
    return result


def optimizer_step(model, optimizer, records, *, tensor_parallel=False,
                   max_norm=1.0, **kwargs):
    check_optimizer(model, optimizer)
    # Clear the model as well: an accidentally stale optimizer must never leave
    # orphaned gradients accumulating silently across steps.
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    result = backward_step(model, records, **kwargs)
    groups = []
    if tensor_parallel:
        from distillkit.parallel.sync import replicated_parameter_groups, sync_replicated_gradients

        sync_replicated_gradients(model)
        groups = list(replicated_parameter_groups(model))
    duplicates = {id(p) for group in groups for p in group[1:]}
    parameters = [p for p in model.parameters() if id(p) not in duplicates]
    if kwargs.get("sparse_stage") is not None:
        from distillkit.models.qwen35.csa2 import router_parameters

        router = {id(p) for _, p in router_parameters(model)}
        partitions = [[p for p in parameters if id(p) not in router],
                      [p for p in parameters if id(p) in router]]
    else:
        partitions = [parameters]
    for partition in partitions:
        torch.nn.utils.clip_grad_norm_(partition, max_norm)
    for group in groups:
        if group[0].grad is not None:
            for replica in group[1:]:
                replica.grad.copy_(group[0].grad.to(replica.device, non_blocking=True))
    optimizer.step()
    return result


def synchronize(model):
    for device in sorted({p.device for p in model.parameters() if p.is_cuda}, key=str):
        torch.cuda.synchronize(device)
