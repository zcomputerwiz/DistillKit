"""The optimizer step shared by the dense-GR trainer, benchmark and tests."""
from __future__ import annotations

import contextlib

import bitsandbytes as bnb
import torch

from teacher_kl import accumulation_shares, grouped_tail_kl, scored_mask


class KahanAdamW8bit(bnb.optim.AdamW8bit):
    """AdamW8bit whose bf16 weights keep what rounding would discard.

    bitsandbytes updates a parameter in whatever dtype it is stored in, rounding to
    nearest inside its kernel, and it keeps no float32 master copy -- the 8-bit part
    compresses Adam's moments, not the weights. On bf16 weights that loses every
    update smaller than half an ulp, `|w| * 2^-9`: at lr 7.3e-6 that is every weight
    with `|w| >= 0.002`. Measured on a 2,000-step run of this model, 13.3% of its
    elements changed, weights above 0.0039 changed none, and no norm moved at all.
    Stochastic rounding for these optimizers was requested upstream in 2024
    (bitsandbytes #1165) and is not implemented.

    This does what optimi does for its own optimizers: each bf16 weight carries a
    bf16 compensation buffer holding the part of every update that rounding would
    have dropped. Per parameter, the step runs bitsandbytes' own fp32 kernel on a
    working copy `weight + compensation`, then rounds back and keeps the remainder.
    Weight decay runs inside the same kernel, so it is inside the compensated sum
    rather than rounded on its own. The 8-bit moments are unchanged -- they do not
    depend on the parameter dtype -- and the kernel is unchanged; only the tensor
    it is handed differs.

    Costs 2 bytes a parameter for the buffer, half of float32 master weights, plus
    one parameter's float32 working copy and gradient at a time during the step.
    Non-bf16 parameters take the stock path.
    """

    # Elements per float32 working chunk: 64M, 256 MiB. Must be a multiple of the
    # 256-element block bitsandbytes quantizes its state in, so every chunk boundary is
    # a block boundary and the chunked update is bit-identical to a single pass.
    #
    # Chunking exists because the working copy is the largest allocation in the step.
    # For the 508M-row embedding it is 1.89 GiB each for weight and gradient, and on
    # Windows `expandable_segments` is not supported -- PyTorch warns and ignores it --
    # so the caching allocator fragments: a run at 6,144 tokens a step died asking for
    # 1.89 GiB with 9.51 GiB reserved but unallocated.
    chunk = 1 << 26

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        if p.dtype != torch.bfloat16:
            return super().update_step(group, p, gindex, pindex)
        state = self.state[p]
        if "compensation" not in state:
            state["compensation"] = torch.zeros_like(p, memory_format=torch.contiguous_format)
        compensation = state["compensation"]
        if state["state1"].dtype == torch.uint8 and p.numel() > self.chunk:
            return self._chunked_8bit_step(group, p, gindex, pindex, state, compensation)
        low, grad = p.data, p.grad
        p.data = low.float().add_(compensation.float())
        p.grad = grad.float()
        try:
            super().update_step(group, p, gindex, pindex)
            work = p.data
        finally:
            p.data, p.grad = low, grad
        low.copy_(work)
        compensation.copy_(work.sub_(low.float()))

    def _chunked_8bit_step(self, group, p, gindex, pindex, state, compensation):
        """The same kernel call bitsandbytes makes, over block-aligned slices.

        Mirrors `Optimizer2State.update_step`'s 8-bit branch, with the step counter
        advanced once for the whole parameter rather than once per slice.
        """
        from bitsandbytes import functional as F

        if self.chunk % 256:
            raise ValueError("chunk must be a multiple of the 256-element state block")
        p.data = p.data.contiguous()
        p.grad = p.grad.contiguous()
        config = self.get_config(gindex, pindex, group)
        state["step"] += 1
        step = state["step"]
        low, grad, residual = p.data.view(-1), p.grad.view(-1), compensation.view(-1)
        first, second = state["state1"].view(-1), state["state2"].view(-1)
        betas = config["betas"]
        for start in range(0, low.numel(), self.chunk):
            end = min(start + self.chunk, low.numel())
            blocks = slice(start // 256, (end + 255) // 256)
            work = low[start:end].float().add_(residual[start:end].float())
            F.optimizer_update_8bit_blockwise(
                self.optimizer_name, grad[start:end].float(), work,
                first[start:end], second[start:end],
                betas[0], betas[1], betas[2] if len(betas) >= 3 else 0.0,
                config.get("alpha", 0.0), config["eps"], step, config["lr"],
                state["qmap1"], state["qmap2"],
                state["absmax1"][blocks], state["absmax2"][blocks],
                config["weight_decay"], gnorm_scale=1.0, skip_zeros=config["skip_zeros"])
            low[start:end].copy_(work)
            residual[start:end].copy_(work.sub_(low[start:end].float()))


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
