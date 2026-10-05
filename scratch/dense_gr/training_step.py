"""The optimizer step shared by the dense-GR trainer, benchmark and tests."""
from __future__ import annotations

import contextlib

import bitsandbytes as bnb
import torch

from shared_head import head_losses
from teacher_kl import accumulation_shares, grouped_tail_kl, scored_mask, unlikelihood_loss
from training_state import position_weight


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


def causal_ce(model, hidden, ids, weight=None):
    """Mean next-token cross entropy; with `weight` ([rows, length], position t's weight for
    predicting token t + 1), the weighted mean, sum(w * nll) / sum(w)."""
    from cut_cross_entropy import linear_cross_entropy

    device = model.lm_head.weight.device
    # CCE's Triton kernels launch on the *current* device, not the tensors' own. With
    # the head moved off home (--embedding-on away) that read another card's memory and
    # returned a loss of exactly 0 while the teacher KL, plain torch, carried on.
    with torch.cuda.device(device):
        if weight is None:
            return linear_cross_entropy(hidden.to(device), model.lm_head.weight,
                                        ids.to(device), shift=1, reduction="mean").to(hidden.device)
        targets = ids.clone()
        targets[:, 1:][weight[:, :-1] <= 0] = -100  # CCE's ignore_index: skips the zeros
        # Per position: [rows, length - 1], position t's loss for token t + 1.
        nll = linear_cross_entropy(hidden.to(device), model.lm_head.weight, targets.to(device),
                                   shift=1, reduction="none")
        w = weight[:, :-1].to(device=device, dtype=nll.dtype)
        return ((nll * w).sum() / w.sum().clamp_min(1e-12)).to(hidden.device)


def response_logprob(model, hidden, ids, start, end):
    """Summed log p of tokens `start:end` given everything before them, one row.

    Through Cut Cross-Entropy, so a 248,320-wide row of logits is never formed -- the
    same reason the causal loss uses it -- and it carries gradients. Scored over the whole
    (right-padded) row and masked, so the kernel sees the row's width, not the response's:
    one shape per padded width rather than a cold autotune per pair."""
    from cut_cross_entropy import linear_cross_entropy

    device = model.lm_head.weight.device
    with torch.cuda.device(device):
        losses = linear_cross_entropy(hidden.to(device), model.lm_head.weight, ids.to(device),
                                      shift=1, reduction="none")
    return -losses[..., start - 1:end - 1].sum().to(hidden.device)


def preference_loss(model, pair, *, beta, sft_weight, logprob=response_logprob):
    """DPO for one (chosen, rejected) pair against precomputed reference log-probs, plus
    cross entropy on the chosen response (RPO-style): with plausible, easy-to-separate
    negatives, DPO alone can lower the chosen answer's likelihood along with the loop's.

    Returns (objective, dpo, margin, chosen mean log p)."""
    sides = {}
    for side in ("chosen", "rejected"):
        ids = pair[side + "_ids"]
        hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             use_cache=False).last_hidden_state
        sides[side] = logprob(model, hidden, ids, pair[side + "_start"], pair[side + "_end"])
    margin = beta * ((sides["chosen"] - pair["ref_chosen"]) - (sides["rejected"] - pair["ref_rejected"]))
    dpo = -torch.nn.functional.logsigmoid(margin)
    tokens = pair["chosen_end"] - pair["chosen_start"]
    chosen_mean = sides["chosen"] / tokens
    return dpo - sft_weight * chosen_mean, dpo, margin, chosen_mean


def ftpo_loss(logits, row, *, clip=2.0, tether=0.4, target_tether=0.05, tau=1.5):
    """Final-token preference (antidoom's FTPO) at one position, from its full logit row.

    Each chosen alternative should beat the rejected loop-start token by `clip` logits; a
    pair that does contributes nothing more, so a separated row stops pulling. Two MSE
    tethers to the reference's logits keep the rest of the distribution where it was:
    `tether` over the reference's top-k minus the targets, and a looser `target_tether`
    on the targets beyond `tau` logits of movement.

    Returns (objective, mean margin, fraction of chosen that beat the rejected)."""
    chosen, rejected = row["chosen"], row["rejected"]
    margins = logits[chosen] - logits[rejected]
    preference = (torch.nn.functional.softplus(-margins) * (margins < clip)).mean()
    ref_ids, ref_logits = row["ref_ids"], row["ref_logits"]
    targets = torch.cat([chosen, rejected.view(1)])
    rest = ~torch.isin(ref_ids, targets)
    drift = (logits[ref_ids] - ref_logits)[rest].pow(2).mean()
    moved = (logits[targets] - row["ref_target_logits"]).abs()
    target_drift = (moved - tau).clamp(min=0).pow(2).mean()
    objective = preference + tether * drift + target_tether * target_drift
    return objective, margins.mean(), (margins > 0).float().mean()


def backward_step(model, records, *, teacher_weight=0.0, indexer_weight=1.0,
                  sparse_stage=None, kl_chunk=256, ce=causal_ce, unlikelihood_weight=1.0,
                  pair_weight=0.0, dpo_beta=0.1, pair_sft_weight=0.2, logprob=response_logprob,
                  ftpo_options=None, shared_head=False, head_chunk=512):
    """Accumulate means over the identical B*(L-1) positions for all three terms.

    Does not clear gradients or update weights, so warm-up exercises this exact path.
    The injected CE callable is only for CPU correctness tests; production uses CCE.
    `shared_head` computes CE, KL and unlikelihood from one projection of the scored rows
    (shared_head.py) instead of a projection each, `head_chunk` rows at a time (each chunk's
    fp32 logits are rows x 248,320 x 4 bytes, ~485 MiB at 512).
    Preference-pair records (`pair`) are left out of the token accounting and add
    `pair_weight` times their mean preference objective; FTPO rows (`ftpo`, also `pair`)
    likewise, with `ftpo_options` passed to `ftpo_loss`.
    """
    from distillkit.models.qwen35.csa2 import isolated_indexer, recorded_attention

    if not 0 <= teacher_weight <= 1 or indexer_weight < 0 or pair_weight < 0:
        raise ValueError("invalid objective weights")
    pairs = [r for r in records if r.get("pair")]
    records = [r for r in records if not r.get("pair")]
    counts, total = accumulation_shares([r["input_ids"] for r in records])
    # Position weights (assistant-only turns, answer spans, padding at 0); an older
    # record's boolean `supervised` mask is the 0/1 case. A record's share of the step is
    # its total weight, so an accumulated step equals one weighted mean over all of it.
    weights = [position_weight(r) for r in records]
    counts = [float(w[:, :-1].sum()) if w is not None else n for w, n in zip(weights, counts)]
    total = max(sum(counts), 1)
    if not (records or pairs) or any(n <= 0 for n in counts):
        raise ValueError("an optimizer step needs nonempty causal targets")
    result = dict(loss=0.0, teacher_kl=0.0, indexer=0.0, unlikelihood=0.0, objective=0.0,
                  targets=sum(counts))
    for record, count, weight in zip(records, counts, weights):
        ids = record["input_ids"]
        mask = queries = scored_mask(ids.shape[1], ids.device, ids.shape[0])
        if weight is not None:
            mask = mask * weight.to(mask.device)  # float: the KL is summed weighted
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
                kl_only = bool(record.get("kl_only", False))
                # The student's own shortest correct rollouts: its text alone. The teacher
                # thinks at length, and KL toward it at every position of a brief thought
                # pulls against closing it.
                ce_only = bool(record.get("ce_only", False))
                distil = bool(teacher_weight or kl_only) and not ce_only
                if distil and "topk_ids" not in record:
                    raise ValueError("teacher weight requires cached targets")
                negative = record.get("negative") if distil else None
                # A looping rollout: no KL on its repeated spans, where the teacher
                # endorses the loop; unlikelihood pushes them down instead.
                kl_mask = None if not distil else (mask if negative is None
                                                   else mask * ~negative.to(mask.device))
                context = record.get("context_kl") if distil else None
                if context is not None:
                    # KL alone on an assistant-only document's sampled context positions.
                    kl_mask = torch.broadcast_to(kl_mask, ids.shape) + context.to(kl_mask.device)
                if shared_head:
                    # One projection of the scored rows for every head loss (shared_head.py).
                    where = model.lm_head.weight.device
                    sums = head_losses(
                        hidden.to(where), model.lm_head.weight, ids.to(where),
                        weight=None if weight is None else weight.to(where),
                        topk_ids=record["topk_ids"].to(where) if distil else None,
                        topk_logprobs=record["topk_logprobs"].to(where) if distil else None,
                        kl_weight=None if kl_mask is None else torch.broadcast_to(kl_mask, ids.shape).to(where),
                        negative=None if negative is None else negative.to(where), chunk=head_chunk,
                        kl_beyond=context is not None)
                    language = (sums["nll"] / sums["weight"].clamp_min(1e-12)).to(hidden.device)
                    carried = (sums["kl"] / count).to(hidden.device)
                    repelled = (sums["unlikelihood"] / count).to(hidden.device)
                else:
                    language = (ce(model, hidden, ids) if weight is None
                                else ce(model, hidden, ids, weight=weight))
                    carried = language.new_zeros(())
                    repelled = language.new_zeros(())
                    if distil:
                        where = model.lm_head.weight.device
                        if negative is not None:
                            repelled = unlikelihood_loss(
                                hidden.to(where), model.lm_head, ids.to(where), negative.to(where),
                                weight=None if weight is None else weight.to(where)).to(hidden.device) / count
                        carried = grouped_tail_kl(
                            hidden.to(where), model.lm_head, record["topk_ids"].to(where),
                            record["topk_logprobs"].to(where), kl_mask.to(where),
                            chunk_length=kl_chunk).to(hidden.device) / count
                objective = language
                aligned = language.new_zeros(())
                if distil:
                    objective = (1 - teacher_weight) * language + teacher_weight * carried
                    if kl_only:
                        # The student's own text: the teacher's view only. The CE is
                        # still reported, and is what an on-policy round should lower.
                        language = language.detach()
                        objective = carried + unlikelihood_weight * repelled
                if sparse_stage is not None:
                    targets = {i: a.last_attention for i, a in sparse_stage}
                    chosen = {i: a.last_allowed for i, a in sparse_stage}
                    borrowed = {i: a.bus.require_latent(a.latent_donor, i)
                                for i, a in sparse_stage}
                    aligned = indexer_loss(model, sparse_stage, seen, targets, borrowed,
                                           selected=chosen, query_mask=queries)  # every query routes
                    objective = objective + indexer_weight * aligned
            share = count / total
            (objective * share).backward()
            for name, value in (("loss", language), ("teacher_kl", carried),
                                ("indexer", aligned), ("unlikelihood", repelled),
                                ("objective", objective)):
                result[name] += float(value.detach()) * share
        finally:
            for handle in handles:
                handle.remove()
    ftpo = [p for p in pairs if p.get("ftpo")]
    pairs = [p for p in pairs if not p.get("ftpo")]
    if ftpo and pair_weight:
        for name in ("ftpo", "ftpo_margin", "chosen_win"):
            result[name] = 0.0
        for row in ftpo:
            ids = row["input_ids"]
            hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 use_cache=False).last_hidden_state
            where = model.lm_head.weight.device
            # Back beside the row's targets: with --embedding-on away the head is on the other card.
            logits = model.lm_head(hidden[0, row["position"]].to(where)).float().to(ids.device)
            objective, margin, win = ftpo_loss(logits, row, **(ftpo_options or {}))
            (objective * pair_weight / len(ftpo)).backward()
            result["ftpo"] += float(objective.detach()) / len(ftpo)
            result["ftpo_margin"] += float(margin.detach()) / len(ftpo)
            result["chosen_win"] += float(win) / len(ftpo)
    if pairs and pair_weight:
        for name in ("dpo", "dpo_margin", "chosen_logp"):
            result[name] = 0.0
        for pair in pairs:
            objective, dpo, margin, chosen_mean = preference_loss(
                model, pair, beta=dpo_beta, sft_weight=pair_sft_weight, logprob=logprob)
            (objective * pair_weight / len(pairs)).backward()
            result["dpo"] += float(dpo.detach()) / len(pairs)
            result["dpo_margin"] += float(margin.detach()) / len(pairs)
            result["chosen_logp"] += float(chosen_mean.detach()) / len(pairs)
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
