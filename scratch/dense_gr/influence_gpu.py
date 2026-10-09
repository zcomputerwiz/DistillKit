# Assisted-by: Codex
"""Discarded, frozen-input influence measurements using the existing trainer.

No checkpoint writes. Full-model gradient geometry counts TP replicas once.
Two copied optimizer steps compare fresh and saved moments on the same gradient.
"""
import gc
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from influence_audit import OUT, RUN, CODE, read, write, digest, teacher_for_run


def family(name):
    import re
    layer = re.search(r"\.layers\.(\d+)\.", name)
    depth = ("early" if int(layer[1]) < 8 else "middle" if int(layer[1]) < 16 else "late") if layer else "global"
    if "embed" in name or "lm_head" in name:
        return "embedding_head"
    if "index" in name:
        return "indexer"
    if "_residual." in name:
        return "residual/" + depth
    if ".mlp." in name:
        return "mlp/" + depth
    if "norm" in name:
        return "norm/" + depth
    return ("deltanet/" if "linear_attn" in name else "mla/") + depth


def dot(a, b):
    # Bounded float64 temporaries, including the 508M-element tied head.
    a, b = a.reshape(-1), b.reshape(-1)
    total = 0.
    for start in range(0, a.numel(), 1 << 20):
        x, y = a[start:start + (1 << 20)].double(), b[start:start + (1 << 20)].double()
        total += float(torch.dot(x, y))
    return total


def geometry(vector, reference=None):
    sums = defaultdict(lambda: [0., 0., 0.])
    for n, g in vector.items():
        q = dot(g, g)
        sums[family(n)][0] += q
        sums["all"][0] += q
        if reference is not None:
            r = reference.get(n)
            if r is not None:
                d = dot(g, r)
                sums[family(n)][2] += d
                sums["all"][2] += d
    if reference is not None:
        for n, r in reference.items():
            q = dot(r, r)
            sums[family(n)][1] += q
            sums["all"][1] += q
    return {f: dict(norm=math.sqrt(q[0]), reference_norm=math.sqrt(q[1]), dot=q[2],
                   cosine=q[2] / math.sqrt(q[0] * q[1]) if q[0] * q[1] else None)
            for f, q in sums.items()}


def add(accumulator, vector, scale=1.):
    for name, value in vector.items():
        if name in accumulator:
            accumulator[name].add_(value, alpha=scale)
        else:
            accumulator[name] = value.clone().mul_(scale)


def move(record):
    return {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in record.items()}


class Diagnostic:
    def __init__(self, plan):
        from benchmark import apply_liger, SpillWatch
        from long_context_probe import load
        from distillkit.parallel.model import shard_model
        from distillkit.parallel.sync import replicated_parameter_groups
        from training_state import parameter_layout
        self.plan = plan
        self.teacher, self.tokenizer, self.args = teacher_for_run()
        self.spill = SpillWatch(interval=5.).start()
        self.model = load(Path(plan["checkpoint"]))
        self.model.train()
        self.model.model.gradient_checkpointing = True
        apply_liger(self.model, self.model.config)
        shard_model(self.model, ["cuda:0", "cuda:1"], shard_embeddings=False, embedding_device="cuda:1")
        state_file = Path(plan["state"]) / "state.pt"
        if not (state_file.parent / "complete.json").exists():
            raise ValueError("incomplete saved optimizer state")
        self.state = torch.load(state_file, mmap=True, map_location="cpu", weights_only=True)
        if self.state["layout"] != parameter_layout(self.model):
            raise ValueError("saved model parameter layout differs from diagnostic")
        self.model.load_state_dict(self.state["model"], strict=True)
        self.groups = list(replicated_parameter_groups(self.model))
        duplicates = {id(p) for group in self.groups for p in group[1:]}
        self.parameters = {n: p for n, p in self.model.named_parameters() if id(p) not in duplicates}
        self.named = dict(self.model.named_parameters())
        self.activations = {}
        self.handles = []
        for index in (0, 7, 15, 23):
            self.handles.append(self.model.model.layers[index].mlp.register_forward_hook(self.activation_hook(index)))
        self.pairs = {r["pair_id"]: r for r in map(__import__("json").loads,
            Path(self.args["pairs"]).read_text().splitlines())}
        print("Loaded candidate + exact TP saved-state layout; state step", self.state["progress"]["steps"], flush=True)

    def activation_hook(self, layer):
        def hook(module, inputs, output):
            # First forward only: checkpoint replay is not a second sample.
            if layer in self.activations:
                return
            result = {}
            for name, value in (("input", inputs[0]), ("output", output)):
                x = value.detach().reshape(-1, value.shape[-1]).float()
                channel = x.square().mean(0).sqrt()
                rms = float(channel.square().mean().sqrt())
                result[name] = dict(rms=rms, abs_peak_over_rms=float(x.abs().max()) / max(rms, 1e-30),
                    max_channel_rms_over_median=float(channel.max() / channel.median().clamp_min(1e-30)),
                    sample_fraction_gt_8rms=float((x.reshape(-1)[::64].abs() > 8 * rms).float().mean()))
            self.activations[layer] = result
        return hook

    def record(self, row, reweight=False):
        doc, width = row["doc_id"], row["width"]
        self.teacher.real_width[doc] = min(self.teacher.cap(doc), width)
        r = self.teacher.read_batch([doc], width)
        if __import__("hashlib").sha256(r["input_ids"][0].numpy().tobytes()).hexdigest() != row["input_sha256"]:
            raise ValueError("diagnostic token prefix changed: " + doc)
        if reweight:
            w = r.get("weight", torch.ones_like(r["input_ids"], dtype=torch.float32)).clone()
            w[:, -1] = 0
            r["weight"] = w * (row["coefficient"] / row["targets"])
        return move(r)

    def pair(self, pid):
        from smoke_train import PairSource
        # Use the production padding/span conversion, without a cycling source.
        source = object.__new__(PairSource)
        source.rows = [self.pairs[pid]]
        source.pad, source.block, source.device = self.tokenizer.pad_token_id or 248044, 256, "cuda:0"
        return source._record(source.rows[0])

    def backward(self, records, teacher_weight=.5, pair_weight=.1, beta=.1, sft=1.):
        from training_step import backward_step, synchronize
        from selection_replay import SelectionReplayCache
        from distillkit.parallel.sync import sync_replicated_gradients
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        self.activations = {}
        with torch.autograd.set_multithreading_enabled(False), SelectionReplayCache(self.model):
            metrics = backward_step(self.model, records, teacher_weight=teacher_weight, pair_weight=pair_weight,
                dpo_beta=beta, pair_sft_weight=sft, shared_head=True, streaming_head=True, head_chunk=64)
        sync_replicated_gradients(self.model)
        synchronize(self.model)
        gradients = {n: p.grad.detach().cpu().float() for n, p in self.parameters.items() if p.grad is not None}
        channels = []
        for n, g in gradients.items():
            if ".mlp." not in n or g.ndim != 2:
                continue
            norms = g.square().sum(1)
            top = max(1, math.ceil(len(norms) * .01))
            channels.append(dict(parameter=n, top_one_percent_row_energy=float(norms.topk(top).values.sum() /
                norms.sum().clamp_min(1e-30)), max_row_norm_over_median=float(norms.max().sqrt() /
                norms.median().sqrt().clamp_min(1e-30))))
        self.check_memory()
        return gradients, metrics, dict(self.activations), channels

    def check_memory(self):
        if self.spill.breached():
            raise RuntimeError("shared GPU memory spill detected")
        for device in (0, 1):
            if torch.cuda.max_memory_allocated(device) > .9 * torch.cuda.get_device_properties(device).total_memory:
                raise RuntimeError("diagnostic exceeded 90% VRAM allocation guard")

    @torch.no_grad()
    def preservation_nll(self):
        from shared_head import head_losses
        self.model.eval()
        self.activations = {}
        rows = []
        for row in self.plan["replay"]:
            if not row["preservation"]:
                continue
            r = self.record(row)
            hidden = self.model.model(input_ids=r["input_ids"], use_cache=False).last_hidden_state
            where = self.model.lm_head.weight.device
            sums = head_losses(hidden.to(where), self.model.lm_head.weight, r["input_ids"].to(where),
                               weight=None if "weight" not in r else r["weight"].to(where), chunk=64)
            rows.append(dict(doc_id=row["doc_id"], nll=float(sums["nll"] / sums["weight"]),
                             coefficient=row["coefficient"]))
        total = sum(r["coefficient"] for r in rows)
        return dict(rows=rows, mean=sum(r["nll"]*r["coefficient"] for r in rows)/total)

    def copied_step(self, gradient, preservation_gradient, baseline_nll, saved):
        from training_step import KahanAdamW8bit, synchronize
        from distillkit.parallel.sync import clip_grad_norm
        self.model.load_state_dict(self.state["model"], strict=True)
        groups = []
        for source, names in zip(self.state["optimizer"]["param_groups"], self.state["optimizer_layout"]):
            group = {k: v for k, v in source.items() if k != "params"}
            group["params"] = [self.named[n] for n in names]
            groups.append(group)
        optimizer = KahanAdamW8bit(groups, betas=(.9, .95), weight_decay=.1)
        if saved:
            # deepcopy prevents diagnostic updates from changing mmap-backed saved moments.
            optimizer.load_state_dict(__import__("copy").deepcopy(self.state["optimizer"]))
        before_comp = {n: optimizer.state[p]["compensation"].detach().cpu().clone()
                       for n, p in self.parameters.items() if "compensation" in optimizer.state[p]}
        self.model.zero_grad(set_to_none=True)
        for n, g in gradient.items():
            p = self.parameters[n]
            p.grad = g.to(device=p.device, dtype=p.dtype)
        for group in self.groups:
            if group[0].grad is not None:
                for p in group[1:]:
                    p.grad = group[0].grad.to(p.device)
        preclip = float(clip_grad_norm(self.model, 1.))
        optimizer.step()
        synchronize(self.model)
        update, observed = {}, {}
        family_parameters = defaultdict(float)
        for n, p in self.parameters.items():
            old = self.state["model"][n].float()
            delta = p.detach().cpu().float() - old
            observed[n] = delta
            c = optimizer.state[p].get("compensation")
            compensated = delta.clone()
            if c is not None:
                compensated += c.detach().cpu().float()
            if n in before_comp:
                compensated -= before_comp[n].float()
            update[n] = compensated
            family_parameters[family(n)] += dot(old, old)
        effective = geometry(update, preservation_gradient)
        visible = geometry(observed, preservation_gradient)
        for f, r in effective.items():
            q = sum(family_parameters.values()) if f == "all" else family_parameters[f]
            r["relative_parameter_movement"] = r["norm"] / max(math.sqrt(q), 1e-3)
        after = self.preservation_nll()
        result = dict(mode="saved_step40_moments" if saved else "fresh_moments",
            preclip_norm=preclip, global_clip_scale=min(1., 1/max(preclip, 1e-30)),
            effective_compensated_update=effective, visible_bf16_update=visible,
            before_preservation_nll=baseline_nll, after_preservation_nll=after,
            preservation_mean_delta=after["mean"]-baseline_nll["mean"],
            rates=[dict(name=g.get("name"), lr=g["lr"]) for g in optimizer.param_groups],
            caveat="Discarded counterfactual step; saved pair/scheduler cursor is not resumed. Same computed gradient in both modes.")
        self.check_memory()
        del optimizer, before_comp, update, observed
        self.model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        return result


def main():
    if (OUT / "gpu-audit.json").exists():
        raise ValueError("completed diagnostic exists; refusing overwrite")
    plan = read(OUT / "plan.json")
    for path, sha in plan["input_sha256"].items():
        if digest(path) != sha:
            raise ValueError("frozen input changed: " + path)
    torch.set_num_threads(4)
    torch.manual_seed(25)
    torch.cuda.set_device(0)
    diag = None
    started = time.monotonic()
    result = dict(status="running", plan_sha256=digest(OUT/"plan.json"), source_sha256=digest(__file__),
                  documents=[], pairs=[], component_checks=[])
    try:
        diag = Diagnostic(plan)
        preserve = [diag.record(r, True) for r in plan["replay"] if r["preservation"]]
        # CE on real preservation tokens is a diagnostic reference only. The actual
        # combined update below retains the inherited raw-code KL-only objective.
        preservation_share = sum(r["coefficient"] for r in plan["replay"] if r["preservation"])
        for record in preserve:
            record["ce_only"], record["kl_only"] = True, False
            record["weight"] = record["weight"] / preservation_share
        reference, metrics, _, _ = diag.backward(preserve, teacher_weight=0.)
        result["preservation_objective"] = metrics
        result["preservation_reference_policy"] = "CE over actual code tokens, normalized to sampled code-source shares; diagnostic only, never added to the optimizer objective."
        del preserve
        replay, pairs = {}, {}
        for row in plan["replay"]:
            g, metrics, activations, channels = diag.backward([diag.record(row)])
            geo = geometry(g, reference)
            add(replay, g, row["coefficient"])
            result["documents"].append(dict(doc_id=row["doc_id"], source=row["source"], coefficient=row["coefficient"],
                metrics=metrics, gradient=geo, activations=activations, mlp_rows=channels,
                weighted_gradient_norm=geo["all"]["norm"]*row["coefficient"]))
            print("Document", len(result["documents"]), row["source"], "norm", geo["all"]["norm"],
                  "code cosine", geo["all"]["cosine"], flush=True)
            del g
            write(OUT/"gpu-progress.json", result)
        for pid in plan["pair_ids"]:
            g, metrics, _, _ = diag.backward([diag.pair(pid)])
            geo = geometry(g, reference)
            add(pairs, g, 1/len(plan["pair_ids"]))
            result["pairs"].append(dict(pair_id=pid, metrics=metrics, gradient=geo))
            print("Pair", pid, "norm", geo["all"]["norm"], "code cosine", geo["all"]["cosine"], flush=True)
            del g
            write(OUT/"gpu-progress.json", result)
        result["replay_gradient"] = geometry(replay, reference)
        result["pair_gradient"] = geometry(pairs, reference)
        result["replay_pair_geometry"] = geometry(pairs, replay)
        del replay, pairs
        # Separate existing objective components for one code and one QA document.
        for source in ("teacher-code", "frontier-qa2"):
            row = next(r for r in plan["replay"] if r["source"] == source)
            record = diag.record(row)
            record["ce_only"], record["kl_only"] = True, False
            ce, _, _, _ = diag.backward([record])
            record["ce_only"], record["kl_only"] = False, True
            kl, _, _, _ = diag.backward([record])
            result["component_checks"].append(dict(source=source, doc_id=row["doc_id"], ce_kl=geometry(kl, ce)))
            del ce, kl
        pid = plan["pair_ids"][0]
        dpo, _, _, _ = diag.backward([diag.pair(pid)], sft=0.)
        chosen, _, _, _ = diag.backward([diag.pair(pid)], beta=0.)
        result["component_checks"].append(dict(pair_id=pid, dpo_chosen_ce=geometry(dpo, chosen)))
        del dpo, chosen
        baseline = diag.preservation_nll()
        records = [diag.record(r, True) for r in plan["replay"]] + [diag.pair(p) for p in plan["pair_ids"]]
        gradient, metrics, _, _ = diag.backward(records)
        del records
        result["combined_objective"] = metrics
        result["combined_gradient"] = geometry(gradient, reference)
        result["copied_steps"] = []
        for saved in (False, True):
            print("Discarded optimizer step:", "saved moments" if saved else "fresh moments", flush=True)
            result["copied_steps"].append(diag.copied_step(gradient, reference, baseline, saved))
            write(OUT/"gpu-progress.json", result)
        result.update(status="complete", seconds=time.monotonic()-started,
                      allocated_peaks_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in (0, 1)])
    finally:
        if diag is not None:
            for handle in diag.handles:
                handle.remove()
            diag.teacher.close()
            diag.spill.stop()
            result.update(diag.spill.report())
        write(OUT/("gpu-audit.json" if result["status"] == "complete" else "gpu-progress.json"), result)
    print("Influence audit complete; no checkpoint saved", flush=True)


if __name__ == "__main__":
    main()
