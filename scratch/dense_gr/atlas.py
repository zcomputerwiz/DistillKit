"""What the student represents where, and what a round changed (TARGETED_TRAINING.md, Phase 1).

Domains are fixed held-out documents, the same tokens for every arm. Units are the hybrid's
own components: each layer's mixer output (Gated DeltaNet or MLA/CSA2) and MLP output, and
the mixers' heads -- the input of `linear_attn.out_proj` / `self_attn.o_proj`, a head a slice.

  domains     the fixed sets: agent traces and llama.cpp source as the long-context probe
              draws them, and the eval splits of the teacher captures
  nll         per-domain loss of each arm, with per-token losses and top-1 hits for 1->0 counts
  change      the change map: every parameter row's and column's share of a round's
              per-domain loss change, 0.5 (g_base + g_tuned) . (theta_tuned - theta_base) --
              the trapezoid rule along the straight path, exact for a quadratic loss --
              checked against the measured change (completeness)
  importance  the atlas: each unit's mean-ablation effect per domain, estimated for every unit
              from one forward and backward a document (attribution patching), the largest
              confirmed by ablating them for real
  lens        where predictions form: each layer's collapsed stream read through the final
              norm and head, against the final distribution

    python scratch/dense_gr/atlas.py domains --output scratch/csa2-eval/atlas/domains.pt
    python scratch/dense_gr/atlas.py nll --arm base=<ckpt> --arm tuned=<ckpt> --output-dir <dir>
    python scratch/dense_gr/atlas.py change --base <ckpt> --tuned <ckpt> --output-dir <dir>
    python scratch/dense_gr/atlas.py importance --arm base=<ckpt> --output-dir <dir>
    python scratch/dense_gr/atlas.py lens --arm base=<ckpt> --arm tuned=<ckpt> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
D = HERE.parents[2]  # HybridModel
DOMAINS = HERE.parent / "csa2-eval" / "atlas" / "domains.pt"
# domain: (capture, documents, token cap); eval splits only, sampled with a fixed seed.
CAPTURES = {"teacher-code": ("teacher-cache-teacher-code", 48, 4096),
            "thinking-math": ("teacher-cache-teacher-math-gen", 48, 4096),
            "nothink-math": ("teacher-cache-teacher-nothink-math", 64, 4096),
            "qa": ("teacher-cache-frontier-qa2", 16, 4096),
            "general": ("teacher-cache-general-pilot-w8", 48, 2048),
            "agent-train": ("teacher-cache-agent-smol-b", 16, 4096)}


def build_domains(args):
    from transformers import AutoTokenizer

    from distillkit.offline_cache import OfflineTeacherCache
    from long_context_probe import TOKENIZER, agent_documents, code_documents

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    domains = {}
    for kind, ids in (agent_documents(tok, args.length, args.long_documents)
                      + code_documents(tok, args.length, args.long_documents // 2)):
        name = "code-llamacpp" if kind == "code" else kind.replace("agent:", "agent-")
        domains.setdefault(name, []).append(torch.tensor(ids, dtype=torch.int32))
    for name, (capture, count, cap) in CAPTURES.items():
        cache = OfflineTeacherCache(D / capture)
        ids = sorted(cache.document_ids("eval"))
        picked = random.Random(0).sample(ids, min(count, len(ids)))
        docs = [cache.read_document(i, tokens_only=True)["input_ids"][:cap] for i in picked]
        domains[name] = [torch.tensor(d.astype(np.int64), dtype=torch.int32) for d in docs if len(d) >= 64]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"domains": domains, "length": args.length}, args.output)
    for name, docs in domains.items():
        print("%-20s %4d documents %8d tokens" % (name, len(docs), sum(len(d) for d in docs)))


def read_domains(path, only=None):
    domains = torch.load(path)["domains"]
    return {k: v for k, v in domains.items() if not only or k in only}


def load(path):
    from long_context_probe import load as load_model

    return load_model(path)


def arms(specs):
    return [tuple(spec.split("=", 1)) for spec in specs]


def doc_loss(model, ids):
    """Sum of next-token cross entropy over a document, and its target count."""
    from shared_head import head_losses

    x = ids.to(model.lm_head.weight.device).long()[None]
    hidden = model.model(input_ids=x, use_cache=False).last_hidden_state
    sums = head_losses(hidden, model.lm_head.weight, x, chunk=1024)
    return sums["nll"], sums["weight"]


def domain_loss(model, docs, backward=False):
    """Mean loss over a domain's targets; with `backward`, its gradient accumulated in .grad."""
    total = float(sum(len(d) - 1 for d in docs))
    loss = 0.0
    for ids in docs:
        with torch.set_grad_enabled(backward):
            nll, _ = doc_loss(model, ids)
            if backward:
                (nll / total).backward()
        loss += float(nll.detach())
    return loss / total


# ---------------------------------------------------------------------------------- nll

@torch.inference_mode()
def token_losses(model, ids, chunk=1024):
    x = ids.to("cuda").long()[None]
    hidden = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
    nll, hit = [], []
    for start in range(0, x.shape[1] - 1, chunk):
        stop = min(start + chunk, x.shape[1] - 1)
        logits = model.lm_head(hidden[start:stop]).float()
        target = x[0, start + 1:stop + 1]
        nll.append(F.cross_entropy(logits, target, reduction="none"))
        hit.append(logits.argmax(-1) == target)
    return torch.cat(nll).cpu().numpy().astype(np.float32), torch.cat(hit).cpu().numpy()


def run_nll(args):
    domains = read_domains(args.domains, args.only)
    summary = {}
    for name, path in arms(args.arm):
        model = load(path)
        arrays = {}
        summary[name] = {}
        for domain, docs in domains.items():
            per = [token_losses(model, ids) for ids in docs]
            arrays[domain + "/nll"] = np.concatenate([p[0] for p in per])
            arrays[domain + "/hit"] = np.concatenate([p[1] for p in per])
            summary[name][domain] = {"nll": float(arrays[domain + "/nll"].mean()),
                                     "top1": float(arrays[domain + "/hit"].mean())}
            print("%-10s %-20s nll %.4f top1 %.3f" % (name, domain, summary[name][domain]["nll"],
                                                     summary[name][domain]["top1"]), flush=True)
        np.savez_compressed(args.output_dir / ("nll-%s.npz" % name), **arrays)
        del model
        torch.cuda.empty_cache()
    names = [n for n, _ in arms(args.arm)]
    if len(names) >= 2:
        # Sample-wise forgetting against the first arm (arXiv 2510.17776): targets it got
        # right (top-1) that the other gets wrong, and the reverse.
        ref = np.load(args.output_dir / ("nll-%s.npz" % names[0]))
        for other in names[1:]:
            cur = np.load(args.output_dir / ("nll-%s.npz" % other))
            for domain in domains:
                a, b = ref[domain + "/hit"], cur[domain + "/hit"]
                summary[other][domain].update(
                    delta_nll=float(cur[domain + "/nll"].mean() - ref[domain + "/nll"].mean()),
                    forgot=int((a & ~b).sum()), learned=int((~a & b).sum()), targets=int(len(a)))
    (args.output_dir / "nll.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


# ------------------------------------------------------------------------------- change

def unit_of(name):
    """(layer, module) a parameter belongs to: 'L3.mixer', 'L3.mlp', 'L3.attn_residual', ..."""
    parts = name.split(".")
    if parts[:2] == ["model", "layers"]:
        module = parts[3]
        if module in ("linear_attn", "self_attn"):
            module = "mixer"
        return "L%s.%s" % (parts[2], module)
    return parts[-2] if len(parts) > 1 else name


def accumulate(acc, params, delta):
    """Add g . delta summed over each row and each column (whole tensor for others), in fp32:
    a document at a time, so bf16 gradients never sum across documents."""
    for name, p in params:
        if p.grad is None:
            continue
        s = p.grad.float() * delta[name].float()
        rows, cols = (s.sum(1), s.sum(0)) if s.ndim == 2 else (s.reshape(-1), None)
        if name in acc:
            rows, cols = acc[name][0] + rows, None if cols is None else acc[name][1] + cols
        acc[name] = (rows, cols)


def path_scores(model, params, delta, docs):
    """The domain's mean loss at the model's current weights, and g . delta per row/column."""
    total = float(sum(len(d) - 1 for d in docs))
    acc, value = {}, 0.0
    for ids in docs:
        model.zero_grad(set_to_none=True)
        nll, _ = doc_loss(model, ids)
        (nll / total).backward()
        value += float(nll.detach())
        accumulate(acc, params, delta)
    model.zero_grad(set_to_none=True)
    return value / total, acc


def run_change(args):
    domains = read_domains(args.domains, args.only)
    model = load(args.base)
    base = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()}
    del model
    torch.cuda.empty_cache()
    model = load(args.tuned)
    params = list(model.named_parameters())
    tuned = {n: p.detach().to("cpu", copy=True) for n, p in params}
    delta = {n: (tuned[n].float() - base[n].float()).to("cuda", torch.bfloat16) for n in tuned}
    for p in model.parameters():
        p.requires_grad_(True)
    loss, scores = {}, {}
    points = ["base", "tuned"] + (["mid"] if args.midpoint else [])
    for which in points:
        with torch.no_grad():
            for n, p in params:
                p.copy_(base[n] if which == "base" else tuned[n] if which == "tuned"
                        else (0.5 * (base[n].float() + tuned[n].float())).to(p.dtype))
        for domain, docs in domains.items():
            started = time.time()
            loss[(which, domain)], acc = path_scores(model, params, delta, docs)
            scores[(which, domain)] = {n: (r.cpu().numpy(), None if c is None else c.cpu().numpy())
                                       for n, (r, c) in acc.items()}
            print("%-6s %-20s loss %.4f  %.0f s" % (which, domain, loss[(which, domain)], time.time() - started),
                  flush=True)
    summary, arrays = {}, {}
    for domain in domains:
        measured = loss[("tuned", domain)] - loss[("base", domain)]
        units, total = {}, 0.0
        for name in scores[("tuned", domain)]:
            # Trapezoid (g_base + g_tuned) / 2, or with a midpoint Simpson's (g_b + 4 g_m + g_t) / 6.
            weights = {"base": 0.5, "tuned": 0.5} if not args.midpoint else {"base": 1 / 6, "tuned": 1 / 6, "mid": 4 / 6}
            rows = sum(w * scores[(p, domain)][name][0] for p, w in weights.items())
            cols = (None if scores[("tuned", domain)][name][1] is None
                    else sum(w * scores[(p, domain)][name][1] for p, w in weights.items()))
            arrays["%s/%s/rows" % (domain, name)] = rows.astype(np.float32)
            if cols is not None:
                arrays["%s/%s/cols" % (domain, name)] = cols.astype(np.float32)
            value = float(rows.sum())
            total += value
            units[unit_of(name)] = units.get(unit_of(name), 0.0) + value
        summary[domain] = {"base": loss[("base", domain)], "tuned": loss[("tuned", domain)],
                           "measured": measured, "attributed": total,
                           "completeness": total / measured if measured else None,
                           "units": dict(sorted(units.items(), key=lambda kv: kv[1]))}
        print("%-20s measured %+.4f attributed %+.4f" % (domain, measured, total), flush=True)
    np.savez_compressed(args.output_dir / "change.npz", **arrays)
    (args.output_dir / "change.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- importance

def units_of(model):
    """(name, module, kind, heads): kind 'out' hooks the module's output, 'in' its input."""
    cfg = model.config
    units = []
    for i, layer in enumerate(model.model.layers):
        linear = layer.block_type == "linear_attention"
        mixer = layer.linear_attn if linear else layer.self_attn
        proj = mixer.out_proj if linear else mixer.o_proj
        heads = cfg.linear_num_value_heads if linear else cfg.num_attention_heads
        units += [("L%d.mixer" % i, mixer, "out", 1), ("L%d.mlp" % i, layer.mlp, "out", 1),
                  ("L%d.heads" % i, proj, "in", heads)]
    return units


class Taps:
    """Forward hooks over the units: record activations, or replace one unit by its mean."""

    def __init__(self, units):
        self.units, self.handles, self.saved, self.means, self.counts = units, [], {}, {}, {}
        self.mode, self.ablate = None, None

    def __enter__(self):
        for name, module, kind, heads in self.units:
            if kind == "out":
                self.handles.append(module.register_forward_hook(self._out(name, heads)))
            else:
                self.handles.append(module.register_forward_pre_hook(self._in(name, heads)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def _visit(self, name, heads, value):
        if self.mode == "mean":
            flat = value.detach().float().reshape(-1, value.shape[-1])
            self.means[name] = self.means.get(name, 0) + flat.sum(0)
            self.counts[name] = self.counts.get(name, 0) + flat.shape[0]
        elif self.mode == "grad":
            value.retain_grad()
            self.saved[name] = value
        elif self.ablate is not None and self.ablate[0] == name:
            mean = self.means[name].to(value.dtype)
            head = self.ablate[1]
            if head is None:
                return mean.expand_as(value).clone()
            width = value.shape[-1] // heads
            value = value.clone()
            value[..., head * width:(head + 1) * width] = mean[head * width:(head + 1) * width]
            return value
        return None

    def _out(self, name, heads):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            new = self._visit(name, heads, tensor)
            if new is None:
                return None
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return hook

    def _in(self, name, heads):
        def hook(module, inputs):
            new = self._visit(name, heads, inputs[0])
            return None if new is None else (new,) + tuple(inputs[1:])
        return hook


def attribution(units, taps, scale):
    """Per unit (and head): sum over positions of (mean - a) . dL/da, the first-order
    estimate of mean-ablating it."""
    out = {}
    for name, _, _, heads in units:
        a = taps.saved[name]
        if a.grad is None:
            continue
        diff = (taps.means[name].to(a.device) - a.detach().float()) * a.grad.float()
        if heads > 1:
            per = diff.reshape(*diff.shape[:-1], heads, -1).sum(-1).reshape(-1, heads).sum(0)
            for h in range(heads):
                out["%s.%d" % (name, h)] = float(per[h]) * scale
        else:
            out[name] = float(diff.sum()) * scale
    return out


def unit_importance(model, units, docs, confirm):
    """Mean-ablation effect of every unit on the docs' mean loss: the first-order estimate for
    all of them, and the exact effect for the `confirm` largest estimates."""
    with Taps(units) as taps:
        taps.mode = "mean"
        with torch.inference_mode():
            for ids in docs:
                doc_loss(model, ids)
        taps.means = {k: v / taps.counts[k] for k, v in taps.means.items()}
        taps.mode = "grad"
        total = float(sum(len(d) - 1 for d in docs))
        estimate, clean = {}, 0.0
        for ids in docs:
            model.zero_grad(set_to_none=True)
            nll, _ = doc_loss(model, ids)
            (nll / total).backward()
            clean += float(nll.detach())
            for k, v in attribution(units, taps, 1.0).items():
                estimate[k] = estimate.get(k, 0.0) + v
            taps.saved.clear()
        model.zero_grad(set_to_none=True)
        clean /= total
        taps.mode = None
        ranked = sorted(estimate, key=lambda k: -abs(estimate[k]))
        exact = {}
        for key in ranked[:confirm]:
            unit, _, head = key.rpartition(".") if key.split(".")[-1].isdigit() else (key, "", "")
            taps.ablate = (unit, int(head) if head else None)
            with torch.inference_mode():
                exact[key] = domain_loss(model, docs) - clean
        taps.ablate = None
    return clean, estimate, exact


def run_importance(args):
    domains = read_domains(args.domains, args.only)
    summary = {}
    for arm, path in arms(args.arm):
        model = load(path)
        units = units_of(model)
        summary[arm] = {}
        for domain, docs in domains.items():
            started = time.time()
            clean, estimate, exact = unit_importance(model, units, docs, args.confirm)
            ranked = sorted(estimate, key=lambda k: -abs(estimate[k]))
            summary[arm][domain] = {"loss": clean, "estimate": estimate, "exact": exact}
            agree = [(estimate[k], exact[k]) for k in exact]
            r = float(np.corrcoef(*zip(*agree))[0, 1]) if len(agree) > 2 else None
            print("%-8s %-20s loss %.4f  top %s  estimate/exact r %s  %.0f s"
                  % (arm, domain, clean, ranked[:3], None if r is None else round(r, 3), time.time() - started),
                  flush=True)
        del model
        torch.cuda.empty_cache()
    (args.output_dir / "importance.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------------- lens

@torch.inference_mode()
def run_lens(args):
    domains = read_domains(args.domains, args.only)
    summary = {}
    for arm, path in arms(args.arm):
        model = load(path)
        norm, head = model.model.norm, model.lm_head
        summary[arm] = {}
        for domain, docs in domains.items():
            layers = None
            for ids in docs:
                x = ids.to("cuda").long()[None]
                states = model.model(input_ids=x, use_cache=False, output_hidden_states=True).hidden_states
                pos = torch.linspace(0, x.shape[1] - 2, min(args.positions, x.shape[1] - 1),
                                     device="cuda").long()
                target = x[0, pos + 1]
                final = F.log_softmax(head(states[-1][0, pos]).float(), -1)
                stack = []
                # states: embeddings, layers 0..22 collapsed, then the final norm's output.
                for h in list(states[1:-1]) + [None]:
                    logp = final if h is None else F.log_softmax(head(norm(h[0, pos])).float(), -1)
                    stack.append(torch.stack([(final.exp() * (final - logp)).sum(-1).mean(),
                                              (logp.argmax(-1) == final.argmax(-1)).float().mean(),
                                              -logp.gather(-1, target[:, None]).mean()]))
                rows = torch.stack(stack).cpu()
                layers = rows if layers is None else layers + rows
            layers = (layers / len(docs)).tolist()
            summary[arm][domain] = [{"layer": i, "kl_to_final": r[0], "top1_with_final": r[1], "nll": r[2]}
                                    for i, r in enumerate(layers)]
            print("%-8s %-20s nll by layer %s" % (arm, domain, " ".join("%.2f" % r[2] for r in layers)),
                  flush=True)
        del model
        torch.cuda.empty_cache()
    (args.output_dir / "lens.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("domains")
    d.add_argument("--output", type=Path, default=DOMAINS)
    d.add_argument("--length", type=int, default=4096, help="agent traces and llama.cpp source")
    d.add_argument("--long-documents", type=int, default=24, help="agent traces (half each harness)")
    for command in ("nll", "change", "importance", "lens"):
        p = sub.add_parser(command)
        p.add_argument("--domains", type=Path, default=DOMAINS)
        p.add_argument("--only", nargs="*", default=None, help="these domains only")
        p.add_argument("--output-dir", type=Path, required=True)
        if command == "change":
            p.add_argument("--base", required=True)
            p.add_argument("--tuned", required=True)
            p.add_argument("--midpoint", action="store_true",
                           help="a third gradient at the midpoint: Simpson's rule, for when the "
                                "trapezoid misses the measured change by more than ~10%%")
        else:
            p.add_argument("--arm", action="append", required=True, help="name=checkpoint")
        if command == "importance":
            p.add_argument("--confirm", type=int, default=20, help="units ablated exactly per domain")
        if command == "lens":
            p.add_argument("--positions", type=int, default=256, help="positions read per document")
    args = parser.parse_args()
    if args.command != "domains":
        args.output_dir.mkdir(parents=True, exist_ok=True)
    {"domains": build_domains, "nll": run_nll, "change": run_change,
     "importance": run_importance, "lens": run_lens}[args.command](args)


if __name__ == "__main__":
    main()
