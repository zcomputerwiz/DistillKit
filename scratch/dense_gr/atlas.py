"""What the student represents where, and what a round changed (TARGETED_TRAINING.md, Phase 1).

Domains are fixed held-out documents, the same tokens for every arm. Units are the hybrid's
own components: each layer's mixer output (Gated DeltaNet or MLA/CSA2) and MLP output, and
the mixers' heads -- the input of `linear_attn.out_proj` / `self_attn.o_proj`, a head a slice.

  domains     the fixed sets, token ids frozen: held-out (eval-split) agent conversations by
              harness, tool use, the teacher's code and math, QA, general text, and llama.cpp
              source; every token labelled with its chat role (roles_of)
  nll         the regression ledger: each arm's loss per domain and role (assistant, thinking,
              tool call, tool result, system, user, plain), against the first arm with a 95%
              bootstrap interval over documents. The long-context probe scores every token, and
              its agent documents are the system prompt to 8K and mostly tool output beyond:
              training scores the model's own turns, so the ledger separates them
  change      the change map, a screening statistic: every parameter row's and column's
              0.5 (g_base + g_tuned) . (theta_tuned - theta_base), the trapezoid rule along the
              straight path (Simpson's with --midpoint). Its sum is compared with the measured
              change, but CSA2's discrete top-k makes the path non-smooth and cancellation can
              hide misranked columns: exact reverts decide (Codex review, codex-review-targeted)
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
# domain: (captures, document-id prefix, documents, token cap); eval splits only (never
# trained), sampled with a fixed seed. The agent conversations keep their system prompt
# (claude-code's is ~12K tokens of tool definitions) and are scored by role.
CAPTURES = {
    "agent-claude-code": (("teacher-cache-agent-smol-a", "teacher-cache-agent-smol-b"), "claude-code:", 12, 16384),
    "agent-codex": (("teacher-cache-agent-smol-b",), "codex:", 12, 16384),
    "agent-opencode": (("teacher-cache-agent-smol-b",), "opencode:", 12, 16384),
    "agent-mini-swe": (("teacher-cache-agent-smol-b",), "mini-swe-agent:", 24, 16384),
    "tools": (("teacher-cache-frontier-tools",), "", 48, 4096),
    "teacher-code": (("teacher-cache-teacher-code",), "", 48, 4096),
    "thinking-math": (("teacher-cache-teacher-math-gen",), "", 48, 4096),
    "nothink-math": (("teacher-cache-teacher-nothink-math",), "", 64, 4096),
    "qa": (("teacher-cache-frontier-qa2",), "", 16, 16384),
    "general": (("teacher-cache-general-pilot-w8",), "", 48, 2048)}
# Chat structure, all single tokens in this vocabulary.
IM_START, IM_END, NEWLINE = 248045, 248046, 198
OPENS = {248068: "think", 248058: "call", 248066: "response"}
CLOSES = {248069, 248059, 248067}
ROLE_TOKENS = {846: "user", 74455: "assistant", 8678: "system", 13766: "tool"}
ROLES = ["structure", "system", "user", "tool-result", "assistant", "thinking", "tool-call", "plain"]


def roles_of(ids):
    """Each token's role: chat structure (turn markers, role names, think/tool tags), system,
    user, tool result (a tool turn, or <tool_response> inside a user turn), assistant text,
    thinking, tool call; 'plain' outside any turn (code, prose)."""
    out = np.full(len(ids), ROLES.index("plain"), np.int8)
    role = inner = None
    state = 0  # 1: next token names the role, 2: then its newline
    for i, t in enumerate(int(x) for x in ids):
        if t == IM_START:
            out[i], state, inner = 0, 1, None
            continue
        if state == 1:
            out[i], role, state = 0, ROLE_TOKENS.get(t, "user"), 2
            continue
        if state == 2 and t == NEWLINE:
            out[i], state = 0, 0
            continue
        state = 0
        if t == IM_END:
            out[i], role = 0, None
        elif t in OPENS:
            out[i], inner = 0, OPENS[t]
        elif t in CLOSES:
            out[i], inner = 0, None
        elif role is not None:
            name = {"system": "system", "tool": "tool-result",
                    "user": "tool-result" if inner == "response" else "user",
                    "assistant": {"think": "thinking", "call": "tool-call"}.get(inner, "assistant")}[role]
            out[i] = ROLES.index(name)
    return out


def build_domains(args):
    from transformers import AutoTokenizer

    from distillkit.offline_cache import OfflineTeacherCache
    from long_context_probe import TOKENIZER, code_documents

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    # llama.cpp's source as the probe reads it, frozen here: the tree is live.
    domains = {"code-llamacpp": [torch.tensor(ids, dtype=torch.int32)
                                 for _, ids in code_documents(tok, args.length, args.code_documents)]}
    for name, (captures, prefix, count, cap) in CAPTURES.items():
        caches = [OfflineTeacherCache(D / c) for c in captures]
        pool = sorted((i, k) for k, cache in enumerate(caches) for i in cache.document_ids("eval")
                      if i.startswith(prefix))
        picked = random.Random(0).sample(pool, min(count, len(pool)))
        docs = [caches[k].read_document(i, tokens_only=True)["input_ids"][:cap] for i, k in picked]
        domains[name] = [torch.tensor(d.astype(np.int64), dtype=torch.int32) for d in docs if len(d) >= 64]
    roles = {name: [torch.from_numpy(roles_of(d.numpy())) for d in docs] for name, docs in domains.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"domains": domains, "roles": roles, "length": args.length}, args.output)
    for name, docs in domains.items():
        mix = torch.cat(roles[name]).bincount(minlength=len(ROLES)).float()
        print("%-18s %4d documents %8d tokens  %s" % (
            name, len(docs), sum(len(d) for d in docs),
            " ".join("%s %.0f%%" % (r, 100 * m / mix.sum()) for r, m in zip(ROLES, mix) if m > 0)))


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


GROUPS = {"all": list(range(len(ROLES))), "own-turns": [ROLES.index(r) for r in ("assistant", "thinking", "tool-call")]}


def ledger_rows(per_doc, labels):
    """Per role (and the 'all' / 'own-turns' groups): per-document sums of loss, top-1 hits
    and targets, so arms can be compared document by document."""
    rows = {}
    for key, members in [(r, [k]) for k, r in enumerate(ROLES)] + list(GROUPS.items()):
        sums = np.array([[nll[np.isin(lab, members)].sum(), hit[np.isin(lab, members)].sum(),
                          np.isin(lab, members).sum()] for (nll, hit), lab in zip(per_doc, labels)], dtype=np.float64)
        if sums[:, 2].sum() > 0:
            rows[key] = sums
    return rows


def paired_delta(ref, cur, draws=2000, seed=0):
    """Loss change, pooled over targets, with a 95% bootstrap interval over documents."""
    delta = lambda idx: (cur[idx, 0].sum() - ref[idx, 0].sum()) / max(ref[idx, 2].sum(), 1)
    rng = np.random.default_rng(seed)
    boots = [delta(rng.integers(0, len(ref), len(ref))) for _ in range(draws)]
    return float(delta(np.arange(len(ref)))), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_nll(args):
    data = torch.load(args.domains)
    domains = {k: v for k, v in data["domains"].items() if not args.only or k in args.only}
    roles = data.get("roles", {})
    rows = {}
    for name, path in arms(args.arm):
        model = load(path)
        rows[name] = {}
        for domain, docs in domains.items():
            per_doc = [token_losses(model, ids) for ids in docs]
            labels = ([r.numpy()[1:] for r in roles[domain]] if domain in roles
                      else [np.full(len(d) - 1, ROLES.index("plain")) for d in docs])
            rows[name][domain] = ledger_rows(per_doc, labels)
            whole = rows[name][domain]["all"]
            print("%-10s %-18s nll %.4f top1 %.3f" % (name, domain, whole[:, 0].sum() / whole[:, 2].sum(),
                                                     whole[:, 1].sum() / whole[:, 2].sum()), flush=True)
        del model
        torch.cuda.empty_cache()
    names = list(rows)
    summary = {}
    for name in names:
        summary[name] = {}
        for domain in domains:
            summary[name][domain] = {}
            for key, sums in rows[name][domain].items():
                entry = {"nll": sums[:, 0].sum() / sums[:, 2].sum(), "top1": sums[:, 1].sum() / sums[:, 2].sum(),
                         "targets": int(sums[:, 2].sum())}
                if name != names[0]:
                    ref = rows[names[0]][domain][key]
                    entry["delta"], entry["low"], entry["high"] = paired_delta(ref, sums)
                summary[name][domain][key] = entry
    (args.output_dir / "nll.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for name in names[1:]:
        print("\n== %s against %s: loss change [95%% interval over documents]" % (name, names[0]))
        for domain in domains:
            cells = ["%s %+.3f [%+.3f,%+.3f]" % (k, e["delta"], e["low"], e["high"])
                     for k, e in summary[name][domain].items() if e["targets"] >= 200 and k != "structure"]
            print("%-18s %s" % (domain, "  ".join(cells)))


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


# ------------------------------------------------------------------------------- revert

# Parameter families, coarse first (Codex review: coherent group reverts before heads).
FAMILIES = {
    "embed-head": r"^model\.embed_tokens\.",  # tied: the input embedding and the readout
    "norms": r"(layernorm|^model\.norm\.|q_norm|kv_a_norm|linear_attn\.norm)",
    "decay": r"linear_attn\.(A_log|dt_bias|in_proj_a)\b",  # g = -exp(A_log) softplus(a + dt_bias)
    "write-strength": r"linear_attn\.in_proj_b\b",
    "deltanet": r"linear_attn\.(in_proj_qkv|in_proj_z|conv1d|out_proj)\b",
    "mla": r"self_attn\.(q_proj|kv_a_proj|kv_b_proj|o_proj)\b",
    "indexer": r"self_attn\.(index_|indexer_)",
    "mlp": r"\.mlp\.",
    "hyper": r"_residual\.",
    "layers-0-7": range(0, 8), "layers-8-15": range(8, 16), "layers-16-23": range(16, 24)}


def family_members(family, names):
    import re

    rule = FAMILIES[family]
    if isinstance(rule, range):
        return [n for n in names if n.startswith("model.layers.") and int(n.split(".")[2]) in rule]
    return [n for n in names if re.search(rule, n)]


def run_revert(args):
    """Each family of the tuned checkpoint put back to the base, scored on the ledger."""
    data = torch.load(args.domains)
    domains = {k: v for k, v in data["domains"].items() if not args.only or k in args.only}
    labels = {d: ([r.numpy()[1:] for r in data["roles"][d]] if d in data.get("roles", {})
                  else [np.full(len(x) - 1, ROLES.index("plain")) for x in docs]) for d, docs in domains.items()}
    model = load(args.base)
    base = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()}
    del model
    torch.cuda.empty_cache()
    model = load(args.tuned)
    params = dict(model.named_parameters())
    tuned = {n: p.detach().to("cpu", copy=True) for n, p in params.items()}

    def evaluate(name):
        started = time.time()
        out = {d: ledger_rows([token_losses(model, ids) for ids in docs], labels[d]) for d, docs in domains.items()}
        print("%-22s %.0f s" % (name, time.time() - started), flush=True)
        return out

    def put(source, names):
        with torch.no_grad():
            for n in names:
                params[n].copy_(source[n])

    rows = {"tuned": evaluate("tuned")}
    for family in args.families:
        members = family_members(family, list(params))
        if not members:
            raise SystemExit("family %s matches no parameter" % family)
        put(base, members)
        rows["revert " + family] = evaluate("revert " + family)
        put(tuned, members)
    put(base, list(params))
    rows["base"] = evaluate("base")
    summary = {}
    for arm, by_domain in rows.items():
        summary[arm] = {}
        for domain, by_role in by_domain.items():
            summary[arm][domain] = {}
            for key, sums in by_role.items():
                entry = {"nll": sums[:, 0].sum() / sums[:, 2].sum(), "targets": int(sums[:, 2].sum())}
                for ref in ("tuned", "base"):
                    if arm != ref:
                        d, lo, hi = paired_delta(rows[ref][domain][key], sums)
                        entry["vs_" + ref] = [d, lo, hi]
                summary[arm][domain][key] = entry
    (args.output_dir / "revert.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n== loss change against the tuned checkpoint (negative: reverting the family helps)")
    for arm in rows:
        if arm == "tuned":
            continue
        cells = ["%s/%s %+.3f" % (d, k, e["vs_tuned"][0]) for d, by_role in summary[arm].items()
                 for k, e in by_role.items() if k in ("all", "own-turns", "tool-result") and e["targets"] >= 200]
        print("%-22s %s" % (arm, "  ".join(cells)))


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
    d.add_argument("--length", type=int, default=4096, help="llama.cpp source documents")
    d.add_argument("--code-documents", type=int, default=12)
    for command in ("nll", "change", "revert", "importance", "lens"):
        p = sub.add_parser(command)
        p.add_argument("--domains", type=Path, default=DOMAINS)
        p.add_argument("--only", nargs="*", default=None, help="these domains only")
        p.add_argument("--output-dir", type=Path, required=True)
        if command in ("change", "revert"):
            p.add_argument("--base", required=True)
            p.add_argument("--tuned", required=True)
        if command == "change":
            p.add_argument("--midpoint", action="store_true",
                           help="a third gradient at the midpoint: Simpson's rule, for when the "
                                "trapezoid misses the measured change by more than ~10%%")
        elif command == "revert":
            p.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
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
     "revert": run_revert, "importance": run_importance, "lens": run_lens}[args.command](args)


if __name__ == "__main__":
    main()
