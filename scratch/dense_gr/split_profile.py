"""Can a prompt's upper layers be predicted instead of run? A preflight for split prefill.

Split prefill runs a prompt exactly through layers 0..s, then fills every layer above s
from the residual stream at s instead of running those layers: each upper layer gets what
decoding will read from the prompt -- a gated-delta layer its pre-convolution q/k/v
(`in_proj_qkv`) and gates a, b, from which its own convolution and recurrence build the
state exactly; a latent-attention layer its `kv_a_proj` output, the latent and the rotary
key before its own norm and RoPE (borrow_profile: rotary keys do not transfer between
layers, so the target layer positions them itself). Decoding then runs the whole stack.

Two measurements, no training -- centered ridge maps from the split's residual stream
(both branches), fitted on calibration text, give a first look at how good a start a
trained projector gets (not a bound on it):

* **reconstruction** of each predicted input on held-out text, per part (q, k, v, a, b;
  latent, rotary): R^2 about the held-out means, and skill against the calibration mean.
* **continuation NLL**, the decision: each held-out document's first 3/4 is the prompt,
  its upper layers filled from the maps, and the rest is scored per token against an exact
  prefill, by distance from the boundary (first token, 2-8, 9-32, 33+), paired per
  document. Errors compound through the stack and the recurrences, so this, not R^2, is
  what a split has to pass (borrow_sweep: the best pattern by isolated reconstruction came
  fourth once assembled). The boundary follows KV Prediction: fills cover the prompt up to
  its second-to-last token, and the last prompt token runs the full stack once reading
  them, so the first continuation token's loss is the one a real split prefill gets.
  An exact recent tail (--tails; the Qwen3.8-27B split-prefill work kept the last ~2K of a
  32K prompt exact and recovered most of the loss: 6.27 -> 5.49 PPL against 5.29 exact)
  and exact anchors every ubatch (--anchor-every) are swept the same way.
  Calibration and probe text mix WikiText with held-out chat, agent and code captures:
  a projector fitted on prose alone does worst on code (the same work).
  A mean-only fill (calibration means) is a baseline; an identity fill checks the wiring,
  and every fill asserts that each target was intercepted.

The whole-sequence emulation equals a real split prefill for every continuation position
(Codex review, codex-review-split/REVIEW.md): the gated-delta z gate and all residual and
hyper-connection work are token-local, and CSA2 derives keys, values and index keys from
the filled latent. It is a mathematical, not a kernel-level, equivalence; a cached-decode
parity check belongs with the real implementation.

Also the arithmetic of the payoff: a dense map costs (4096+1) x outputs per token, set
against the parameters of the layers it replaces (not a latency estimate).

    python scratch/dense_gr/split_profile.py --checkpoint scratch/dense_gr/merges-long1/u50 \\
        --splits 11 15 19 --output scratch/csa2-eval/split-profile-u50.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from borrow_profile import documents  # noqa: E402  (imports smoke_train's shims)

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402

RIDGE = 1e-4
BINS = (("first", 0, 1), ("2-8", 1, 8), ("9-32", 8, 32), ("33+", 32, None))


def targets(model, split):
    """{name: (module, parts)} for every cache-feeding projection above `split`."""
    found = {}
    for index, layer in enumerate(model.model.layers):
        if index <= split:
            continue
        attn = getattr(layer, "linear_attn", None)
        if attn is not None:
            key = attn.key_dim if hasattr(attn, "key_dim") else attn.in_proj_qkv.out_features // 3
            value = attn.in_proj_qkv.out_features - 2 * key
            found["%d.qkv" % index] = (attn.in_proj_qkv, dict(q=(0, key), k=(key, 2 * key),
                                                               v=(2 * key, 2 * key + value)))
            found["%d.a" % index] = (attn.in_proj_a, dict(a=(0, attn.in_proj_a.out_features)))
            found["%d.b" % index] = (attn.in_proj_b, dict(b=(0, attn.in_proj_b.out_features)))
        else:
            attn = layer.self_attn
            latent = attn.latent_dim
            found["%d.kv" % index] = (attn.kv_a_proj, dict(latent=(0, latent),
                                                           rotary=(latent, attn.kv_a_proj.out_features)))
    return found


class Hooks:
    """Record the split's residual stream and every target's output, or overwrite the
    targets' prompt positions with predictions from the residual stream."""

    def __init__(self, model, splits):
        self.model, self.splits = model, splits
        self.mode, self.rows, self.maps, self.means, self.split = "off", None, {}, {}, None
        self.stream, self.outputs, self.filled = {}, {}, set()
        self.handles = []
        for s in splits:
            self.handles.append(model.model.layers[s].register_forward_hook(self._stream(s)))
        self.patched = []
        for name, (module, _) in targets(model, min(splits)).items():
            if name.endswith(".kv"):
                # CSA2 fuses kv_a_proj into one multiply with the query projections
                # (`_project`), so the module's own hook never fires: wrap the fused call.
                attn = model.model.layers[int(name.split(".")[0])].self_attn
                if attn.mode != "full":
                    raise ValueError("layer %s borrows its latent; nothing to fill" % name)
                attn._project = self._fused(name, attn._project)
                self.patched.append(attn)
            else:
                self.handles.append(module.register_forward_hook(
                    lambda _m, _a, output, name=name: self._apply(name, output)))

    def _stream(self, s):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.stream[s] = hidden[0].reshape(hidden.shape[1], -1).float()
        return hook

    def _fused(self, name, project):
        def wrapped(hidden_states):
            parts = list(project(hidden_states))
            parts[1] = self._apply(name, parts[1])  # [q_proj, kv_a_proj, index_q_proj]
            return tuple(parts)
        return wrapped

    def _apply(self, name, output):
        if self.mode == "record":
            self.outputs[name] = output[0].float()
        elif self.mode in ("fill", "mean", "identity") and name in self.maps:
            rows = self.rows.to(output.device)  # the approximated prompt positions
            if self.mode == "identity":
                fill = output[0, rows].float()
            else:
                x = self.stream[self.split][rows]
                weight = self.maps[name]
                fill = (x @ weight[:, :-1].T + weight[:, -1] if self.mode == "fill"
                        else self.means[name].to(x.device).expand(len(rows), -1))
            output = output.clone()
            output[0, rows] = fill.to(output.dtype)
            self.filled.add(name)
        return output

    def remove(self):
        for handle in self.handles:
            handle.remove()
        for attn in self.patched:
            del attn._project  # back to the class method


@torch.no_grad()
def forward(model, ids):
    return model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state


@torch.no_grad()
def continuation_losses(model, state, ids, prompt):
    """Per-token NLL of ids[prompt:], from the states at prompt-1 .. -2."""
    logits = model.lm_head(state[0, prompt - 1:-1]).float()
    return torch.nn.functional.cross_entropy(logits, ids[0, prompt:], reduction="none").cpu()


def mean_se(values):
    if not values:
        return [None, None]
    mean = sum(values) / len(values)
    if len(values) < 2:
        return [mean, None]
    return [mean, (sum((v - mean) ** 2 for v in values) / (len(values) - 1) / len(values)) ** 0.5]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chat", type=Path, nargs="+",
                        default=[Path("../teacher-cache-expand-chat"), Path("../teacher-cache-agent-smol-b"),
                                 Path("../teacher-cache-expand-code-w8")],
                        help="held-out captures for the second half of the documents (chat, agent, code)")
    parser.add_argument("--tails", type=int, nargs="+", default=[1, 128, 512],
                        help="prompt tokens kept exact at the end (1: only the boundary token)")
    parser.add_argument("--anchor-every", type=int, default=0,
                        help="also keep every Nth prompt token exact (the last of each ubatch)")
    parser.add_argument("--splits", type=int, nargs="+", default=[11, 15, 19])
    parser.add_argument("--count", type=int, default=16, help="documents a kind, calibration and probe each")
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--prompt-share", type=float, default=0.75)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = Qwen35WidenedForCausalLM.from_pretrained(args.checkpoint, dtype=dtype).to(args.device).eval()
    depth = len(model.model.layers)
    if len(set(args.splits)) != len(args.splits) or not all(0 <= s < depth - 1 for s in args.splits):
        raise SystemExit("splits must be distinct layer indices below the last layer")
    if not 0 < args.prompt_share < 1:
        raise SystemExit("--prompt-share must be between 0 and 1")
    if min(args.tails) < 1:
        raise SystemExit("tails must be at least 1 (the boundary token runs exactly)")
    calibration, probe = documents(tokenizer, args.chat, args.count, args.length, args.device)
    # documents() gives `count` WikiText windows then `count` chat documents in each set;
    # with too few chat documents the halves would mislabel, so refuse.
    if len(calibration) != 2 * args.count or len(probe) != 2 * args.count:
        raise SystemExit("expected %d calibration and probe documents, got %d and %d"
                         % (2 * args.count, len(calibration), len(probe)))
    domains = ["wiki"] * args.count + ["captures"] * args.count
    hooks = Hooks(model, args.splits)
    per_split = {s: targets(model, s) for s in args.splits}

    # Centered ridge with an unpenalized intercept, from sums accumulated in float64 (the
    # products too) and held on the CPU.
    gram, cross, xsum, ysum, tokens = {}, {}, {}, {}, 0
    hooks.mode = "record"
    for ids in calibration:
        forward(model, ids)
        tokens += ids.shape[1]
        for s in args.splits:
            x = hooks.stream[s].double()
            gram[s] = gram.get(s, 0) + (x.T @ x).cpu()
            xsum[s] = xsum.get(s, 0) + x.sum(0).cpu()
            for name in per_split[s]:
                y = hooks.outputs[name].double()
                cross[s, name] = cross.get((s, name), 0) + (x.T @ y).cpu()
                if s == min(args.splits):
                    ysum[name] = ysum.get(name, 0) + y.sum(0).cpu()
    print("calibration: %d documents, %d tokens" % (len(calibration), tokens), flush=True)
    maps, diagnostics = {}, {}
    for s in args.splits:
        mx = xsum[s] / tokens
        g = gram[s] - tokens * torch.outer(mx, mx)
        lam = float(g.diagonal().mean()) * RIDGE
        eig = torch.linalg.eigvalsh(g)
        factor = torch.linalg.cholesky(g + lam * torch.eye(g.shape[0], dtype=g.dtype))
        diagnostics[s] = dict(ridge_lambda=lam, eig_min=float(eig[0]), eig_max=float(eig[-1]),
                              condition_ridged=float((eig[-1] + lam) / (eig[0].clamp_min(0) + lam)))
        for name in per_split[s]:
            my = ysum[name] / tokens
            w = torch.cholesky_solve(cross[s, name] - tokens * torch.outer(mx, my), factor)
            if not torch.isfinite(w).all():
                raise SystemExit("non-finite map for split %d %s" % (s, name))
            bias = my - mx @ w
            maps[s, name] = torch.cat([w.T, bias[:, None]], 1).float().to(args.device)
    width = gram[args.splits[0]].shape[0] + 1  # residual stream, both branches, plus the intercept
    del gram, cross
    hooks.means = {name: (total / tokens).float() for name, total in ysum.items()}

    report = dict(checkpoint=str(args.checkpoint), captures=[str(c) for c in args.chat], tails=args.tails,
                  anchor_every=args.anchor_every, calibration_docs=len(calibration),
                  calibration_tokens=tokens, probe_docs=len(probe), prompt_share=args.prompt_share,
                  ridge=RIDGE, length=args.length, splits={})
    # Reconstruction on the probe documents, per target part: R^2 about the probe's own
    # per-channel means, and skill against the calibration mean (what the mean fill uses).
    errors, spread, ssum, ssq, count = {}, {}, {}, {}, 0
    hooks.mode = "record"
    for ids in probe:
        forward(model, ids)
        count += ids.shape[1]
        for s in args.splits:
            x = hooks.stream[s]
            for name in per_split[s]:
                weight, y = maps[s, name], hooks.outputs[name]
                miss = (x @ weight[:, :-1].T + weight[:, -1] - y).pow(2).sum(0).double().cpu()
                off = (y - hooks.means[name].to(y.device)).pow(2).sum(0).double().cpu()
                errors[s, name] = errors.get((s, name), 0) + miss
                spread[s, name] = spread.get((s, name), 0) + off
                if s == min(args.splits):
                    ssum[name] = ssum.get(name, 0) + y.double().sum(0).cpu()
                    ssq[name] = ssq.get(name, 0) + y.double().pow(2).sum(0).cpu()
    layers = model.model.layers
    for s in args.splits:
        rows = {}
        for name, (_, parts) in per_split[s].items():
            centered = ssq[name] - ssum[name].pow(2) / count
            rows[name] = {}
            for part, (a, b) in parts.items():
                err = float(errors[s, name][a:b].sum())
                rows[name][part] = dict(r2=1 - err / max(float(centered[a:b].sum()), 1e-12),
                                        skill=1 - err / max(float(spread[s, name][a:b].sum()), 1e-12))
            print("split %2d  layer %-7s %s" % (s, name, "  ".join(
                "%s R2 %.3f" % (part, v["r2"]) for part, v in rows[name].items())), flush=True)
        outputs = sum(m.out_features for m, _ in per_split[s].values())
        skipped = sum(p.numel() for layer in layers[s + 1:] for p in layer.parameters())
        report["splits"][s] = dict(reconstruction=rows, solve=diagnostics[s], map_parameters=width * outputs,
                                   skipped_layer_parameters=skipped)

    # Continuation NLL against an exact prefill, per token, by distance from the boundary.
    docs = []
    for doc, ids in enumerate(probe):
        prompt = int(ids.shape[1] * args.prompt_share)
        if not 1 < prompt < ids.shape[1]:
            raise SystemExit("probe document %d too short for the prompt share" % doc)
        hooks.mode = "record"  # also records the streams the fills below read
        exact = continuation_losses(model, forward(model, ids), ids, prompt)
        row = dict(domain=domains[doc], tokens=int(ids.shape[1]), prompt=prompt,
                   exact=float(exact.mean()), splits={})
        for s in args.splits:
            hooks.maps = {name: maps[s, name] for name in per_split[s]}
            hooks.split = s
            runs = [("fill", t) for t in args.tails] + [("mean", 1)] + ([("identity", 1)] if doc == 0 else [])
            for mode, tail in runs:
                rows = torch.arange(max(prompt - tail, 0))
                if args.anchor_every:
                    rows = rows[(rows + 1) % args.anchor_every != 0]
                if not len(rows):
                    continue
                hooks.mode, hooks.rows, hooks.filled = mode, rows, set()
                delta = continuation_losses(model, forward(model, ids), ids, prompt) - exact
                if hooks.filled != set(hooks.maps):
                    raise SystemExit("split %d %s: filled %s of %s"
                                     % (s, mode, sorted(hooks.filled), sorted(hooks.maps)))
                if mode == "identity":
                    if float(delta.abs().max()) > 1e-3:
                        raise SystemExit("identity fill moved the loss by %.5f" % float(delta.abs().max()))
                    continue
                cell = {name: float(delta[a:b].mean()) for name, a, b in BINS if len(delta) > a}
                cell["all"] = float(delta.mean())
                row["splits"].setdefault(str(s), {})["%s-tail%d" % (mode, tail)] = cell
        hooks.maps, hooks.split = {}, None
        docs.append(row)
    report["documents"] = docs
    for s in args.splits:
        out = report["splits"][s]
        for key in sorted({k for d in docs for k in d["splits"].get(str(s), {})}):
            for domain in ("wiki", "captures"):
                out["%s_%s" % (key, domain)] = {
                    name: mean_se([d["splits"][str(s)][key][name] for d in docs
                                   if d["domain"] == domain and name in d["splits"][str(s)].get(key, {})])
                    for name in ("all",) + tuple(b[0] for b in BINS)}
            w, c = out["%s_wiki" % key], out["%s_captures" % key]
            print("split %2d  %-12s dNLL  wiki: all %+.4f first %+.4f 2-8 %+.4f 33+ %+.4f | "
                  "captures: all %+.4f first %+.4f 2-8 %+.4f 33+ %+.4f"
                  % (s, key, w["all"][0], w["first"][0], w["2-8"][0], w["33+"][0],
                     c["all"][0], c["first"][0], c["2-8"][0], c["33+"][0]), flush=True)
        print("split %2d  map/skipped params %.2f" % (s, out["map_parameters"] / out["skipped_layer_parameters"]),
              flush=True)
    hooks.remove()
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
