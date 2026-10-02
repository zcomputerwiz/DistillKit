"""Can a prompt's upper layers be predicted instead of run? A preflight for split prefill.

Split prefill runs a prompt exactly through layers 0..s, then fills every layer above s
from the residual stream at s instead of running those layers: each upper layer gets what
decoding will read from the prompt -- a gated-delta layer its pre-convolution q/k/v
(`in_proj_qkv`) and gates a, b, from which its own convolution and recurrence build the
state exactly; a latent-attention layer its `kv_a_proj` output, the latent and the rotary
key before its own norm and RoPE (borrow_profile: rotary keys do not transfer between
layers, so the target layer positions them itself). Decoding then runs the whole stack.

Two measurements, no training -- ridge maps from the split's residual stream (both
branches), fitted on calibration text, bound how good a start a trained projector gets:

* **R^2** of each predicted input on held-out text, per part (q, k, v, a, b; latent,
  rotary).
* **continuation NLL**, the decision: each held-out document's first 3/4 is the prompt,
  its upper layers filled from the maps, and the last quarter is scored against an exact
  prefill, paired per document. Errors compound through the stack and the recurrences, so
  this, not R^2, is what a split has to pass (borrow_sweep: the best pattern by isolated
  reconstruction came fourth once assembled). A mean-only fill (the maps' intercepts) is
  the floor, and an identity fill checks the wiring (must be 0).

Also the arithmetic of the payoff: a dense map costs (4096+1) x outputs per token, which
is set against the parameters of the layers it replaces.

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
GDN_PARTS = ("q", "k", "v")


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
        self.mode, self.prompt, self.maps, self.means, self.split = "off", 0, {}, {}, None
        self.stream, self.outputs = {}, {}
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
            p = self.prompt
            if self.mode == "identity":
                fill = output[0, :p].float()
            else:
                x = self.stream[self.split][:p]
                weight = self.maps[name]
                fill = (x @ weight[:, :-1].T + weight[:, -1] if self.mode == "fill"
                        else self.means[name].to(x.device).expand(p, -1))
            output = output.clone()
            output[0, :p] = fill.to(output.dtype)
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
def continuation_nll(model, state, ids, prompt):
    hidden = state[0, prompt - 1:-1].float()
    logits = model.lm_head(hidden.to(model.lm_head.weight.dtype)).float()
    return float(torch.nn.functional.cross_entropy(logits, ids[0, prompt:]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chat", type=Path, default=Path("../teacher-cache-expand-chat"))
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
    calibration, probe = documents(tokenizer, [args.chat], args.count, args.length, args.device)
    hooks = Hooks(model, args.splits)
    per_split = {s: targets(model, s) for s in args.splits}

    # Normal equations, accumulated per document in float64 on the CPU: [x | 1] for the
    # residual stream at each split, against every target above it.
    gram, cross, sums, tokens = {}, {}, {}, 0
    hooks.mode = "record"
    for ids in calibration:
        forward(model, ids)
        tokens += ids.shape[1]
        for s in args.splits:
            x = torch.cat([hooks.stream[s], torch.ones_like(hooks.stream[s][:, :1])], 1)
            gram[s] = gram.get(s, 0) + (x.T @ x).double().cpu()
            for name in per_split[s]:
                y = hooks.outputs[name]
                cross[s, name] = cross.get((s, name), 0) + (x.T @ y).double().cpu()
                if s == min(args.splits):
                    sums[name] = sums.get(name, 0) + y.sum(0).double().cpu()
    print("calibration: %d documents, %d tokens" % (len(calibration), tokens), flush=True)
    maps = {}
    for s in args.splits:
        g = gram[s].clone()
        g += torch.eye(g.shape[0], dtype=g.dtype) * g.diagonal().mean() * RIDGE
        factor = torch.linalg.cholesky(g)
        for name in per_split[s]:
            maps[s, name] = torch.cholesky_solve(cross[s, name], factor).T.float().to(args.device)
    width = gram[args.splits[0]].shape[0]  # residual stream, both branches, plus the intercept
    del gram, cross
    means = {name: (total / tokens).float() for name, total in sums.items()}
    hooks.means = means

    report = dict(checkpoint=str(args.checkpoint), calibration_docs=len(calibration),
                  calibration_tokens=tokens, probe_docs=len(probe), prompt_share=args.prompt_share,
                  ridge=RIDGE, splits={})
    # R^2 on the probe documents, per target part.
    errors, totals = {}, {}
    hooks.mode = "record"
    for ids in probe:
        forward(model, ids)
        for s in args.splits:
            x = hooks.stream[s]
            for name, (_, parts) in per_split[s].items():
                weight, y = maps[s, name], hooks.outputs[name]
                miss = (x @ weight[:, :-1].T + weight[:, -1] - y).pow(2)
                spread = (y - means[name].to(y.device)).pow(2)
                for part, (a, b) in parts.items():
                    errors[s, name, part] = errors.get((s, name, part), 0.0) + float(miss[:, a:b].sum())
                    totals[s, name, part] = totals.get((s, name, part), 0.0) + float(spread[:, a:b].sum())
    first = model.model.layers
    for s in args.splits:
        rows = {}
        for name, (_, parts) in per_split[s].items():
            rows[name] = {part: 1 - errors[s, name, part] / totals[s, name, part] for part in parts}
            print("split %2d  layer %-7s %s" % (s, name, "  ".join(
                "%s R2 %.3f" % (part, value) for part, value in rows[name].items())), flush=True)
        outputs = sum(m.out_features for m, _ in per_split[s].values())
        skipped = sum(p.numel() for layer in first[s + 1:] for p in layer.parameters())
        report["splits"][s] = dict(r2=rows, map_parameters=width * outputs,
                                   skipped_layer_parameters=skipped)

    # Continuation NLL: exact prefill against each split's fill, paired per document.
    hooks.split = None
    deltas = {}
    for doc, ids in enumerate(probe):
        prompt = int(ids.shape[1] * args.prompt_share)
        hooks.mode = "record"  # also fills hooks.stream for the fills below
        exact = continuation_nll(model, forward(model, ids), ids, prompt)
        for s in args.splits:
            hooks.maps = {name: maps[s, name] for name in per_split[s]}
            hooks.split, hooks.prompt = s, prompt
            for mode in ("fill", "mean") + (("identity",) if doc == 0 else ()):
                hooks.mode = mode
                deltas.setdefault((s, mode), []).append(
                    continuation_nll(model, forward(model, ids), ids, prompt) - exact)
        hooks.maps, hooks.split = {}, None
    half = len(probe) // 2
    for s in args.splits:
        out = report["splits"][s]
        out["identity_delta"] = deltas[s, "identity"][0]
        for mode in ("fill", "mean"):
            for kind, part in (("wiki", slice(0, half)), ("chat", slice(half, None))):
                d = deltas[s, mode][part]
                mean = sum(d) / len(d)
                se = (sum((v - mean) ** 2 for v in d) / max(len(d) - 1, 1) / len(d)) ** 0.5
                out["%s_%s" % (mode, kind)] = [mean, se]
        print("split %2d  continuation dNLL  ridge fill: wiki %+.4f (se %.4f) chat %+.4f (se %.4f)  "
              "mean fill: wiki %+.4f chat %+.4f  identity %+.5f  map/skipped params %.2f"
              % (s, *out["fill_wiki"], *out["fill_chat"], out["mean_wiki"][0], out["mean_chat"][0],
                 out["identity_delta"], out["map_parameters"] / out["skipped_layer_parameters"]), flush=True)
    hooks.remove()
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
