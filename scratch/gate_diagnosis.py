"""Ask a frozen checkpoint what its admission gate should have done.

The layer-1 sidecar helps on `\\n` and `<think>` and costs +0.003512 on the other 94.5%
of assistant tokens (``scratch/row-novelty/RESULTS.md``). Three structural facts in
``ple_gated_sidecar.py`` could each explain that, and all three are testable on the
existing C1 checkpoint without training anything:

1. **The gate never sees the table.** ``_admission(stream)`` reads only the pre-sidecar
   residual stream and the learned directions, so real rows and shuffled rows get the
   same admission decision on the same input. The table enters only through
   ``value_proj``. A context-only gate can learn "this looks like a newline context ->
   open"; it cannot learn "this n-gram disagrees with what I currently believe -> close".
2. **The convolution is ungated.** The forward is literally
   ``stream + gate * value + short_conv(value)``. The gate's absence there is sound --
   it cancels inside the RMS normalisation it would pass through -- but sound is not the
   same as intended, and the branch escapes admission control entirely.
3. **The gate's neutral level is 1.0.** ``2 * mean_k sigmoid(z)`` spans (0, 2) and sits
   at 1.0 for an undecided gate, so "no opinion" writes the value at full strength.

So rather than redesign the gate, measure. Four diagnostic scalars are fitted to a
frozen checkpoint,

    h' = h + alpha * g_{b,t}(h) * v + beta * c(v),    g_{b,t} = 2 mean_k sigmoid(t z_k + b)

where ``z_k`` is the existing signed-square-root score before its sigmoid. alpha << 1
says the value write is too strong; beta ~ 0 says the convolution is hurting; b < 0 says
admission is globally too open; t > 1 says the gate ranks contexts correctly but is not
selective enough; alpha and beta both ~ 0 says the representation itself is not useful
for content. alpha and beta are held >= 0: the question is whether these paths should be
attenuated, not whether an inverted sidecar can rescue NLL.

Then one level deeper. With a per-token additive perturbation on the gate, a single
content-only backward pass gives every token an oracle:

    q_t = dL_content / dg_t,    q_t > 0 -> closing would have helped, q_t < 0 -> opening

which is an admission label for every token without training a second model. Whether the
*existing* gate score predicts that label is then an AUC. Near 0.5 and no amount of bias
or temperature can help, because the gate does not hold the information. Well above 0.5
and the mechanism is right and only the threshold is wrong.

The table-aware candidate costs one dot product rather than resurrecting the 26.2M
parameter ``key_proj``: ``value_proj`` already puts the table read in the residual space,
so ``cos(rms(h), rms(v))`` is available for free.

Calibration fits on the bundle's ``confirmation`` split and reports on ``screen``, which
every other result in this project is measured on and which nothing here has touched.

    python scratch/gate_diagnosis.py verify    --checkpoint ...
    python scratch/gate_diagnosis.py oracle    --checkpoint ...
    python scratch/gate_diagnosis.py calibrate --checkpoint ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # `scratch` as a package

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from distillkit.ngram_table import GGUFNGramTable
from scratch.row_novelty import BUNDLE, LAYOUT_TOKEN_IDS, TABLE, load_json

OUT = Path("scratch/gate-diagnosis")


# --- the instrumented forward -------------------------------------------------


def admission_score(ple, stream):
    """``_admission`` up to but not including its sigmoid.

    Reimplemented rather than hooked because the diagnostic needs ``z`` itself. Every
    line is the module's; ``verify`` asserts the pair agree bit for bit, which is what
    licenses reading anything off this.
    """
    query = stream.float()
    query = query * torch.rsqrt(query.pow(2).mean(-1, keepdim=True) + ple.eps)
    direction = ple.gate.float()
    direction = direction * torch.rsqrt(direction.pow(2).mean(-1, keepdim=True) + ple.eps)
    direction = direction * (1.0 + ple.sharpness_delta.float()).unsqueeze(-1)
    raw = torch.einsum("...hd,hkd->...hk", query, direction) / math.sqrt(ple.hidden_size)
    return raw.abs().clamp_min(1e-6).sqrt() * raw.sign()


class Calibration(nn.Module):
    """alpha, beta, bias, temperature, plus optional per-token oracle perturbations.

    The perturbations are additive and start at zero, so ``d loss / d perturbation`` is
    exactly ``dL/dg_t`` at the checkpoint's own operating point -- a multiplicative
    probe would return ``g_t dL/dg_t`` and rescale every token by its own gate.
    """

    def __init__(self):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.raw_beta = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.raw_temperature = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.gate_delta = None          # [batch, seq, hc, 1] when probing
        self.conv_delta = None          # [batch, seq, hc, 1] when probing
        self.recorded = {}

    def project(self):
        """alpha, beta >= 0 and temperature > 0, applied after each optimiser step."""
        with torch.no_grad():
            self.raw_alpha.clamp_(min=0.0)
            self.raw_beta.clamp_(min=0.0)
            self.raw_temperature.clamp_(min=1e-3)

    def arm_probes(self, shape, device):
        self.gate_delta = torch.zeros(shape, dtype=torch.float32, device=device,
                                      requires_grad=True)
        self.conv_delta = torch.zeros(shape, dtype=torch.float32, device=device,
                                      requires_grad=True)
        return self.gate_delta, self.conv_delta


def calibrated_forward(ple, cal, stream, features):
    """``h + alpha g_{b,t} v + beta c(v)``, with the checkpoint's own weights."""
    features = features.to(dtype=stream.dtype)
    value = ple.value_proj(features)
    z = admission_score(ple, stream)
    gate = torch.sigmoid(cal.raw_temperature * z + cal.bias).mean(-1, keepdim=True) * 2.0
    conv = ple._short_conv(value)
    if cal.gate_delta is not None:
        gate = gate + cal.gate_delta
        conv = conv * (1.0 + cal.conv_delta.to(conv.dtype))
        cal.recorded = {"z": z.detach(), "gate": gate.detach(),
                        "value": value.detach(), "stream": stream.detach()}
    written = cal.raw_alpha * gate.to(value.dtype) * value.unsqueeze(-2)
    return stream + written + cal.raw_beta.to(conv.dtype) * conv


class instrumented:
    """Swap the sidecar's forward for the calibrated one for the duration of a block."""

    def __init__(self, model, cal):
        self.ple = model.model.layers[model.config.sidecar_layer_index].sidecar.ple
        self.cal = cal

    def __enter__(self):
        self.saved = self.ple.forward
        self.ple.forward = lambda stream, features: calibrated_forward(
            self.ple, self.cal, stream, features)
        return self.ple

    def __exit__(self, *exception):
        self.ple.forward = self.saved
        return False


# --- the corpus ---------------------------------------------------------------


def content_targets(record, start=1, grade="content"):
    """Assistant target indices under one grading set.

    `content` is the default and the grade every verdict here is read on: `\\n` and
    the think tags are 5.5% of assistant tokens and carry 2.45x the whole measured
    win, so grading on them measures line-break placement
    (`scratch/row-novelty/RESULTS.md`). `layout` is those tokens alone, which is the
    contrast that settles it -- if the fitted scalars want the module switched off for
    content and left on for layout, it is a layout device, and the two grades say so
    in the same units.
    """
    ids = record["ids"]
    picked = [i for low, high in record["roles"].get("assistant", [])
              for i in range(max(low, start), min(high, len(ids)))]
    if grade == "all":
        keep = picked
    elif grade == "layout":
        keep = [i for i in picked if ids[i] in LAYOUT_TOKEN_IDS]
    elif grade == "content":
        keep = [i for i in picked if ids[i] not in LAYOUT_TOKEN_IDS]
    else:
        raise ValueError("grade must be content, layout or all")
    return np.array(keep, dtype=np.int64)


def corpus(split, limit=None, grade="content"):
    """Records from one bundle split that have something to score under this grade."""
    records = load_json(BUNDLE)["splits"][split]["nll"]
    kept = [(r, content_targets(r, grade=grade)) for r in records]
    kept = [(r, t) for r, t in kept if len(t)]
    return kept[:limit] if limit else kept


def content_loss(model, collator, record, targets, device, reduction="sum"):
    """Content-only CE, with the graph intact so the probes and scalars can be read."""
    ids = record["ids"]
    batch = {k: v.to(device) for k, v in collator([record]).items()}
    positions = torch.arange(0, len(ids) - 1, device=device)
    logits = model(**batch, use_cache=False, logits_to_keep=positions,
                   sidecar_enabled=True).logits[0].float()
    index = torch.as_tensor(targets - 1, device=device)
    target = torch.as_tensor([ids[i] for i in targets], device=device)
    return F.cross_entropy(logits[index], target, reduction=reduction)


def load(checkpoint, device):
    from distillkit.independent_eval import load_checkpoint, make_collator
    model, audit = load_checkpoint(checkpoint, device, torch.bfloat16)
    if audit["variant"] != "ple_gated":
        raise ValueError("this diagnostic is specific to the ple_gated sidecar, got %r"
                         % audit["variant"])
    collator = make_collator(load_json(BUNDLE)["pad_token_id"], GGUFNGramTable(TABLE))
    return model, collator


# --- is the reimplementation faithful? ----------------------------------------


def verify(checkpoint, device="cuda:0"):
    """The calibrated forward at alpha=beta=t=1, b=0 must BE the module's forward.

    Everything downstream reads `z` off a reimplementation of `_admission`. If that
    drifts from the module by so much as a normalisation constant, every number this
    script prints is about a different model than the checkpoint.
    """
    model, collator = load(checkpoint, device)
    ple = model.model.layers[model.config.sidecar_layer_index].sidecar.ple
    record, _ = corpus("screen", limit=1)[0]
    batch = {k: v.to(device) for k, v in collator([record]).items()}
    sidecar = model.model.layers[model.config.sidecar_layer_index].sidecar

    captured = {}
    handle = ple.register_forward_pre_hook(
        lambda module, args: captured.update(stream=args[0], features=args[1]))
    with torch.inference_mode():
        model(**batch, use_cache=False, logits_to_keep=1, sidecar_enabled=True)
    handle.remove()
    stream, features = captured["stream"], captured["features"]

    with torch.inference_mode():
        reference = ple(stream, features)
        mine = calibrated_forward(ple, Calibration().to(device), stream, features)
        gate_reference = ple._admission(stream)
        gate_mine = 2.0 * torch.sigmoid(admission_score(ple, stream)).mean(-1, keepdim=True)

    forward_gap = (reference.float() - mine.float()).abs().max().item()
    gate_gap = (gate_reference - gate_mine).abs().max().item()
    print(json.dumps({"stream": list(stream.shape), "features": list(features.shape),
                      "max_forward_difference": forward_gap,
                      "max_gate_difference": gate_gap,
                      "gate_mean": gate_reference.mean().item(),
                      "gate_std": gate_reference.std().item(),
                      "gate_min": gate_reference.min().item(),
                      "gate_max": gate_reference.max().item()}, indent=2))
    if forward_gap or gate_gap:
        raise SystemExit("the calibrated forward is not the module's forward")
    print("exact")


# --- ranking, without scipy ---------------------------------------------------


def average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    ordered = values[order]
    start = 0
    for stop in range(1, len(ordered) + 1):
        if stop == len(ordered) or ordered[stop] != ordered[start]:
            if stop - start > 1:                      # ties share their mean rank
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return ranks


def auc(scores, labels):
    """P(score of a positive > score of a negative), ties counted as half."""
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return float("nan")
    ranks = average_ranks(np.asarray(scores, dtype=np.float64))
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives)


# --- what does every token want the gate to do? -------------------------------


def oracle(checkpoint, device="cuda:0", split="screen", limit=None, grade="content"):
    """One backward pass per document under one grade, recording the gate's oracle."""
    model, collator = load(checkpoint, device)
    cal = Calibration().to(device)
    records = corpus(split, limit, grade)

    columns = {name: [] for name in
               ("q", "r", "gate", "z0", "z1", "zmean", "cos", "value_norm", "ratio",
                "stream_index", "document")}
    for number, (record, targets) in enumerate(records):
        with instrumented(model, cal) as ple:
            probe_shape = (1, len(record["ids"]), ple.hc_count, 1)
            gate_delta, conv_delta = cal.arm_probes(probe_shape, device)
            with torch.enable_grad():
                loss = content_loss(model, collator, record, targets, device)
                q, r = torch.autograd.grad(loss, [gate_delta, conv_delta])
        kept = cal.recorded
        # A target at index i was predicted from position i-1, which is where the
        # sidecar wrote; that is the position whose admission decision this token's
        # loss is a verdict on.
        rows = torch.as_tensor(targets - 1, device=device)
        stream = kept["stream"][0].index_select(0, rows).float()       # [n, hc, d]
        value = kept["value"][0].index_select(0, rows).float()         # [n, d]
        normed_stream = F.normalize(stream, dim=-1)
        normed_value = F.normalize(value, dim=-1).unsqueeze(-2)
        cosine = (normed_stream * normed_value).sum(-1)                # [n, hc]
        z = kept["z"][0].index_select(0, rows).float()                 # [n, hc, k]
        gate = kept["gate"][0].index_select(0, rows).float().squeeze(-1)
        heads = stream.shape[1]
        columns["q"].append(q[0].index_select(0, rows).float().squeeze(-1).cpu().numpy())
        columns["r"].append(r[0].index_select(0, rows).float().squeeze(-1).cpu().numpy())
        columns["gate"].append(gate.cpu().numpy())
        columns["z0"].append(z[..., 0].cpu().numpy())
        columns["z1"].append(z[..., 1].cpu().numpy() if z.shape[-1] > 1
                             else z[..., 0].cpu().numpy())
        columns["zmean"].append(z.mean(-1).cpu().numpy())
        columns["cos"].append(cosine.cpu().numpy())
        columns["value_norm"].append(
            value.norm(dim=-1, keepdim=True).expand(-1, heads).cpu().numpy())
        columns["ratio"].append(
            (value.norm(dim=-1, keepdim=True) / stream.norm(dim=-1)).cpu().numpy())
        columns["stream_index"].append(
            np.tile(np.arange(heads), (len(targets), 1)).astype(np.float32))
        columns["document"].append(np.full((len(targets), heads), number, dtype=np.int32))
        if (number + 1) % 32 == 0:
            print("oracle %d/%d" % (number + 1, len(records)), flush=True)

    packed = {name: np.concatenate(values).ravel() for name, values in columns.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    name = Path(checkpoint).name
    np.savez(OUT / ("oracle-%s-%s-%s.npz" % (name, split, grade)), **packed)
    report_oracle(packed, name, "%s/%s" % (split, grade))


def report_oracle(packed, name, split):
    q = packed["q"]
    opens = q < 0
    print("\n%s on %s: %d stream-token verdicts, %.1f%% want the gate opened further"
          % (name, split, len(q), 100 * opens.mean()))
    print("  gate now: mean %.4f  std %.4f  min %.4f  max %.4f"
          % (packed["gate"].mean(), packed["gate"].std(),
             packed["gate"].min(), packed["gate"].max()))
    # What the gate is admitting, in the stream's own terms. A write that is tiny and
    # orthogonal cannot be fixed by admitting it more selectively, so these two belong
    # beside the AUCs rather than in a follow-up.
    print("  value write: |v|/|h| median %.4f, cos(h, v) mean %+.4f std %.4f"
          % (np.median(packed["ratio"]), packed["cos"].mean(), packed["cos"].std()))
    # An oracle of pure numerical noise would make every AUC 0.5 by construction, so
    # say how much of it is signal before reporting that none of it is predictable.
    print("  oracle: median |q| %.3e, %.1f%% of verdicts below 1e-6, %.1f%% want closing"
          % (np.median(np.abs(q)), 100 * (np.abs(q) < 1e-6).mean(), 100 * (q > 0).mean()))

    # Tokens whose gradient is in the noise carry no verdict; |q| below the median is
    # mostly value ~ 0 positions where the gate cannot matter either way.
    decisive = np.abs(q) >= np.median(np.abs(q))
    for label, mask in (("all tokens", np.ones(len(q), bool)), ("|q| above median", decisive)):
        print("\n  %s (%d)" % (label, mask.sum()))
        for predictor in ("z0", "z1", "zmean", "gate", "cos", "value_norm", "ratio"):
            score = packed[predictor][mask]
            if predictor in ("value_norm", "ratio"):
                score = np.log(np.maximum(score, 1e-12))
            print("    %-12s oracle AUC %.4f" % (predictor, auc(score, opens[mask])))

    if packed["z0"].std() and packed["z1"].std():
        correlation = np.corrcoef(packed["z0"], packed["z1"])[0, 1]
        print("\n  correlation between the two gate directions: %+.4f" % correlation)
    for stream in np.unique(packed["stream_index"]).astype(int):
        mask = packed["stream_index"] == stream
        print("  stream %d: gate mean %.4f, %.1f%% want opening, zmean AUC %.4f"
              % (stream, packed["gate"][mask].mean(), 100 * opens[mask].mean(),
                 auc(packed["zmean"][mask], opens[mask])))

    r = packed["r"]
    print("\n  convolution: dL/dbeta_t mean %+.6g, %.1f%% of tokens want it attenuated"
          % (r.mean(), 100 * (r > 0).mean()))
    print("  summed over tokens, dL/dbeta = %+.6g (positive means beta should fall)" % r.sum())
    print("  summed over tokens, dL/dalpha-like gate pressure = %+.6g" % (q * packed["gate"]).sum())


# --- fitting the four scalars -------------------------------------------------


def evaluate_content(model, collator, records, device, cal=None, bypass=False):
    """Summed content-only NLL and its token count, under a given calibration."""
    total, tokens = 0.0, 0
    with torch.inference_mode():
        for record, targets in records:
            ids = record["ids"]
            batch = {k: v.to(device) for k, v in collator([record]).items()}
            positions = torch.arange(0, len(ids) - 1, device=device)
            logits = model(**batch, use_cache=False, logits_to_keep=positions,
                           sidecar_enabled=not bypass).logits[0].float()
            index = torch.as_tensor(targets - 1, device=device)
            target = torch.as_tensor([ids[i] for i in targets], device=device)
            total += F.cross_entropy(logits[index], target, reduction="sum").item()
            tokens += len(targets)
    return total, tokens


def scored(model, collator, records, device, cal, bypass=False):
    if bypass or cal is None:
        return evaluate_content(model, collator, records, device, bypass=bypass)
    with instrumented(model, cal):
        return evaluate_content(model, collator, records, device)


def calibrate(checkpoint, device="cuda:0", steps=150, batch=6, lr=0.05,
              calibration_documents=128, grade="content", seed=20260911):
    """Fit alpha, beta, bias, temperature on `confirmation`, report on `screen`."""
    model, collator = load(checkpoint, device)
    fitting = corpus("confirmation", calibration_documents, grade)
    testing = corpus("screen", grade=grade)
    cal = Calibration().to(device)
    cal.gate_delta = cal.conv_delta = None
    optimiser = torch.optim.Adam(cal.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    history = []
    for step in range(steps):
        optimiser.zero_grad(set_to_none=True)
        picked = rng.choice(len(fitting), size=min(batch, len(fitting)), replace=False)
        total, tokens = 0.0, 0
        with instrumented(model, cal):
            for index in picked:
                record, targets = fitting[index]
                with torch.enable_grad():
                    loss = content_loss(model, collator, record, targets, device)
                (loss / len(picked)).backward()
                total += loss.item()
                tokens += len(targets)
        optimiser.step()
        cal.project()
        history.append(total / tokens)
        if (step + 1) % 10 == 0:
            print("step %3d  batch content NLL %.5f  alpha %.4f beta %.4f bias %+.4f t %.4f"
                  % (step + 1, np.mean(history[-10:]), cal.raw_alpha.item(),
                     cal.raw_beta.item(), cal.bias.item(), cal.raw_temperature.item()),
                  flush=True)

    fitted = {"alpha": cal.raw_alpha.item(), "beta": cal.raw_beta.item(),
              "bias": cal.bias.item(), "temperature": cal.raw_temperature.item()}
    print("\nfitted on %d confirmation documents, graded on %s: %s"
          % (len(fitting), grade,
             json.dumps({k: round(v, 5) for k, v in fitted.items()})))

    neutral = Calibration().to(device)
    neutral.gate_delta = neutral.conv_delta = None
    rows = []
    for label, configuration, bypass in (("bypassed", None, True),
                                         ("checkpoint", neutral, False),
                                         ("calibrated", cal, False)):
        total, tokens = scored(model, collator, testing, device, configuration, bypass)
        rows.append((label, total, tokens))
        print("  %-12s content NLL %.6f over %d tokens" % (label, total / tokens, tokens))
    baseline = dict((label, total) for label, total, _ in rows)
    tokens = rows[0][2]
    print("\n  sidecar cost on screen %s, enabled - bypassed:" % grade)
    for label in ("checkpoint", "calibrated"):
        print("    %-12s %+.6f" % (label, (baseline[label] - baseline["bypassed"]) / tokens))

    OUT.mkdir(parents=True, exist_ok=True)
    write = {"checkpoint": str(checkpoint), "fitted": fitted, "grade": grade,
             "calibration_documents": len(fitting), "steps": steps,
             "screen": {label: {"sum_nll": total, "tokens": count}
                        for label, total, count in rows}}
    (OUT / ("calibration-%s-%s.json" % (Path(checkpoint).name, grade))).write_text(
        json.dumps(write, indent=2), encoding="utf-8")


# --- would a table-aware gate have done better? -------------------------------


def fit_logistic(features, labels, steps=400, lr=0.1, weight_decay=1e-4):
    """Plain logistic regression; there is no sklearn in this environment."""
    x = torch.as_tensor(features, dtype=torch.float32)
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-8)
    y = torch.as_tensor(labels, dtype=torch.float32)
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimiser = torch.optim.Adam([weight, bias], lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(x @ weight + bias, y).backward()
        optimiser.step()
    return weight.detach(), bias.detach(), x.mean(0), x.std(0)


def logistic(checkpoint, split="screen", grade="content", folds=5, seed=20260911):
    """Does adding the value read to the admission decision buy predictive power?

    Cross-validated by document, never by token: tokens within a document share a
    context and a fold split that ignored that would report a leak as a result.
    """
    name = Path(checkpoint).name
    packed = dict(np.load(OUT / ("oracle-%s-%s-%s.npz" % (name, split, grade)),
                          allow_pickle=True))
    labels = (packed["q"] < 0).astype(np.float32)
    documents = packed["document"].astype(int)
    unique = np.unique(documents)
    rng = np.random.default_rng(seed)
    assignment = {doc: index % folds for index, doc in enumerate(rng.permutation(unique))}
    fold = np.array([assignment[d] for d in documents])

    candidates = {
        "gate score only": ["zmean"],
        "value agreement only": ["cos"],
        "value norm only": ["log_value_norm"],
        "gate + agreement": ["zmean", "cos"],
        "gate + agreement + norm": ["zmean", "cos", "log_value_norm"],
    }
    packed["log_value_norm"] = np.log(np.maximum(packed["value_norm"], 1e-12))
    print("%s on %s: %d verdicts, %d documents, %.1f%% want opening"
          % (name, split, len(labels), len(unique), 100 * labels.mean()))
    print("\n  model                        held-out AUC   weights")
    for label, names in candidates.items():
        matrix = np.stack([packed[n] for n in names], axis=1)
        held_out = np.empty(len(labels))
        for index in range(folds):
            train, test = fold != index, fold == index
            weight, bias, centre, scale = fit_logistic(matrix[train], labels[train])
            x = (torch.as_tensor(matrix[test], dtype=torch.float32) - centre) / scale
            held_out[test] = (x @ weight + bias).numpy()
        weight, _, _, _ = fit_logistic(matrix, labels)
        print("  %-28s %.4f         %s"
              % (label, auc(held_out, labels.astype(bool)),
                 ", ".join("%s %+.3f" % (n, w) for n, w in zip(names, weight.tolist()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "oracle", "calibrate", "logistic"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--device", default="cuda:0")
        if name in ("oracle", "logistic"):
            command.add_argument("--split", default="screen")
        if name != "verify":
            command.add_argument("--grade", default="content",
                                 choices=("content", "layout", "all"))
        if name == "oracle":
            command.add_argument("--limit", type=int, default=None)
        if name == "calibrate":
            command.add_argument("--steps", type=int, default=150)
            command.add_argument("--batch", type=int, default=6)
            command.add_argument("--lr", type=float, default=0.05)
            command.add_argument("--calibration-documents", type=int, default=128)
    args = parser.parse_args()
    if args.command == "verify":
        return verify(args.checkpoint, args.device)
    if args.command == "oracle":
        return oracle(args.checkpoint, args.device, args.split, args.limit, args.grade)
    if args.command == "calibrate":
        return calibrate(args.checkpoint, args.device, args.steps, args.batch, args.lr,
                         args.calibration_documents, args.grade)
    return logistic(args.checkpoint, args.split, args.grade)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
