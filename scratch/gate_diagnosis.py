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

    #: Whether the write scales may go negative. Off by default: the question is
    #: whether these paths should be attenuated, not whether an inverted sidecar can
    #: rescue NLL. Turned on deliberately, it answers a different and narrower
    #: question -- alpha pinned at the 0 boundary only says the optimum is <= 0, and
    #: an optimum meaningfully below 0 would mean the value direction is systematically
    #: wrong-signed, which is a parameterisation bug rather than a useless read.
    signed = False

    def project(self):
        """alpha, beta >= 0 and temperature > 0, applied after each optimiser step."""
        with torch.no_grad():
            if not self.signed:
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
              calibration_documents=128, grade="content", signed=False, seed=20260911):
    """Fit alpha, beta, bias, temperature on `confirmation`, report on `screen`."""
    model, collator = load(checkpoint, device)
    fitting = corpus("confirmation", calibration_documents, grade)
    testing = corpus("screen", grade=grade)
    cal = Calibration().to(device)
    cal.signed = signed
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
             "signed": signed,
             "calibration_documents": len(fitting), "steps": steps,
             "screen": {label: {"sum_nll": total, "tokens": count}
                        for label, total, count in rows}}
    (OUT / ("calibration-%s-%s%s.json"
            % (Path(checkpoint).name, grade, "-signed" if signed else ""))).write_text(
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


# --- how sharp is this instrument? --------------------------------------------


def sensitivity(checkpoint, split="screen"):
    """Score the oracle against a distinction already known to be large.

    An AUC near 0.5 only means "no signal" if the instrument could have shown one.
    The layout/content split is the reference: those two token classes differ by
    -0.103 against +0.0035 nats, a 30x effect in the direction the sidecar is supposed
    to work, established independently of any gradient. Whatever AUC the oracle gives
    *that* is the ceiling a per-token predictor could plausibly reach here, and every
    other AUC in this script should be read against it rather than against 1.0.

    Needs the `--grade all` oracle, since the content grade has no layout tokens in it.
    """
    name = Path(checkpoint).name
    packed = dict(np.load(OUT / ("oracle-%s-%s-all.npz" % (name, split)), allow_pickle=True))
    records = corpus(split, grade="all")
    targets = np.concatenate([np.array([record["ids"][i] for i in picked])
                              for record, picked in records])
    heads = len(packed["q"]) // len(targets)
    targets = np.repeat(targets, heads)
    if len(targets) != len(packed["q"]):
        raise ValueError("the oracle and the bundle disagree about which tokens were scored")

    is_layout = np.isin(targets, LAYOUT_TOKEN_IDS)
    print("%s on %s: %d layout verdicts, %d content"
          % (name, split, is_layout.sum(), (~is_layout).sum()))
    for label, mask in (("layout", is_layout), ("content", ~is_layout)):
        q = packed["q"][mask]
        print("  %-8s mean q %+.4e  %.1f%% want opening  median |q| %.2e"
              % (label, q.mean(), 100 * (q < 0).mean(), np.median(np.abs(q))))
    print("\n  the oracle's own AUC on a known 30x effect: %.4f"
          % auc(-packed["q"], is_layout))
    print("  read every predictor AUC against that, not against 1.0")


# --- is |v|/|h| an admission variable, or newline identity leaking? -----------


def deciles(checkpoint, split="screen", bins=10):
    """Sidecar cost against the value write's relative size, within each token class.

    `|v|/|h|` beat every other feature at predicting that the next token is layout
    (0.5362, against the gradient oracle's own 0.5181 on the same question), which makes
    it the cheapest admission variable on offer -- one norm ratio, no projection. But
    the same number would appear if large writes simply happen at newlines, in which
    case it is an identity detector wearing a confidence detector's clothes. The test
    that separates those is whether the cost still varies across deciles *within
    content*, where there are no newlines left to detect.

    Joins the per-token NLL from `row_novelty score` to the per-stream norms recorded by
    `oracle --grade all`: both replay the same bundle records over the same assistant
    targets in the same order, asserted below rather than assumed.
    """
    from scratch.row_novelty import OUT as ROWS, by_document, bootstrap, target_token_ids

    name = Path(checkpoint).name
    recorded = dict(np.load(OUT / ("oracle-%s-%s-all.npz" % (name, split)), allow_pickle=True))
    scores = dict(np.load(ROWS / ("tokens-%s.npz" % name), allow_pickle=True))
    records = load_json(BUNDLE)["splits"][split]["nll"]
    targets = target_token_ids(records)
    heads = len(recorded["q"]) // len(scores["k"])
    if len(scores["k"]) != len(targets) or heads * len(targets) != len(recorded["q"]):
        raise ValueError("the oracle and the per-token scores cover different tokens")

    # |v| is shared across streams and |h| is not, so the ratio is per stream; the mean
    # is what a single admission variable would have to work from.
    ratio = recorded["ratio"].reshape(len(targets), heads).mean(axis=1)
    is_layout = np.isin(targets, LAYOUT_TOKEN_IDS)
    documents = len(scores["doc_ids"])

    for label, keep in (("all assistant tokens", np.ones(len(targets), bool)),
                        ("content only", ~is_layout), ("layout only", is_layout)):
        edges = np.quantile(ratio[keep], np.linspace(0, 1, bins + 1))
        edges[-1] = np.inf
        print("\n%s (%d tokens)" % (label, keep.sum()))
        print("  decile  |v|/|h| range        tokens   layout%%   sidecar cost  [95%]")
        for index in range(bins):
            mask = keep & (ratio >= edges[index]) & (ratio < edges[index + 1])
            if not mask.sum():
                continue
            enabled, bypassed, tokens = by_document(scores, mask, documents)
            cost, low, high = bootstrap(enabled - bypassed, tokens)
            print("  %4d    %.4f - %.4f  %7d   %5.1f%%   %+.6f [%+.6f, %+.6f]"
                  % (index + 1, edges[index], min(edges[index + 1], ratio[keep].max()),
                     int(mask.sum()), 100 * is_layout[mask].mean(), cost, low, high))


# --- what is the convolution actually doing? ----------------------------------


#: Which source position each kernel index reads, at kernel_size 4 and dilation 3.
#: The module pads (K-1)*dilation = 9 on the left, so output[t] = sum_k w[k] x[t+3k-9]:
#: index 3 is the instantaneous tap and 0 is the oldest. Derived, then pinned by
#: `tests/test_gate_diagnosis.py` with an impulse, because an off-by-one here would
#: reverse the conclusion it is used to draw.
TAP_OFFSETS = (-9, -6, -3, 0)

#: Cumulative ablations, newest tap first.
TAP_SETS = (("conv off", ()), ("t only", (3,)), ("t, t-3", (2, 3)),
            ("t, t-3, t-6", (1, 2, 3)), ("full", (0, 1, 2, 3)))


@torch.inference_mode()
def per_token_nll(model, collator, records, device, bypass=False):
    """Assistant-target NLL, one value per token, in the bundle's order."""
    values, counts = [], []
    for record, targets in records:
        ids = record["ids"]
        batch = {k: v.to(device) for k, v in collator([record]).items()}
        positions = torch.arange(0, len(ids) - 1, device=device)
        logits = model(**batch, use_cache=False, logits_to_keep=positions,
                       sidecar_enabled=not bypass).logits[0].float()
        index = torch.as_tensor(targets - 1, device=device)
        target = torch.as_tensor([ids[i] for i in targets], device=device)
        values.append(F.cross_entropy(logits[index], target, reduction="none").cpu().numpy())
        counts.append(len(targets))
    return np.concatenate(values), np.array(counts)


def taps(checkpoint, device="cuda:0", split="screen"):
    """Does the layout gain come from the table read, or from the delayed taps?

    The convolution is dilated by ngram_size, so it mixes table-derived features across
    t, t-3, t-6 and t-9 -- a tiny causal sequence model over n-gram memory rather than a
    lookup. If the instantaneous tap alone recovers the layout gain, the mechanism is
    "inject n-gram knowledge". If the gain needs the delayed taps, the mechanism is
    "run a temporal filter over n-gram memory", which is a different thing to build.
    """
    from scratch.row_novelty import by_document, bootstrap

    model, collator = load(checkpoint, device)
    ple = model.model.layers[model.config.sidecar_layer_index].sidecar.ple
    records = corpus(split, grade="all")
    targets = np.concatenate([np.array([r["ids"][i] for i in t]) for r, t in records])
    is_layout = np.isin(targets, LAYOUT_TOKEN_IDS)

    bypassed, counts = per_token_nll(model, collator, records, device, bypass=True)
    documents = len(counts)
    document_of = np.repeat(np.arange(documents), counts)
    print("%d documents, %d assistant tokens, %d layout"
          % (documents, len(targets), is_layout.sum()))
    print("kernel index -> source position: %s"
          % ", ".join("%d: t%+d" % (k, TAP_OFFSETS[k]) for k in range(len(TAP_OFFSETS))))

    saved = ple.conv1d.weight.detach().clone()
    rows = []
    try:
        for label, keep in TAP_SETS:
            with torch.no_grad():
                ple.conv1d.weight.copy_(saved)
                for index in range(saved.shape[-1]):
                    if index not in keep:
                        ple.conv1d.weight[..., index] = 0.0
            enabled, _ = per_token_nll(model, collator, records, device)
            rows.append((label, enabled))
            print("  scored %s" % label, flush=True)
    finally:
        with torch.no_grad():
            ple.conv1d.weight.copy_(saved)

    for name, mask in (("all", np.ones(len(targets), bool)),
                       ("content", ~is_layout), ("layout", is_layout)):
        print("\n%s (%d tokens): sidecar cost against bypassed" % (name, mask.sum()))
        for label, enabled in rows:
            packed = {"doc": document_of, "enabled": enabled, "bypassed": bypassed}
            a, b, tokens = by_document(packed, mask, documents)
            cost, low, high = bootstrap(a - b, tokens)
            print("  %-14s %+.6f [%+.6f, %+.6f]" % (label, cost, low, high))


# --- score a fixed calibration, with an interval ------------------------------


def apply_scalars(checkpoint, alpha, beta, bias, temperature, device="cuda:0",
                  split="screen", grade="content"):
    """Evaluate one (alpha, beta, bias, temperature) with a paired bootstrap.

    `calibrate` reports sums, which is enough to see alpha and beta go to a boundary but
    not enough to say whether a small calibrated gain is distinguishable from bypass.
    The signed content fit lands at -0.000524, which is exactly the size where that
    matters -- and the same scalars have to be scorable on the shuffled arm, since an
    inverted write that helps there too is a generic regulariser rather than content.
    """
    from scratch.row_novelty import by_document, bootstrap

    model, collator = load(checkpoint, device)
    records = corpus(split, grade=grade)
    cal = Calibration().to(device)
    cal.gate_delta = cal.conv_delta = None
    with torch.no_grad():
        cal.raw_alpha.fill_(alpha)
        cal.raw_beta.fill_(beta)
        cal.bias.fill_(bias)
        cal.raw_temperature.fill_(temperature)

    bypassed, counts = per_token_nll(model, collator, records, device, bypass=True)
    with instrumented(model, cal):
        enabled, _ = per_token_nll(model, collator, records, device)
    documents = len(counts)
    packed = {"doc": np.repeat(np.arange(documents), counts),
              "enabled": enabled, "bypassed": bypassed}
    mask = np.ones(len(enabled), bool)
    a, b, tokens = by_document(packed, mask, documents)
    cost, low, high = bootstrap(a - b, tokens)
    print(json.dumps({"checkpoint": Path(checkpoint).name, "split": split, "grade": grade,
                      "alpha": alpha, "beta": beta, "bias": bias,
                      "temperature": temperature, "tokens": int(tokens.sum()),
                      "bypassed_nll": float(b.sum() / tokens.sum()),
                      "calibrated_nll": float(a.sum() / tokens.sum()),
                      "cost": cost, "ci": [low, high]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "oracle", "calibrate", "logistic", "sensitivity", "deciles",
                 "taps", "apply"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--device", default="cuda:0")
        if name in ("oracle", "logistic", "sensitivity", "deciles", "taps", "apply"):
            command.add_argument("--split", default="screen")
        if name in ("oracle", "calibrate", "logistic", "apply"):
            command.add_argument("--grade", default="content",
                                 choices=("content", "layout", "all"))
        if name == "apply":
            command.add_argument("--alpha", type=float, required=True)
            command.add_argument("--beta", type=float, required=True)
            command.add_argument("--bias", type=float, default=0.0)
            command.add_argument("--temperature", type=float, default=1.0)
        if name == "oracle":
            command.add_argument("--limit", type=int, default=None)
        if name == "calibrate":
            command.add_argument("--steps", type=int, default=150)
            command.add_argument("--batch", type=int, default=6)
            command.add_argument("--lr", type=float, default=0.05)
            command.add_argument("--calibration-documents", type=int, default=128)
            command.add_argument("--signed", action="store_true",
                                 help="let alpha and beta go negative")
    args = parser.parse_args()
    if args.command == "verify":
        return verify(args.checkpoint, args.device)
    if args.command == "oracle":
        return oracle(args.checkpoint, args.device, args.split, args.limit, args.grade)
    if args.command == "calibrate":
        return calibrate(args.checkpoint, args.device, args.steps, args.batch, args.lr,
                         args.calibration_documents, args.grade, args.signed)
    if args.command == "logistic":
        return logistic(args.checkpoint, args.split, args.grade)
    if args.command == "deciles":
        return deciles(args.checkpoint, args.split)
    if args.command == "taps":
        return taps(args.checkpoint, args.device, args.split)
    if args.command == "apply":
        return apply_scalars(args.checkpoint, args.alpha, args.beta, args.bias,
                             args.temperature, args.device, args.split, args.grade)
    return sensitivity(args.checkpoint, args.split)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
