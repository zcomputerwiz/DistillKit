"""14A: does a cached boundary gradient predict the real change in assistant NLL?

The whole boundary-replay design rests on one first-order claim: for a student that
differs from the teacher only inside layers 20-27,

    L_CE(S) - L_CE(T)  ~=  g_28 . (h_28^S - h_28^T),     g_28 = dL_CE/dh_28 at T

If that holds over the displacement range local training actually produces, the frozen
tail can be replaced by a cached vector and the inner loop gets cheap. If it does not,
every later phase is measuring an approximation error.

This tests it before any cache is built, which is the point: cache generation over the
corpus is hours, and this is minutes. Layers 0-19 are frozen and untouched by the
perturbation, so h_20 is identical between T and S and one full forward per perturbed
model yields both `h_28^S` and the *exact* L_CE(S) -- no replay approximation anywhere
in the measurement of the thing being approximated.

Perturbations stand in for local training: Gaussian noise on the weights of layers
20-27, scaled relative to each tensor's own norm, over a range wide enough to bracket
where the linearisation stops working. What comes out is the usable trust radius, which
is what sets lambda_R in the Phase-A objective.

    python scratch/boundary_replay_check.py --docs 16
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath("scratch/boundary-replay/triton-cache"))

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from distillkit.independent_eval import make_collator, role_spans  # noqa: E402
from distillkit.models.qwen35_widened import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.ngram_table import GGUFNGramTable  # noqa: E402
from distillkit.signals import OfflineHiddenStateSignalSource  # noqa: E402
from distillkit.tp_model import shard_model  # noqa: E402

CHECKPOINT = Path("../runs/widened-plegated-L24-stage1-1m")
CONFIG_FILE = "scratch/plegated-L24-stage1-1m.yml"
#: The replay boundary. Layer 28's input is layer 27's output, so the trainable window
#: is layers 20..27 inclusive and everything from 28 up is the frozen tail.
WINDOW = (20, 28)


def use_reference_recurrence():
    """Unwrap flash-linear-attention so the tail's backward runs in pure torch.

    The boundary gradient needs a backward through layers 28-30, which are
    linear_attention, and fla's Triton autotuner trips over its own disk cache there
    ("'NoneType' object is not a mapping"). The graph here is four layers deep, so the
    reference recurrence costs little; `scratch/rho_diagnostic.py` takes the same route
    for the same reason.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
        operator = getattr(qwen, name, None)
        while operator is not None and hasattr(operator, "__wrapped__"):
            operator = operator.__wrapped__
        if operator is not None:
            setattr(qwen, name, operator)


def load_model():
    use_reference_recurrence()
    config = AutoConfig.from_pretrained(CHECKPOINT)
    config = getattr(config, "text_config", config)
    model, info = Qwen35WidenedForCausalLM.from_pretrained(
        CHECKPOINT, config=config, dtype=torch.bfloat16,
        attn_implementation="sdpa", output_loading_info=True)
    saved = {}
    with safe_open(str(CHECKPOINT / "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if any(p in key for p in (".sidecar.", ".attn_residual.", ".mlp_residual.")):
                saved[key] = handle.get_tensor(key)
    actual = model.state_dict()
    for key, value in saved.items():
        if not torch.equal(actual[key].cpu(), value.to(actual[key].dtype)):
            raise ValueError(f"adapter tensor did not load exactly: {key}")
    del actual, saved
    shard_model(model, ["cuda:0", "cuda:1"])
    model.requires_grad_(False).eval()
    return model, info


def assistant_targets(tokenizer, ids):
    """Predictor positions whose target is an assistant token, and those targets."""
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != list(ids):
        raise ValueError("role tokenisation does not reproduce the cached ids")
    spans = role_spans(text, encoded["offset_mapping"])
    keep = torch.zeros(max(0, len(ids) - 1), dtype=torch.bool)
    for start, stop in spans.get("assistant", []):
        keep[max(start, 1) - 1:max(stop, 1) - 1] = True
    positions = keep.nonzero().flatten()
    return positions, torch.tensor(ids, dtype=torch.long)[positions + 1]


class Boundary:
    """Cuts the activation graph at the window's top edge and keeps the handle.

    The cut is a forward *pre*-hook on the consumer layer rather than a forward hook
    replacing the producer's output: replacing a layer's return value inside a
    tensor-parallel stack left the autograd engine raising "invalid unordered_map key"
    on the second backward, and rewriting a callee's arguments is the supported path
    for the same effect.

    Detaching here means the stack below builds no graph -- every parameter is frozen
    and the inputs are integers, so nothing above the cut would contribute anyway --
    while the tail above it does, making `.grad` on this handle exactly dL/dh_28.
    """

    def __init__(self, model, index, detach=True):
        self.layer = model.model.layers[index]
        self.value = None
        #: Detaching makes h_28 a leaf, which is how dL/dh is obtained when every
        #: parameter is frozen and no graph exists below it. When the window's own
        #: parameters are trainable the graph already reaches h_28, and detaching
        #: would sever exactly the path whose gradient is wanted -- so record instead.
        self.detach = detach
        self.handle = self.layer.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _cut(self, tensor):
        if not self.detach:
            self.value = tensor
            return tensor
        cut = tensor.detach().requires_grad_(True)
        self.value = cut
        return cut

    def _hook(self, module, args, kwargs):
        if args:
            return (self._cut(args[0]),) + tuple(args[1:]), kwargs
        return args, {**kwargs, "hidden_states": self._cut(kwargs["hidden_states"])}

    def close(self):
        self.handle.remove()


@torch.enable_grad()
def forward_once(model, batch, boundary, positions, targets, head_chunk=256):
    """Assistant NLL, the boundary activation, and dL/d(boundary) in one pass."""
    output = model(**batch, use_cache=False, logits_to_keep=0, return_dict=True)
    logits = output.logits
    device = logits.device
    selected = logits[0, positions.to(device)]
    loss = torch.nn.functional.cross_entropy(
        selected.float(), targets.to(device), reduction="mean")
    grad, = torch.autograd.grad(loss, boundary.value)
    value = boundary.value.detach()
    boundary.value = None
    return float(loss.detach()), value, grad.detach()


@torch.no_grad()
def perturb(model, window, scale, generator):
    """Relative Gaussian noise on every weight inside the window. Returns an undo."""
    undo = []
    for index in range(*window):
        for parameter in model.model.layers[index].parameters():
            if parameter.ndim < 1:
                continue
            noise = torch.randn(parameter.shape, generator=generator,
                                dtype=torch.float32).to(parameter.device)
            step = noise * (scale * parameter.float().norm() / max(noise.norm(), 1e-12))
            parameter.add_(step.to(parameter.dtype))
            undo.append((parameter, step.to(parameter.dtype)))
    return undo


@torch.no_grad()
def restore(undo):
    for parameter, step in undo:
        parameter.sub_(step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=16)
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", default="scratch/boundary-replay/check.json")
    arguments = parser.parse_args()

    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(arguments.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.autograd.set_multithreading_enabled(False)

    model, info = load_model()
    tokenizer = AutoTokenizer.from_pretrained("../student-hf")
    source = OfflineHiddenStateSignalSource("../teacher-cache-1m")
    raw = yaml.safe_load(Path(CONFIG_FILE).read_text(encoding="utf-8"))
    table = GGUFNGramTable(raw["sidecar"]["table_path"])
    collator = make_collator(tokenizer.pad_token_id, table)

    records = [r for r in source.cache.manifest["documents"] if r["split"] == "eval"]
    order = np.random.default_rng(arguments.seed).permutation(len(records))
    chosen = [records[i] for i in order[: arguments.docs]]

    boundary = Boundary(model, WINDOW[1])
    documents = []
    for record in chosen:
        ids = source.cache.read_document(record["doc_id"],
                                         include_hidden_states=False)["input_ids"].tolist()
        positions, targets = assistant_targets(tokenizer, ids)
        if len(positions) == 0:
            continue
        batch = collator([{"ids": ids}])
        batch = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in batch.items()}
        loss, h_T, g_T = forward_once(model, batch, boundary, positions, targets)
        documents.append({"id": record["doc_id"], "batch": batch, "positions": positions,
                          "targets": targets, "loss_T": loss,
                          "h_T": h_T.float().cpu(), "g_T": g_T.float().cpu(),
                          "tokens": int(len(positions))})
        print("cached %s  assistant %4d  L_T %.6f  |g| %.3e"
              % (record["doc_id"][:10], len(positions), loss, float(g_T.float().norm())),
              flush=True)

    rows = []
    generator = torch.Generator().manual_seed(arguments.seed)
    for scale in arguments.scales:
        for repeat in range(arguments.repeats):
            undo = perturb(model, WINDOW, scale, generator)
            try:
                for document in documents:
                    loss_S, h_S, _ = forward_once(
                        model, document["batch"], boundary,
                        document["positions"], document["targets"])
                    delta = (h_S.float().cpu() - document["h_T"])
                    predicted = float((document["g_T"] * delta).sum())
                    rows.append({
                        "scale": scale, "repeat": repeat, "id": document["id"],
                        "tokens": document["tokens"],
                        "predicted": predicted,
                        "actual": loss_S - document["loss_T"],
                        "displacement": float(delta.norm()),
                        "relative_displacement": float(delta.norm() / document["h_T"].norm()),
                    })
            finally:
                restore(undo)
            print("scale %.4f repeat %d done" % (scale, repeat), flush=True)
    boundary.close()

    print("\n%-8s %8s %12s %12s %9s %9s %9s"
          % ("scale", "rel disp", "predicted", "actual", "ratio", "corr", "sign ok"))
    summary = []
    for scale in arguments.scales:
        subset = [r for r in rows if r["scale"] == scale]
        predicted = np.array([r["predicted"] for r in subset])
        actual = np.array([r["actual"] for r in subset])
        disp = np.mean([r["relative_displacement"] for r in subset])
        corr = float(np.corrcoef(predicted, actual)[0, 1]) if len(subset) > 2 else float("nan")
        sign = float(np.mean(np.sign(predicted) == np.sign(actual)))
        ratio = float(np.sum(actual * predicted) / max(np.sum(predicted ** 2), 1e-30))
        entry = {"scale": scale, "relative_displacement": disp, "n": len(subset),
                 "mean_predicted": float(predicted.mean()), "mean_actual": float(actual.mean()),
                 "calibration_slope": ratio, "correlation": corr, "sign_agreement": sign}
        summary.append(entry)
        print("%-8.4f %8.5f %12.6f %12.6f %9.3f %9.3f %9.2f"
              % (scale, disp, predicted.mean(), actual.mean(), ratio, corr, sign))

    report = {"checkpoint": str(CHECKPOINT.resolve()), "window": list(WINDOW),
              "documents": len(documents), "loading_info": {k: list(v) if isinstance(v, (set, list)) else v
                                                            for k, v in info.items()},
              "summary": summary, "rows": rows}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote", output)
    print("\nA usable trust radius is the largest scale where correlation and sign")
    print("agreement stay high and the calibration slope stays near 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
