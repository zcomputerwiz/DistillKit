"""Read-only CPU review: saved-score bootstrap and tiny algebra counterexamples.

No models/checkpoints are loaded, no optimizer is created, and nothing is trained.
Selected definitions are compiled directly from the reviewed source via AST.
"""
from __future__ import annotations

import ast
import copy
import json
import math
import os
from pathlib import Path
import threading
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
watchdog = threading.Timer(180, lambda: os._exit(124))
watchdog.daemon = True
watchdog.start()

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.set_default_device("cpu")
ROOT = Path(__file__).resolve().parent.parent


def definitions(relative_path, names, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


ns = {"np": np}
definitions("distillkit/independent_eval.py", {"paired_interval"}, ns)
definitions("scratch/paired_arms.py", {"compare"}, ns)
scores = ROOT / "scratch/independent-eval"
comparisons = {}
whole_window = {}
for stage, left, right in [
    ("stage1", "reply-widened-ple-stage1-1m.json", "reply-ple-stage1-1m.json"),
    ("stage2", "reply-widened-ple-stage2-5m.json", "reply-ple-control-stage2-5m.json"),
]:
    a, b = [json.loads((scores / name).read_text(encoding="utf-8")) for name in (left, right)]
    assert a["complete"] and b["complete"]
    for mode in ("enabled", "bypassed"):
        whole_window[f"{stage}_{mode}"] = ns["compare"](a, b, mode)["nll"]
    # Despite the reply-* filenames, top-level NLL includes all roles.
    # Select the assistant subsection explicitly, preserving IDs and pairing.
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    for arm in (a, b):
        for row in arm["records"]["nll"]:
            for mode, score in list(row["modes"].items()):
                row["modes"][mode] = score["by_role"]["assistant"]
    for mode in ("enabled", "bypassed"):
        comparisons[f"{stage}_{mode}"] = ns["compare"](a, b, mode)["nll"]

upstream = ".venv/Lib/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py"
ns = {"torch": torch, "nn": nn, "F": F, "Qwen4ExpTextConfig": object}
definitions(upstream, {"Qwen4ExpTextRMSNorm", "Qwen4ExpTextGatedResidual"}, ns)
config = SimpleNamespace(hc_count=2, hidden_size=2, hc_lowrank=2, rms_norm_eps=1e-6)
mixer = ns["Qwen4ExpTextGatedResidual"](config, use_combine=False)
with torch.no_grad():
    for parameter in mixer.parameters():
        parameter.zero_()
    h = torch.tensor([1.0, 0.0])
    v = torch.tensor([0.0, 1.0])
    split = h + torch.tensor([0.2, 0.8])[:, None] * v
    tied = h + torch.tensor([0.5, 0.5])[:, None] * v
    assert torch.allclose(split.mean(0), tied.mean(0))
    split_out, tied_out = mixer(split.flatten()), mixer(tied.flatten())
    counterexample = {
        "constant_mix_weight": 0.5,
        "same_raw_stream_mean": split.mean(0).tolist(),
        "split_gates": [0.2, 0.8],
        "tied_gates": [0.5, 0.5],
        "split_mixer_output": split_out.tolist(),
        "tied_mixer_output": tied_out.tolist(),
        "max_abs_difference": (split_out - tied_out).abs().max().item(),
    }

ns = {"torch": torch, "nn": nn, "math": math}
definitions("scratch/table_capacity_probe.py", {"Readout"}, ns)
torch.manual_seed(7)
readout = ns["Readout"](8, 4)
features, stream = torch.randn(16, 8), torch.randn(16, 8)
with torch.no_grad():
    readout.value.weight.normal_(std=0.1)
    original = readout(features, stream)
    normed = stream * torch.rsqrt(stream.square().mean(-1, keepdim=True) + 1e-6)
    raw = normed @ readout.gate.T / math.sqrt(8)
    gates = torch.sigmoid(raw.abs().clamp_min(1e-6).sqrt() * raw.sign())
    mean_rescaled = (4 * readout.value(features)) * gates.mean(-1, keepdim=True)
    scale_error = (original - mean_rescaled).abs().max().item()
    complement_error = (gates + torch.sigmoid(-raw.abs().clamp_min(1e-6).sqrt() * raw.sign()) - 1).abs().max().item()

result = {"assistant_only_paired_widened_minus_control": comparisons,
          "whole_window_reply_bundle_paired_widened_minus_control": whole_window,
          "constant_mixer_normalization_counterexample": counterexample,
          "sum_to_mean_with_W_times_four_max_abs_error": scale_error,
          "opposite_gate_complement_max_abs_error": complement_error}
output = ROOT / "scratch/hc_gate_review_cpu.json"
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
watchdog.cancel()
