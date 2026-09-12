"""Does the native-table student load, hash, train a step, and reload? Nothing more.

This is the pre-experiment check, not an experiment: a handful of steps on a handful of
documents, to establish that the machinery is wired end to end and to measure the one
number that decides whether the dense table is affordable -- the optimizer's state for it.

No 0.8B checkpoint exists on this machine, so the geometry under test is built rather than
downloaded: a 1024-wide Qwen3.5 config with the real vocabulary, which is what fixes the
16 heads of 64 dims the task specifies. `--student` runs the same check against the real
2560-wide student instead, where the same code gives 16 heads of 160.

    python scratch/native_table/smoke.py --base 131072
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

ROOT = Path(__file__).resolve().parents[2]
STUDENT = str((ROOT / ".." / "student-hf").resolve())


def build_config(student, hidden_size, base, layers):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(STUDENT, local_files_only=True)
    config = getattr(config, "text_config", config)
    if not student:
        # The 0.8B shape, constructed: same vocabulary and tokenizer, a 1024-wide stream,
        # and enough depth to be a real forward. Weights are random -- this checks the
        # machinery, and every property it checks is weight-independent.
        config.hidden_size = hidden_size
        config.intermediate_size = hidden_size * 3
        config.num_hidden_layers = layers
        # Qwen3.5 interleaves attention and linear-attention layers and validates that
        # the list matches the depth, so shortening the model means shortening the plan.
        if getattr(config, "layer_types", None):
            config.layer_types = list(config.layer_types[:layers])
        config.head_dim = hidden_size // config.num_attention_heads
        config.linear_key_head_dim = config.head_dim
        config.linear_value_head_dim = config.head_dim
    config.sidecar_variant = "ple"
    config.sidecar_table_mode = "native"
    config.sidecar_ngram_vocab_size_base = base
    config.sidecar_layer_index = 1
    config.use_cache = False
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=int, default=131072)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--student", action="store_true",
                        help="load the real 2560-wide student instead of a built config")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="train only the sidecar, as the first experiment will")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.native_ple import native_hash_config
    from distillkit.ngram_hash import NGramHasher

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    config = build_config(args.student, args.hidden_size, args.base, args.layers)
    if args.student:
        model = Qwen35SidecarForCausalLM.from_pretrained(
            STUDENT, config=config, dtype=torch.bfloat16, local_files_only=True).to(args.device)
    else:
        model = Qwen35SidecarForCausalLM(config).to(args.device, torch.bfloat16)
    model.config.use_cache = False
    sidecar = model.model.layers[config.sidecar_layer_index].sidecar
    hasher = NGramHasher(native_hash_config(config))
    if args.freeze_backbone:
        model.requires_grad_(False)
        sidecar.requires_grad_(True)

    report = {"geometry": sidecar.geometry(), "hidden_size": config.hidden_size}
    report["geometry"].pop("head_vocab_sizes")
    report["geometry"].pop("head_offsets")
    table = sidecar.table.weight
    total = sum(p.numel() for p in model.parameters())
    report["parameters"] = {
        "table": table.numel(),
        "model_total": total,
        "model_without_table": total - table.numel(),
        "table_share": table.numel() / total,
    }
    report["table_bytes"] = {name: table.numel() * size for name, size in
                             (("bf16", 2), ("fp16", 2), ("fp32", 4), ("fp8", 1))}
    # AdamW keeps two fp32 moments per element, and a bf16 parameter still gets fp32
    # state. That is the number that decides whether the dense table is affordable at
    # all, so it is measured from the optimizer rather than predicted.
    report["adamw_state_bytes_predicted"] = table.numel() * 4 * 2

    text = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tokenizer(text)["input_ids"][:args.tokens]
    rows = hasher.row_indices(torch.tensor([ids], dtype=torch.long))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    model.train()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    losses = []
    before = {"rho": float(sidecar.rho), "table": float(table.detach().float().norm())}
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=torch.tensor([ids], device=args.device),
                       attention_mask=torch.ones(1, len(ids), dtype=torch.long,
                                                 device=args.device),
                       ngram_ids=rows.to(args.device)).logits[0, :-1].float()
        target = torch.as_tensor(ids[1:], device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        if step == 0:
            report["step0_gradients"] = {
                "rho": float(sidecar.rho.grad.abs().item()),
                # Zero at step 0 by construction: the write is multiplied by rho, and rho
                # is exactly zero. The table can only start learning once rho has moved,
                # which is the derivative structure, not a wiring fault.
                "table": float(table.grad.abs().max()) if table.grad is not None else 0.0,
                "value_proj": float(sidecar.ple.value_proj.weight.grad.abs().max())
                if sidecar.ple.value_proj.weight.grad is not None else 0.0,
            }
        optimizer.step()
        losses.append(float(loss.detach()))
    report["loss"] = losses
    report["after"] = {"rho": float(sidecar.rho), "table": float(table.detach().float().norm())}
    report["rho_moved"] = report["after"]["rho"] != before["rho"]
    report["table_moved"] = report["after"]["table"] != before["table"]
    report["final_gradients"] = {
        "rho": float(sidecar.rho.grad.abs().item()),
        "table": float(table.grad.abs().max()) if table.grad is not None else 0.0,
        "value_proj": float(sidecar.ple.value_proj.weight.grad.abs().max()),
    }
    report["metrics"] = sidecar.gate_report()

    if args.device.startswith("cuda"):
        state = sum(v.numel() * v.element_size()
                    for group in optimizer.state.values() for v in group.values()
                    if torch.is_tensor(v))
        table_state = sum(v.numel() * v.element_size()
                          for v in optimizer.state[table].values() if torch.is_tensor(v))
        report["adamw_state_bytes_measured"] = {"total": state, "table": table_state}
        report["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())

    model.eval()
    with torch.no_grad():
        on = model(input_ids=torch.tensor([ids], device=args.device),
                   ngram_ids=rows.to(args.device)).logits
        off = model(input_ids=torch.tensor([ids], device=args.device),
                    sidecar_enabled=False).logits
    report["ablation_on_off_differs"] = not torch.equal(on, off)
    with torch.no_grad():
        again_same = model(input_ids=torch.tensor([ids], device=args.device),
                           ngram_ids=rows.to(args.device)).logits
    # The control for the reload comparison below: this model's own forward, twice. The
    # linear-attention kernels are not bitwise deterministic in bf16 on this box, so a
    # reload difference is only meaningful against this floor.
    report["repeat_forward_max_difference"] = float((again_same.float() - on.float()).abs().max())

    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        # Same dtype as the model that wrote it: reloading bf16 weights into fp32 and
        # comparing logits measures the cast, not the round trip.
        reloaded = Qwen35SidecarForCausalLM.from_pretrained(
            directory, dtype=torch.bfloat16).to(args.device).eval()
        restored = reloaded.model.layers[config.sidecar_layer_index].sidecar
        with torch.no_grad():
            again = reloaded(input_ids=torch.tensor([ids], device=args.device),
                             ngram_ids=rows.to(args.device)).logits
        # The weights are the claim; the logits are the noise floor. Every tensor comes
        # back bitwise identical, and the residual logit delta is one bf16 unit in the
        # last place -- two model objects holding identical weights can still schedule
        # these kernels differently once the allocator has been fragmented by training.
        # `repeat_forward_max_difference` above is 0.0 within a single object.
        saved_state, restored_state = model.state_dict(), reloaded.state_dict()
        report["reload"] = {
            "geometry_matches": restored.geometry() == sidecar.geometry(),
            "rho_matches": float(restored.rho) == float(sidecar.rho),
            "missing_or_extra_tensors": sorted(set(saved_state) ^ set(restored_state)),
            # Bitwise, and without promoting anything: a full-model diff in fp32 on the
            # device allocates gigabytes for the embedding alone.
            "tensors_differing": [
                name for name in saved_state if name in restored_state
                and not torch.equal(saved_state[name], restored_state[name])],
            "logits_max_difference": float((again.float() - on.float()).abs().max()),
        }

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
