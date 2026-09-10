"""Bounded real-student identity and TP training-cost verification.

Synthetic teacher signals exercise the actual KL/hidden-state backward paths;
their losses are plumbing checks, never measures of language-model quality.
Each invocation has a nine-minute watchdog and performs at most three updates.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import threading
import time

import torch
import yaml

from distillkit.anchor_tap import AnchorTap
from distillkit.chunked_head import HeadContext
from distillkit.configuration import DistillationRunConfig
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.hidden_state import compute_hs_loss
from distillkit.lossfuncs.kl import KLDLoss
from distillkit.main import load_student_model
from distillkit.signals import SparseSignal
from distillkit.tp_model import shard_model, sync_replicated_gradients


def synchronize():
    for index in range(2):
        torch.cuda.synchronize(index)


def configuration(branches):
    data = yaml.safe_load(Path("examples/qwen35_sidecar_stage2_tp.yml").read_text())
    # Keep the sidecar identical between arms; its weights are dormant in student-hf.
    if branches > 1:
        data["residual_stream"] = {"num_branches": branches, "lowrank": 64}
    return DistillationRunConfig.model_validate(data)


def load(branches, tp):
    torch.manual_seed(42)
    cfg = configuration(branches)
    model = load_student_model(cfg, 248077, 248320)
    if branches > 1:
        from distillkit.widened_residual import WidenedResidual
        routers = [module for module in model.modules() if isinstance(module, WidenedResidual)]
        assert len(routers) == 2 * model.config.num_hidden_layers, "Every subblock must route branches"
        assert all(module.num_branches == branches for module in routers), "Widening was ignored"
    model = shard_model(model, ["cuda:0", "cuda:1"]) if tp else model.to("cuda:0")
    model.config.use_cache = False
    return cfg, model


def identity(tp):
    references = None
    results = []
    for branches in (1, 2):
        _, model = load(branches, tp)
        model.eval()
        ids = torch.arange(128, device="cuda:0").reshape(2, 64) + 100
        branch_shapes = []
        handles = []
        if branches > 1:
            hidden_size = model.config.hidden_size
            def check_branches(module, inputs, output):
                assert output.shape == (2, 64, branches, hidden_size)
                assert torch.equal(output[..., 0, :], output[..., 1, :])
                branch_shapes.append(list(output.shape))
            handles = [layer.register_forward_hook(check_branches) for layer in model.model.layers]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, sidecar_enabled=False, logits_to_keep=4).logits.cpu()
        for handle in handles:
            handle.remove()
        results.append({"branches": branches, "logits_shape": list(logits.shape),
                        "widened_layers_observed": len(branch_shapes)})
        if references is None:
            references = logits
        else:
            results[-1]["max_abs_difference"] = (logits.float() - references.float()).abs().max().item()
            results[-1]["bit_identical"] = torch.equal(logits, references)
            torch.testing.assert_close(logits, references, rtol=0, atol=0)
        del model, ids, logits
        gc.collect()
        for index in range(2):
            torch.cuda.empty_cache()
    return {"kind": "identity", "tensor_parallel": tp, "comparisons": results}


def performance(branches, batch, sequence, steps):
    import bitsandbytes as bnb

    cfg, model = load(branches, True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    mapping = HiddenStateMapping(model, 5120, cfg.layer_mapping)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
    anchors = [a for a, _ in cfg.layer_mapping] + [model.config.num_hidden_layers]
    sidecar = model.model.layers[cfg.sidecar.layer_index].sidecar
    torch.manual_seed(123)
    ids = torch.randint(0, 248000, (batch, sequence), device="cuda:0")
    mask = torch.ones(batch, sequence, 1, dtype=torch.bool, device="cuda:0")
    raw = torch.zeros(batch, sequence, sidecar.num_heads, sidecar.bytes_per_head, dtype=torch.uint8)
    signal = SparseSignal(
        sparse_ids=torch.randint(0, 248320, (batch, sequence, 64), device="cuda:0"),
        sparse_values=torch.log_softmax(torch.randn(batch, sequence, 64, device="cuda:0"), -1),
        log_values=True, generation_temperature=1.0,
        hidden_states=tuple(torch.randn(batch, sequence, 5120, device="cuda:0", dtype=torch.bfloat16)
                            for _ in cfg.layer_mapping),
        vocab_size=248320,
    )
    records = []
    for step in range(steps):
        synchronize()
        for index in range(2):
            torch.cuda.reset_peak_memory_stats(index)
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with AnchorTap(model, anchors) as tap:
                out = model(input_ids=ids, ngram_raw=raw, logits_to_keep=1)
            out.hidden_states = tap.states()
            kl = KLDLoss(temperature=1.0, sparse_chunk_length=256)(
                out, signal, mask=mask, hidden_state_mapping=mapping,
                head_context=HeadContext(out.hidden_states[model.config.num_hidden_layers],
                                         model.lm_head, vocab_size=248320, chunk_length=256))
            loss = 0.7 * kl.to("cuda:0") + 0.3 * compute_hs_loss("cosine", out, signal, mask, mapping)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite smoke loss")
        loss.backward()
        sync_replicated_gradients(model)
        routing_grads = {name: parameter.grad.float().norm().item()
                         for name, parameter in model.named_parameters()
                         if "lambda" in name and parameter.grad is not None}
        if branches > 1:
            assert len(routing_grads) == 4 * model.config.num_hidden_layers
            assert all(torch.isfinite(torch.tensor(value)) for value in routing_grads.values())
            assert max(routing_grads.values()) > 0, "Routing cannot learn"
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        record = {"step": step + 1, "seconds": time.perf_counter() - start,
                  "loss": loss.item(), "routing_grad_count": len(routing_grads),
                  "routing_grad_max": max(routing_grads.values(), default=0),
                  "peak_gib": [torch.cuda.max_memory_allocated(i) / 1024**3 for i in range(2)],
                  "reserved_gib": [torch.cuda.max_memory_reserved(i) / 1024**3 for i in range(2)]}
        records.append(record)
        print(json.dumps(record), flush=True)
        del out, loss, kl, tap
    return {"kind": "training_cost", "branches": branches, "batch": batch, "sequence": sequence,
            "steps": records, "parameters": sum(p.numel() for p in model.parameters()),
            "quality_evaluation": False}


def trainer_smoke(output):
    from distillkit.main import do_distill

    cfg = configuration(2)
    cfg.output_path = str(output.parent.resolve() / "smoke-checkpoint")
    cfg.training_args.update(
        max_steps=3, gradient_accumulation_steps=1, per_device_train_batch_size=4,
        train_sampling_strategy="group_by_length", length_column_name="length",
        eval_strategy="no", save_strategy="no", report_to="none", logging_steps=1,
        warmup_steps=0, lr_scheduler_type="constant",
    )
    cfg.optimizer.log_every_n_steps = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "smoke-config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    started = time.perf_counter()
    do_distill(cfg)
    return {"kind": "trainer_smoke", "max_steps": 3, "seconds": time.perf_counter() - started,
            "checkpoint": cfg.output_path, "quality_evaluation": False}


def reload_smoke(output):
    from safetensors import safe_open
    from distillkit.models.qwen35_widened import Qwen35WidenedForCausalLM
    from distillkit.independent_eval import validate_loading

    checkpoint = output.parent / "smoke-checkpoint"
    model, info = Qwen35WidenedForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16,
        output_loading_info=True, attn_implementation="sdpa")
    validate_loading(info, has_sidecar=True)
    actual = {key: value for key, value in model.state_dict().items()
              if ".attn_residual." in key or ".mlp_residual." in key}
    checked = set()
    for path in sorted(checkpoint.glob("model*.safetensors")):
        with safe_open(path, framework="pt") as handle:
            for key in set(handle.keys()) & actual.keys():
                assert torch.equal(actual[key].cpu(), handle.get_tensor(key)), key
                checked.add(key)
    assert checked == actual.keys() and checked, "Export omitted routing tensors"
    lambdas = [value.item() for key, value in actual.items() if "lambda" in key]
    assert any(value != 0 for value in lambdas), "Trainer never updated the routing"
    model.to("cuda:0").eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=torch.arange(32, device="cuda:0")[None] + 100,
                    sidecar_enabled=False, use_cache=False, logits_to_keep=2)
    assert torch.isfinite(out.logits).all()
    return {"kind": "reload_smoke", "checkpoint": str(checkpoint.resolve()),
            "exact_routing_tensors": len(checked), "nonzero_lambdas": sum(v != 0 for v in lambdas),
            "logits_shape": list(out.logits.shape), "finite_logits": True,
            "ignored_loss_only_keys": sorted(info.get("unexpected_keys", []))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["identity", "performance", "trainer", "reload"])
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--tp", action="store_true")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--steps", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    timer = threading.Timer(540, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    for index in range(2):
        torch.cuda.set_per_process_memory_fraction(0.95, index)
    if args.kind == "identity":
        result = identity(args.tp)
    elif args.kind == "trainer":
        result = trainer_smoke(args.output)
    elif args.kind == "reload":
        result = reload_smoke(args.output)
    else:
        result = performance(args.branches, args.batch, args.sequence, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    timer.cancel()


if __name__ == "__main__":
    main()
