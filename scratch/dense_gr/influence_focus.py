# Assisted-by: Codex
"""Complement the frozen influence audit with existing aggregation continuations.

Uses training pairs only. No optimizer step or checkpoint write; preserves the
original audit's inputs, objectives and source files.
"""
import gc
import time

import torch

from influence_audit import OUT, read, write, digest
from influence_gpu import Diagnostic, geometry, family


PAIR_IDS = [
    "completion-v2:train:aggregate:2:turn:2",
    "completion-v2:train:aggregate:2:turn:3",
    "completion-v2:train:aggregate_overlap:4:turn:2",
    "completion-v2:train:aggregate_overlap:4:turn:3",
    "completion-v2:train:aggregate:2:turn:6",
]


def main():
    destination = OUT / "focused-audit.json"
    if destination.exists():
        raise ValueError("completed focus audit exists; refusing overwrite")
    plan = read(OUT / "plan.json")
    for path, sha in plan["input_sha256"].items():
        if digest(path) != sha:
            raise ValueError("frozen input changed: " + path)
    torch.set_num_threads(4)
    torch.manual_seed(25)
    torch.cuda.set_device(0)
    result = dict(status="running", plan_sha256=digest(OUT / "plan.json"),
                  source_sha256=digest(__file__), pair_ids=PAIR_IDS, pairs=[],
                  caveat="Existing training continuations; not an independent transfer evaluation. No updates.")
    started = time.monotonic()
    diag = None
    try:
        diag = Diagnostic(plan)
        preserve = [diag.record(row, True) for row in plan["replay"] if row["preservation"]]
        share = sum(row["coefficient"] for row in plan["replay"] if row["preservation"])
        for record in preserve:
            record["ce_only"], record["kl_only"] = True, False
            record["weight"] = record["weight"] / share
        reference, _, _, _ = diag.backward(preserve, teacher_weight=0.)
        del preserve
        for pid in PAIR_IDS:
            gradient, metrics, _, _ = diag.backward([diag.pair(pid)])
            indexer = []
            for name, parameter in diag.parameters.items():
                if family(name) == "indexer":
                    g = gradient.get(name)
                    indexer.append(dict(parameter=name, requires_grad=parameter.requires_grad,
                        gradient_present=g is not None, gradient_abs_max=None if g is None else float(g.abs().max())))
            result["pairs"].append(dict(pair_id=pid, metrics=metrics,
                gradient=geometry(gradient, reference), indexer_gradients=indexer))
            print("Continuation", pid, result["pairs"][-1]["gradient"]["all"], flush=True)
            del gradient
            write(OUT / "focused-progress.json", result)
        pid = PAIR_IDS[0]
        dpo, _, _, _ = diag.backward([diag.pair(pid)], sft=0.)
        chosen, _, _, _ = diag.backward([diag.pair(pid)], beta=0.)
        result["components"] = dict(pair_id=pid, dpo_chosen_ce=geometry(dpo, chosen),
                                    dpo_code_ce=geometry(dpo, reference), chosen_ce_code_ce=geometry(chosen, reference))
        del dpo, chosen, reference
        # A zero gradient still enters AdamW; None skips that parameter. Inspect
        # the saved moments and compensation without changing their backing file.
        rows = []
        for group, names in zip(diag.state["optimizer"]["param_groups"], diag.state["optimizer_layout"]):
            for pid, name in zip(group["params"], names):
                if family(name) != "indexer":
                    continue
                state = diag.state["optimizer"]["state"].get(pid, {})
                rows.append(dict(parameter=name, lr=group["lr"], weight_decay=group["weight_decay"],
                    optimizer_state_present=bool(state), step=state.get("step"),
                    first_absmax_max=None if "absmax1" not in state else float(state["absmax1"].max()),
                    second_absmax_max=None if "absmax2" not in state else float(state["absmax2"].max()),
                    compensation_abs_max=None if "compensation" not in state else float(state["compensation"].abs().max())))
        result["saved_indexer_state"] = rows
        result["policy"] = {k: diag.args[k] for k in ("sparse_stage", "freeze_router", "router_lr", "indexer_weight")}
        result["router_bias"] = bool(diag.model.config.csa2_router_bias)
        result.update(status="complete", seconds=time.monotonic()-started,
            allocated_peaks_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in (0, 1)])
        diag.model.zero_grad(set_to_none=True)
        gc.collect()
    finally:
        if diag is not None:
            for handle in diag.handles:
                handle.remove()
            diag.teacher.close()
            diag.spill.stop()
            result.update(diag.spill.report())
        write(destination if result["status"] == "complete" else OUT / "focused-progress.json", result)
    print("Focused audit complete; no updates", flush=True)


if __name__ == "__main__":
    main()
