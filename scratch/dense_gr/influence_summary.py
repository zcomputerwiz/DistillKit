# Assisted-by: Codex
"""Summarize completed receipts and inspect wrapped saved router moments on CPU."""
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

from influence_audit import OUT, read, write, digest


def main():
    main_audit = read(OUT / "gpu-audit.json")
    focus = read(OUT / "focused-audit.json")
    targets = read(OUT / "target-audit.json")
    plan = read(OUT / "plan.json")
    provenance = read(OUT / "provenance.json")
    assert main_audit["status"] == focus["status"] == "complete"
    assert main_audit["source_sha256"] == digest(Path(__file__).with_name("influence_gpu.py"))
    assert focus["source_sha256"] == digest(Path(__file__).with_name("influence_focus.py"))
    for path, sha in plan["input_sha256"].items():
        assert digest(path) == sha, path
    by_source = defaultdict(Counter)
    roles = Counter()
    for row in targets["documents"]:
        for role, stats in row["roles"].items():
            roles[role] += stats["targets"]
            by_source[row["source"]].update(stats)
    assert roles["system"] == roles["user"] == roles["tool-result"] == 0
    assert sum(roles.values()) == targets["total_targets"]
    profile = {}
    for source, totals in by_source.items():
        k = totals["kl_weight"]
        profile[source] = dict(targets=totals["targets"], share=totals["targets"]/targets["total_targets"],
            teacher_top1_match=None if not k else totals["teacher_top1_correct_weight"]/k,
            actual_absent=None if not k else totals["teacher_actual_absent_weight"]/k,
            premature_stop=None if not k else totals["premature_stop_weight"]/k)
    code = read(OUT / "code-execution-audit.json")
    assert all(row["statuses"]["served"] == "passed" for row in code["rows"])
    rejected = sum(status != "passed" for row in code["rows"]
                   for variant, status in row["statuses"].items() if variant != "served")
    # Quantized moments are nested deliberately by training_state.py, so torch's
    # generic optimizer loader does not cast uint8 state to the parameter dtype.
    # The focus receipt's top-level absmax fields are None for these wrappers.
    state_file = Path(plan["state"]) / "state.pt"
    receipt = provenance["saved_state"]
    assert state_file.stat().st_size == receipt["size"]
    assert state_file.stat().st_mtime_ns == receipt["mtime_ns"]
    state = torch.load(state_file, mmap=True, weights_only=True, map_location="cpu")
    router, decay_squared = [], [0., 0.]
    zero_names = {r["parameter"] for r in focus["pairs"][0]["indexer_gradients"] if r["gradient_present"]}
    for row in focus["pairs"]:
        assert {r["parameter"] for r in row["indexer_gradients"] if r["gradient_present"]} == zero_names
        assert all(r["gradient_abs_max"] == 0 for r in row["indexer_gradients"] if r["gradient_present"])
    for group, names in zip(state["optimizer"]["param_groups"], state["optimizer_layout"]):
        for pid, name in zip(group["params"], names):
            if name not in zero_names:
                continue
            entry = state["optimizer"]["state"][pid]
            quant = entry.get("__bnb_optimizer_quant_state__", entry)
            first, second = float(quant["absmax1"].max()), float(quant["absmax2"].max())
            assert first == second == 0, "nonzero saved router moments: " + name
            p = state["model"][name].double()
            c = entry["compensation"].double()
            shrink = group["lr"] * group["weight_decay"]
            decay_squared[0] += float(p.square().sum()) * shrink**2
            decay_squared[1] += float((p+c).square().sum()) * shrink**2
            router.append(dict(parameter=name, step=entry["step"], lr=group["lr"],
                weight_decay=group["weight_decay"], first_moment_absmax=first,
                second_moment_absmax=second, gradient_abs_max=0., compensation_abs_max=float(c.abs().max())))
    expected = list(map(math.sqrt, decay_squared))
    measured = [s["effective_compensated_update"]["indexer"]["norm"] for s in main_audit["copied_steps"]]
    error = [abs(a-b)/b for a,b in zip(measured, expected)]
    # FP32 working arithmetic plus BF16 compensation rounds the decay increment.
    assert max(error) < .005, (measured, expected)
    value = dict(source_sha256=digest(__file__), document_visits=len(targets["documents"]),
        weighted_targets=targets["total_targets"], excluded_visits=targets["excluded_historical_visits"],
        role_targets=dict(roles), source_profile=profile, code_native_passes=len(code["rows"]),
        rejected_mutants=rejected, router_state=router,
        router_decay=dict(expected_fresh_saved_norms=expected, measured_fresh_saved_norms=measured,
            relative_errors=error, interpretation="Six query projections receive zero gradients and have zero saved moments; compensated movement matches decay alone. Other indexer gradients are None in these five focused pairs."),
        target_mass=dict(min_captured=min(r["min_captured_mass"] for r in targets["documents"]),
            max_captured=max(r["max_captured_mass"] for r in targets["documents"]),
            visits_with_any_overshoot=sum(r["max_captured_mass"] > 1.00001 for r in targets["documents"]),
            caveat="Document maxima do not measure the fraction of affected positions. FP16 cache rounding is a plausible explanation; original full-precision values are unavailable. Existing loss clamps the tail and does not renormalize the head."))
    write(OUT / "audit-summary.json", value)
    print("Audit summary complete", len(router), "zero-moment query projections; decay errors", error)


if __name__ == "__main__":
    main()
