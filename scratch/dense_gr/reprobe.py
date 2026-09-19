"""Re-score saved arms, with a substitution control that measures content.

The seed-0 endpoint matrix left two cells uninterpretable. `scrambled` permutes the
module's output across positions *within a sequence*, so every candidate it emits is still
a token of that sequence -- on the probe, one of the 256 tokens of the repeated block
against a 32,768 vocabulary. `scrambled` enabled landed within 0.17 nats of the real
module because of it, and the same permutation was used for evaluation-time substitution,
so "content dependence" was never actually measured.

`randomized` fixes that: features untouched, candidates drawn uniformly from the
vocabulary. The channel activates at exactly the same positions with exactly the same
strength and says something wrong.

Both controls are reported side by side rather than the old one being replaced, because
the difference between them is the measurement.

`enabled` is recomputed and checked against the value the training run recorded. The probe
is seeded identically, so a mismatch means the checkpoint did not come back as it went in,
and everything below it would be measuring the reload instead of the model.

    python scratch/dense_gr/reprobe.py --checkpoint scratch/dense_gr/checkpoints/copy-s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from benchmark import apply_liger  # noqa: E402
from copy_module import (Connector, build_inputs, module_output,  # noqa: E402
                         randomize_candidates, scramble)
from copy_probe import copy_probe, format_probe  # noqa: E402
from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-half", type=int, default=256)
    parser.add_argument("--probe-windows", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7_000,
                        help="seeds the substitution draws only; the probe sequences "
                             "themselves are fixed so every arm is scored on the same text")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("scratch/dense_gr") / ("reprobe-%s.json" % args.checkpoint.name)

    record = json.loads((args.checkpoint / "milestone.json").read_text(encoding="utf-8"))
    arm, seed = record["arm"], record["seed"]
    print("%s: arm %s seed %d, %d tokens, sha256 %s"
          % (args.checkpoint.name, arm, seed, record["scored_tokens"],
             record["sha256"][:16]), flush=True)

    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, local_files_only=True).to("cuda")
    # The training run swapped in Liger's RMSNorm and SwiGLU. They are mathematically
    # equivalent to the stock modules and numerically not, so a re-probe without them
    # scores a slightly different computation -- measured at 5.7e-04 nats, which is
    # negligible against the probe's 0.39 sd but is not zero, and the point of the reload
    # check is to catch differences rather than absorb them.
    apply_liger(model, model.config)
    model.eval()

    connector, vocab, order, max_length = None, model.config.vocab_size, 3, 8
    blob_path = args.checkpoint / "connector.pt"
    if blob_path.exists():
        blob = torch.load(blob_path, map_location="cpu", weights_only=False)
        connector = Connector(blob["hidden"]).to(device="cuda", dtype=torch.bfloat16)
        connector.load_state_dict(blob["state_dict"])
        connector.eval()
        vocab, order, max_length = blob["vocab"], blob["order"], blob["max_length"]

    def hidden_for(ids_tensor, treatment):
        ids = ids_tensor.detach().cpu().numpy()
        generator = np.random.default_rng(args.seed)
        features, candidate = module_output(ids, arm, vocab, generator, order, max_length)
        if features is not None:
            if treatment == "substituted":
                features, candidate = scramble(features, candidate, generator)
            elif treatment == "randomized":
                features, candidate = randomize_candidates(features, candidate,
                                                           generator, vocab)
            inputs = build_inputs(model, ids_tensor,
                                  torch.from_numpy(features).to("cuda"),
                                  torch.from_numpy(candidate).to("cuda"),
                                  connector, ablate=(treatment == "ablated"))
        else:
            inputs = model.model.embed_tokens(ids_tensor)
        return model.model(inputs_embeds=inputs,
                           attention_mask=torch.ones_like(ids_tensor),
                           use_cache=False).last_hidden_state

    treatments = ["enabled"]
    if connector is not None:
        treatments += ["ablated", "substituted", "randomized"]

    results = {}
    for treatment in treatments:
        results[treatment] = copy_probe(
            model, vocab, half=args.probe_half, windows=args.probe_windows,
            batch=args.batch, hidden_fn=lambda ids, t=treatment: hidden_for(ids, t))
        print("  %-12s %s" % (treatment, format_probe(results[treatment])), flush=True)

    # The reload check. Same probe seed, same weights, so `enabled` must reproduce.
    recorded = record.get("endpoint", {}).get("enabled", {}).get("gain")
    check = None
    if recorded is not None:
        drift = abs(results["enabled"]["gain"] - recorded)
        # An arm whose module consumes randomness cannot reproduce bitwise: the training
        # run's probe drew its permutation from a generator that training had already
        # advanced, and a fresh one draws a different permutation. That is a different
        # sample of the same quantity, so the tolerance is the probe's own scale rather
        # than machine epsilon. Everything else must land exactly.
        # A save/reload round trip moves the probe by about 5.4e-04 nats and the cause was
        # not found. Ruled out: the probe is bitwise deterministic within a process
        # (0.00e+00 over three trials), Liger's kernels (applying them moved the drift
        # from 5.69e-04 to 5.38e-04), and fp32 rounding of the branch gains (they reach
        # disk as fp32 with zero loss). What is left is consistent with a few bf16 ULPs
        # somewhere in 46M parameters. It is 0.14% of the probe's own 0.39 sd, so the
        # tolerance sits above it and well below anything a conclusion could rest on.
        stochastic = arm == "scrambled"
        tolerance = 0.05 if stochastic else 2e-3
        check = {"recorded_gain": recorded, "reprobed_gain": results["enabled"]["gain"],
                 "drift": drift, "tolerance": tolerance, "stochastic_arm": stochastic,
                 "faithful": drift < tolerance}
        print("  reload check: recorded %+.6f, reprobed %+.6f, drift %.2e "
              "(tolerance %.0e%s) -> %s"
              % (recorded, results["enabled"]["gain"], drift, tolerance,
                 ", stochastic arm" if stochastic else "",
                 "faithful" if check["faithful"] else "MISMATCH"), flush=True)
        if not check["faithful"]:
            raise SystemExit("checkpoint did not reload identically; the cells below it "
                             "would be measuring the reload rather than the model")

    args.output.write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "arm": arm, "seed": seed,
         "sha256": record["sha256"], "scored_tokens": record["scored_tokens"],
         "substitution_seed": args.seed, "reload_check": check,
         "endpoint": results}, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
