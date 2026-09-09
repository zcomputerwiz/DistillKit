"""Did the PLE module actually learn, and does its gate discriminate?

`architecture_metrics` only recognises `GatedResidual` and `W_side_proj`, so a PLE run
logs no gate statistics -- and the gate is the whole reason for the port. Rather than
patch the trainer mid-curriculum, which would have left the arms running different code,
this reads the answer back out of the saved checkpoints afterwards.

Two levels, cheapest first:

* **Weight norms**, straight from the checkpoint, no forward pass. value_proj and conv1d
  start at exactly zero and the three norms start at exactly zero deviation, so any
  nonzero value is movement. This is the PLE counterpart of the 5M pilot's finding that
  `W_side_proj` reached 4.007 while the gate stayed at 0.5013.
* **The gate distribution**, by running real evaluation documents through the module. A
  computed gate cannot sit at its initialisation the way a learned one can, but it can
  still be useless: if it returns the same value for every token it is a constant scale,
  not a selector. Spread and the split between open and shut are what distinguish those.

    python scratch/ple_gate_report.py runs/ple-stage1-1m [runs/ple-stage2-5m ...]
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path("D:/DeepThought/Projects/HybridModel")
PLE_KEYS = ("value_proj", "key_proj", "conv1d", "norm_key", "norm_query", "norm_conv")


def weight_norms(checkpoint: Path) -> dict[str, float]:
    path = checkpoint / "model.safetensors"
    if not path.is_file():
        raise SystemExit("no model.safetensors under %s" % checkpoint)
    out = {}
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():
            if ".sidecar.ple." in key and any(k in key for k in PLE_KEYS):
                tensor = handle.get_tensor(key).float()
                out[key.split(".sidecar.ple.")[-1]] = tensor.norm().item()
    return out


def gate_distribution(checkpoint: Path, documents: int = 8, positions: int = 512):
    """Run real evaluation text through the module and describe its gate."""
    import yaml

    from distillkit.configuration import DistillationRunConfig
    from distillkit.main import load_student_model
    from distillkit.ngram_hash import NGramHasher
    from distillkit.ngram_table import NGramTable
    from distillkit.offline_cache import OfflineTeacherCache

    config_path = next(Path("examples").glob("qwen35_ple_stage*.yml"))
    raw = yaml.safe_load(open(config_path, encoding="utf-8"))
    raw["model"] = str(checkpoint)
    raw["sidecar"]["resident"] = False
    raw["sidecar"]["prefault"] = False
    config = DistillationRunConfig.model_validate(raw)

    model = load_student_model(config, 248077, 248320).to("cuda:0").eval()
    sidecar = model.model.layers[config.sidecar.layer_index].sidecar
    table = NGramTable(config.sidecar.table_path, resident=False, prefault=False)
    hasher = NGramHasher(table.spec, vocab_size=248320)
    cache = OfflineTeacherCache(config.teacher.cache_path)

    gates = []
    for doc_id in cache.document_ids("eval")[:documents]:
        tokens = cache.read_document(doc_id, tokens_only=True)["input_ids"][:positions]
        ids = torch.tensor(tokens, dtype=torch.long, device="cuda:0").unsqueeze(0)
        raw_rows = table.gather(hasher.rows_for(ids.cpu())).to("cuda:0")
        with torch.no_grad():
            hidden = model.model.embed_tokens(ids)
            features = sidecar.dequant(raw_rows).flatten(-2).to(hidden.dtype)
            ple = sidecar.ple
            key = ple.norm_key(ple.key_proj(features))
            query = ple.norm_query(hidden)
            import math
            raw_gate = (key * query).sum(-1) / math.sqrt(ple.hidden_size)
            gates.append(torch.sigmoid(
                raw_gate.abs().clamp_min(1e-6).sqrt() * raw_gate.sign()).float().flatten())
    gate = torch.cat(gates)
    return {
        "tokens": gate.numel(),
        "mean": gate.mean().item(),
        "std": gate.std().item(),
        "p05": gate.quantile(0.05).item(),
        "p95": gate.quantile(0.95).item(),
        "frac_open": (gate > 0.6).float().mean().item(),
        "frac_shut": (gate < 0.4).float().mean().item(),
    }


def main(argv):
    targets = argv or ["runs/ple-stage1-1m"]
    for target in targets:
        checkpoint = ROOT / target if not Path(target).is_absolute() else Path(target)
        print("\n=== %s ===" % checkpoint.name)
        norms = weight_norms(checkpoint)
        if not norms:
            print("  no ple.* tensors -- is this a gated_residual run?")
            continue
        print("  weight norms (value_proj and conv1d start at exactly 0;")
        print("                norms store a deviation, so they start at 0 too)")
        for name in PLE_KEYS:
            for key, value in sorted(norms.items()):
                if key.startswith(name):
                    print("    %-24s %.5f%s" % (key, value, "   <- still at init" if value == 0 else ""))
        try:
            stats = gate_distribution(checkpoint)
        except Exception as error:                      # noqa: BLE001 - report, do not abort
            print("  gate distribution unavailable: %s: %s" % (type(error).__name__, error))
            continue
        print("  gate over %d real evaluation tokens:" % stats.pop("tokens"))
        print("    mean %.4f  std %.4f  p05 %.4f  p95 %.4f" %
              (stats["mean"], stats["std"], stats["p05"], stats["p95"]))
        print("    open (>0.6) %.1f%%   shut (<0.4) %.1f%%" %
              (100 * stats["frac_open"], 100 * stats["frac_shut"]))
        if stats["std"] < 1e-3:
            print("    -> effectively constant: a scale, not a selector")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
