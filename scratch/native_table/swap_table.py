"""Which half learned: the rows, or the block that reads them?

The trained sidecar improves held-out NLL where the bootstrapped one degrades it, but
three things moved during the run -- the table rows, the PLE projections, and the
admission scalar -- and the aggregate cannot say which carried the gain. A block that has
merely learned to exploit *random* memory would be a much weaker claim than a table whose
rows hold content.

So build a chimera: every trained parameter except the table, whose rows are restored to
the bootstrap checkpoint's random values. Evaluating that against both parents separates
the two. If the gain survives the swap, the block learned to use arbitrary rows; if it
collapses, the rows are where the content lives.

    python scratch/native_table/swap_table.py --output ../runs/native-ple-2b-randomtable
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TABLE = "model.layers.1.sidecar.table.weight"


def load(path: Path) -> dict:
    with safe_open(str(path / "model.safetensors"), framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", type=Path,
                        default=Path("../runs/native-ple-2b-ce"))
    parser.add_argument("--bootstrap", type=Path,
                        default=Path("../runs/native-ple-2b-bootstrap"))
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    trained = load(args.trained)
    donor_rows = load(args.bootstrap)[args.table]
    if trained[args.table].shape != donor_rows.shape:
        raise SystemExit("table geometry differs between the two checkpoints")

    report = {
        "trained": str(args.trained), "bootstrap": str(args.bootstrap),
        "rows_restored": int((trained[args.table] != donor_rows).any(-1).sum()),
        "rows_total": int(donor_rows.shape[0]),
    }
    trained[args.table] = donor_rows

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(trained, str(args.output / "model.safetensors"),
              metadata={"format": "pt"})
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja"):
        source = args.trained / name
        if source.exists():
            shutil.copy2(source, args.output / name)

    # Everything except the table must be bit-identical to the trained checkpoint,
    # or this is not the ablation it claims to be.
    written = load(args.output)
    report["only_the_table_differs"] = all(
        torch.equal(written[key], value)
        for key, value in load(args.trained).items() if key != args.table)
    report["table_is_the_bootstrap_one"] = bool(
        torch.equal(written[args.table], donor_rows))
    print(json.dumps(report, indent=2))
    return 0 if report["only_the_table_differs"] and report["table_is_the_bootstrap_one"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
