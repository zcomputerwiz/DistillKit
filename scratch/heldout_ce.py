"""Cross-entropy on text the model has never seen, and no teacher was ever run on.

Every number this project has tuned against is the distillation objective -- sparse KL
against cached teacher logits plus a hidden-state cosine through learned projections. The
sidecar learning-rate sweep showed why that is dangerous: eval_loss fell 0.5445 -> 0.3435
while the KL term barely moved, because `distillation_projections` are free parameters
whose only job is to make the hidden-state term small, and raising their learning rate let
them do exactly that. The tech report warns about the same disagreement from the other
direction: "enlarging the n-gram vocabulary lowers loss monotonically while downstream
accuracy saturates".

Plain next-token cross-entropy avoids both problems. It involves no projections, no
teacher, and no cached anything -- just whether the model predicts real text. It is not a
downstream benchmark and does not claim to be; it is the cheapest metric that cannot be
fitted by a parameter whose purpose is to fit the metric.

The documents are drawn from the corpus *beyond* those any cache used. capture-data/
heldout.jsonl is a 7,200-document slice of which 1,602 appear in no cache manifest; the
source corpus has 890,684, so unseen text is not scarce.

    python scratch/heldout_ce.py runs/lr-sweep-base runs/lr-sweep-1e3 [...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/DeepThought/Projects/HybridModel")
DOCUMENTS, POSITIONS = 64, 1024


def unseen_documents(tokenizer, count):
    """Corpus documents whose doc_id appears in no cache manifest."""
    used = set()
    for manifest in ROOT.glob("teacher-cache-*/manifest.json"):
        used.update(d["doc_id"] for d in json.loads(manifest.read_text())["documents"])
    out = []
    with open(ROOT / "capture-data" / "heldout.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["doc_id"] in used:
                continue
            tokens = tokenizer(record["text"], add_special_tokens=True)["input_ids"]
            if len(tokens) >= 128:
                out.append(tokens[:POSITIONS])
            if len(out) >= count:
                break
    if len(out) < count:
        raise SystemExit(
            "only %d unseen documents in heldout.jsonl; prepare a larger slice with "
            "distillkit.prepare_corpus --limit beyond what the caches used" % len(out)
        )
    return out


def evaluate(checkpoint: Path, documents, sidecar_enabled=True):
    import yaml

    from distillkit.configuration import DistillationRunConfig
    from distillkit.main import load_student_model
    from distillkit.ngram_hash import NGramHasher
    from distillkit.ngram_table import GGUFNGramTable

    raw = yaml.safe_load(open("examples/_lr_sweep_base.yml", encoding="utf-8"))
    raw["model"] = str(checkpoint)
    raw["sidecar"]["resident"] = False
    raw["sidecar"]["prefault"] = False
    config = DistillationRunConfig.model_validate(raw)

    model = load_student_model(config, 248077, 248320).to("cuda:0").eval()
    sidecar = model.model.layers[config.sidecar.layer_index].sidecar
    # Memory-mapped, not resident: this walks a few thousand rows, not an epoch.
    table = GGUFNGramTable(config.sidecar.table_path)
    hasher = NGramHasher()

    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for tokens in documents:
            ids = torch.tensor(tokens, dtype=torch.long, device="cuda:0").unsqueeze(0)
            rows = hasher.row_indices(ids.cpu())
            ngram_raw = torch.from_numpy(
                np.ascontiguousarray(table.gather_raw(rows))).to("cuda:0")
            logits = model(input_ids=ids, ngram_raw=ngram_raw,
                           sidecar_enabled=sidecar_enabled).logits.float()
            # Standard causal shift: position t predicts token t+1.
            nll = torch.nn.functional.cross_entropy(
                logits[0, :-1], ids[0, 1:], reduction="sum")
            total_nll += nll.item()
            total_tokens += ids.shape[1] - 1
    del model
    torch.cuda.empty_cache()
    return total_nll / total_tokens


def main(argv):
    from transformers import AutoTokenizer

    if not argv:
        raise SystemExit(__doc__)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "student-hf"), local_files_only=True)
    documents = unseen_documents(tokenizer, DOCUMENTS)
    tokens = sum(len(d) for d in documents)
    print("%d unseen documents, %d tokens\n" % (len(documents), tokens))

    print("%-24s %12s %12s %12s" % ("checkpoint", "CE (nats)", "perplexity", "sidecar off"))
    for target in argv:
        checkpoint = ROOT / target if not Path(target).is_absolute() else Path(target)
        with_sidecar = evaluate(checkpoint, documents, True)
        without = evaluate(checkpoint, documents, False)
        print("%-24s %12.4f %12.2f %12.4f   (sidecar worth %+.4f)"
              % (checkpoint.name, with_sidecar, pow(2.718281828, with_sidecar),
                 without, with_sidecar - without))
    print("\nLower is better. The last column is the same model with the sidecar bypassed:")
    print("if it is not worse, the sidecar is not earning its place regardless of what")
    print("the distillation objective says.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
