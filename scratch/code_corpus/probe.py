"""Prove the Stack v2 data path end to end before committing to tens of millions of tokens.

Five links have to hold, and each can fail in a way that looks like success further
down: Hugging Face authentication, metadata streaming, unsigned retrieval from the
Software Heritage object store, gzip decompression, and checksum agreement between
the bytes we got and the identifier that addressed them. This measures all five on a
few hundred files and reports the throughput, so the cost of the real corpus is a
number rather than a guess.

Nothing is written except the report. The corpus builder reuses this exact retrieval
path -- see ``stack.BlobFetcher`` -- so what is measured here is what will run.

    python scratch/code_corpus/probe.py --records 300
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stack import (CONFIG, DATASET, BlobFetcher, dataset_revision, iter_row_groups,
                   load_tokenizer, tokenizer_identity)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=300)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/code_corpus/manifests/probe.json"))
    args = parser.parse_args()

    from huggingface_hub import whoami

    who = whoami()
    print("hf user: %s (%s)" % (who["name"], who["type"]))

    started = time.monotonic()
    revision = dataset_revision()
    groups = iter_row_groups(args.shard)
    _, rows = next(groups)
    records = rows[:args.records]
    metadata_seconds = time.monotonic() - started
    print("metadata: %d records in %.1f s (revision %s)"
          % (len(records), metadata_seconds, revision[:12]))

    print("\nschema of one record:")
    for key, value in records[0].items():
        print("  %-22s %-12s %s" % (key, type(value).__name__, str(value)[:60]))

    fetcher = BlobFetcher(workers=args.workers)
    statuses = collections.Counter()
    sources = []
    bytes_retrieved = 0
    started = time.monotonic()
    for record, source, status in fetcher.map(records):
        statuses[status] += 1
        if status == "ok":
            sources.append(source)
            bytes_retrieved += len(source.encode("utf-8"))
    retrieval_seconds = time.monotonic() - started

    tokenizer = load_tokenizer()
    started = time.monotonic()
    encoded = tokenizer(sources, add_special_tokens=False)["input_ids"]
    tokenize_seconds = time.monotonic() - started
    tokens = sum(len(row) for row in encoded)

    licenses = collections.Counter(r.get("license_type") for r in records)
    flags = {"is_vendor": sum(bool(r.get("is_vendor")) for r in records),
             "is_generated": sum(bool(r.get("is_generated")) for r in records)}

    report = {
        "dataset": DATASET, "config": CONFIG, "revision": revision,
        "hf_user": who["name"],
        "records_attempted": len(records),
        "retrieval": dict(statuses),
        "success_rate": statuses["ok"] / max(len(records), 1),
        "checksum_mismatches": statuses["checksum"],
        "workers": args.workers,
        "retrieval_seconds": retrieval_seconds,
        "files_per_second": statuses["ok"] / retrieval_seconds,
        "megabytes_per_second": bytes_retrieved / 2 ** 20 / retrieval_seconds,
        "tokenize_seconds": tokenize_seconds,
        "model_tokens": tokens,
        "model_tokens_per_second_end_to_end":
            tokens / (retrieval_seconds + tokenize_seconds),
        "mean_tokens_per_file": tokens / max(len(sources), 1),
        "bytes_per_token": bytes_retrieved / max(tokens, 1),
        "tokenizer": tokenizer_identity(tokenizer),
        "license_types": dict(licenses),
        "metadata_flags": flags,
        "schema": {key: type(value).__name__ for key, value in records[0].items()},
    }

    print("\nretrieval: %s" % dict(statuses))
    print("  %.1f files/s   %.2f MB/s   %.0f model tokens/s end to end"
          % (report["files_per_second"], report["megabytes_per_second"],
             report["model_tokens_per_second_end_to_end"]))
    print("  %d model tokens, %.0f tokens/file, %.2f bytes/token"
          % (tokens, report["mean_tokens_per_file"], report["bytes_per_token"]))
    print("  checksum mismatches: %d" % statuses["checksum"])
    print("\nlicense_type: %s" % dict(licenses))
    print("metadata flags: %s" % flags)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote %s" % args.output)
    return 0 if statuses["checksum"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
