"""A capture's own documents as capture input: the same token ids and splits, to capture
again with a different teacher configuration (here int8 weights with bf16 activations in
place of LLM.int8, whose targets depend on the document's length).

    python scratch/dense_gr/export_capture_inputs.py ../teacher-cache-thinking ../capture-data/thinking.recapture.jsonl
"""
import json
import sys

from distillkit.offline_cache import OfflineTeacherCache


def main(source, output):
    cache = OfflineTeacherCache(source)
    with open(output, "w", encoding="utf-8") as out:
        for split in ("train", "eval"):
            for doc_id in cache.document_ids(split):
                ids = cache.read_document(doc_id, tokens_only=True)["input_ids"]
                out.write(json.dumps({"doc_id": doc_id, "split": split, "input_ids": [int(t) for t in ids]}) + "\n")
    print("%d documents -> %s" % (len(cache.documents), output))


if __name__ == "__main__":
    main(*sys.argv[1:])
