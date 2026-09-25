"""Where does the student's "Wait, let me re-read" come from: the text, or the teacher?

The teacher never generates; it scores corpus documents. So hedging reaches the student by
cross entropy on text that hedges, or by KL toward a teacher that expects hedging even
where the text does not. Measured per capture:

* text      hedge words per 1000 words of the assistant turns, by source dataset
* teacher   at each line or sentence start in the assistant turns, the teacher's cached
            top-k probability of a hedge opener, against how often the text opens one

    python scratch/downstream/code_bench/hedge_sources.py
"""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from distillkit.offline_cache import OfflineTeacherCache  # noqa: E402

HEDGE = re.compile(r"\b(wait|actually|hold on|hmm|let me re-?read|let me reconsider|"
                   r"let me re-?check|let me double[- ]check|but wait)\b", re.I)
OPENERS = ("Wait", "Actually", "Hmm", "Hold")
ROOT = Path("D:/DeepThought/Projects/HybridModel")
CAPTURES = [("capture-data/run5m.jsonl", "teacher-cache-5m"),
            ("capture-data/expand-chat.jsonl", "teacher-cache-expand-chat"),
            ("capture-data/expand-code.jsonl", "teacher-cache-expand-code")]
ASSISTANT = "<|im_start|>assistant"
PER_CAPTURE = 600


def main():
    tok = AutoTokenizer.from_pretrained(ROOT / "teacher-hf")
    opener_ids = set()
    for word in OPENERS:
        opener_ids.add(tok(word, add_special_tokens=False)["input_ids"][0])
        opener_ids.add(tok(" " + word, add_special_tokens=False)["input_ids"][0])
    for jsonl, cache_name in CAPTURES:
        cache = OfflineTeacherCache(ROOT / cache_name)
        by_source = collections.defaultdict(lambda: [0, 0, 0])
        mass, hits, starts = 0.0, 0, 0
        rows = [json.loads(line) for line in open(ROOT / jsonl, encoding="utf-8")]
        rows = [r for r in rows if r.get("split", "train") == "train"]
        step = max(1, len(rows) // PER_CAPTURE)
        for row in rows[::step][:PER_CAPTURE]:
            text = row["text"]
            answer = text.split(ASSISTANT, 1)[1] if ASSISTANT in text else text
            entry = by_source[row.get("source", "?")]
            entry[0] += len(HEDGE.findall(answer))
            entry[1] += len(answer.split())
            entry[2] += int("<think>" in answer)
            try:
                record = cache.read_document(row["doc_id"], include_hidden_states=False)
            except (KeyError, ValueError):
                continue
            ids = np.asarray(record["input_ids"])
            pieces = tok.convert_ids_to_tokens(ids.tolist())
            marker = tok(ASSISTANT, add_special_tokens=False)["input_ids"]
            begin = next((i + len(marker) for i in range(len(ids) - len(marker))
                          if ids[i:i + len(marker)].tolist() == marker), 1)
            topk_ids = record["topk_ids"]
            probs = np.exp(record["topk_logprobs"].astype(np.float32))
            for position in range(max(begin, 1), len(ids)):
                previous = pieces[position - 1]
                # a line start, or a sentence start after ". "
                if not ("\u010a" in previous or previous.endswith(".")):
                    continue
                starts += 1
                row_ids = topk_ids[position - 1]
                mass += float(probs[position - 1][np.isin(row_ids, list(opener_ids))].sum())
                hits += int(ids[position]) in opener_ids
        print("%s: %d line/sentence starts in assistant turns" % (cache_name, starts))
        print("   teacher P(hedge opener) %.4f   text rate %.4f   teacher/text %.2f"
              % (mass / max(starts, 1), hits / max(starts, 1),
                 (mass / max(starts, 1)) / max(hits / max(starts, 1), 1e-9)))
        for source, (h, words, think) in sorted(by_source.items(), key=lambda kv: -kv[1][1])[:10]:
            print("   %-34s hedges/1k words %6.2f  think blocks %3d  (%d words)"
                  % (source[:34], 1000 * h / max(words, 1), think, words))


if __name__ == "__main__":
    main()
