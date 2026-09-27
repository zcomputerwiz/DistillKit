"""What the teacher said about the student's own rollouts.

On-policy distillation only helps if the teacher's distribution, read at the student's
positions, points somewhere better than where the student went. This reads a rollout
capture and reports, for finished and unfinished (cut at the cap) rollouts: the teacher's
probability of the token the student actually sampled, how often the teacher's top choice
agrees, the teacher's mass on closing the thought (`</think>`) and on ending the turn, and
the same inside repeated stretches (a position whose 8-gram already occurred) and at a loop's
onset (the first repeated position), where a teacher that says "stop looping" would put little
mass on the repeat.

Round 1's answer: it does not. At onset and inside loops the teacher gives the student's
repeated token ~0.86-0.94 and agrees with it as top-1 ~90% of the time -- in-context
copying -- so KL on looping rollouts reinforces the loop.

    python scratch/dense_gr/onpolicy_signal.py ../teacher-cache-onpolicy-r1 ../capture-data/onpolicy-r1.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distillkit.offline_cache import OfflineTeacherCache  # noqa: E402

START = "<|im_start|>assistant\n<think>\n"


def main():
    cache_path, rollouts = Path(sys.argv[1]), Path(sys.argv[2])
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("D:/DeepThought/Projects/HybridModel/teacher-hf")
    close, end = tok.convert_tokens_to_ids("</think>"), tok.convert_tokens_to_ids("<|im_end|>")
    meta = {}
    for line in open(rollouts, encoding="utf-8"):
        row = json.loads(line)
        meta[row["doc_id"]] = (row["finished"], row["text"])
    cache = OfflineTeacherCache(cache_path)
    stats = {}
    for doc_id in cache.document_ids():
        finished, text = meta[doc_id]
        record = cache.read_document(doc_id, include_hidden_states=False)
        ids = np.asarray(record["input_ids"], dtype=np.int64)
        top_ids = np.asarray(record["topk_ids"], dtype=np.int64)
        top_lp = np.asarray(record["topk_logprobs"], dtype=np.float32)
        prompt = len(tok(text[:text.rfind(START) + len(START)], add_special_tokens=False)["input_ids"])
        seen, repeat = set(), np.zeros(len(ids), bool)
        for i in range(prompt + 8, len(ids)):
            gram = tuple(ids[i - 8:i])
            repeat[i] = gram in seen
            seen.add(gram)
        thinking = True
        for i in range(prompt, len(ids) - 1):  # position i predicts token i + 1
            nxt = ids[i + 1]
            thinking = thinking and ids[i] != close
            match = top_ids[i] == nxt
            p_actual = float(np.exp(top_lp[i][match][0])) if match.any() else 0.0
            p = np.exp(top_lp[i])
            key = ("finished" if finished else "cut", "onset" if repeat[i + 1] and not repeat[i] else "repeat" if repeat[i + 1] else "fresh",
                   "think" if thinking else "answer")
            s = stats.setdefault(key, np.zeros(5))
            s += (1, p_actual, float(top_ids[i][0] == nxt), float(p[top_ids[i] == close].sum()),
                  float(p[top_ids[i] == end].sum()))
    print("%-30s %8s %9s %7s %10s %9s" % ("rollouts / stretch / part", "tokens", "p(actual)",
                                          "top1=", "p(</think>)", "p(end)"))
    for key in sorted(stats):
        n, pa, t1, pc, pe = stats[key]
        print("%-30s %8d %9.3f %7.3f %10.4f %9.4f" % (" / ".join(key), n, pa / n, t1 / n, pc / n, pe / n))


if __name__ == "__main__":
    main()
