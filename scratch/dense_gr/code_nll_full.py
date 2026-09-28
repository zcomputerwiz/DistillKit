"""Held-out code NLL on the whole expand-code eval split, by source dataset, for several checkpoints.

The trainer scores 20 stratified documents and `merge_proxy.py` the first 64 by id; on
round 5 they disagreed by 0.09 nats about whether code NLL rose at all, so conclusions
drawn from either (stripping the effort prompt, the blends) need the full split.

    python scratch/dense_gr/code_nll_full.py think=<ckpt> r5=<ckpt> ...
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402


def main():
    from cut_cross_entropy import linear_cross_entropy
    from transformers import AutoTokenizer

    from distillkit.models import Qwen35WidenedForCausalLM
    from smoke_train import ANSWER_MARKER
    from teacher_kl import CachedTeacher

    arms = [a.split("=", 1) for a in sys.argv[1:]]
    tok = AutoTokenizer.from_pretrained(arms[0][1])
    held = CachedTeacher("../teacher-cache-expand-code", "eval", device="cuda", max_length=1024,
                         answer_marker=tok(ANSWER_MARKER, add_special_tokens=False)["input_ids"],
                         min_answer_tokens=2)
    docs = sorted(held.ids)
    source = lambda doc_id: doc_id.split(":")[0]
    print("%d documents: %s" % (len(docs), json.dumps({s: sum(source(d) == s for d in docs)
                                                       for s in sorted({source(d) for d in docs})})))
    for name, path in arms:
        model = Qwen35WidenedForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        by = {}
        for doc_id in docs:
            ids = held.read(doc_id)["input_ids"]
            with torch.no_grad():
                state = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    use_cache=False).last_hidden_state
                loss = float(linear_cross_entropy(state, model.lm_head.weight, ids, shift=1,
                                                  reduction="sum"))
            for key in (source(doc_id), "ALL"):
                row = by.setdefault(key, [0.0, 0])
                row[0] += loss
                row[1] += ids.shape[1] - 1
        print("%-14s " % name + "  ".join("%s %.4f" % (k, t / n) for k, (t, n) in sorted(by.items())),
              flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
