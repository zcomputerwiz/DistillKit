"""Does the teacher teach the student not to stop thinking?

At every position where a held-out document's next token closes a reasoning block
(`</think>`) or ends the assistant turn (`<|im_end|>`), the probability each model gives
that token -- and the teacher's, from its cached top-k (0 when outside it). A teacher that
would have kept going puts little there; KL toward it teaches a student to postpone.

    python scratch/dense_gr/closing_propensity.py source=../student-2b-hf control=<ckpt> ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import smoke_train  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

from hedge_propensity import load  # noqa: E402
from teacher_kl import CachedTeacher  # noqa: E402

CACHES = ["../teacher-cache-5m", "../teacher-cache-expand-code", "../teacher-cache-expand-chat"]


def main():
    from transformers import AutoTokenizer

    arms = [a.split("=", 1) for a in sys.argv[1:]]
    tok = AutoTokenizer.from_pretrained(arms[0][1])
    closers = {name: tok(text, add_special_tokens=False)["input_ids"]
               for name, text in (("think", "</think>"), ("turn", "<|im_end|>"))}
    assert all(len(v) == 1 for v in closers.values()), closers
    closers = {k: v[0] for k, v in closers.items()}
    docs, teacher_p = [], {k: [] for k in closers}
    for cache in CACHES:
        teacher = CachedTeacher([cache], "eval", device="cpu", max_length=1024)
        for _, doc in teacher.stratified(60):
            record = teacher.read(doc)
            ids = record["input_ids"][0]
            spots = {}
            for kind, token in closers.items():
                where = (ids[1:] == token).nonzero().flatten().tolist()
                spots[kind] = where
                topk = record["topk_ids"][0]
                probs = record["topk_logprobs"][0].exp()
                for p in where:
                    hit = (topk[p] == token)
                    teacher_p[kind].append(float(probs[p][hit].sum()))
            if any(spots.values()):
                docs.append((ids, spots))
    print("%d documents; %s closing positions"
          % (len(docs), {k: len(v) for k, v in teacher_p.items()}))
    print("%-10s " % "teacher" + "  ".join("P(%s) %.3f" % (k, np.mean(v)) for k, v in teacher_p.items()))
    for name, path in arms:
        model = load(path)
        got = {k: [] for k in closers}
        with torch.no_grad():
            for ids, spots in docs:
                logits = model(input_ids=ids.view(1, -1).cuda(), use_cache=False).logits[0].float()
                probs = logits.softmax(-1)
                for kind, where in spots.items():
                    for p in where:
                        got[kind].append(float(probs[p, closers[kind]]))
        print("%-10s " % name + "  ".join("P(%s) %.3f" % (k, np.mean(v)) for k, v in got.items()))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
