"""How much a model wants to open a line with "Wait", where the text does not.

At every line or sentence start in the assistant turns of held-out capture documents, the
model's probability of a hedge opener, averaged -- only where the actual next token is not
one, so this measures the inclination the teacher's distribution injects, not agreement
with text that hedges. Paired per document against the first model named.

    python scratch/dense_gr/hedge_propensity.py source=../student-2b-hf finish=<ckpt> ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import smoke_train  # noqa: E402,F401
import torch  # noqa: E402

from teacher_kl import CachedTeacher, first_response, hedge_token_ids  # noqa: E402

CACHES = ["../teacher-cache-expand-code", "../teacher-cache-expand-chat", "../teacher-cache-5m"]


def load(path):
    import json
    from transformers import AutoModelForCausalLM

    config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    if config.get("csa2_enabled") or config.get("residual_stream_enabled"):
        from distillkit.models import Qwen35WidenedForCausalLM as Model
    else:
        Model = AutoModelForCausalLM
    return Model.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()


def main():
    from transformers import AutoTokenizer

    arms = [a.split("=", 1) for a in sys.argv[1:]]
    tok = AutoTokenizer.from_pretrained(arms[0][1])
    hedges = torch.tensor(hedge_token_ids(tok), device="cuda")
    marker = tok("<|im_start|>assistant", add_special_tokens=False)["input_ids"]
    docs = []
    for cache in CACHES:
        teacher = CachedTeacher([cache], "eval", device="cpu", max_length=1024)
        for _, doc in teacher.stratified(40):
            ids = teacher.read(doc)["input_ids"][0]
            start = first_response(ids.tolist(), marker)
            if start is None:
                continue
            pieces = tok.convert_ids_to_tokens(ids.tolist())
            rows = [p for p in range(max(start, 1), len(ids) - 1)
                    if ("Ċ" in pieces[p - 1] or pieces[p - 1].endswith("."))
                    and int(ids[p]) not in set(hedges.tolist())]
            if rows:
                docs.append((Path(cache).name, ids, rows))
    print("%d documents, %d line starts" % (len(docs), sum(len(r) for _, _, r in docs)))
    table = {}
    for name, path in arms:
        model = load(path)
        per_doc = []
        with torch.no_grad():
            for _, ids, rows in docs:
                logits = model(input_ids=ids.view(1, -1).cuda(), use_cache=False).logits[0]
                at = torch.tensor([p - 1 for p in rows], device="cuda")
                probs = logits[at].float().softmax(-1)[:, hedges].sum(-1)
                per_doc.append(float(probs.mean()))
        table[name] = per_doc
        del model
        torch.cuda.empty_cache()
    reference = arms[0][0]
    for name, _ in arms:
        values = table[name]
        by_cache = {}
        for (cache, _, _), v in zip(docs, values):
            by_cache.setdefault(cache, []).append(v)
        parts = "  ".join("%s %.4f" % (c.replace("teacher-cache-", ""), sum(v) / len(v))
                          for c, v in by_cache.items())
        ratio = (sum(values) / max(sum(table[reference]), 1e-12))
        print("%-10s P(hedge opener) %.4f  (%.2fx %s)   %s"
              % (name, sum(values) / len(values), ratio, reference, parts))


if __name__ == "__main__":
    main()
