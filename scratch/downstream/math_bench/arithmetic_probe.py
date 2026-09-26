"""Exact-match arithmetic, by operation and digit count, for any checkpoint.

Did the conversion cost precise computation, or was the source already weak? Seeded random
problems -- add, subtract, multiply, divide with integer quotients -- at 2 to 6 digits,
asked for the bare answer in non-thinking mode, greedy, scored by exact integer match.
The same problems for every model.

    python scratch/downstream/math_bench/arithmetic_probe.py source=../student-2b-hf think=<ckpt>
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dense_gr"))

import smoke_train  # noqa: E402,F401
import torch  # noqa: E402

from hedge_propensity import load  # noqa: E402

OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
       "/": lambda a, b: a // b}


def problems(per_cell=40, seed=0):
    rng = random.Random(seed)
    out = []
    for op in OPS:
        for digits in range(2, 7):
            for _ in range(per_cell):
                a = rng.randrange(10 ** (digits - 1), 10 ** digits)
                b = rng.randrange(10 ** (digits - 1), 10 ** digits)
                if op == "*" and digits > 4:
                    b = rng.randrange(10, 1000)  # keep products readable
                if op == "/":
                    b = rng.randrange(2, 10 ** min(digits - 1, 3))
                    a = b * rng.randrange(10 ** (digits - 2), 10 ** digits // b + 1)
                out.append((op, digits, a, b, OPS[op](a, b)))
    return out


def main():
    arms = [a.split("=", 1) for a in sys.argv[1:]]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(arms[0][1])
    tok.padding_side = "left"
    items = problems()
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": "Compute %d %s %d. Reply with only the number." % (a, op, b)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for op, _, a, b, _ in items]
    table = {}
    for name, path in arms:
        model = load(path)
        model.config.use_cache = True
        answers = []
        for start in range(0, len(prompts), 64):
            batch = tok(prompts[start:start + 64], return_tensors="pt", padding=True,
                        add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                out = model.generate(**batch, max_new_tokens=24, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
            answers += tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        cells = {}
        for (op, digits, a, b, truth), text in zip(items, answers):
            found = re.findall(r"-?\d[\d,]*", text)
            ok = bool(found) and int(found[0].replace(",", "")) == truth
            cells.setdefault((op, digits), []).append(ok)
        table[name] = cells
        del model
        torch.cuda.empty_cache()
    names = [n for n, _ in arms]
    print("op digits  " + "  ".join("%8s" % n for n in names))
    for key in sorted(table[names[0]], key=lambda k: (list(OPS).index(k[0]), k[1])):
        print("%2s %6d  " % key + "  ".join("%7.0f%%" % (100 * sum(table[n][key]) / len(table[n][key]))
                                           for n in names))
    for n in names:
        all_ok = [x for cell in table[n].values() for x in cell]
        print("%-8s overall %.1f%%" % (n, 100 * sum(all_ok) / len(all_ok)))
    Path(__file__).with_name("arithmetic-probe.json").write_text(json.dumps(
        {n: {"%s%d" % k: sum(v) / len(v) for k, v in cells.items()} for n, cells in table.items()},
        indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
