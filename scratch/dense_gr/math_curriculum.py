"""A drilled math curriculum: basics worked in-head, anything heavy handed to Python.

The policy (memory: drill-basics-outsource-rest): the student should be reliable at basic
arithmetic and algebra in its own reasoning, and hand larger computation to a tool. Its
thinking-mode loops start from exactly the slips this targets -- a wrong Fibonacci value,
a subtraction from an element not in the list.

Every item is generated fresh from a seeded RNG and exact by construction, so the model
learns procedures rather than answers. In-head items are short worked scratchpads -- column
carries, partial products, long-division steps, one algebra move per line -- in the native
thinking format. Outsource items reason briefly that the computation is too large to trust
by hand, call `run_python`, and answer from its result, which is computed here.

Each item carries a random answer-format instruction (boxed, bare answer, "Answer:" line, or
none, meaning plain prose) and the reply follows it; a share are non-thinking, with the worked
steps in the reply. The non-thinking student had learned to drop `\\boxed{}` when asked for it.

    python scratch/dense_gr/math_curriculum.py --tokens 1500000 --output ../capture-data/math-curriculum.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
from fractions import Fraction
from pathlib import Path

TEACHER = "D:/DeepThought/Projects/HybridModel/teacher-hf"
TOOLS = [{"type": "function", "function": {
    "name": "run_python", "description": "Execute Python code and return its stdout.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string",
                   "description": "Python source to run"}}, "required": ["code"]}}}]


def digits(n):
    return [int(c) for c in reversed(str(n))]


def add(rng):
    a, b = (rng.randrange(10 ** (k - 1), 10 ** k) for k in (rng.randint(2, 4), rng.randint(2, 4)))
    lines, carry = [], 0
    da, db = digits(a), digits(b)
    places = ["ones", "tens", "hundreds", "thousands", "ten-thousands"]
    for i in range(max(len(da), len(db))):
        x, y = (da[i] if i < len(da) else 0), (db[i] if i < len(db) else 0)
        total = x + y + carry
        lines.append("%s: %d + %d%s = %d, write %d%s" % (
            places[i].capitalize(), x, y, " + %d carried" % carry if carry else "", total,
            total % 10, ", carry 1" if total >= 10 else ""))
        carry = total // 10
    if carry:
        lines.append("Final carry: write 1")
    return "What is %d + %d?" % (a, b), lines, a + b


def subtract(rng):
    a = rng.randrange(100, 100000)
    b = rng.randrange(10, a)
    lines, borrow = [], 0
    da, db = digits(a), digits(b)
    for i in range(len(da)):
        x, y = da[i] - borrow, (db[i] if i < len(db) else 0)
        paid = "%d, less the 1 borrowed, is %d; " % (da[i], x) if borrow else ""
        if x < y:
            lines.append("Place %d: %s%d - %d needs a borrow: %d - %d = %d" % (i + 1, paid, x, y, x + 10, y, x + 10 - y))
            borrow = 1
        else:
            lines.append("Place %d: %s%d - %d = %d" % (i + 1, paid, x, y, x - y))
            borrow = 0
    return "Compute %d - %d." % (a, b), lines, a - b


def multiply(rng):
    a, b = rng.randrange(12, 1000), rng.randrange(2, 100)
    lines, total = [], 0
    for i, d in enumerate(digits(b)):
        part = a * d * 10 ** i
        lines.append("%d x %d%s = %d" % (a, d, " x %d" % 10 ** i if i else "", part))
        total += part
    if len(lines) > 1:
        lines.append("Sum of partial products: %s = %d" % (" + ".join(
            str(a * d * 10 ** i) for i, d in enumerate(digits(b))), total))
    return "What is %d times %d?" % (a, b), lines, a * b


def divide(rng):
    b = rng.randrange(2, 13)
    a = rng.randrange(20, 5000)
    q, r = divmod(a, b)
    lines, rest = [], 0
    for c in str(a):
        rest = rest * 10 + int(c)
        if rest >= b or lines:
            lines.append("Bring down to %d: %d goes into it %d times, remainder %d" % (
                rest, b, rest // b, rest % b))
        rest %= b
    answer = "%d remainder %d" % (q, r) if r else str(q)
    return "Divide %d by %d." % (a, b), lines, answer


def fractions(rng):
    a, b, c, d = rng.randrange(1, 10), rng.randrange(2, 13), rng.randrange(1, 10), rng.randrange(2, 13)
    lcm = b * d // math.gcd(b, d)
    x, y = a * (lcm // b), c * (lcm // d)
    result = Fraction(a, b) + Fraction(c, d)
    lines = ["Common denominator of %d and %d: %d" % (b, d, lcm),
             "%d/%d = %d/%d and %d/%d = %d/%d" % (a, b, x, lcm, c, d, y, lcm),
             "%d/%d + %d/%d = %d/%d" % (x, lcm, y, lcm, x + y, lcm)]
    g = math.gcd(x + y, lcm)
    if g > 1:
        lines.append("Divide top and bottom by %d: %s" % (g, result))
    return "Add %d/%d + %d/%d and simplify." % (a, b, c, d), lines, str(result)


def percent(rng):
    p, n = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 75]), rng.randrange(20, 2000, 10)
    value = Fraction(p * n, 100)
    lines = ["%d%% means %d/100" % (p, p), "%d x %d = %d" % (n, p, n * p),
             "%d / 100 = %s" % (n * p, float(value) if value.denominator != 1 else value)]
    return "What is %d%% of %d?" % (p, n), lines, str(float(value) if value.denominator != 1 else value)


def linear(rng):
    a, x, b = rng.choice([2, 3, 4, 5, 6, 7, 8, 9]), rng.randrange(-20, 21), rng.randrange(-30, 31)
    c = a * x + b
    sign = "+" if b >= 0 else "-"
    lines = ["Start: %dx %s %d = %d" % (a, sign, abs(b), c),
             "%s %d on both sides: %dx = %d" % ("Subtract" if b >= 0 else "Add", abs(b), a, c - b),
             "Divide both sides by %d: x = %d" % (a, x)]
    return "Solve for x: %dx %s %d = %d" % (a, sign, abs(b), c), lines, str(x)


def word(rng):
    name = rng.choice(["Maya", "Omar", "Lena", "Ravi", "Sofia", "Kenji"])
    item = rng.choice(["apples", "notebooks", "tickets", "marbles", "stickers"])
    packs, per, given = rng.randrange(2, 12), rng.randrange(3, 25), rng.randrange(1, 20)
    have = packs * per
    given = min(given, have - 1)
    lines = ["%d packs of %d: %d x %d = %d %s" % (packs, per, packs, per, have, item),
             "After giving away %d: %d - %d = %d" % (given, have, given, have - given)]
    return ("%s buys %d packs of %s with %d in each pack, then gives away %d. How many %s "
            "does %s have left?" % (name, packs, item, per, given, item, name)), lines, str(have - given)


IN_HEAD = [add, subtract, multiply, divide, fractions, percent, linear, word]

# The answer format follows the prompt's instruction, and no instruction means plain prose, so
# the model learns to obey the request rather than to box everything that looks like math.
FORMATS = [
    ("\n\nPut the final answer in \\boxed{}.", lambda a: "\\boxed{%s}" % a),
    ("\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
     lambda a: "The final answer is \\boxed{%s}." % a),
    ("\n\nReply with only the answer.", lambda a: str(a)),
    ("\n\nEnd your reply with a line of the form 'Answer: <value>'.", lambda a: "Answer: %s" % a),
    ("", lambda a: "The answer is %s." % a),
]


def formatted(messages, rng, thinking):
    """Give the item a random answer-format instruction and a matching final reply.

    Non-thinking items move the worked steps into the reply, ahead of the formatted answer."""
    instruction, render = rng.choice(FORMATS)
    messages[0]["content"] += instruction
    final = messages[-1]
    answer = final.pop("answer")
    if thinking:
        final["content"] = render(answer)
    else:
        steps = final.pop("reasoning_content")
        final["content"] = steps + "\n\n" + render(answer)
    return messages


def outsourced(rng):
    kind = rng.choice(["product", "power", "prime", "sqrt", "sum"])
    if kind == "product":
        a, b = rng.randrange(10 ** 4, 10 ** 6), rng.randrange(10 ** 3, 10 ** 5)
        q, code, why = "What is %d x %d?" % (a, b), "print(%d * %d)" % (a, b), "a product this large"
        out = str(a * b)
    elif kind == "power":
        a, e = rng.randrange(3, 40), rng.randrange(8, 25)
        q, code, why = "Compute %d^%d." % (a, e), "print(%d ** %d)" % (a, e), "a power this large"
        out = str(a ** e)
    elif kind == "prime":
        n = rng.randrange(10 ** 6, 10 ** 9) | 1
        q = "Is %d a prime number?" % n
        code = ("n = %d\nprint(n > 1 and all(n %% d for d in range(2, int(n ** 0.5) + 1)))" % n)
        why = "trial division up to the square root of a nine-digit number"
        out = str(n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)))
    elif kind == "sqrt":
        n = rng.randrange(1000, 10 ** 7)
        q, code, why = ("What is the square root of %d, to 4 decimal places?" % n,
                        "print(round(%d ** 0.5, 4))" % n, "a square root to four places")
        out = str(round(n ** 0.5, 4))
    else:
        values = [rng.randrange(1000, 100000) for _ in range(rng.randrange(8, 15))]
        q = "Add these numbers: %s." % ", ".join(map(str, values))
        code, why, out = "print(sum(%s))" % values, "a long column of five-digit numbers", str(sum(values))
    reason = ("This needs %s. Doing it by hand risks a slip, so I will compute it with Python "
              "and use the result." % why)
    if kind == "prime":
        out = "yes" if out == "True" else "no"
    return [{"role": "user", "content": q},
            {"role": "assistant", "content": "", "reasoning_content": reason,
             "tool_calls": [{"type": "function", "function": {"name": "run_python",
                                                              "arguments": {"code": code}}}]},
            {"role": "tool", "name": "run_python", "content": out},
            {"role": "assistant", "reasoning_content": "The tool returned %s." % out, "answer": out}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1_500_000)
    parser.add_argument("--outsource-share", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--non-thinking-share", type=float, default=0.4)
    parser.add_argument("--reasoning-effort", default="medium",
                        help="the teacher template injects a system prompt for xhigh (its default) "
                             "and low; medium injects none, matching the student's template")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEACHER)
    rng = random.Random(args.seed)
    total = n = 0
    counts = {}
    with open(args.output, "w", encoding="utf-8") as out:
        while total < args.tokens:
            if rng.random() < args.outsource_share:
                messages, kind, tools = outsourced(rng), "outsource", TOOLS
            else:
                generator = rng.choice(IN_HEAD)
                question, lines, answer = generator(rng)
                messages = [{"role": "user", "content": question},
                            {"role": "assistant", "reasoning_content": "\n".join(lines),
                             "answer": answer}]
                kind, tools = generator.__name__, None
            thinking = rng.random() >= args.non_thinking_share
            messages = formatted(messages, rng, thinking)
            if not thinking:
                for m in messages[1:-1]:  # a tool-call turn says why it calls, in the open
                    if m["role"] == "assistant":
                        m["content"] = m.pop("reasoning_content")
            text = tok.apply_chat_template(messages, tools=tools, tokenize=False, enable_thinking=thinking,
                                           reasoning_effort=args.reasoning_effort)
            kind += "" if thinking else ":nothink"
            length = len(tok(text, add_special_tokens=False)["input_ids"])
            split = "eval" if n % args.eval_every == 0 else "train"
            out.write(json.dumps({"doc_id": "curriculum:%s:%d" % (kind, n), "text": text,
                                  "split": split, "source": "curriculum", "domain": kind,
                                  "tokens": length}) + "\n")
            counts[kind] = counts.get(kind, 0) + 1
            total += length
            n += 1
    print("wrote %d items, %d tokens: %s" % (n, total, json.dumps(counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
