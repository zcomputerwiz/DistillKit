"""Are the 145 flagged documents really benchmark questions, or just similar text?

The screen flags a document for sharing a single 8- or 13-word run with any scored
question. One shared run is weak evidence: templated stems ("which of the following
best describes the") and stock science phrasing recur across unrelated questions. This
measures, per flagged document, how much of the best-matching question it actually
contains, whether the answer options and the correct answer are there too, and which
benchmark split the question comes from -- only MMLU test and ARC-Challenge test are
scored by the eval.
"""
import json
import re
import sys
from collections import Counter, defaultdict

from datasets import load_dataset

WORDS = re.compile(r"[a-z0-9]+")
CORPUS = sys.argv[1] if len(sys.argv) > 1 else "../capture-data/run5m.jsonl"


def words(text):
    return WORDS.findall(text.lower())


def grams(tokens, width):
    return {" ".join(tokens[i:i + width]) for i in range(len(tokens) - width + 1)}


# --- the bank, with each question kept whole so a match can be measured -------------
questions = []  # (bank, question words, choices, answer text)
for repo, config, split, bank in [
        ("cais/mmlu", "all", "test", "MMLU test (scored)"),
        ("allenai/ai2_arc", "ARC-Challenge", "test", "ARC-Challenge test (scored)"),
        ("allenai/ai2_arc", "ARC-Easy", "test", "ARC-Easy test (not scored)")]:
    for row in load_dataset(repo, config, split=split):
        if repo == "cais/mmlu":
            choices = list(row["choices"])
            answer = choices[row["answer"]]
        else:
            choices = list(row["choices"]["text"])
            labels = list(row["choices"]["label"])
            answer = choices[labels.index(row["answerKey"])] if row["answerKey"] in labels else ""
        questions.append((bank, words(row["question"]), choices, answer, row["question"]))

index = defaultdict(set)
for qi, (_, qw, _, _, _) in enumerate(questions):
    width = 13 if len(qw) >= 13 else 8
    if len(qw) < 8:
        continue
    for gram in grams(qw, width):
        index[gram].add(qi)


def coverage(qw, doc_grams):
    """Fraction of the question's words inside a run it shares with the document."""
    width = 13 if len(qw) >= 13 else 8
    covered = [False] * len(qw)
    for i in range(len(qw) - width + 1):
        if " ".join(qw[i:i + width]) in doc_grams:
            for j in range(i, i + width):
                covered[j] = True
    return sum(covered) / max(len(qw), 1)


def present(option, doc_text):
    ow = " ".join(words(option))
    return bool(ow) and (" " + ow + " ") in (" " + doc_text + " ")


def user_turn(text):
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.S)
    return (match.group(1) if match else text).strip()


rows = []
with open(CORPUS, encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        dw = words(record["text"])
        doc_grams = grams(dw, 8) | grams(dw, 13)
        hits = set()
        for gram in doc_grams:
            hits |= index.get(gram, set())
        if not hits:
            continue
        doc_text = " ".join(dw)
        best = None
        for qi in hits:
            bank, qw, choices, answer, raw = questions[qi]
            cov = coverage(qw, doc_grams)
            long_choices = [c for c in choices if len(words(c)) >= 1]
            shown = sum(present(c, doc_text) for c in long_choices)
            candidate = (cov, shown, qi)
            if best is None or candidate > best:
                best = candidate
        cov, shown, qi = best
        bank, qw, choices, answer, raw = questions[qi]
        answer_in = present(answer, doc_text) and len(words(answer)) >= 1
        rows.append({"doc_id": record["doc_id"], "source": record.get("source"),
                     "bank": bank, "coverage": cov, "choices_shown": shown,
                     "choices": len(choices), "answer_in": answer_in,
                     "question": raw, "answer": answer,
                     "user": user_turn(record["text"])})


def verdict(row):
    all_choices = row["choices_shown"] == row["choices"]
    if row["coverage"] >= 0.9 and all_choices:
        return "same question, same options"
    if row["coverage"] >= 0.9:
        return "same question text"
    if row["coverage"] >= 0.5:
        return "most of the question"
    return "incidental run"


for row in rows:
    row["verdict"] = verdict(row)

print("flagged documents: %d\n" % len(rows))
table = Counter((row["verdict"], row["bank"]) for row in rows)
order = ["same question, same options", "same question text", "most of the question",
         "incidental run"]
banks = ["MMLU test (scored)", "ARC-Challenge test (scored)", "ARC-Easy test (not scored)"]
print("%-30s %8s %8s %8s %6s" % ("verdict", "MMLU", "ARC-C", "ARC-E", "total"))
for name in order:
    counts = [table[(name, bank)] for bank in banks]
    print("%-30s %8d %8d %8d %6d" % (name, *counts, sum(counts)))
print()
print("answer text present, by verdict:")
for name in order:
    group = [row for row in rows if row["verdict"] == name]
    if group:
        print("   %-30s %d of %d" % (name, sum(row["answer_in"] for row in group), len(group)))
print()
print("by corpus source:")
for (source, name), count in sorted(Counter((row["source"], row["verdict"])
                                            for row in rows).items()):
    print("   %-24s %-30s %d" % (source, name, count))

json.dump(rows, open("../capture-data/run5m-contamination-audit.json", "w", encoding="utf-8"), indent=1)


def show(name, limit):
    group = [row for row in rows if row["verdict"] == name]
    for row in group[:limit]:
        print("\n--- [%s] %s  (%s, coverage %.2f, options %d/%d, answer %s)"
              % (name, row["doc_id"][:16], row["source"], row["coverage"],
                 row["choices_shown"], row["choices"], "yes" if row["answer_in"] else "no"))
        print("    benchmark: %s" % row["question"][:220].replace("\n", " "))
        print("    answer:    %s" % row["answer"][:120])
        print("    corpus:    %s" % row["user"][:300].replace("\n", " "))


for name, limit in (("same question, same options", 3), ("same question text", 3),
                    ("most of the question", 4), ("incidental run", 6)):
    show(name, limit)
