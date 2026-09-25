"""What were the completions that never produced a closed code block doing?

    python scratch/downstream/code_bench/no_code_audit.py <completions dir> [...]

Per set: completions with no closed fence, and for them against the ones that did write
code -- hesitation markers per 1000 words, the longest repeated 8-word run, the share of
repeated lines, and whether an unclosed fence had been opened (it was writing code when
the cap hit).
"""
import collections
import json
import re
import sys
from pathlib import Path

FULL = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
OPEN = re.compile(r"```(?:python|py)?\s*\n")
MARKERS = re.compile(r"\b(wait|actually|hold on|hmm|hm|let me re-?read|let me reconsider|"
                     r"let me re-?check|let me double[- ]check|let me verify|let me trace|"
                     r"on second thought|no,? that|i think|but wait)\b", re.I)


def profile(text):
    words = text.split()
    grams = collections.Counter(tuple(words[i:i + 8]) for i in range(max(0, len(words) - 8)))
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    repeated_lines = 1 - len(set(lines)) / max(len(lines), 1)
    return dict(markers=1000 * len(MARKERS.findall(text)) / max(len(words), 1),
                repeat8=max(grams.values()) if grams else 0,
                repeated_lines=repeated_lines,
                opened=bool(OPEN.search(text)) and not FULL.search(text))


def summarize(rows):
    if not rows:
        return "none"
    mean = lambda k: sum(r[k] for r in rows) / len(rows)
    loop = sum(r["repeat8"] >= 4 or r["repeated_lines"] > 0.3 for r in rows)
    return ("n %3d  markers/1k words %5.1f  max 8-gram repeat %4.1f  repeated lines %4.0f%%  "
            "looping %3d  cut mid-code %3d"
            % (len(rows), mean("markers"), mean("repeat8"), 100 * mean("repeated_lines"),
               loop, sum(r["opened"] for r in rows)))


for directory in sys.argv[1:]:
    rows = [json.loads(l) for l in open(Path(directory) / "completions.jsonl", encoding="utf-8-sig")]
    none = [profile(r["raw"]) for r in rows if not FULL.search(r["raw"])]
    code = [profile(r["raw"]) for r in rows if FULL.search(r["raw"])]
    print(Path(directory).name)
    print("  no code    ", summarize(none))
    print("  wrote code ", summarize(code))
