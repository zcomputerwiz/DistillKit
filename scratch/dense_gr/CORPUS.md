# Expanding the capture corpus, and what screening it found (2026-09-21)

The teacher cache the distillation arms train against is 5,598 documents and 4,999,608
tokens of one dataset's `sft_balanced` config. Two things prompted a second look at it:
the ARC screen turned out to be partly contaminated by it, and the corpus is thin in
exactly the two places the model is meant to be used.

## What the existing corpus is

| domain | documents | largest sources |
| --- | --- | --- |
| code | 1,440 | Evol-Code 842, CodeAlpaca 560 |
| math | 1,413 | MetaMathQA 625, NuminaMath-CoT 268, OrcaMath 226 |
| reasoning | 1,040 | SciQ 372, CommonsenseQA 302, QASC 167 |
| instruction | 767 | tulu-3 555, Dolly 191 |
| agent_tool | 514 | glm5.2-agent-tool-synthetic 494 |

Almost every row is one instruction and one answer, so plain multi-turn chat is nearly
absent, and the code is Evol-Code and CodeAlpaca -- short synthetic snippets rather than
the file-scale code the model gets asked about.

## The screen

`expand_corpus.py` screens by n-gram containment, the standard decontamination test: a
document is dropped when it shares an 8- or 13-word run with a benchmark question.
Thirteen is the usual width; 8 exists because many benchmark questions are shorter than
13 words and would otherwise be invisible. Questions under 8 words contribute nothing,
because a run that short is not evidence of anything -- 1,340 of MMLU's and ARC's test
questions fall into that gap and are not screened at all.

Choices and answers are deliberately out of the bank. They are short, formulaic and
shared between unrelated questions, so including them drops documents that merely
discuss the same subject.

The banks are split by whether the eval actually scores that split, because reading the
two as one number overstates the problem:

* **scored** -- `cais/mmlu` all test, `allenai/ai2_arc` ARC-Challenge and ARC-Easy test.
  17,590 questions, 443,175 shingles. Overlap here is contamination.
* **related** -- the validation, dev and train splits of the same three. 6,055
  questions, 101,445 shingles. Overlap here is a near-duplicate of the benchmark's
  distribution, which is a much weaker objection. Dropped anyway, since it costs almost
  nothing.

### The existing corpus, screened

| bank | documents | share |
| --- | --- | --- |
| scored | 145 | 2.59% |
| related | 3 | 0.05% |

By source: ARC-Easy 92, ARC-Challenge 41, then a tail of MetaMathQA 3, NuminaMath-CoT 2,
MATH 6, SciQ 1, Evol-Code 1.

### Checked by hand: which flags are contamination, and what the screen missed

One shared 8- or 13-word run is weak evidence, and an earlier screen here had already
counted similar text as contamination. So every flag was checked: `contamination_audit.py`
finds the best-matching benchmark question for each flagged document, measures how much
of it the document contains, and whether the answer options and the correct answer are
there too. The twelve that were not ARC were then read one by one, because the coverage
score gets them wrong in both directions.

| verdict | ARC-Challenge test (scored) | ARC-Easy test (not scored) | MMLU test (scored) |
| --- | --- | --- | --- |
| contamination | 46 | 99 | 6 |
| same problem, reworded | -- | -- | 1 |
| not contamination | -- | 1 | 4 |

**Every ARC document in the corpus is an ARC test question.** The 133 the screen flagged
carry the question, all four options and the answer, verbatim. The corpus holds 145 ARC
documents, and the other 12 were never screened, because their questions are 4 to 7
words -- under the 8-word floor -- but each matches an ARC test question exactly. Those
12 were matched on question text only, not options. With 133 of 133 screenable ones
being full test items, the upstream mixture evidently drew its ARC rows from the test
split. So the screen *undercounted* ARC: 46 scored ARC-Challenge questions, not 41.

**Six MMLU test questions are really in the corpus.** Five are verbatim MATH problems
that MMLU's mathematics subjects share -- `Express 0.1(7) as a common fraction`, the
toothpaste unit-price problem, the least perfect square with three prime factors, the
45-degree triangle with a 10-inch hypotenuse, and the point on `h(x) = g(x)^2`. The sixth
is MetaMathQA's inversion of a remainder problem: the same question with one number
replaced by X, and the answer, 37, stated in it. One more is the same problem reworded:
"the remainder when 9! is divided by 10" is "the ones digit of 1 x 2 x ... x 9".

**Five flags are not contamination.** Three are templated competition problems with
different numbers and different answers -- a 45-degree triangle with an 8*sqrt(2)
hypotenuse, the 150th Fibonacci term mod 9 against the 100th mod 4, and a ball-drawing
game with different stakes whose answer is 15, not 3. One shares only the stem "what is
the value of the expression". The last is the 9-word generic question "What is at the
center of our solar system?" with different options, in a bank the eval does not score.
It had coverage 1.00, which is why a coverage threshold is not a verdict.

The verified list, with a reason for each entry, is `capture-data/run5m-contamination.json`
-- 157 documents, 152 of them contamination or a reworded equivalent. That list, not the
145 flags, is what to exclude if the training corpus itself is to be clean.

This does not change any MMLU result. None of the seven MMLU questions with a training
copy is in either split of `q512-bundle.json` or `q512-clean-bundle.json`, checked
against the prompts directly. It does mean a resample of the MMLU bank would need
screening, and that the 8-word floor has to be covered by exact-text matching.

**The expanded corpus, checked under the floor.** The same gap applies to `expand.jsonl`,
so its 9,932 documents were searched for the 1,279 scored questions under 8 words,
verbatim. That finds 154 documents, and they show exactly why this cannot be a verdict:
MMLU has "questions" that are one word or a fragment -- `gluten`, `inflation`,
`theories`, `4 3` -- whose meaning lives in the options, and those words occur in any
chat or code corpus. Requiring what the real ARC cases had -- the question plus at least
two of its options -- leaves none. The only hit of four or more words is the stem "which
of the following statements is true", with none of its options present. The expanded
corpus is clean against MMLU and ARC test at every question length.

## The expansion

Four sources, 1.25M tokens each, drawn round-robin so a run cut short stays balanced.
Rendered exactly as the existing corpus is: the teacher's own chat template, whole
documents one per row, unpadded, cut at the capture's `sequence_length` of 4096.

| source | kept | tokens | scored | related | duplicate |
| --- | --- | --- | --- | --- | --- |
| Magicoder-OSS-Instruct-75K | 2,198 | 1,250,180 | 0 | 0 | 0 |
| self-oss-instruct-sc2-exec-filter-50k | 3,627 | 1,250,125 | 0 | 0 | 0 |
| tulu-3-sft-mixture | 3,100 | 1,250,002 | 0 | 0 | 125 |
| ultrachat_200k | 1,007 | 1,250,213 | 0 | 1 | 0 |
| **total** | **9,932** | **5,000,797** | **0** | **1** | **125** |

**Not one document out of 9,932 overlaps a scored benchmark split.** That is worth
stating plainly because it was not the expected result: tulu-3-sft-mixture is a large
public mixture with benchmark-derived subsets in it, and it was the most likely source
of a repeat of the ARC problem. Its benchmark-derived rows are GSM8K- and MATH-shaped
rather than MMLU- or ARC-shaped, so they do not touch these two banks. The screen is
demonstrably able to find contamination when it is there -- it found 145 documents in
the existing corpus using the same bank in the same run.

The 125 duplicates are exact repeats inside tulu-3-sft-mixture itself, after
normalisation.

497 documents, every twentieth of each source, are marked `eval`. The split has to be
written explicitly: the capture applies its own every-Nth fallback only to records that
carry no split at all, so a corpus written as entirely `train` would leave the held-out
loss measured on the old corpus while training ran on both.

### Two bugs worth recording

The first run reported 137 duplicates, including 11 from Magicoder and 1 from
self-oss-instruct. They were not duplicates. `fingerprint` hashed the first 400
characters of the normalised text, and these documents open with the chat template and
its default system prompt, so the first 400 characters are shared by every row from a
given source. The same bug made the deduplication against the existing corpus useless in
the other direction: 5,598 documents collapsed to 11 distinct keys. Hashing the whole
normalised word sequence fixes both.

The held-out split was then assigned on a global counter, one document in twenty. The
stream is a round-robin over four sources, so a stride of twenty is a stride of five
whole cycles and lands on the same source every time. Magicoder got 262 held-out
documents, tulu-3 150, self-oss 85, and **ultrachat none at all** -- a held-out set that
contained none of the one source added for multi-turn chat. Counting per source instead
of globally is immune to it, and to the cycle length changing as sources hit their
quota. The tables above are from the rerun that has both fixes.

## Why the code half is worth capturing rather than assumed harmful

The obvious objection to adding 2.5M tokens of code is that code has hurt this model
before. It is worth being precise about what was actually measured, because the two
pieces of evidence say different things and only one of them is about training.

**The calibration result does not bear on this at all.** Calibrating the conversion on
Python instead of chat costs 5.7 MMLU points, and mixing the two costs 3.7 against pure
chat at matched size. Neither arm involves a teacher: `convert_full.py` solves each
layer's projections by least squares against the *source dense model's* own keys, values
and rotary key on the calibration tokens. That result is about which input distribution
the fit sees, and it is already acted on -- the conversion this is all built from is
chat-calibrated. It says nothing about what to train on afterwards.

**The training result is confounded, in exactly the way that matters.** The 30M tokens of
code that destroyed control tokens were cross-entropy alone, because no teacher capture
exists for `scratch/code_training/tokens-v2` -- both caches on disk are the chat mixture.
And DISTILLATION.md separately establishes, on the chat corpus with the corpus held
fixed, that cross entropy alone costs MMLU and that carrying the teacher's distribution
is what stops it: the blend beats cross-entropy-only by 5.5 points,
[+0.021484, +0.089844]. Its own conclusion is that the teacher term "protects, not
recovers".

So "code hurt" and "cross entropy alone hurt" have never been separated. Nothing measured
here holds the objective fixed and varies the corpus.

Two further differences make the old result a poor guide to the new corpus. The old code
was raw source files from the-stack-v2, which DISTILLATION.md notes "carries none of this
tokenizer's protocol tokens" -- that is the other half of why it wrecked control tokens,
and it does not apply here, because Magicoder and self-oss-instruct are rendered through
the teacher's chat template like everything else. And the old run was a 30M-token
pre-training pass, not 2.5M inside a 10M mixture.

That is the argument for capturing it rather than dropping it: it converts the one
confounded piece of evidence into a clean test. The two captures are kept separate so the
mixture stays a training-time argument -- code-with-teacher against chat-only at a matched
token budget is then one run, and dropping the code half costs nothing if it loses.

## Does the prefix cap ever cut the answer off?

Both objectives score every position but the last -- `scored_mask` masks only the final
token -- so a document's system prompt and user turn are trained on exactly like its
answer. That is fine while the answer is in there. The prefix cap makes it a question:
`--teacher-max-length 1024` keeps a document's first 1024 tokens, so a document whose
framing outruns the cap is scored entirely on framing, and the model is trained to
reproduce a prompt it will never be asked to produce.

`prefix_audit.py` counts it, locating the answer by the token run for
`<|im_start|>assistant` in the ids rather than by searching decoded text.

| corpus | cap | all prompt | share | answer share of scored tokens |
| --- | --- | --- | --- | --- |
| `teacher-cache-5m` train (5,303) | 1024 | 115 | 2.17% | 60.6% |
| | 2048 | 83 | 1.57% | 64.9% |
| | 4096 | 72 | 1.36% | 65.0% |
| `teacher-cache-5m` eval (295) | 1024 | 8 | 2.71% | 61.1% |
| the 128 the training loop scores | 1024 | 3 | 2.34% | 62.8% |
| `expand.jsonl` (9,932) | 1024 | 26 | 0.26% | 61.8% |
| | 2048 | 1 | 0.01% | 64.2% |
| | 4096 | 0 | **0.00%** | 64.6% |

**It happens, and it is small.** About 2.2% of the documents and 3.5% of the scored
tokens in the corpus currently being trained on are spent on documents whose answer
never arrives. No cap that fits in 24 GiB fixes it: the prompt length before the first
answer token runs median 185, p90 456, p99 4,848, max 7,131, and that tail is the
`k3_grounded_long_context` documents, whose answers sit beyond any reachable cap.

**The new corpus does not have the problem.** Its worst prompt is 2,210 tokens against
7,131, so at a 4096 cap nothing is cut off at all, and even at 1024 it is 0.26%.

`--min-answer-tokens` drops the affected documents, and reports how many it dropped. It
defaults to zero -- keeping every document -- because turning it on changes the corpus,
and every arm measured so far ran without it. It is worth enabling for a whole
generation of runs at once, not for one arm of a comparison.

**The evaluation was already guarded.** `independent_eval.prepare` applies
`--min-assistant-tokens` to the NLL bank, with the note that at a 512-token window 21 of
384 documents were all prompt. Every one of the 32 NLL documents in `q512-bundle.json`
has assistant tokens -- minimum 15, median 262 -- and scoring is broken out by role, so
`nll content` is assistant tokens minus layout while `nll all` is the whole window. The
per-role columns were already the right ones to read.

**What is not a bug but is worth knowing.** Roughly 39% of the scored tokens are system,
user and template tokens, by design. That is ordinary full-document SFT and the teacher's
distribution over those positions is just as valid, but it does mean about two fifths of
the token budget teaches prompt reproduction -- which is a plausible part of why control
tokens came out eight times better than the source model's.

### The split field in run5m.jsonl does not match its capture

`run5m.jsonl` marks all 5,598 documents `split: "train"`, while `teacher-cache-5m` holds
5,303 train and 295 eval. The 295 are scattered through the file, not every Nth, and are
drawn from the same source mix. The cache is what training reads and its two sets are
disjoint, so the held-out number is sound -- but the JSONL cannot be used to reconstruct
that split, and recapturing from it would produce a cache with no held-out documents at
all. `expand.jsonl` carries its split explicitly for this reason.

## What this does not check

* **The teacher's framing.** The template's default system prompt asks for reasoning at
  "xhigh", and these documents' answers are ordinary SFT answers with no reasoning
  trace. The teacher is therefore being asked to score text that does not look like what
  that instruction would produce. The existing corpus has exactly the same mismatch, so
  the two are consistent with each other, which is what matters for a comparison -- but
  neither is what the teacher would generate unprompted.
* **Overlap with benchmarks we do not currently score.** The bank is MMLU and ARC. A
  later HumanEval or MBPP claim would need those banks added first, and the two code
  sources here are the ones most likely to trip it.
* **Questions under 8 words.** 1,340 scored questions are too short to screen.
