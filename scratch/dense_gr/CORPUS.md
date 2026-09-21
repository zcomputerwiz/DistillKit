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

The 133 ARC hits are the ones already known: 145 documents in the corpus carry an ARC
label, and this recovers 133 of them by text alone. The remaining 12 are the interesting
ones -- math and science documents that share a run with a scored question without
carrying a benchmark label, which no label-based filter would have caught.

This does not change any MMLU result. The 512 MMLU questions in the screen bundle were
checked individually against the corpus earlier and none of them appear in it; these 12
hits are against the other 13,530 MMLU test questions and the ARC test sets. It does
mean a resample of the MMLU bank would need screening, which is what
`q512-clean-bundle.json` and `independent_eval --decontaminate` are for.

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
