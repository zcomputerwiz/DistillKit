# CSA2 conversion screen (2026-09-20)

The converted 2B against the model it was converted from, on the independent screen,
with the conversion and the training separated so each is answerable on its own.

Three checkpoints:

* `../../../student-2b-hf` -- the source. Dense attention, never touched.
* `scratch/dense_gr/checkpoints-2b/warmed-scaled` -- converted to MLA + CSA2, then the
  indexer warm-up. The warm-up freezes every parameter but the router, so this is the
  conversion's own cost with nothing trained on top of it.
* `scratch/dense_gr/checkpoints-2b-sparse/smoke-r1-1-gr-s0-csa2` -- the above plus the
  sparse stage: 30M tokens at lr 7.3e-6 on the `dense_gr` code/text corpus.

## Results

Token-weighted NLL in nats on 384 held-out documents, accuracy on 256 questions each.
Lower NLL is better, higher accuracy is better.

| metric | source | converted | + 30M tokens |
| --- | --- | --- | --- |
| nll all | **1.3713** | 1.4817 | 1.4464 |
| nll content | **1.7572** | 1.8978 | 1.8081 |
| nll layout | **0.5096** | 0.5228 | 0.6281 |
| nll punctuation | **0.6109** | 0.6733 | 0.6426 |
| nll control | **0.6327** | 0.7860 | 2.8020 |
| mmlu acc | **0.5977** | 0.4727 | 0.5000 |
| arc acc | 0.3906 | **0.4062** | 0.3750 |
| arc acc_token_norm | **0.4141** | 0.4102 | 0.4023 |
| arc acc_char_norm | **0.4102** | 0.3945 | 0.3867 |

Paired bootstrap, 10,000 resamples, converted-with-training against the source:

| comparison | estimate | 95% CI |
| --- | --- | --- |
| nll | +0.075134 | [+0.067536, +0.082750] |
| mmlu acc | -0.097656 | [-0.160156, -0.035156] |
| arc acc | -0.015625 | [-0.058594, +0.027344] |
| arc acc_token_norm | -0.011719 | [-0.054688, +0.031250] |
| arc acc_char_norm | -0.023438 | [-0.062500, +0.015625] |

NLL and MMLU regress by intervals that exclude zero. ARC does not move by more than its
sampling noise on 256 questions.

## What the decomposition says

The two regressions have different causes, and the middle column is what separates them.

**MMLU is the conversion.** It falls 12.5 points before a single token of language-model
training, and the 30M tokens then recover 2.7 of them. Routing is not the reason: 253 of
the 256 questions fit entirely inside the 256-token budget plus the 128-token local
window, so on almost every question the converted model reads the whole prompt and still
answers worse. What it loses is fit -- the least-squares refit onto MLA and CSA2 does not
reproduce the attention it replaced closely enough, and 30M tokens is not enough training
to close that. The direction is right, which is the useful part: more of the same recovers
it slowly, so the question is whether a better objective recovers it faster.

ARC, over the same conversion, does not move at all: 0.3906 to 0.4062 unnormalized, and
the two normalized forms within a point and a half in the other direction. So the
conversion is not losing general capability; it is losing something MMLU asks for and ARC
does not. The two differ in what they score. MMLU enumerates the four options inside the
prompt and scores the single letter that follows `Answer:`, so answering means comparing
four candidates already in context and emitting one token. ARC scores each answer text as
a continuation, which is fluency over a short span. That the delicate one breaks and the
robust one does not is consistent with a fit that reproduces attention well on average and
badly in its sharpest cases -- but it is a reading of two benchmarks, not a measurement of
the mechanism, and it should be checked before anything is built on it.

**Control tokens are the corpus.** The conversion costs them +0.153 nats, which is in
line with everything else. The sparse stage then takes them from 0.786 to 2.802, a 3.6x
blow-up, while *improving* content by 0.090 and punctuation by 0.031 over the same
tokens. The `dense_gr` corpus is code and prose and contains essentially none of this
tokenizer's protocol tokens, so nothing held their rows in place across 30M tokens. This
is ordinary catastrophic forgetting of a distribution the training data does not
represent, and layout moving the same way (+0.105) is the same effect on whitespace
conventions.

Both point at the same fix: distill the sparse stage against the source model rather than
running plain cross-entropy on a new corpus. That holds the output distribution -- control
tokens included -- and transfers capability faster than fitting a fresh corpus from
scratch, which is the other half of the MMLU gap. Mixing the source's own distribution
into the corpus would address the forgetting alone; distillation addresses both, and the
teacher is already on disk.

## Against the success criterion

The target was to match the source on these benchmarks while costing less to run. It does
not match yet. NLL is +0.075 nats and MMLU is -9.8 points, both with intervals excluding
zero; ARC is a wash. The resource side is where it was: 2.29x less cache, 3.17x faster
decode at 16K, and a length ceiling of 32768. So the trade is real and the parity is not,
and the two gaps above are what stand between them.

## Scoring, and why it changed

`independent_eval` scored a question's choices as one right-padded batch. For a dense
model that is exact: causality keeps a real query off a pad key, and the padded and
unpadded logits agree to 0.000000. A routed model is different. Its indexer scores every
position, ReLUs them, and takes a top-k; where scores tie at the cutoff, which of the tied
positions wins depends on how wide the row is, so choices of different lengths route
differently batched than alone. Measured on this checkpoint, the share of reachable cells
scoring exactly zero runs from 5.3% at layer 23 to 31.7% at layer 15, and up to 10.2% of
queries have a tie at the cutoff.

The evaluator now scores one sequence per forward for every task and every arm, so nothing
is padded anywhere. It costs four forwards per question instead of one and removes the
question rather than arguing about it.

The reference implementations share the exposure and mostly outrun it. llama.cpp does not
pad token sequences at all -- it packs sequences into a unified KV cache with per-cell
sequence ids and an `-INFINITY` mask -- and it masks before the top-k the same way we do
(`models/deepseek4.cpp`), so no pad key can be selected there either. But its sorts are no
more stable than ours: `std::partial_sort` on CPU, whose equal-key order depends on the
full range length, and a bitonic network on CUDA padded to the next power of two. Its
`n_kv` varies with cache occupancy rather than with batch padding, which is the same
problem from the other side. What saves the reference is head count: a cell is exactly
zero only when every indexer head scores at or below zero, so the tie mass falls roughly
as `2^-heads`. DeepSeek runs 64 of them. This model runs 4.

## Gotchas for a rerun

* `full-bundle-384.json` was prepared before `--min-assistant-tokens` existed, so its
  aggregate NLL is a whole-window number over rendered conversations. The class rows are
  the ones to read.
* Each evaluation process has a 540-second watchdog and `--max-seconds` cannot exceed 570,
  so tasks run one per process. At one sequence per forward, 256 MMLU questions take about
  112 seconds on the converted model and about 80 on the source.
* `checkpoints-2b/warmed-scaled` ships without tokenizer files, and the evaluator builds
  its token classes from the checkpoint's own tokenizer. Copy `tokenizer.json`,
  `tokenizer_config.json` and `chat_template.jinja` beside it first.
* `student-2b-hf` and `student-hf` have byte-identical tokenizers, so the 4B bundle scores
  the 2B without rebuilding.

## Reproduce

```powershell
foreach ($t in @('nll','mmlu','arc')) {
  .venv\Scripts\python.exe -m distillkit.independent_eval evaluate `
    --bundle scratch/independent-eval/full-bundle-384.json `
    --checkpoint ../student-2b-hf --tasks $t --device cuda:1 `
    --output "scratch/csa2-eval/stock-2b.$t.json"
}
```

Same for the converted checkpoint on `cuda:0`, then merge the three per-task files per
checkpoint the way `scratch/score_widening_full.sh` does and run `report --reference` the
source against both.
