# Two cards against one, on the 2B (2026-09-21)

The training loop runs at 1,700-2,150 tokens per second on one card and cannot fit a
micro-batch of 4. This asks whether splitting across both cards helps, and the answer is
yes -- but for the memory, not the arithmetic.

## What was in the way

`TensorParallelAttention` shards `q_proj`, `k_proj`, `v_proj` by head and reduces
`o_proj`. MLA has no `k_proj` or `v_proj`: it compresses the keys and values into a single
latent with `kv_a_proj` and expands that latent per head with `kv_b_proj`. Borrowing CSA2
layers have neither, because they read another layer's latent off the bus. So every
attention layer in this model raised `AttributeError` before the first step.

`distillkit/parallel/latent_attention.py` shards them by their own rules:

| projection | treatment |
| --- | --- |
| `q_proj` | split by head; query and gate are head-major, so a contiguous cut lands on a boundary |
| `kv_a_proj` | **replicated** -- one latent shared by every head and every borrowing layer |
| `kv_b_proj` | split by head |
| `o_proj` | split by input channel, reduced home |
| `index_*`, `indexer_proj` | **replicated** |

The router is replicated because it decides *which positions* every head reads. Split, the
two ranks could reach different top-k sets wherever the indexer's scores tie at the
cutoff -- and on this checkpoint 5.3% to 31.7% of reachable cells score exactly zero, with
up to 10.2% of queries tied at the budget. The heads would attend to different histories
and the concatenation would be incoherent. It is 5.7M parameters against the body's
1,375M, so replicating costs nothing worth counting.

Outputs are gathered rather than kept split. The CSA2 forward routes, borrows latents
across layers, optionally absorbs the up-projection into the query, and switches between a
gathered path and a FlexAttention block path on conditions computed inside it.
Re-deriving that in a sharded forward would be a second implementation of the hard part.
Sharding the parameters and gathering the result keeps one implementation and still moves
weights, gradients and optimizer state off the home card -- which is what the ceiling is
made of.

The embedding stays whole (`shard_embeddings=False`). Cut Cross-Entropy never forms the
logits, which is the only reason a 248,320-wide vocabulary is affordable; a
vocabulary-parallel loss has to form them, and CCE's `VocabParallelOptions` wants a
`torch.distributed` process group this design does not have.

## Results

1.915B parameters, length 1024, the real sparse-stage step -- forward, CCE, the indexer
KL, backward, AdamW8bit. Six timed steps after two warm-up steps, each arm in its own
process.

**Each micro-batch in its own process.** Running them in sequence contaminates the
measurement: a previous batch's fragmentation and retained state made both arms report an
OOM one step earlier than the truth, which understated each of them. The first version of
this table said one card stopped at micro-batch 2 and two cards at micro-batch 4. Both
were wrong.

| cards | micro-batch | tokens/s | ms/step | peak GiB, card 0 / 1 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1732 | 591.3 | 12.09 |
| 1 | 2 | 2151 | 952.2 | 15.79 |
| 1 | 3 | **2308** | 1330.8 | 19.45 |
| 1 | 4 | OOM | - | 21.24 |
| 2 | 1 | 1642 | 623.8 | 8.33 / 3.91 |
| 2 | 2 | 2389 | 857.3 | 10.82 / 5.00 |
| 2 | 4 | 2876 | 1424.1 | 15.69 / 7.32 |
| 2 | 6 | **3090** | 1988.5 | 20.61 / 9.64 |
| 2 | 8 | OOM | - | 21.25 / 10.40 |

Parameters resident: 1,231M on card 0, 684M on card 1.

**Best that fits against best that fits: 3,090 against 2,308, +33.9%.**

At a matched micro-batch the split is nearly free and sometimes negative: -5.2% at
micro-batch 1, +11.1% at micro-batch 2. That reproduces the prior conclusion from the
10-layer toy -- "split when the micro-batch is large; do not split a small one" -- and it
is not where the gain comes from. The gain is that 3.76 GiB leaves the home card, so
micro-batch 6 fits where micro-batch 4 did not.

**What micro-batch 8 is short of.** Not 32 MiB, which is what the error names: that is the
size of the allocation that happened to fail, not the deficit. Raising the allocator's
share of the card from 0.9 to 0.94 -- 21.60 GiB to 22.56 -- lets it climb to 22.40 GiB and
fail there instead. The real requirement is about a gigabyte more than micro-batch 6, and
spending the margin to chase it would trade a measured model for a measurement of the
PCIe bus, because Windows serves VRAM out of system RAM rather than failing.

## Balancing the two cards, and why the obvious lever loses

Card 0 peaks at 20.61 GiB and card 1 at 9.64. Eleven gigabytes sit unused next to a card
that is a gigabyte short, so it is worth asking what moves.

Parameters are the smaller half of the answer. The split is 1,231M against 684M, which at
six bytes each accounts for 3.3 GiB of the 11 GiB gap. The rest is activations, and they
are on home because the sharded projections gather their outputs there:

| stage, micro-batch 6 | card 0 | card 1 |
| --- | --- | --- |
| model resident | 2.29 | 1.27 |
| after forward | 17.18 | 8.26 |
| after the indexer loss | 18.00 | 8.26 |
| after backward | 5.06 | 2.57 |
| after the optimizer step | 7.39 | 3.87 |

Moving the tied embedding and head to card 1 balances the parameters and loses:

| arrangement | resident | tokens/s | peak GiB |
| --- | --- | --- | --- |
| embedding home | 1,231M / 684M | **3090** | 20.61 / 9.64 |
| embedding on card 1 | 722M / 1,193M | 1460 | 18.65 / 12.50 |

It does what it was supposed to -- 1.96 GiB leaves home and the parameter split inverts --
and costs 2.1x the step time, and micro-batch 8 still does not fit. The wrapper is not the
problem: pointing the same code at home, where its moves become no-ops, gives 3,091 tokens
per second against the unwrapped 3,090. The cost is the two cross-device synchronisations
it introduces, at the first operation of the forward and the last before backward, on a
graph whose other traffic was already overlapping.

So the parameters can be balanced and it is not worth doing. The imbalance that matters is
the activations, and moving those means not gathering the sharded outputs -- the split
forward this design deliberately avoided, which would require a second implementation of
the CSA2 routing. That is the honest next lever, and it is a large one.

## Caveats

* The gather means home still holds each projection's full output, so activation memory is
  unchanged by the split; only parameter-resident memory moves. A split forward would take
  the activations too, at the cost of a second implementation of the CSA2 routing.
* `q_proj` pays a weight gather on every forward, because `_project` fuses the query, the
  latent and the index queries into one multiply and reads `.weight` to do it. 16 MiB per
  full-attention layer against a step of hundreds of milliseconds.
* The autotuner warm-up is per *shape*, not per run. A new micro-batch is a new shape, and
  a cold autotune under sharding puts two backward threads through one Triton autotuner,
  one clearing `nargs` while the other benchmarks. `tp_bench.py` warms each batch size.
* These are throughput measurements on random tokens. No trained-quality claim is made,
  and the sharded path has not run a full training job yet.

## Does this make more training viable?

It makes it a third cheaper, which is worth having. It does not change what the
conversion ladder said: the residual MMLU gap sits in the least-squares refit rather than
in training volume, a latent of 768 fits *worse* than 512 with twice the room, and 10M
tokens of distillation moved MMLU by +2.5 points with the interval spanning zero. A 1.34x
throughput gain buys about a third more tokens per hour against a curve that was already
flat. Worth using for every future run; not on its own a reason to expect the gap to close.

## Reproduce

```powershell
.venv\Scripts\python.exe scratch\dense_gr\tp_bench.py --cards 1 --batches 1 2 4 --output scratch/dense_gr/tp-bench-1card.json
.venv\Scripts\python.exe scratch\dense_gr\tp_bench.py --cards 2 --batches 1 2 4 6 --output scratch/dense_gr/tp-bench-2card.json
```

Run them separately, with both cards otherwise idle. `CUDA_VISIBLE_DEVICES=0` for the
one-card arm; leave it unset for two.
