# Dense GR: the reference substrate

The architecture the standard-parts work attaches to. It is not under test. Nothing here
is compared against a single-stream or MoE alternative, and no result in this program
should be read as evidence about the substrate itself -- it is held fixed so that
everything else can vary.

## What it is

A dense decoder-only transformer with a four-stream gated residual.

Every sublayer reads a gated mean over four persistent residual streams, runs the ordinary
attention or FFN block at full `d_model`, and writes back to all four streams through
learned per-stream multipliers. Attention interleaves linear and full attention. Every
layer has one FFN at the full intermediate width and every token uses all of it.

Structurally it is Flash-Next with the mixture-of-experts removed. Worth being precise
about what that removal is, because the narrowing is the point rather than the sparsity:

| | per-expert FFN width | experts | active per token |
| --- | ---: | ---: | ---: |
| `qwen3_5_moe` defaults | 512 | 256 (+1 shared) | 8 of 256 |
| dense equivalent | 4304 | -- | all |

An MoE expert is roughly 8.4x narrower than the dense FFN it replaces; MoE holds active
FLOPs near dense while multiplying stored parameters. "Full width all the way through"
means the FFN width, and it means no router, no top-k, no shared expert and no per-expert
narrowing.

The read/write machinery is Hyper-Connections (Zhu et al., arXiv 2409.19606); the shipped
variant is Qwen's Gated Residual, technical report section 2.2, equations 30--34. Lead
with the structural description rather than the lineage: naming it after Flash-Next
invites readers to assume Qwen's PLE results carry over, which is exactly the factorial
`docs/standard_parts.md` records as unestablished.

### Two widths, and they are separable

- **Compute width** `d_model`. What attention and the FFN operate on. GR does not change
  it: the read collapses four streams to one `d`-wide vector, the block runs at `d`, and
  the result is written back to all four.
- **Residual width** `n * d_model`. The persistent state carried between sublayers, and
  4x the activation memory at `n = 4`.

Keeping these apart is the whole premise of the native-capacity track. "Dense" negates
MoE; it does not mean single-stream.

## Implementation

Already present. `Qwen35WidenedForCausalLM` with

```json
{
  "residual_stream_enabled": true,
  "residual_stream_routing": "flash_next",
  "residual_stream_num_branches": 4,
  "residual_stream_lowrank": "<r>",
  "residual_stream_sidecar": false,
  "mlp_only_layers": []
}
```

on a `Qwen3_5TextConfig`. The 2B student is already dense hybrid-attention with a single
stream; the widened wrapper adds the other three. From-scratch is the same class with
random initialization instead of `from_pretrained` followed by `recipient_initialize`.

Staying inside this config family is most of the argument for the choice: the 248,320
vocabulary, token classes, `independent_eval`, the evaluation bundles, chunked CE and the
collators all work unchanged, and the module ABI declared in `docs/standard_parts.md`
stays honest.

### From-scratch deltas

Two initializations in the tree exist to preserve a pretrained backbone's behavior. Both
are wrong here, for the same reason, and both are the zero-multiplier pattern that
`docs/standard_parts.md` argues against -- a channel that starts inert has to be talked
into mattering.

**`_WidenedWeightInit` zeroes `branch_gain_delta`.** Correct for identity-preserving
conversion, wrong for scratch: initialize the gains at the norm's ordinary initialization
instead.

**`native_ple.py` puts the PLE block behind an admission scalar**, `h' = h + rho *
PLEWrite(h, E[n])` with `rho_0 = 0`, so the block is dormant at load. Flash-Next trains
PLE jointly from step 0 and there is no pretrained behavior to preserve here, so drop the
outer scalar and keep everything inside the block at reference initialization. The file's
own reasoning -- that a zero inside a product is a place gradients cannot leave -- applies
to its own outer scalar once the reason for it is gone.

## PLE geometry

Use the existing implementation rather than reimplementing it.

- `distillkit/experimental/ngram_hash.py` -- the hasher, geometry from the Flash-Next
  config: `ngram_size=3`, `heads_per_ngram=8` giving 16 heads, bigrams first, seed 1234,
  one prime address space per head above `ngram_vocab_size_base`, running global offsets,
  EOS-aware history reset.
- `distillkit/experimental/native_ple.py` -- the model learns its own table jointly.
  `ple_embed_dim` follows hidden size, each of the 16 heads contributes `d / 16`
  dimensions, and the concatenation lands at stream width with no projection. **This is
  the one to use.**
- `distillkit/experimental/ngram_table.py` -- serves the donor's frozen 51.2-billion-element
  table from a GGUF. Not for from-scratch work.

### Sizing

The head count cancels, since each of the 16 heads holds `d / 16` dimensions:

```
table parameters  ~=  ngram_vocab_size_base * d_model
```

Checked against the donor: 20,000,000 * 2560 = 51.2e9.

The donor default is unusable below its own scale. At `d_model = 2048`:

| `ngram_vocab_size_base` | table parameters |
| ---: | ---: |
| 20,000,000 (donor) | 41e9 |
| 1,000,000 | 2.0e9 |
| 100,000 | 205e6 |
| 10,000 | 20e6 |

This is the first number to fix in any configuration, and it should be chosen rather than
inherited.

## Budget

Three formulas cover the levers. At `n = 4` and `r = d/8`, the GR read/write machinery
costs `d^2` per sublayer:

```
GR routing        ~=  2 * L * d^2
PLE table         ~=  ngram_vocab_size_base * d
residual activation  =  4x single-stream
```

At `d = 2048`, `L = 24`: routing is about 201e6 parameters, and a table at
`base = 100,000` is about 205e6 -- comparable, and a legible split between dense backbone,
routing and memory.

Note `r = 320` is inherited from Flash-Next at `d = 2560`, where it is `d/8`. Fix the
ratio, not the constant.

## Measured configuration

Everything below is measured on one RTX 3090 at sequence 1024, bf16 weights, 8-bit AdamW,
random inputs and real optimizer steps: `scratch/dense_gr/benchmark.py`, with the full
sweep in `benchmark-full-stack.json` and the intermediate states in `benchmark.json`,
`benchmark-chunked-head.json` and `benchmark-cce.json`.

### What to run

```
Cut Cross-Entropy, filter_eps="auto"      the loss never forms the logits
Flash-Attention 2                         free, though only worth 1%
Liger fused SwiGLU and RMSNorm            not for anything needing bitwise agreement
gradient checkpointing OFF                at 125M and below
torch.compile                             off
batch 32 to 64                            above the launch-bound crossover
```

| configuration | params | core | batch | tok/s | peak VRAM | 1B tokens | 3B tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d1536-L12, v248k | 778.8M | 51% | 16 | 11,651 | 66% | 23.8 h | 71.5 h |
| d1280-L18, v248k | 732.0M | 57% | 16 | 11,796 | 79% | 23.5 h | 70.6 h |
| d768-L12, v32k | 124.7M | 80% | 32 | 50,664 | 58% | 5.5 h | 16.4 h |
| d512-L8, v16k | 38.0M | 78% | 64 | 132,762 | 49% | 2.1 h | 6.3 h |

Three arms at 3B tokens each: 8.9 days at 0.8B, 2.05 days at 125M, 0.79 days at 38M.

### What each change was worth

Measured separately, because crediting a stack to whichever piece was added last is how a
1% change gets mistaken for the reason something got faster.

| change | gain at 0.8B, v248k |
| --- | ---: |
| chunked head over materialized logits | 1.41x |
| CCE over chunked head | 1.45x |
| Liger fused SwiGLU and RMSNorm | 1.20x |
| gradient checkpointing off | 1.17x at 125M, 1.21x at 38M, 1.07x at 0.8B |
| Flash-Attention 2 over SDPA | 1.01x |
| `expandable_segments` | 1.005x |
| `torch.compile` on the decoder | 1.00x, and +1.2 GiB reserved |

Total against the materialized-logits path: **2.58x** at the 248,320 vocabulary and
**1.89x** at 16,384. The gain tracks vocabulary size throughout, because the head is what
most of it removes.

Depth against width at a fixed budget is worth 1% (`d1536-L12` against `d1280-L18`), so
that choice can be made on other grounds.

### Where a step goes

`scratch/dense_gr/profile_step.py`, at 125M and batch 32:

| group | ms/step | share | launches |
| --- | ---: | ---: | ---: |
| matmul | 339.98 | 44.7% | 272 |
| elementwise | 211.42 | 27.8% | 1,646 |
| loss (cce) | 109.58 | 14.4% | 2 |
| other | 36.25 | 4.8% | 444 |
| attention | 28.71 | 3.8% | 24 |
| linear attention | 13.04 | 1.7% | 66 |
| optimizer | 9.29 | 1.2% | 189 |

Two things this settles. **The four-stream route is not where the time is**: the same
profile at one branch costs 725.44 ms against 761.19, so the whole GR route is 4.7% of a
step and its per-branch Python loops are not worth fusing. And **the step is already 97%
GPU-busy** -- 761 ms of device time against 781 ms of wall clock -- so there is at most 3%
of launch gap at this batch size.

### Measured and rejected

**`torch.compile`** produces 58 graph breaks, so dynamo never gets a contiguous graph and
is 17% *slower* with checkpointing off. One break is ours: `Tensor.item()` from
`alpha = float(self.blend)` in `HyperConnection.read`, which is deliberate -- the CPU pin
avoids a device synchronisation on each of the model's sublayers, and is right for eager
and poison for dynamo. The rest are spread across transformers, fla and the custom
autograd Functions. Chasing them to attack a bucket whose largest identified contributor
is 4.7% is not a good trade.

**CUDA graphs** are worth 1.66x at batch 4, 1.05x at batch 8, and *cost* 5% at batch 32
(`cuda-graphs.json`). The crossover sits between batch 8 and 16, which is exactly where
the sublinearity put it -- the 38M model takes 1.36x the time for twice the work from
batch 8 to 16, and 1.90x from 16 to 32. Above the crossover, replay has its own fixed cost
and no launch gap to amortise it against.

A graphed step is unavailable regardless, and `graph_capture_probe.py` isolates why by
running each component in its own process, since a failed capture invalidates the CUDA
context and one traceback from a combined run names nothing. Plain SwiGLU captures. The
decoder captures under sdpa, including fla's gated delta rule, `causal_conv1d` and the GR
route. Flash-Attention 2 does not. Eager attention does not, and says why: *"Cannot copy
between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned"*.
CCE does not. So the choice is graphs or CCE, and CCE is worth 1.45x against graphs
costing 5% here.

Keep the crossover rule even though the answer is no: **launch-bound below batch ~12,
compute-bound above.** A configuration forced to small batch by size or sequence length
would find graphs a 1.66x lever, at the price of sdpa and no CCE.

### Two ways to measure this wrong

**Spilling.** On Windows WDDM an over-budget run does not raise: the driver pages CUDA
allocations to shared system memory and serves them over PCIe, at 100% reported GPU
utilisation. The first pass of this benchmark reserved 35.27 GiB on a 24 GiB card and
reported 1,714 tok/s against 4,491 at half the batch, with performance counters showing
23.74 GiB dedicated alongside 12.22 GiB shared. `nvidia-smi` does not report shared memory
and on this machine mirrors GPU 0's statistics onto GPU 1, so it cannot detect this at
all. The benchmark now caps the allocator with `set_per_process_memory_fraction`, samples
the WDDM counter before any model is built, and discards any run whose shared usage climbs
more than 0.25 GiB above that baseline. `max_vram_fraction` does the same for training
runs.

**Budgeting the head chunk per sequence.** The position budget counts *total* positions,
so at a budget of 4096 with batch 4 over a 1024-token sequence the chunk is the whole
sequence and nothing is chunked -- 14.82 GiB against 9.87 at a budget of 2048. The
"chunked" head was silently not chunking.

### Liger's numerical boundary

`LigerRMSNorm(offset=1.0, casting_mode="gemma")` is exactly Qwen3.5's convention: a stored
deviation applied as `1 + weight`, multiplied in fp32 before casting back. In isolation
and in fp32 it agrees with `Qwen3_5RMSNorm` to 1.6e-7, one ulp, which is what establishes
the convention is right rather than merely plausible -- the same check that would have
caught the `2 * weight` trap in the GR conversion. In bf16 that becomes 1.6e-4 and
accumulates over twelve layers to 1.5e-2 on the logits, with 98.05% argmax agreement
(`liger-equivalence.json`).

For training from scratch that is a different but equally valid trajectory. It is not
acceptable anywhere a bitwise gate is in play, and the swap replaces the module class, so
a Liger-patched model no longer satisfies `recipient_initialize`'s `Qwen3_5RMSNorm` type
check. That refusal exists to catch a guessed gain convention and it correctly catches
this one.

RoPE is not swapped: Qwen3.5 uses interleaved mRoPE with a partial rotary factor against
Liger's standard formulation, and matching them is rework rather than a swap. The GR
branch norms are left alone because they are a custom autograd Function written to avoid
holding fp32 copies until backward, and the route is 4.7% of a step.

### CCE gradient filtering

`linear_cross_entropy` defaults to `filter_eps="auto"`, which skips vocabulary entries
whose contribution falls below a dtype-derived threshold in the backward pass. Measured at
4,096 tokens, hidden 1,536, vocabulary 248,320 (`cce-variants.json`):

| variant | ms | peak GiB | dW zeroed |
| --- | ---: | ---: | ---: |
| reference fp32 | 263.4 | 11.367 | 0% |
| cce, `filter_eps="auto"` | 127.3 | 0.724 | 11.13% |
| cce, `filter_eps=None` | 283.5 | 0.722 | 0.03% |
| `impl="cce_exact"` | 429.1 | 2.155 | 0% |
| `impl="torch_compile"` | 187.1 | 2.618 | 0% |

The loss is unchanged to bf16 rounding in every variant, and the ~1e-3 relative gradient
differences are precision rather than approximation -- `cce_exact` shows them too.

**Memory is flat between `auto` and `None`**, so the filtering buys time (2.2x) and
nothing else; exact gradients cost time, not memory, and `filter_eps=None` is the route
rather than `impl="cce_exact"`, which is worse on both axes. And **`impl="torch_compile"`
is not a cheaper CCE** -- it is the fallback for systems without Triton, it materializes
the logits, and its good timing comes from doing the expensive thing efficiently.

`auto` is kept. What it drops is the small gradient pushing away from tokens the model
already assigns near-zero probability, the 11.13% was measured on random weights (the
near-uniform regime that is worst case for a threshold), and a failure would show up as
rare classes not improving in the per-class evaluation breakdown.

### Installing CCE and Liger on Windows

Both declare `triton>=3.0.0` while the platform package is `triton-windows`, so both need
`uv pip install --no-deps` against the Triton already present. CCE additionally reads
`importlib.metadata.version("triton")` inside `is_triton_3_2` at *run* time rather than
import time, so it imports cleanly and then raises `PackageNotFoundError` on the first
step; the harness resolves the alias rather than hard-coding a version, which leaves CCE's
own comparison intact.

FLA and `causal_conv1d` need no action. `install_device_aware_linear_attention` reports
patching nothing, which reads like a failure and is not: this version of transformers
binds the fused implementations directly, and `fla.ops.gated_delta_rule.backends.flash_qla`
loads during the forward. That dispatch exists for CPU fallback.

### Sequence length and the sortish sampler

`distillkit/core/sortish_sampler.py` groups by length to save padding, after finding that
HF's sampler cost 0.0145 of `eval_loss` through ordering alone. It does not apply to
pretraining here: slicing fixed 1024-token windows out of a packed token store produces no
padding for it to save. It stays relevant to the SFT and distillation paths, where
documents arrive at their own lengths.

## One interaction with the first experiment

The substrate contains an n-gram memory, and that competes with two of the candidate
standard parts.

The first experiment asks whether a supplied exact copy module stops the backbone
rebuilding induction. On a PLE-equipped backbone there are three routes to that function:
attention heads, the MLP, and a context-hashed table that is already most of an n-gram
predictor. If the ablated copy probe holds up, the result would not distinguish a backbone
that kept building induction heads from one leaning on PLE.

**Run the first experiment with PLE disabled**, and use the full reference from the second
experiment onward. Same architecture, one switch. The same caution applies to the n-gram
prior in the candidate parts list, which is partly redundant with PLE by construction.

## Fixed versus free

Fixed, because the tooling and the ABI depend on them: the `Qwen3_5TextConfig` family, the
248,320 vocabulary, RMSNorm storing a deviation and applying `1 + weight` at eps 1e-6,
four streams, dense SwiGLU everywhere, hybrid linear/full attention.

Free, and to be decided before anything runs: `d_model`, depth, the linear-to-full
attention ratio, `r` as a ratio of `d`, `ngram_vocab_size_base`, whether the final
collapse is a mean or a learned mixer (Qwen uses a learned mixer; this repository has
deliberately used a mean collapse), and whether PLE is present in a given run.


### Smoke test on real tokens

`scratch/dense_gr/smoke_train.py` runs the smallest configuration on the actual Python
token store with the recommended stack, which exercises what random ids cannot: the
vocabulary remap, the data path, and whether the loss falls.

It works. The remap round-trips at every vocabulary size tested, loss falls from 9.78 to
5.81 over 2M tokens at 16,384, throughput reproduces the benchmark, and nothing spilled.

It also corrected a cost that had only been thought of as a quality issue. A token below
the cut becomes two to four byte tokens, so a small vocabulary does not merely lose
fidelity -- it *inflates the corpus*. Measured over the same 30,760,040-token store:

| vocab | coverage | expanded | stream tokens | tok/s | s / corpus pass | params |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | 93.641% | 6.359% | 40,652,874 | 126,812 | 320.6 | 38.0M |
| 32,768 | 97.628% | 2.372% | 34,596,431 | 119,974 | **288.4** | 46.3M |
| 65,536 | 99.596% | 0.404% | 31,489,910 | 97,975 | 321.4 | 63.1M |
| 248,320 | 100.000% | 0% | 30,760,040 | 59,857 | 513.9 | 156.7M |

**Seconds per corpus pass is the metric, not tokens per second**, because each vocabulary
turns the same text into a different number of tokens. On that measure 32,768 wins, and
16,384 and 65,536 tie for opposite reasons -- one processes 32% more tokens quickly, the
other fewer tokens slowly. The full vocabulary is 1.78x slower per unit of text than
32,768.

**The loss column is deliberately absent, and the per-run losses must not be compared
across vocabularies.** A 16,384-way softmax has a lower entropy floor than a 248,320-way
one and starts near `log(V)`, so at 30 steps a larger vocabulary is merely further from
convergence. Choosing on those numbers would pick the smallest vocabulary every time for
no reason but arithmetic. Quality has to come from the baseline qualification run at a
real budget.

The 32,768 figure is measured on 30.7M tokens of Python. Recompute the ranking on the full
corpus before freezing it: the number of distinct ids in use grows with corpus size, so
the expansion rates above are a lower bound.

### Vocabulary at a real budget

The smoke test ranked vocabularies on throughput alone. This trains each one for **three
passes over the same corpus** -- equal text rather than equal tokens, which is the only
fair budget when a cut changes how many tokens the text becomes -- and scores held-out
loss on the calibration split, which shares no repository with train. Loss is reported in
**nats per original token**, so a cut is charged for its own expansion instead of being
rewarded with an easier softmax. `sweep-v*.json`.

| vocab | params | inflation | scored | held-out | normalized | tok/s | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | 38.0M | 1.3216 | 121,896,960 | 2.0280 | 2.6802 | 135,123 | 902 |
| **32,768** | 46.3M | 1.1247 | 103,743,488 | 2.2953 | **2.5816** | 116,837 | **888** |
| 65,536 | 63.1M | 1.0237 | 94,437,376 | 2.7490 | 2.8143 | 100,397 | 941 |
| 248,320 | 156.7M | 1.0000 | 92,274,688 | 3.0122 | 3.0122 | 54,249 | 1,701 |

**32,768 wins on both axes**, best normalized loss and fastest wall clock, so there is no
trade-off to adjudicate. It also confirms the throughput-only prediction independently:
288.4 s per pass predicted 865 s for three, against 888 measured.

The curve is **not** monotonic and a partial run says otherwise. At half a pass the
ordering was 16,384 < 65,536 < 248,320 and looked like "smaller is better"; by three
passes 16,384 has fallen behind 32,768 by 0.0986. Reading a vocabulary comparison off an
early checkpoint gives the wrong answer.

**Vocabulary is confounded with model size here, and larger cuts are penalized twice.**

| vocab | tokens/parameter | train | held-out | gap |
| ---: | ---: | ---: | ---: | ---: |
| 16,384 | 3.21 | 1.9154 | 2.0280 | +0.1126 |
| 32,768 | 2.24 | 2.3060 | 2.2953 | -0.0107 |
| 65,536 | 1.50 | 2.6390 | 2.7490 | +0.1100 |
| 248,320 | 0.59 | 2.9221 | 3.0122 | +0.0901 |

A bigger vocabulary adds embedding parameters *and* yields fewer tokens from the same
text, so tokens per parameter falls 5.4x across the sweep and every arm is far below
compute-optimal. The 248,320 result is therefore partly "this model is undertrained"
rather than purely "this vocabulary is worse" -- the same confound that made the Phase 1
parameter-matched arm uninformative. Separating them needs either equal parameter counts
at different vocabularies, or a budget where all arms converge.

So 32,768 is the right choice *at this scale and budget*, which is what the copy
experiment will run at. It is not established as the right choice in general, and the
ranking should be recomputed on the 3B corpus before it is treated as settled.

## Sidecar on code, at tiny-model scale

The structural sidecar was run here in both regimes and both are null. Jointly trained from
scratch it moves held-out by 0.0003 nats, a fifth of the seed spread, while the content
delta climbs unconstrained to +0.035. Fitted post hoc to the frozen `plain` checkpoint it
gives a token-weighted structural aggregate of +0.00016 with content -0.00052.

**The comparator is `0d075d7`, not `ab68bef`.** `ab68bef` was general text; this is code.
`0d075d7` refitted the same module to Python against a frozen code-trained backbone at full
scale and got +0.0075 aggregate with newline still regressing, verdict PARTIAL STRUCTURAL
TRANSFER, and recorded why: "B_code's structural NLLs are already low enough that a
hash-addressed bias has little to add beyond noise." This result has the same sign and is
about forty-seven times smaller.

So the finding is: the sidecar again fails to add value on code, now at tiny-model scale,
consistent with what was already on record. It does not separate model scale from vocabulary
reduction from code-domain saturation, and it does not test the general-text result at all.

Two checks that the port reads the corpus correctly rather than differently. Control is 0.26%
of structural targets, exactly the proportion `0d075d7` reports for one EOS per document. And
the strength selector here is token-weighted, so it avoids the defect `0d075d7` found in the
canonical one, where control at 0.26% of targets carried the same weight as punctuation at
60.32% and the selector improved monotonically while the token-weighted NLL worsened.
