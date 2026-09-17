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

### Measured throughput

`scratch/dense_gr/benchmark.py`, one RTX 3090, sequence 1024, bf16 weights, 8-bit AdamW,
gradient checkpointing, chunked cross-entropy, real optimizer steps. Best batch per
configuration; full sweep in `scratch/dense_gr/benchmark.json`.

With the head folded into the loss loop (`--loss chunked_head`, position budget 2048), the
configuration in `benchmark-chunked-head.json`:

| configuration | params | core | best batch | tok/s | peak VRAM | 1B tokens | 3B tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d1536-L12, v248k | 778.8M | 51% | 32 | 6,484 | 78% | 42.8 h | 128.5 h |
| d1280-L18, v248k | 732.0M | 57% | 32 | 6,543 | 73% | 42.5 h | 127.4 h |
| d768-L12, v32k | 124.7M | 80% | 32 | 30,552 | 23% | 9.1 h | 27.3 h |
| d512-L8, v16k | 38.0M | 78% | 32 | 75,440 | 11% | 3.7 h | 11.0 h |

**Where the loss is computed matters more than the architecture.** Against
`chunked_causal_lm_loss`, which chunks the fp32 upcast but is still handed logits the head
has already materialized, folding the head into the loop is worth 1.41x at the 248,320
vocabulary and 1.07x at 16,384 -- the gain tracks vocabulary size, because the head is
what it removes. It also removes an inversion: with the logits materialized the 0.8B
configurations peaked at batch 2 and got *slower* by batch 8 (3,671 tok/s), where folding
the head makes them climb monotonically to batch 32 and fit at 78% of VRAM, which the
other path could not reach at all.

Depth against width at a fixed budget is worth 1% (`d1536-L12` against `d1280-L18`), so
that choice can be made on other grounds.

Per arm, multiplied by three arms at 1B tokens each: 5.4 days at 0.8B, 1.1 days at 125M,
half a day at 38M.

[Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) goes further -- it never forms the
logits at all, computing the log-sum-exp in SRAM, for `O(N + |V|)` memory instead of
`O(N|V|)`. It is not yet used here. Folding the head recovers most of the available gain
with no new dependency, and throughput is nearly flat from batch 16 to 32 (+1.9%), so the
remaining headroom is memory rather than speed.

**Measure with the allocator capped.** `torch.cuda.set_per_process_memory_fraction`, or
`max_vram_fraction` for a real run. An uncapped first pass of this benchmark reserved
35.27 GiB on a 24 GiB card and reported 1,714 tok/s against 4,491 at half the batch: on
Windows WDDM the driver pages to shared system memory rather than raising, and the result
is a PCIe bandwidth measurement that still reports 100% GPU utilisation.

**Activations bind before parameters do.** Four streams carry 4x the residual state. The
`_BranchNorm` docstring records that two branches at batch 2 x 4096 held about 670 MB per
widened layer during backward, the largest single item in the recompute working set; four
branches roughly doubles that, against 48 GB across two cards with no NCCL on Windows.
Sequence length and batch are the constrained quantities, not parameter count.

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
