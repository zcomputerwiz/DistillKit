# Identity-preserving residual widening

Enable the two-branch retrofit by adding this to a DistillationRunConfig YAML:

```yaml
residual_stream:
  num_branches: 2
  lowrank: 64
```

Omitting `residual_stream` keeps the existing model class and single-stream behavior.
The extension works with a plain Qwen3.5 text student or the existing GR/PLE sidecar.
The sidecar implementations and independent evaluator are unchanged.

## Architecture

`distillkit/models/qwen35_widened.py` carries `[batch, tokens, branches, hidden]`
through every decoder layer. Each attention and MLP block has a separate
`WidenedResidual` from `distillkit/widened_residual.py`. The static read selects
`layer_index % num_branches`; the static write adds the block output to every
branch. The residual connection is identity, with no learned branch-mixing matrix.
Sigmoid read gates are per branch/channel and write gates per branch. Their
learnable scales start at zero. Random dynamic projections let the scales receive
gradients immediately, followed by projection gradients once the scales move.

Both static routes are stored as **offsets** from the identity route, `read_offset`
and `write_offset`, rather than as the route itself. This is not cosmetic: BF16
spacing near 1.0 is 0.0078, so an all-ones static write cannot record the ~1e-5
steps this project trains with. The first version stored it at 1.0 and three real
trainer steps left `write_deviation` at exactly 0 while `lambda_write`, which starts
at zero, had reached 2.7e-5. With offsets the same three steps move it to 3.9e-5.
`tests/test_widened_residual.py::test_identity_routes_are_stored_where_bfloat16_is_dense`
pins this for every non-projection routing parameter.

This is the identity-initialized HC form of report section 2.2, with the finer read
granularity of GR. It deliberately retains static terms. Copying the upstream
GR-from-scratch module would halve the normalized input at its zero gate logits.
This retrofit uses the requested rank 64, rather than the report's `d/8` rank.

Each branch is RMS-normalized once using the pretrained layernorm weights under
their original checkpoint names. Zero-initialized per-branch gain offsets multiply
that result by `1 + offset`, allowing independent gains without resetting trained
weights or introducing another RMS normalization. The initial read/write arithmetic
preserves the original computation exactly. The final head and hidden-state losses
receive a branch mean expressed around branch zero, which preserves equal-branch
BF16 values exactly. `AnchorTap` uses the same readout for intermediate anchors.

Routing and the residual branches remain on the TP home card. The existing sharded
attention/MLP modules receive the mixed `d`-wide input and return `d`-wide output;
the number of residual branches therefore does not multiply these TP transfers.

Because the stream lives entirely on the home card, widening charges the whole
retrofit to card 0 while card 1 sits several gigabytes below it, and card 0 has no
room to spare. `offload_stream_boundaries` parks every second checkpoint boundary on
the peer card with `torch.autograd.graph.saved_tensors_hooks`. Under non-reentrant
checkpointing the only large tensors those hooks see in the layer loop are the
per-layer stream inputs -- recompute temporaries never leave the checkpoint frame --
so alternating them splits the stream evenly. It is active only while training with
gradient checkpointing under TP; single-card and inference runs are untouched.

New routing parameters use the architecture/AdamW optimizer group and remain
trainable when the backbone is frozen. Architecture metrics include routing scales
and static read/write offsets.

## Checkpoints

Portable TP export preserves the original tensor names and all new routing tensors.
Load a widened checkpoint explicitly with:

```python
from distillkit.models import Qwen35WidenedForCausalLM
model = Qwen35WidenedForCausalLM.from_pretrained(checkpoint_path)
```

For the training entry point, retain the matching `residual_stream` and sidecar
configuration. The loader rejects missing or mismatched architecture settings for
a widened checkpoint, avoiding silently discarded or reinitialized routes. A stock
Transformers model class is not the loader for a trained widened checkpoint.

The existing independent-evaluation CLI does not yet select this model class; its
strict unexpected-key check rejects widened checkpoints. No quality claim is made
from the smoke losses. Future quality comparisons must use the independent text
and benchmark scoring with the correct widened loader, rather than distillation
loss or a stock class that omits routing.

## Verification commands

Run from the DistillKit repository root. Every probe has a nine-minute watchdog;
the trainer probe sets `max_steps=3`, disables external reporting and intermediate
evaluation/saves, then exports its final checkpoint beneath the result directory.

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.8"
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe scratch/widened_residual_probe.py identity --output scratch/widened-residual/identity-single.json
.venv/Scripts/python.exe scratch/widened_residual_probe.py identity --tp --output scratch/widened-residual/identity-tp.json
.venv/Scripts/python.exe scratch/widened_residual_probe.py performance --branches 1 --output scratch/widened-residual/baseline.json
.venv/Scripts/python.exe scratch/widened_residual_probe.py performance --branches 2 --output scratch/widened-residual/widened.json
.venv/Scripts/python.exe scratch/widened_residual_probe.py trainer --output scratch/widened-residual/trainer.json
.venv/Scripts/python.exe scratch/widened_residual_probe.py reload --output scratch/widened-residual/reload.json
```

The performance probe uses the real 4B student, TP, non-reentrant checkpointing,
AdamW8bit updates, folded-head KL and hidden-state cosine. It uses synthetic input
and teacher tensors to make the two arms comparable, so its memory/timing includes
optimizer state but not corpus collation or real table fetching. The separate
trainer smoke uses the real cache and sidecar collator.

The automated tests cover exact FP32/BF16 pretrained logits and hidden states,
nonzero trained norm gains, trained GR/PLE sidecars, persistent divergent branches,
gradient wakeup, checkpoint replay, intermediate-anchor gradients, incremental
decode, optimizer grouping, config guards, and exact trained tensor round trips
through direct save/load and TP export.

## Measured results

The real-student identity probes passed on one GPU and under two-GPU TP: eight
output positions across the complete 248,320-token vocabulary have maximum absolute
logit difference **0.0**. All 32 layer outputs were checked for two persistent,
identical branches at initialization. See `scratch/widened-residual/identity-*.json`.

### Making batch 4 x 4096 fit

The first attempt at batch 4 x 4096 completed one update and ran out of memory in the
second backward, with card 0 holding 19.31 GiB allocated and 3.37 GiB reserved but
unallocated under the 22.8 GiB cap
(`scratch/widened-residual/initial-memory-failure.json`). The final failure was 90 MiB
short of fitting, so the fix had to buy real headroom rather than close the gap.

Windows does not support `expandable_segments` -- torch 2.11 warns
`expandable_segments not supported on this platform` and ignores it -- so the
fragmentation had to be removed rather than absorbed. What was tried, in order:

| change | card 0 allocated | card 0 reserved | outcome |
| --- | ---: | ---: | --- |
| `max_split_size_mb:256` | 19.51 | > 22.80 | still OOM |
| `empty_cache()` between updates | 19.50 | > 22.80 | still OOM; the pool fragments *within* one backward, so returning it between updates buys nothing |
| fewer, more uniform temporaries in read/write | 19.42 | > 22.80 | still OOM, and only ~0.1 GiB off allocated |
| alternate boundaries on the peer card | 19.64 | **20.97** | runs |

The transient cuts stayed in -- dropping a redundant `.contiguous()` per branch,
applying `1/n` to the narrow projection outputs instead of a second `[B,T,n,d]` copy,
in-place `sigmoid_`, and `addcmul` in the write -- because they are free and remove
about 1.3 GiB of churn per layer. But they were not what fixed it. Peer offload cut
card 0's reserved-but-unallocated pool from 3.32 GiB to 1.34 GiB, and that is what
brought reserved under the cap.

Offloading *every* boundary rather than every second one was also measured: card 0
went to 19.56 GiB (0.08 lower) while card 1 rose from 14.21 to 15.86. Card 0's peak
is not in the layer loop, so the extra traffic buys nothing; `every=2` is the setting.

### Measured cost of the widening

Batch 4 x 4096, TP across both cards, non-reentrant checkpointing, AdamW8bit:

| | seconds/update | card 0 peak / reserved | card 1 peak / reserved |
| --- | ---: | --- | --- |
| `num_branches: 1` | 9.80 | 16.19 / 18.41 GiB | 13.87 / 15.48 GiB |
| `num_branches: 2` | 12.75 | 19.64 / 20.97 GiB | 14.21 / 16.48 GiB |

About +30% per update, of which the peer offload is roughly 0.19 s: the every-boundary
arm moved 2.6 GiB more per update for 0.19 s more wall clock. The rest is the widening
itself, which is bandwidth-bound on `[B,T,n,d]` elementwise work rather than FLOP-bound
-- the routing projections come to about 4 TFLOP per update against roughly 140 TFLOP/s
of bf16 across the pair.

The real trainer smoke, on the actual cache with the sidecar collator, is quieter still
at 12.6-13.2 s/step with card 0 at 17.40 / 18.58 GiB and card 1 at 13.57 / 16.76 GiB.

### Checkpoint round trip

`reload` reloaded the smoke checkpoint through `Qwen35WidenedForCausalLM`, compared all
**512** routing tensors against the exported safetensors bit for bit, and confirmed all
**128** lambdas were nonzero, so the trainer had actually moved the routing before the
export. See `scratch/widened-residual/reload.json`.
