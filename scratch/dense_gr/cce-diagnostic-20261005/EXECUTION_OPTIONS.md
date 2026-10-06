# Assisted-by: Codex

# Can head memory savings enable a better execution schedule?

Yes, this remains plausible and deserves a full-step test. The measured 2.373 GiB
head-phase saving is a useful lower-bound result, not yet a complete implementation
or guaranteed reduction in training's global peak. The current shared head remains
the controlled context-KL recipe.

## Quantified limits

`larger.json` measures physical GPU1 only: at 4,096 active rows the shared head's
additional forward/backward allocator peak is 3.33594 GiB; public unfiltered CCE's
weighted NLL/LSE lower bound is 0.96319 GiB. Selected teacher scores/corrections are
absent from that lower bound. It saves 2.37276 GiB and about 0.146 seconds at this
shape, before those additions. The original direct-tail prototype took 2.8
seconds forward. The newer bucketed stable full/tail forward takes 84-85 ms at
4096 rows; CPU preparation and upload are separate costs. Complete KL/UL
backward is still absent. See BUCKETED_FORWARD.md for newer measurements.

The actual r5 report (`train-2b-long-r5.json`) records peak allocated GPU0/GPU1 of
15.68715/15.55147 GiB and home-GPU reserved memory of 19.44727 GiB. Reserved memory
is allocator capacity, not the live activation count, and the report does not give
away-GPU reserved memory. Its JSON `batch=64,length=1024` are generic CLI fields;
teacher-cache batches come from the 32K micro-token plan, not those fields. The run's
443 recorded step times average 23.301 seconds and range from 10.991 to 43.381 seconds.
The context controls are about 22.8 seconds per step. Dynamic records and fresh kernel
shapes make those broad means unsuitable for attributing a 0.146-second head change.

The tied V=248320, H=2048 head contains 508,559,360 parameters. Assuming the current
BF16 weight, BF16 gradient, two 8-bit moments and BF16 Kahan compensation, it retains
8 bytes/parameter, about 3.789 GiB, plus moment scales. Those persistent bytes and
optimizer work do not disappear with CCE. The away-head placement preserves the tie
(`distillkit/parallel/model.py:103-111`). GPU0 retains the residual stream and gathered
TP outputs, so GPU1's local head saving does not by itself relieve GPU0's peak.

At 32K tokens, a BF16 two-branch H2048 residual boundary is 256 MiB. The measured
head saving nominally fits nine such boundaries, before other temporaries, fragmentation
or a completed CCE backward's state. A BF16 32K x3072 MLP shard result is 192 MiB;
twelve such results use 2.25 GiB. These are capacity estimates, not measured savings
from any checkpoint policy. New retained activations can coexist with head work,
so global peak is a maximum over phases rather than the old peak minus 2.373 GiB.

For scale, linear extrapolation of the measured direct kernel gain to 65,536 physical positions per
optimizer step and all positions active, is 16*0.146 seconds, about 2.3 seconds/step
before teacher corrections. Sparse assistant/QA objectives have fewer active head
rows. This is a scaling estimate, not an end-to-end speedup prediction; fixed overhead
and different shapes prevent treating it as an exact bound.

## Most useful execution change to test

Use saved memory for a narrow selective checkpoint policy before trying a larger
microbatch or removing all layer checkpoints. Current outer checkpoints cover every
layer (`widened.py:250-264`), while CSA2 also checkpoints query chunks. The existing
offload hook alternates large saved tensors using peer copies. Their actual
device wait time has not been established by the failed CUDA traces. Removing an
outer checkpoint changes which tensors it sees: overlapping K/V prefix views can
become independent peer copies. One 128 MiB K/V series can then accumulate around
2 GiB of prefix copies across 32 query chunks. Whole-layer uncheckpointing can consume
the proposed saving quickly and worsen scheduling.

First measure logical saved bytes and unique live storage, per device, plus copy and
recompute time. The installed PyTorch selective checkpoint API can retain chosen
expensive operations while recomputing cheaper ones. Whether the fused FLA/Triton
operations expose useful dispatcher operations must be established by a trace. Cache
the highest measured recomputation cost per retained byte that fits the actual away
device peak; do not choose solely by layer type. Reuse the existing boundary offload
path carefully, and only add asynchronous prefetch if the trace shows copy waits.

Vocabulary sharding is a larger alternative: it can spread the head's persistent
states and loss work across both cards, but the current shared-head API takes one
whole head tensor. A correct extension needs global full/omitted normalizers and
teacher-gradient reductions. It can also burden GPU0, which owns the residual stream.
Naive microbatch overlap additionally conflicts with the mutable TP bus and two live
backward graphs. Neither is a free consequence of replacing the normalizer.

Increasing micro-tokens from 32K to 64K while reducing accumulation changes the teacher
packing and can change which examples/weights share an optimizer update. It also
increases long-context body activations on GPU0. Test that only with a frozen real
accumulation cycle and matched loss weights, not as an assumed head-memory win.

## Bounded full-step experiment

Reuse `tp_bench.py`, which now supports frozen records/shapes and a profile directory,
and the actual `smoke_train.py` objective. `training_profile.py` supplies record hashes,
saved unique-storage accounting, phase scopes and CPU/CUDA traces. Avoid the old toy
CE-only `profile_step.py` as evidence for this question.

After the complete CCE objective and backward pass numerical validation, compare:

| Run | Head | Checkpoint policy | Purpose |
| --- | --- | --- | --- |
| A | Approved shared | Current full outer checkpoint | Matched baseline |
| B | Complete verified CCE | Same full checkpoint | Isolate the head change |
| C | Approved shared | Narrow measured selective policy | Baseline schedule benefit |
| D | Complete verified CCE | Same selective policy | Measure the newly usable schedule |

Use fresh processes and output/profile directories, identical checkpoint, teacher-cache
records and record hashes, source/objective weights, optimizer/LR, and accumulation
cycle. Select both a representative compact shape and a real long 32K shape. Run at
least two real optimizer warm-up steps and three measured steps per configuration;
the frozen shapes avoid first-use autograd/kernel behavior. Do not change context
sampling, teacher clamp arithmetic or packing between the configurations.

Compare complete loss/gradient checks first, then median full optimizer-step time,
per-device phase allocated/reserved peaks, unique saved activation storage, copy waits,
recompute time and shared-memory spill. A new schedule can win even if B's isolated
head is slightly slower; D must demonstrate that the changed full execution is faster
within both GPUs' actual capacity. All GPU execution of this follow-up is deferred
until the controlled context run releases the cards.
