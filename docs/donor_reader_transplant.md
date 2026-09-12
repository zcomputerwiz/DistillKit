# Donor-reader transplant probe

This experiment treats Flash-Next's PLE reader as a frozen feature extractor. It does
not instantiate hyper-connections, MoE experts, a key/query gate, or the direct value
residual. Every arm injects at decoder layer index 1 and computes:

```text
table rows -> frozen value_proj -> frozen RMS + dilated depthwise conv banks
           -> stream collapse -> rho -> ordinary 4B residual
```

`rho` initializes to exactly zero. Loading a stock 4B checkpoint therefore preserves
its logits and hidden states exactly even after the nonzero C1/donor tensors are loaded.
For donor-conv arms, `reader_collapse: mixer` adds one coefficient per stream/channel:
4 x 2560 = 10,240 parameters. Together with `rho`, those are the only trainable model
parameters. Donor `value_proj`, `norm_conv`, and all four convolution banks are frozen.
C1's two diagnosed convolution banks are also frozen and use a fixed equal mean.

## Factorial arms

| Value projection | Temporal filters | Configuration |
| --- | --- | --- |
| C1 | C1 | `examples/qwen35_transplant_c1_c1.yml` |
| donor | C1 | `examples/qwen35_transplant_donor_c1.yml` |
| C1 | donor | `examples/qwen35_transplant_c1_donor.yml` |
| donor | donor | `examples/qwen35_transplant_donor_donor.yml` |

The YAML files share the L1 injection point, frozen stock backbone, teacher cache,
training schedule, and real table. For a matched shuffled-table control, copy an arm
and set `sidecar.shuffle_context` to the same nonzero roll in every control (the existing
C controls use 7). Do not compare arms produced from different evaluation bundles.

## CPU analysis before training

Donor weight analysis is CPU-only and writes the complete spectrum/tap report:

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
python -m distillkit.analyze_donor_reader weights `
  --donor ../flash-next-ple/ple_layer.pt `
  --output scratch/donor-reader-weights.json
```

Capture held-out dequantized rows directly from the frozen evaluation bundle and table
(no model forward and no GPU):

```powershell
python -m distillkit.analyze_donor_reader capture-rows `
  --bundle ../DistillKit/scratch/independent-eval/reply-bundle-384.json `
  --table <table.gguf> --seen-rows ../DistillKit/scratch/row-novelty/seen-rows-1m.npy `
  --output scratch/donor-reader-heldout.npz
```

The resulting NPZ contains `features[...,2560]`, aligned `token_ids`, all sixteen
`row_ids`, and row-seen labels when supplied. Then run:

```powershell
python -m distillkit.analyze_donor_reader representations `
  --rows scratch/donor-reader-heldout.npz `
  --c1 ../runs/win-C1-L1-real-stage1-1m `
  --donor ../flash-next-ple/ple_layer.pt `
  --output scratch/donor-reader-representations.json
```

This reports C1/donor output norms, row cosine, linear CKA, content versus layout norms,
and row-seen versus row-novel norms. It reads only the C1 sidecar tensor from the large
safetensors file.

To test no-training collapses, capture donor conv outputs with temporal context intact.
The optional C1 target stores C1-value + C1-conv mean on the same positions as an
offline compatibility target (it is not a substitute for content NLL):

```powershell
python -m distillkit.analyze_donor_reader capture-streams `
  --bundle <bundle.json> --table <table.gguf> `
  --donor ../flash-next-ple/ple_layer.pt `
  --value-reference ../flash-next-ple/ple_layer.pt `
  --c1-target-reference ../runs/win-C1-L1-real-stage1-1m `
  --output scratch/donor-reader-streams.npz
```

The output has `streams[...,4,2560]` and optional `target[...,2560]`. Then run:

```powershell
python -m distillkit.analyze_donor_reader collapse `
  --streams scratch/donor-reader-streams.npz `
  --output scratch/donor-reader-collapses.json
```

The report includes equal mean, every single stream and the best one, a global
least-squares four-scalar mixture when `target` is present, and a PCA rank-1 collapse.
Copy fitted scalar/PCA coefficients into `sidecar.reader_collapse_weights` and select
`reader_collapse: scalar` or `pca_rank1` for a frozen evaluation.

## Frozen rho sweep and grading

Save each initialized arm before optimizer steps, then evaluate the same bundle at a
small scalar grid. `--rho` changes only the in-memory scalar and records the override in
the audit:

```powershell
python -m distillkit.independent_eval evaluate --bundle <bundle.json> `
  --checkpoint <initialized-arm> --table <table.gguf> --rho 0.05 `
  --tasks nll --output <arm-rho-005.json>
```

Repeat for a common signed/nonnegative grid including zero. The evaluator reports
`nll@content` as the primary assistant-content metric, `nll@layout` separately for
newline/think-tag targets, and `nll@structural` for layout plus template tokens. It also
retains the ordinary assistant/system/user/template metrics and the enabled-versus-
bypassed comparison.

Full-4B checkpoint initialization, downstream residual/logit capture, rho sweeps, and
adapter training require GPU time and must wait while the GPUs are occupied. Reader-only
feature capture, weight SVD/tap analysis, mathematical collapses, and unit tests are safe
CPU work.
