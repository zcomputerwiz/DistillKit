**Files changed or created by this task**

All changes are uncommitted. Paths are relative to `D:/DeepThought/Projects/HybridModel/DistillKit`. This lists all retained intentional edits and generated task artifacts. Interpreter bytecode and pytest-managed temporary files are not source changes.

| File | Why |
| --- | --- |
| `distillkit/borrowed_routing.py` | Load the existing extracted tensors while preserving the new blend; label historical diagnosis. |
| `distillkit/configuration.py` | Opt-in routing and validated blend/warmup configuration. |
| `distillkit/hyper_connection.py` | New donor equations, memory-conscious read, exact identity endpoint, and scheduled FP32 blend. |
| `distillkit/main.py` | Carry routing type into model config, refuse incompatible checkpoint routing, and attach warmup. |
| `distillkit/models/qwen35_widened.py` | Construct and initialize the selected route without changing the legacy class. |
| `distillkit/optimizers.py` | Recognize the new routing as architecture parameters for AdamW and telemetry. |
| `scratch/hyper-connection/FILES.md` | This complete retained source/artifact inventory and purpose list. |
| `scratch/hyper-connection/LAYER_RMS.md` | All 32 collapsed-stream RMS measurements for every blend and both long steps. |
| `scratch/hyper-connection/REPORT.md` | Findings, sources, equations, decisions, measurements, assumptions and limitations. |
| `scratch/hyper-connection/cpu-tests.log` | Focused regression run output (17 cases before the final TP case was added). |
| `scratch/hyper-connection/datasets-cache/generator/default-20b4bdfff511be75/0.0.0/dataset_info.json` | Probe-local datasets cache metadata. |
| `scratch/hyper-connection/datasets-cache/generator/default-20b4bdfff511be75/0.0.0/generator-train.arrow` | Probe-local cached dataset records; production cache unchanged. |
| `scratch/hyper-connection/datasets-cache/generator/default-c615bdc2e9076e33/0.0.0/dataset_info.json` | Probe-local datasets cache metadata. |
| `scratch/hyper-connection/datasets-cache/generator/default-c615bdc2e9076e33/0.0.0/generator-train.arrow` | Probe-local cached dataset records; production cache unchanged. |
| `scratch/hyper-connection/donor_config.json` | Requested pinned donor configuration, used to identify architecture, rank, branches and epsilon. |
| `scratch/hyper-connection/full-tests.log` | Final complete regression suite output: 506 passed. |
| `scratch/hyper-connection/initial-long-config.yml` | Exact input configuration for the initial one-step allocator check. |
| `scratch/hyper-connection/initial-long-console.log` | Full console output for the initial one-step long probe. |
| `scratch/hyper-connection/initial-long-measurements.jsonl` | Raw per-layer, NLL, gradient, update and memory evidence for the initial one-step long probe. |
| `scratch/hyper-connection/long-config.yml` | Exact input configuration for the bounded long probe. |
| `scratch/hyper-connection/long-console.log` | Full console output for the long probe. |
| `scratch/hyper-connection/long-measurements.jsonl` | Raw per-layer, NLL, gradient, update and memory evidence for the long probe. |
| `scratch/hyper-connection/long-trainer-output/distillkit_config.yaml` | Exact configuration written by the real production trainer during the bounded probe. |
| `scratch/hyper-connection/modeling_qwen4_exp.py` | Pinned Transformers reference modeling source; read and sampled, not installed. |
| `scratch/hyper-connection/nemo_layers.py` | Independent NeMo reference source; read, not installed. |
| `scratch/hyper-connection/next-run.yml` | Concrete recommendation for a separately launched 1% blend run; not executed. |
| `scratch/hyper-connection/paired-screen.json` | Descriptive bootstrap results on the eight-document screen, with absolute controls. |
| `scratch/hyper-connection/probe.py` | Reproducible bounded real-student/table/cache and allocator probes; exports suppressed. |
| `scratch/hyper-connection/short-config.yml` | Exact input configuration for the bounded short probe. |
| `scratch/hyper-connection/short-console.log` | Full console output for the short probe. |
| `scratch/hyper-connection/short-measurements.jsonl` | Raw per-layer, NLL, gradient, update and memory evidence for the short probe. |
| `scratch/hyper-connection/short-trainer-output/distillkit_config.yaml` | Exact configuration written by the real production trainer during the bounded probe. |
| `scratch/hyper-connection/source-oracle.json` | Bitwise FP32/BF16 reference arithmetic results. |
| `scratch/hyper-connection/source_oracle.py` | Execute only the downloaded reference norm/routing classes for an independent arithmetic comparison. |
| `scratch/hyper-connection/sources.json` | Downloaded source URLs and SHA256 hashes. |
| `tests/test_hyper_connection.py` | 18 regression cases covering donor equations, identity, gradients, schedule, dtype, and checkpoints. |

A downloaded `scratch/hyper-connection/test_modeling_qwen4_exp.py` was inspected for source discovery and removed because it was not needed; no pre-existing file was removed.

`PROGRESS.md` and `distillkit/widened_residual.py` were read but not edited. The initial untracked `scratch/gate-update-diagnosis/datasets-cache/`, `scratch/gate-update-diagnosis/trainer-output/`, and `scratch/plegated-smoke.yml` remain untouched. Existing trained models, extraction files, production YAML files and cache manifests were not modified.
