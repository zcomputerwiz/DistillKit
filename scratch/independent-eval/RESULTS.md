# Independent evaluation, 2026-09-09

Status: all four required checkpoints completed the independent-text screening
run. Real MMLU/ARC-Challenge runs are **not complete**: direct downloads failed
with Windows socket permissions, and browser access to the dataset server was
denied by browser security. No benchmark scores or accuracy intervals have been
fabricated. The likelihood scorer and normalization logic have automated tests.

The text screen contains 32 documents / 14,396 scored tokens, selected from 1,602
documents absent from both cache manifests. Each document contributes its first
512 tokens at most. Confirmation has 32 separately assigned documents; no
confirmation model scores were inspected. CIs below are 10,000 paired percentile
bootstrap resamples over documents, preserving token-weighted aggregation.

| Checkpoint | Arm | NLL (nats/token) | 95% bootstrap CI |
|---|---|---:|---|
| student-hf | Pre-retrofit / enabled / bypassed (same stock model) | 1.110749 | [0.904510, 1.356853] |
| gr-stage1-1m | Enabled | 1.676555 | [1.473412, 1.902243] |
| gr-stage1-1m | Flag-only bypassed | 1.521476 | [1.325510, 1.741692] |
| gr-stage1-1m | Complete-adapter bypassed | 1.110749 | [0.904510, 1.356853] |
| ple-stage1-1m | Enabled | 1.562295 | [1.351140, 1.800831] |
| ple-stage1-1m | Bypassed | 1.110749 | [0.904510, 1.356853] |
| lr-sweep-1e3 | Enabled | 1.829387 | [1.632912, 2.041033] |
| lr-sweep-1e3 | Bypassed | 1.021348 | [0.837215, 1.238876] |

The pre-retrofit comparison for every checkpoint is the same student-hf row.

| Checkpoint | Delta NLL, enabled minus bypassed [95% CI] | Delta NLL, enabled minus pre-retrofit [95% CI] | Screen verdict |
|---|---|---|---|
| student-hf | 0 [0, 0] | 0 [0, 0] | Indistinguishable by construction: no adapter |
| gr-stage1-1m | +0.155079 [0.118509, 0.201342] | +0.565806 [0.450630, 0.686913] | Hurting |
| ple-stage1-1m | +0.451546 [0.376977, 0.526646] | +0.451546 [0.376977, 0.526646] | Hurting |
| lr-sweep-1e3 | +0.808038 [0.692499, 0.930007] | +0.718638 [0.580543, 0.858106] | Hurting |

lr-sweep-1e3's bypassed backbone improves on the student by -0.089401
[-0.121993, -0.062410] nats/token. Its enabled sidecar more than reverses that
improvement. This is a source-specific screening result, not a claim about all
knowledge/reasoning performance or a confirmation result.

## Self-check and anomaly resolution

PLE flag-only bypass and GR complete-adapter bypass match student-hf exactly on
**every document**, with maximum absolute document sum-NLL difference 0.0.
Both aggregate paired delta CIs are [0, 0]. GR flag-only bypass does not match:
it still applies the trained gated-residual branches, giving +0.410727
[0.327493, 0.493915] relative to student-hf. No training code was changed to
alter these existing semantics.

The retired sketch read `examples/_lr_sweep_base.yml`, which sets `variant: ple`.
`load_student_model` overwrites the checkpoint's `sidecar_variant` with that
setting. The CPU reproduction in `legacy-loading-audit.json` proves that loading
gr-stage1-1m this way discards all seven saved GR tensors, initializes six PLE
tensors, and leaves both the new PLE value projection and convolution at norm
zero. It therefore measures an inert, newly initialized adapter and produces the
spurious +0.0000. The trained GR adapter was never being evaluated.

The corrected load preserves `gated_residual`, verifies all seven adapter tensors
exactly against disk, and observes up to 3.3046875 change in a probed logit with
the flag enabled. Its saved final W_side_proj norm is 2.100469 (the task quoted
2.077; this report records the actual current checkpoint tensor).

## Validation and limitations

375 tests pass with both GPUs visible, including 15 new evaluator cases. Temporary
and dataset cache directories were placed inside the workspace to resolve
filesystem permission failures; no tests were intentionally skipped. Ruff and
git diff whitespace checks pass. Four complete GPU text jobs took 13.0, 22.8,
18.9, and 19.4 seconds (student, GR, PLE, LR respectively); the preceding tiny GR
job took 11.2 seconds. The watchdog caps each evaluation process at 540 seconds.

This measures heldout rendered-conversation next-token prediction independently
of teacher loss and learned projections. It does not reproduce Qwen's
out-of-domain Uncheatable PPL, prove pretraining/benchmark decontamination,
measure generation or reasoning traces, or justify a downstream verdict without
the missing benchmark runs. Existing tokenizer regex warnings are preserved
rather than silently changing tokenization relative to the repository.

See `../../docs/independent_eval.md` for the exact protocol and rerun commands;
`text-report.json` contains all absolute and delta comparisons, and the four
`*.text.json` files retain per-document observations, loading audits, and probes.
