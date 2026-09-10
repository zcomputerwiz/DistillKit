# Independent evaluation, 2026-09-09

Status: all four required checkpoints completed the independent-text screening
run. Real MMLU/ARC-Challenge runs are **not complete**: direct downloads failed
with Windows socket permissions, and browser access to the dataset server was
denied by browser security. No benchmark scores or accuracy intervals have been
fabricated.

**Correction, 2026-09-09.** That was the sandbox, not the machine. Both datasets
download anonymously here on the first attempt -- `cais/mmlu` `all` 14,042 test rows,
`allenai/ai2_arc` `ARC-Challenge` 1,172 -- and no Hugging Face token is required.
`full-bundle-384.json` now carries 384 documents plus 256 MMLU and 256 ARC questions
per split. Benchmark scoring is queued behind the stage-2 training runs, which own
the GPU; see `scratch/score_widening_full.sh`. The likelihood scorer and normalization logic have automated tests.

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

---

# Widened curriculum, stage 1, on a 384-document screen (2026-09-09)

The 32-document screen above has wide absolute intervals (student-hf
[0.904510, 1.356853]) -- fine for "does this adapter hurt", too coarse to separate
two arms of a curriculum. `text-bundle-384.json` draws 384 of the same 1,602 eligible
unseen documents, 175,526 scored tokens, twelve times the original. Reproduce with
`bash scratch/score_widening.sh`; each checkpoint takes 57-167 s, well inside the
540-second watchdog. Benchmark tasks are still absent for the same reason as above.

Reference `student-hf` = 1.150513 nats/token. All intervals are 10,000 paired
percentile bootstrap resamples over documents.

| Checkpoint | Arm | NLL | Delta vs student-hf [95% CI] |
|---|---|---:|---|
| **widened-stage1-1m** | widening only, no sidecar | **1.124669** | **-0.025844 [-0.028280, -0.023403]** |
| widened-ple-stage1-1m | bypassed | 1.136468 | **-0.014045 [-0.014968, -0.013129]** |
| widened-ple-stage1-1m | enabled | 1.641075 | +0.490562 [+0.463239, +0.518762] |
| ple-stage1-1m | bypassed | 1.150513 | +0.000000 |
| ple-stage1-1m | enabled | 1.593521 | +0.443008 [+0.417380, +0.469528] |
| gr-stage1-1m | enabled | 1.686505 | +0.535992 [+0.499463, +0.574419] |
| gr-stage1-1m | flag-only bypassed | 1.533189 | +0.382677 [+0.357176, +0.409036] |
| gr-stage1-1m | complete bypass | 1.150513 | +0.000000 |
| lr-sweep-1e3 | enabled | 1.835426 | +0.684913 [+0.641930, +0.729746] |
| lr-sweep-1e3 | bypassed | 1.057720 | -0.092793 [-0.101844, -0.084066] |

Two readings, both new:

**The widening helps.** `widened-stage1-1m` is the first checkpoint in this project to
score below the pre-retrofit student, with the whole interval below zero. Its only
trainable model parameters are 42.6M of routing; the backbone is frozen and the
architecture is exactly the identity at initialisation.

**The sidecar still hurts, independently.** Enabled-minus-bypassed is +0.504607 with the
widening and +0.443008 without. A wider residual stream did not give the n-gram table
somewhere cheaper to write; it cost marginally more. The two effects are separable
because the widening starts at identity, which no earlier arm did.

The wider screen also reproduces every earlier verdict at four times the precision --
gr +0.5360 against +0.5658, ple +0.4430 against +0.4515, lr-sweep-1e3 +0.6849 against
+0.7186, and lr-sweep-1e3's bypassed backbone still ahead of the student at -0.0928.

## A probe defect this run found

`plumbing_probe` required exactly two sidecar calls across the enabled and bypassed
forwards. A widened decoder layer applies the sidecar once per residual branch, so
`widened-ple-stage1-1m` raised "evaluation silently bypassed the sidecar or failed to
forward its data/flag" and produced no score at all. It now expects
`residual_stream_num_branches` calls per forward, and reports the count it used. The
failure was safe -- the probe refused to score rather than scoring a bypassed model --
and it is covered by
`tests/test_independent_eval.py::test_probe_counts_one_sidecar_call_per_residual_branch`.
