# Round-five MLP intersection reverts (2026-10-05)

The bottom-eight-layer MLPs carry most of the observed tool-result NLL drift,
while the remaining MLPs carry more of the assistant-turn gains. The specific
intersection is now tested rather than inferred from separate depth/family reverts.
Restoring these shallow MLPs is a selective repair candidate, not a complete repair
or a tested training policy.

## What ran

Four forward-only evaluations on physical GPU 0, with `CUDA_VISIBLE_DEVICES=0`:
the full round-five model, the two new reverts, and the base. The process completed
with exit 0 at 15:17:17; GPU 0 was released. Evaluation times were 163, 163, 165 and
170 seconds. The 25-minute process limit was not reached.

- Base: `scratch/dense_gr/merges-long1/u50`.
- Tuned: `scratch/dense_gr/checkpoints-2b-long-r5/smoke-r1-1-gr-s25-csa2`.
- Documents: the existing, unchanged `scratch/csa2-eval/atlas/domains.pt`.
- New families: MLP tensors in layers 0-7 (24 tensors), and MLP tensors in layers
  8-23 (48 tensors). These partition the 72 MLP tensors; no norms, HC residuals,
  attention tensors or embeddings were reverted.
- Loaded parameters were changed only in memory. No model or training source was
  changed and no checkpoint was written.

The new base and tuned baseline NLLs match the previous `atlas/long5/revert.json`
**exactly across all 84 cells of each baseline**. The prior full-MLP and whole
layers-0-7 reverts were not rerun. Existing artifacts contained only aggregate
statistics, so these two baseline evaluations were necessary for paired token
transitions and retention intervals.

## Causal results

Numbers below are NLL changes from the full tuned model. Negative helps; positive
is the gain lost by the revert. Intervals are 95% paired document bootstraps.

| Held-out role | Revert MLPs 0-7 | Revert MLPs 8-23 |
|---|---:|---:|
| Claude-Code tool results | -0.234105 [-0.279232,-0.193861] | -0.035288 [-0.047522,-0.025737] |
| Codex tool results | -0.221372 [-0.263100,-0.183171] | -0.055303 [-0.068532,-0.043400] |
| Claude-Code own turns | +0.009161 [+0.006086,+0.011604] | +0.026200 [+0.021248,+0.031559] |
| Codex own turns | +0.009422 [+0.007278,+0.012293] | +0.029410 [+0.020785,+0.035147] |
| Teacher code own turns | +0.001394 [+0.000343,+0.002465] | +0.006275 [+0.004426,+0.008071] |
| llama.cpp plain source | -0.000302 [-0.001161,+0.000434] | +0.002838 [+0.000987,+0.004379] |
| QA documents | -0.072578 [-0.076761,-0.068350] | -0.003651 [-0.005188,-0.002058] |
| QA own turns | +0.053742 | +0.045587 |

The shallow-MLP revert removes approximately 72% of Claude/Codex tool-result drift.
Residual changes against the base remain **+0.089636** and **+0.085826**, so it does
not meet the proposed +0.01 context criterion. Claude/Codex system NLL also remains
above base (+0.030931 and +0.059316). These interventions were evaluated separately;
their effects must not be added as though the model were linear.

## Gain retention and paired transitions

Let `G = NLL_base - NLL_arm`. A positive signed contrast
`G_revert - 0.8 * G_tuned` means at least 80% of the tuned gain is retained.
All three models are bootstrapped jointly over the same documents.

| Role | Shallow-MLP gain retained | Signed contrast [95% interval] |
|---|---:|---:|
| Claude-Code own turns | 90.85% | +0.010868 [+0.008191,+0.013163] |
| Codex own turns | 91.52% | +0.012788 [+0.008706,+0.015250] |
| Teacher code own turns | 94.55% | +0.003722 [+0.002926,+0.004637] |
| llama.cpp plain source | 101.57% | +0.004144 [+0.002504,+0.005801] |
| Thinking math own turns | 91.44% | +0.006870 [+0.005393,+0.008192] |
| Non-thinking math own turns | 88.06% | +0.004085 [+0.003218,+0.004988] |
| QA own turns | 88.10% | +0.036563 [+0.029338,+0.041385] |
| Tools own turns | 80.26% | +0.000223 [-0.001655,+0.002147] |

Tools retention is inconclusive at the 80% boundary. QA has only **3 documents
with own-turn targets out of 16 documents**, totaling 1,502 targets; its interval
does not establish broad QA generalization. Bootstrap samples with zero targets
for a role are excluded rather than given a zero-valued estimator.

For the shallow revert, token top-1 transitions against the base are:

| Role | Base right -> arm wrong | Base wrong -> arm right |
|---|---:|---:|
| Claude-Code own turns | 107 | 209 |
| Codex own turns | 415 | 916 |
| Teacher code own turns | 346 | 593 |
| llama.cpp plain source | 559 | 702 |
| Claude-Code tool results | 230 | 152 |
| Codex tool results | 592 | 232 |

Thus net assistant-turn top-1 gains coexist with individual-token forgetting;
the remaining tool-result regression appears in transitions as well as NLL.

## Evidence and reusable tooling

`revert.json` holds paired NLL differences. `token_evidence.npz` (18.3 MB) holds
aligned per-token NLL/hits plus original IDs and target-token roles for all four
arms. `token_evidence.json` records the domain SHA256, model-source/config hashes,
checkpoint paths/file metadata and exact family membership. `paired_comparison.json`
holds per-role gain contrasts, confidence intervals and both transition directions.
`run_status.json` records process completion and GPU release.

Only the existing atlas diagnostic was extended: two optional family selectors and
`--save-token-evidence` for `nll`/`revert`. Existing default families and default
outputs are unchanged. `scratch/dense_gr/atlas_compare.py` consumes this evidence
on CPU and can compare later control/context arms without model reruns.
Three relevant existing CPU tests passed; synthetic contrast, transition and
zero-target bootstrap checks passed. The saved real evidence was analyzed and
its transition totals were checked against aggregate top-1 changes.

No broader atlas, new control-model evaluation, long-context extension, generation
benchmark or training run was launched. Agent documents remain capped at 16K;
these results do not establish the proposed 16-32K criterion. This is one fixed
held-out sample and one full round-five checkpoint, with no seed replication.
No commits or pushes were made.
