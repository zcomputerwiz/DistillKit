# Preservation replay corrections (2026-10-09)

Assisted-by: Codex

The two corrections recommended by the
[influence audit](INFLUENCE_AUDIT_RESULTS.md) are applied and validated: freeze
the selection-only indexer for preservation training and quarantine the exact
confirmed nonfunctional email target. No new training stage or checkpoint was
written. The completion candidate remains experimental; long1-u50 remains the
reference.

## Indexer preservation

The existing trainer's `--freeze-router` covers `index_q_proj`, `index_k_proj`,
`indexer_proj` and `index_gate`. It removes their weights from optimizer groups
before tensor parallel sharding. This prevents the six present-but-zero query
gradients from causing AdamW decay when this stage has no routing objective.
The audit does not establish that this decay caused the historical regression.

[agentic_arm.py](../dense_gr/agentic_arm.py) enables this existing flag by default
for new CLI plans, records its checkpoint name, and uses the trainer's existing
`-frozen` suffix when finding checkpoints for evaluation. Programmatic recipe
defaults and archived serialized plans retain their historical behavior for
audit reconstruction. Future completion/preference builders must pass
`freeze_router=True` and the new exclusions explicitly; an old archived plan is
not the next-phase recipe.

Freezing preserves logits and selected positions before backbone updates. It
preserves indexer weights throughout training, but selected positions can still
change as the trainable backbone changes its inputs. A separate router alignment
stage would be a different objective and needs its own plan. A new frozen stage
needs a fresh optimizer layout; the old optimizer state is not a compatible
resume target.

The freeze also avoids query gradient buffers and optimizer/compensation state
for these inactive weights. This is a modest memory opportunity, not a measured
throughput gain. Attention selection and backbone execution are unchanged.

## Exact quality quarantine

The existing Docker sandbox was reused to requalify
`onpolicy:kodcode:Filter_62706_I:s0`. The inherited tests accept a function that
simply returns `False`. A stronger fake-SMTP test requires one serialized message
to be sent to the requested recipients:

| Case | Inherited tests | Stronger test |
| --- | --- | --- |
| Captured target | Pass | Fail |
| No-op mutation | Pass | Fail |
| Serialized-message positive control | Not needed | Pass |

The positive control validates the harness only; it is not a new training
target. All checks ran in the existing sandbox with networking disabled.
The [requalification receipt](../dense_gr/replay-ready-20261009/email-requalification.json)
records the input hashes, container image, harness and outcomes. The source
captures, historical labels and teacher caches retain their original contents.

The separate
[quality exclusion](../dense_gr/replay-ready-20261009/exclude-quality-confirmed.json)
contains this one document. The existing `exclusion_for.py` combines it with the
4,279-ID confirmed benchmark/run index to produce
[4,280 exclusions](../dense_gr/replay-ready-20261009/exclude-replay-next-20261009.json)
present in the 16 active caches. The benchmark policy remains confirmed matches
only. Cache-manifest review found one active alias of this quality failure.

## Execution and target checks

The new planner uses the same exclusion file for mixture measurement and trainer
arguments, and verifies its recorded SHA256 before launching a frozen plan.
The [readiness receipt](../dense_gr/replay-ready-20261009/readiness.json) records
the replay component, source counts and bounded target checks:

- The quarantined document is absent from teacher IDs and all 8,805 planned
  replay microbatches.
- All 21 bounded records retain their audited token-prefix hashes. Scored
  system, user and tool-result targets remain zero.
- Replay repeats, CE/KL/unlikelihood policies and assistant masks match the
  current saved recipe. The template uses its code multiplier 3 and rate scale
  0.1375; this is not a new mixture or learning-rate experiment.

The focused suite passed **37 tests** in 81.43 seconds. The new freeze test runs
on CPU and two GPUs. It verifies identical initial logits/selection, frozen
weights with no gradients or optimizer state, and nonzero backbone gradients
with actual backbone updates. The GPU case uses production tensor parallel
sharding, the remote tied streaming head, checkpointing, gradient synchronization
and Kahan AdamW8bit. See the
[test receipt](../dense_gr/replay-ready-20261009/tests.log).

Two initial test-fixture errors were corrected: the tiny DeltaNet geometry
needed two key heads for two-way sharding, and the test needed the production
remote-head backward path. Neither required a model or trainer-core change.

## Next phase boundary

This is a validated replay component, not a complete launch plan. The next phase
still needs broader executed aggregation trajectories, disjoint evaluation
worlds, the preference/replay schedule, and frozen retention gates. Preserve
empty-search, direct-answer, recovery, code, math and knowledge coverage while
expanding aggregation. Do not repeat the old four aggregation worlds and infer
transfer from their training results.

No root loss, sample equalization or projection was added. The audit did not
provide evidence for those objective changes. Execution diversity and target
quality remain the better-supported next adjustments.
