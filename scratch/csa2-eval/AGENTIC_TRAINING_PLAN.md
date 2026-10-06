# Agentic follow-through and discovery pilot

Assisted-by: Codex

Date: 2026-10-06. Authorized: research, plan, and begin training with regression
protection. This is a bounded experiment, not a replacement checkpoint decision.

## Evidence and objective

The [completed behavior screen](tool-behavior/RESULTS.md) separates reading tool
results from acting on them. Simple extraction survives all arms, while call
omission and tool-availability denial become common in round 5 and context KL.
Tool-result next-token NLL therefore cannot select an agentic checkpoint.

Relevant primary research:

- [Distilling LLM Agent into Small Models with Retrieval and Code Tools](https://arxiv.org/html/2505.17612v1),
  section 4, trains actions/reasoning conditioned on environment observations,
  excluding observation tokens from loss. It also reports interference with
  existing reasoning behavior. This supports assistant-target supervision and
  separate reasoning retention checks; it does not establish our mixture or LR.
- [ToolACE-MT](https://arxiv.org/html/2508.12685v1) constructs complete multi-turn
  trajectories, refines them, and verifies them with rule/model checks. We adopt
  complete, locally executable references for a small pilot. Our deterministic
  templates are much narrower than that paper's synthesis pipeline.
- [Self-Synthesized Rehearsal](https://arxiv.org/html/2403.01244v1) studies replay
  for retaining earlier capabilities. We already have the earlier data, so we
  reuse protected real caches instead of generating substitutes for them.

The desired policy is: retrieve an absent fact when a supplied tool can resolve
it; ask only about a remaining unresolved choice or unavailable information;
emit a call with an announcement of an immediate action; use returned IDs;
recover from actionable errors; stop after success. Training can improve this
policy but cannot guarantee every future announcement is followed by a call.
A serving protocol can enforce structured action commitments separately.

## Data and loss

New generator: `scratch/dense_gr/agentic_curriculum.py`. Version 1 creates 720
training conversations across six synthetic schema domains and 40 held-out
conversations across two other domains. It executes reference actions against
a local finite state machine; no external APIs or generated code are executed.
It covers discovery, known IDs, resolvable and genuine ambiguity, no matches,
stale-session recovery, direct explanations, lookup then read, announcements
with calls, and already-completed tasks. Some prompts carry explicit policy;
others do not. Successful reports follow success observations.

The training conversations contain 512,261 input tokens and 69,321 assistant
targets. System, user, and tool messages remain visible but have zero loss in
this new data. Hard-label CE supervises all assistant turns and their endings.
No teacher KL is applied to the synthetic trajectories. The existing cache
container has required top-k fields; these are explicitly marked placeholders.
The scratch cache adapter rejects their use without CE-only and assistant-only
flags, and the trainer rejects combining them with context KL.

Existing replay is preserved from `control_arms.ps1`: agent traces, tool traces,
QA (answer weight 8), code, raw-code teacher targets, thinking/nonthinking math,
general instruction data, and the anti-loop caches. Existing objective choices
are retained, including assistant-only agent/tool/QA masks, CE-only short-code,
KL-only raw-code, and loop unlikelihood. This does not silently change the role
masking of other legacy caches. Keep their known contamination exclusions and
effort-prompt cleanup. No benchmark examples are added to training.

Target new-data share: approximately 15% of weighted supervised targets per
epoch, not 15% of documents. `agentic_arm.py plan` computes the repeat factor
and records the exact first 50-step mixture, unique new examples, and maximum
visits. Token accounting includes masks and QA answer weights. This proportion
is an experimental starting point, not a value established by a paper.

The frozen first 50-step plan contains **837,410 weighted targets**, of which
**127,935 (15.28%)** are new agentic targets. It visits 546 of the 720 new
conversations, at most six times each. The repeat factor is 196 because the
legacy replay corpus is large; this bounded run does not make 196 passes over
the new data. The replay-only control contains 1,228,275 targets in 50 steps.
This unequal exposure is a material limitation of the equal-step comparison;
do not attribute every difference to the new examples alone.

## Bounded arms

Both arms start from **u50**, whose generated tool behavior is stronger than
the round-5/context checkpoints. The prior flat-rate result informs the LR,
but is not treated as the matched control for a changed data recipe.

1. Agentic pilot: existing replay plus the new CE-only conversations.
2. Replay control: same initialization, LR, seed, and 50 optimizer steps;
   existing replay alone.

Use the existing compensated optimizer, body LR `1.83e-6` scaled flat by 0.55,
ten-step warmup, and existing DeltaNet rate multiplier. The shorter warmup is
explicitly a pilot choice and differs from the earlier 100-step controls.
Keep the original 10M-target LR schedule (this run is only a prefix), accumulation
2, 32K micro-token cap, full-layer checkpointing, head chunk 512, streaming head
backward, and per-frame CSA2 selection reuse. No architecture changes, layer
freezing, new normalizer, or router objective changes. Both GPUs are available.

These are equal-step controls, not identical-token or identical-example controls:
adding length/objective buckets changes sample order and target counts. Report
those differences. A favorable pilot needs confirmation before a full run.

Exact commands and measured mixture are serialized in
`scratch/dense_gr/agentic-pilot-v1/plan.json`. Training writes fresh checkpoints
and optimizer state. The runner stops on child failure and runs no probes in
parallel with training.

## Evaluation and promotion gates

- Freeze and evaluate the new schema-transfer cases at base, agentic, and replay.
  Shared templates mean this is a transfer screen, not unseen-family evidence.
  Clarification scoring is a lexical screen; inspect its failures manually.
- Run `agentic_live_eval.py` in the same finite environments. This allows an
  extra read before an update and verifies the resulting state. The initial
  base next-call screen scored 62/80, but many mismatches were valid extra
  reads, so that number is not task success. One actual failure claimed a
  session had been refreshed without a call. Closed-loop evaluation prevents
  penalizing a valid longer action sequence merely for differing from the
  reference's shortest next call.
- Repeat the untouched original tool-behavior and post-tool fixtures. These
  were not used to create new training records. Compare exact calls, omission,
  schema errors, incorrect arguments, no-call cases, and unnecessary questions.
- Run the existing role-split ledger including extended 32K QA, code, general,
  agent/tool assistant turns, and long-context material. Context NLL remains
  diagnostic. Compare per-source assistant/answer loss, not the mixed average.
- Run the existing 256-example code/thinking NLL and math generation proxy on
  all three arms. The current proxy uses a 1024-token MATH cap; rerun base and
  control rather than compare against historical 512-token results. Report
  truncations and formatting failures alongside accuracy.
- Before promotion, require a reproducible reduction in real held-out tool
  omissions/incorrect calls, with no new controlled no-call or prerequisite
  violations. Review paired document uncertainty. Treat an assistant/answer
  NLL regression over 0.02 nats with paired evidence, or a generation decline
  over 2 percentage points, as a stop-and-investigate trigger, not as proof
  that smaller changes are safe. Small-sample uncertainty can also block a
  decision. Existing stronger u50 skills remain a comparison, not just replay.

Do not automatically repair a failure with extra post-training or checkpoint
averaging: either can undo tool improvements. If retention degrades, first
adjust the replay mix or reduce the training budget, then repeat both tool and
retention evaluations. If this narrow curriculum saturates without improving
the original held-out traces, expand verified multi-domain trajectories and
evaluate closed-loop tool execution before considering on-policy correction
or preference/RL training. Never reward announcements or raw call count.

## Status

Data, role-mask, CE-only guard, schema, and finite-environment tests passed
(16 tests across the curriculum, tool evaluator, and repeat-dispersal suites;
the existing six unlikelihood tests also passed in the earlier validation).
Baseline closed-loop success is **32/40**: all four genuine-choice cases fail,
three of four complete-empty-search cases loop, and one recovery case reports
completion without completing recovery. The other categories pass all four.
This is a small deterministic screen, not a population estimate.

The sequential pipeline has been launched. Live state is in
`scratch/dense_gr/agentic-pilot-v1/status.json`; detailed training logs are
`train-agentic.log` and `train-replay.log`. It runs both bounded arms followed
by finite-environment, frozen next-call, ledger, paired, and generation checks.
No trained checkpoint is selected yet. Existing model implementation and the
llama.cpp tree are untouched.
