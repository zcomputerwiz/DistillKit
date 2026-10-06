# Role loss and the next training experiment

Reviewed 2026-10-05. This is a research and experiment-design review. No model,
training objective, cache, or checkpoint was changed; no training was launched.

Follow-up: the [completed behavior evaluation](tool-behavior/RESULTS.md) found
retained simple result reading but tool-call omissions, partly recoverable by
explicit system instructions. It favors flat as the next confirmation candidate
and provides no reason to advance teacher context-KL for preservation.

## Recommendation

Keep the existing assistant-only CE/KL objective for agent conversations,
including assistant tool calls and answers after tool results. Keep system,
user, and tool messages visible as inputs, with zero direct prediction loss.
Do not extend the current teacher context-KL arm to a longer run on the basis
of context-token NLL. First measure whether the existing arms actually use
returned results correctly. Context prediction is a diagnostic, not an agent
success criterion.

This revises the earlier suggestion to make a tool-excluded context-KL arm next.
That ablation is interpretable, but it would still impose teacher prediction
targets on system/user messages. There is no evidence yet that this remaining
objective addresses a behavioral regression. It is a secondary diagnostic,
not the preferred next training recipe.

## What the actual implementation does

`scratch/dense_gr/teacher_kl.py:assistant_tokens` selects assistant content after
the assistant marker, through that assistant turn's closing token.
`position_weight` shifts those labels correctly: position t predicts token t+1.
Answer spans can increase their weight. Input tokens remain in the forward
pass; the loss mask is not an attention mask or a detached context state.
Consequently, assistant losses can train the representations and attention used
to read previous user/tool tokens without asking the head to predict them.

Context-KL eligibility is implemented as zero ordinary loss weight on real
positions in the selected assistant-only caches. It is not a separate semantic
role classifier: structural/header positions also qualify. The arm samples
every eighth eligible position, with a deterministic document-dependent offset,
and multiplies sampled weights by eight. Padding and the final no-target
position are excluded. Sampling is an approximation to the dense auxiliary
loss; it does not determine which roles deserve supervision.

`training_step.py` adds that weight only to teacher KL, not CE, and normalizes
by the original supervised weight total. With teacher weight 0.5 and context
weight 0.01, a sampled context row has coefficient 0.5*0.01*8 divided by that
total. Thus 0.01 is not directly comparable to a prompt CE weight in a paper.
The aggregate auxiliary influence depends on context/assistant token counts,
answer weights, and loss magnitudes. A tiny per-token coefficient does not
guarantee a tiny total effect in tool-heavy records.

Earlier data repairs also matter: `agent_corpus.py` retains tool definitions
and serialized calls that an older renderer dropped; `broken_tool_docs.py`
identifies affected historical records. Do not undo these repairs or use
malformed old traces as evidence about role masking. `framing_check.py` already
exists to audit the teacher on the same code rendered as user, tool, or raw text.

## Primary research

- [Distilling LLM Agent into Small Models with Retrieval and Code Tools](https://arxiv.org/html/2505.17612v1), section 4, equation 4: the student is trained on reasoning/actions, excluding observations from loss while retaining them in trajectory history. Its small-model agent setting is especially relevant here. This supports our original observation mask; it is not a controlled proof that every observation auxiliary loss is harmful.
- [ToolLLM / official ToolBench trainer](https://raw.githubusercontent.com/OpenBMB/ToolBench/master/toolbench/train/train.py), lines 82-153: the implementation recognizes system, user, function, and assistant roles and masks context to train assistant outputs. Its exact history/last-reply convention differs from our all-assistant-turn supervision, but function output is context rather than a prediction target.
- [ARTIST](https://arxiv.org/html/2505.01441v1), section 2.2: masks tool-output tokens during GRPO optimization and trains generated reasoning/actions. This is RL evidence, where environment observations are not sampled policy actions; it should not be presented as an SFT masking ablation.
- [MUA-RL](https://arxiv.org/html/2508.18669v1), section 4.1: explicitly masks both tool execution results and user messages. This also concerns agent policy optimization, not a comparison of our CE versus teacher context-KL.
- [Instruction Fine-Tuning: Does Prompt Loss Matter?](https://arxiv.org/html/2401.13586): reports benefits from tuned nonzero prompt weights in short-completion instruction tuning, with effects depending on completion/prompt ratio. It studies likelihood of actual prompt tokens, not distillation of a chat teacher's hypothetical tool outputs. Its weights cannot be imported into our normalization unchanged.
- [Instruction Tuning With Loss Over Instructions](https://papers.nips.cc/paper/2024/file/7ffb43adf37b3eeaba559098bc084cc6-Paper-Conference.pdf), NeurIPS 2024: finds prompt modelling useful especially with long instructions/short outputs and limited training examples, attributing improvement to reduced overfitting. It supports an auxiliary prompt-likelihood ablation if behavior warrants one; it does not establish tool-result teacher KL as the preferred objective.
- [Toolformer](https://papers.nips.cc/paper/2023/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf), section 2: filters calls by whether their results improve prediction of subsequent text, then uses a standard language-modelling objective on augmented text. This is an important counterexample to claiming universal observation masking. Its text-continuation training differs from our assistant policy distillation. Its filtering criterion motivates measuring usefulness of results for later assistant outputs.
- [On-Policy Distillation / GKD](https://arxiv.org/abs/2306.13649): addresses the mismatch between fixed training outputs and student-generated outputs. This supports evaluating generated action trajectories; it does not support imitating an environment's tokens as student policy actions.
- [API-Bank](https://arxiv.org/abs/2304.08244) evaluates runnable API planning/retrieval/calling. [Tau-bench](https://arxiv.org/abs/2406.12045) evaluates final environment state and repeated-trial reliability. Both support behavior/outcome evaluation beyond call-token likelihood.

The literature supports assistant-only supervision as our default, with genuine
exceptions for auxiliary language modelling. It does not supply a universal
optimal role mask, a validated W for this student, or evidence that matching a
teacher's next-token distribution on externally supplied text preserves reading.

## Reassessment of the completed experiments

1. Role-split ledgers were the right correction to aggregate chat NLL. Round 5
   improved assistant tokens; its apparent agent regression was largely in
   other speakers' text. Calling this a context-prediction regression is valid.
   Calling it impaired context understanding is not established.
2. Family and shallow-MLP reverts genuinely localize the cause of that NLL
   change. They do not prove that those layers lost comprehension, that a
   freeze will reproduce a revert, or that repairing this NLL improves agents.
3. Depth ramp versus flat 0.55 was a useful matched control. Ramp changes tool
   prediction more than a comparable global rate reduction, but the broader QA
   evaluation confirms a cost: about 75% versus 90% gain retention. It is not
   automatically the best recipe when context-prediction NLL is demoted.
4. The context-KL arm was a reasonable diagnostic hypothesis but a poorly
   matched preservation target. It transfers the teacher distribution, not the
   frozen student's behavior. On tool content the actual next token is absent
   from teacher top-64 at 46%/49% of positions; teacher top-1 matches only
   26.5%/21.2%. Most guesses are content, not turn endings. The tool NLL increase
   of about 1.5 nats is consistent with conflicting supervision, although this
   does not establish its entire causal mechanism.
5. That arm retained QA gains and improved tool-call token NLL. It must remain
   in the behavioral comparison rather than being discarded solely for tool
   output NLL. Conversely, those token improvements do not prove executable
   call correctness or correct use of observations.
6. The 512-token math proxies are length-limited screens, not decisive full
   benchmark results. The wider QA set was an appropriate repair to sparse
   answer coverage. One seed/100 steps still cannot select a long-run winner.

## Next evaluation and conditional experiments

Reuse base u50, round-5 step 100, ramp, flat, and context checkpoints. Freeze
held-out tasks before viewing generated outcomes, exclude training/cache IDs,
and use identical templates, stop rules, context lengths, and generation caps.
Report paired task-level uncertainty and truncation separately. First run a
bounded screen, then broaden only when the result warrants the expense.

The repository already has role-split `atlas.py`, token evidence and paired
`atlas_compare.py`, generation/load paths, and `frontier/tool_tasks.py` schema
validators. The latter verifies synthetic data rather than evaluating model
generations. No turnkey BFCL/API-Bank/tau execution runner was found in the
searched scratch/docs paths. Extend existing loading/generation/validation
pieces; do not equate schema validity with task success.

Measure separately:

- Initial call selection, required arguments, no-call decisions, and requests
  for missing information.
- After a supplied result, extraction/copy of novel values and propagation
  into a dependent call; error recovery; final answer grounded in the result.
- Longer histories with distractors and results beyond the local window.
- Deterministic sandbox task completion and repeated-trial reliability, when
  execution is available. Never execute captured arbitrary shell/code calls
  against the live workspace as an evaluation.

Use controlled pairs where a result value changes and the correct downstream
answer/call changes accordingly. Update reference targets with the result,
rather than penalizing a correct response to altered context. Teacher-forced
assistant NLL after tool messages can be a cheap screen; generated correctness
and task outcomes remain the primary measures. Removing/shuffling results
alone is a sensitivity probe, not a correctness score.

If assistant behavior is retained, continue an assistant-only recipe chosen on
QA/code/agent outcomes; there is no demonstrated need to repair tool NLL.
If behavior actually regresses, first test frozen-u50 KL on protected assistant
responses conditioned on the real user/tool history. This anchors the behavior
we intend to preserve and can also resist desired improvements, so restrict it
to protected replay and measure the tradeoff. It is a new objective, not an
already adopted change.

A low-weight CE auxiliary on actual user/document text is a secondary
regularization experiment supported by prompt-loss research. Separate role
weights and normalization, keep the original assistant objective fixed, and
audit gradient/loss contributions. Tool-result CE is lower priority: a model
may be unable to infer fresh database values or execution output before seeing
them. A role-excluded teacher-KL ablation can diagnose which part of the failed
context hypothesis mattered, but should not take precedence over behavioral
evaluation. Do not combine a new anchor, depth ramp, and data mixture change
in the same first arm.

Assisted-by: Codex
