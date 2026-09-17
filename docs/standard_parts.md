# Standard parts: modules trained once and reused across backbones

A program proposal, not a result. Nothing here has been run.

The premise is that some structure every language model builds is cheap to compute
exactly, and that such structure is better specified once than relearned by every model.
This document fixes what is being claimed, what the interface has to look like for the
claim to be testable, which parts are worth building first, and the one experiment that
tests the premise at its cleanest point.

## 1. What is being claimed

Three claims, in order of how easily they can be defended. They are separable and should
be reported separately; the program does not need all three.

**Reuse (primary).** One module, trained once, installs into independently trained
backbones -- different sizes, different data, different seeds -- **within a declared
module ABI**, and each becomes dependent on it. Nobody has shown this for an
interpretability-identified structure. It needs no efficiency argument and no capability
argument, and it fails fast if the structure is not real.

The ABI qualifier is load-bearing and belongs in the claim rather than in a footnote. A
vocabulary-space part ports only within a tokenizer family, so its ABI includes tokenizer
identity; a fixed-width input code is substantially more portable and its ABI may be only
the code width. Stating this up front is what stops a future positive result from being
read as universal portability when it in fact depends on one 248,320-entry vocabulary.
Each part in section 4 should carry its own ABI when it is built.

**Specification stability (secondary, and free once reuse holds).** Behavior on the
module's domain is specified rather than learned, so it does not regress between model
generations and does not vary by seed. For the algorithmically trivial parts this is
true by construction. It is also the claim that survives if the compute argument fails.

**Amortized compute (tertiary, hardest).** Total compute to reach a given quality is lower
when the module's one-time training cost is spread over the models that use it. This is
the claim Phase 1 tried to measure and could not, and it requires matched-compute curves
rather than a parameter-matched arm.

### What is not being claimed

Not capability. If a structure is universal then the backbone builds it anyway, so a
module supplying it cannot buy something otherwise unreachable. The parts list is chosen
for things transformers *can* learn and shouldn't have to. Any result phrased as "the
module lets the model do X it could not do" is out of scope and probably measuring an
undertrained backbone.

### Falsifiers, declared in advance

- The backbone builds the structure anyway, at the same rate, with the module installed.
- The module is load-bearing under ablation but not under substitution -- the backbone is
  consuming its presence, not its content. See the `wrong_pointer` result in
  `scratch/modular-phase1/REPORT.md`: ablation cost 0.0592 accuracy and 0.7959 answer NLL
  while a wrong pointer cost 0.0007.
- A module trained against backbone A fails to install into backbone B without retraining
  the module itself.

## 2. Interface constraints

These are consequences of "reusable across new models regardless of size", not
preferences. Getting them wrong makes the program untestable rather than merely awkward.

**The tokenizer is the ABI.** Anything delivered at the output end is vocabulary-shaped
and ports only within a tokenizer family. This work already sits on one fixed 248,320
vocabulary. That should be a declared boundary rather than an accident, because a future
tokenizer change silently invalidates every output-end part.

**A reusable module must not expose backbone-specific residual width.** This is the rule,
and it is narrower than "no residual-space outputs" -- the problem is `d_model`, not
vectors. A module may perfectly well emit a fixed-width code `c_t` in R^256 while each
backbone owns its own projection `P_B` from R^256 to R^d_B. The module stays portable and
the integration stays backbone-specific, which is the same split as the connector
principle below. Vocabulary-space priors are the other admissible form. What is forbidden
is a module whose output shape is a property of the model it was first attached to.

That distinction matters for what comes later: a richer standard part, of the kind a
PLE-like component would need, is much easier to build against a fixed-width code than
against a vocabulary prior.

**Frozen module, disposable connector, present from step 0.** The pattern that works in
practice -- frozen CLIP, SigLIP and T5 encoders reused across many independently trained
downstream models -- is always a frozen module plus a small per-backbone connector trained
jointly with the backbone. The connector is cheap and is not the reusable artifact.

The "from step 0" half is not optional, and two independent results in this repository say
so. Phase 1b handed a decoder the exact retrieved value and moved the probability of the
required answer by -0.000110: a channel introduced after the backbone has learned the task
gets ignored. The GR retrofit reaches the same rule from the other side -- conversion is
built so the route carries the recipient's existing function from the first step, and
`scratch/gr_retrofit/REPORT.md` records why zeroing both factors of a bottleneck strands
it with no gradient path.

**The output end is easy and capped.** Phase 1c composed at the output and worked
immediately with six parameters, reaching conditional accuracy 1.0000. Phase 1e then
failed to move the *category* decision without degrading whitespace NLL by 0.144 against
a 0.02 gate. An output-side module redistributes mass inside the distribution the backbone
produced; it cannot repair a backbone that put the mass in the wrong place. Use the output
end for priors and constraints, the input end for features the model would otherwise have
to construct.

## 3. Selection criteria for a part

A candidate qualifies only if all four hold.

1. **Universal.** Evidence it forms across architectures, scales and seeds -- not a
   plausible story that it should.
2. **Exactly computable.** An algorithm produces it without learning, or a cheap learned
   code reaches it with an objective that needs no oracle.
3. **Load-bearing under substitution.** Both controls, always: ablate it, and feed it
   wrong-but-well-formed content. A part that survives substitution unchanged is an
   activation flag.
4. **Portable.** Expressible in vocabulary space or as a fixed-width code.

## 4. Candidate parts

Ordered by how ready each is to be built, not by expected value.

### Input end

| Part | Computes | Why it qualifies | Cost |
| --- | --- | --- | --- |
| **Repetition index** | Positions where the current n-gram suffix occurred before | Exactly what induction heads compute; the most replicated circuit in the literature | Hash, O(1) per token |
| **Canonicalization codes** | Identity classes under whitespace, case, unicode, and identifier renaming | Many surface forms, one code, with no semantics written down | Lexical pass |
| **Structural counters** | Bracket depth, list index, running counts | Trivial algorithmically, a known attention weakness | Stack, O(1) per token |
| **Lexical typing** | Number / identifier / string-literal / URL span membership | Stable across every corpus; already implicit in tokenizer behavior | Lexical pass |

Canonicalization is the one to build as a ladder rather than a model. Each invariance
obtainable by normalization costs no learning at all -- whitespace is trivial, and
identifier renaming collapses by replacing each identifier with the index of its first
occurrence. Learn a code only above what normalization already gives, so the contribution
of learning is the measured gap rather than the whole number.

The headroom is real and already measured. At `l6_w256`, renaming agreement is 0.5455 and
whitespace agreement 0.5078 -- near chance on a binary answer. The backbones do not have
these invariances.

### Output end

| Part | Emits | Why it qualifies | Cost |
| --- | --- | --- | --- |
| **N-gram prior** | Distribution over next token from context statistics | Already built here, and the one module in this program that unambiguously earned its place (`ab68bef`) | Hash lookup |
| **Copy distribution** | P(next = continuation of a matched prefix) | The induction head's output made explicit; pairs with the repetition index | Hash lookup |
| **Constraint mask** | Format and grammar validity over the vocabulary | Exactly computable, universally needed, and a prior is the correct delivery vehicle | Parser state |

### Degeneracies to design against

An equivalence code has two trivial solutions and the useful one sits between them. A
constant code is perfectly invariant and useless; a hash of the raw tokens is perfectly
discriminative with no invariance, and the backbone already has the tokens. Both
constraints have to bind, which makes this a contrastive objective -- positives are
transformed variants of the same prefix, negatives are other prefixes -- rather than an
invariance objective with a free parameter.

## 5. First experiment: does an installed copy module suppress induction heads?

The flagship pair is the repetition index and the copy distribution, and they admit the
cleanest possible test of the whole premise.

Induction heads form in essentially every transformer above two layers, at an
identifiable phase change during training. If a supplied part can stop a backbone from
rebuilding a structure, this is where it should be visible, because the structure is
universal, the phase change is sharply timed and early, and the algorithm is a suffix
match.

### Question

With an exact copy module present from step 0, does the induction phase change still
occur -- and is quality preserved?

### Arms

| Arm | Description |
| --- | --- |
| `plain` | Backbone alone |
| `copy` | Backbone + frozen copy module + connector, both present from step 0 |
| `scrambled` | Identical to `copy`, with the module's output permuted across positions throughout training |

**Train-time scrambling and evaluation-time substitution are different controls and must
not be conflated.** An earlier draft of this document used one word for both.

`scrambled` as an *arm* holds bandwidth, activation timing, connector capacity and
parameter count fixed while destroying the content, for the whole of training. Its
expected behavior is to converge toward `plain`, because a backbone trained against a
useless channel learns to ignore it. That makes it two controls at once: it bounds what
the connector costs, and it answers whether mere presence of a signal is doing the work.

Evaluation-time substitution is a different measurement, applied to the *trained* `copy`
model at the end: feed it permuted content it never saw during training. That measures
content dependence. A model can be strongly dependent at evaluation while a train-time
scrambled arm converges to plain -- those are consistent, and they answer different
questions.

### Measurements

- **Induction onset (signature).** Per-head prefix-matching score against training step.
  Whether the phase change occurs, when, and how strongly.
- **Copy function (behavior).** A dedicated probe, separate from the signature, because
  the two can disagree in both directions: sample a sequence of *random* tokens, repeat
  it, and score the second occurrence. Random rather than natural text, so the answer
  cannot be supplied by n-gram statistics that both the module and the backbone have
  access to by other means.
- **Quality.** Held-out NLL against matched compute, not matched parameters.
- **Cost.** Module training cost stated separately from backbone cost, with the
  amortization arithmetic shown rather than asserted.

### The endpoint matrix

The signature is not the finding. A phase change can fail to appear while the function is
present by another implementation, and heads can carry the signature without carrying the
load -- which is the whole point of
[Induction Signatures Are Not Enough](https://arxiv.org/pdf/2509.22947). So the decisive
measurements are behavioral, taken at the end, on every arm:

| | held-out NLL | copy probe |
| --- | --- | --- |
| `plain` | reference | **what the endogenous circuit achieves unaided** |
| `copy`, module enabled | reference | reference |
| `copy`, module ablated | | **the decisive cell** |
| `copy`, content permuted at evaluation | | content dependence |
| `scrambled`, module enabled | connector cost | connector cost |

`plain`'s copy probe is what makes the rest readable, and it is missing if only the three
conditions on the `copy` arm are measured. Without it there is no way to tell a backbone
that stopped building the circuit from a backbone at a scale where the circuit was never
going to work.

**The decisive cell is `copy` with the module ablated, on the copy probe.** If that
collapses toward chance while `plain` performs well, the backbone genuinely stopped
building the function -- which is the displacement claim, stated behaviorally rather than
through a signature. If it holds up, the backbone built the function anyway and the
amortization premise fails at its cleanest test case.

Ablation and substitution answer different questions and both are required. If ablation
hurts and substitution does not, the dependence is on activation rather than content, and
the result is void -- the same failure the `wrong_pointer` control caught in Phase 1.

### Predictions, registered before running

This program has been bitten repeatedly by thresholds and readings chosen after seeing
numbers. These are cheap to write down and make the outcome interpretable.

- `scrambled` converges to within noise of `plain` on held-out NLL. If it is materially
  worse, the connector is injecting harmful noise; if materially better, something is
  wrong with the comparison.
- `plain` shows a clear induction phase change at a token budget that admits one.
- `copy` reaches held-out NLL at or below `plain` under matched compute.
- The copy probe with the module enabled is near ceiling for `copy`, at whatever `plain`
  reaches for `plain`.

### Outcomes

Read the behavioral cells first and the signature second.

| Result | Reading |
| --- | --- |
| Ablated copy probe collapses, `plain`'s does not, evaluation-time substitution hurts, quality held | The premise holds at its cleanest point, stated behaviorally. Proceed to a second backbone size for the reuse claim. |
| Ablated copy probe holds up | The backbone built the function anyway. Either the module is not load-bearing or there is a compensating implementation -- which would be [Induction Signatures Are Not Enough](https://arxiv.org/pdf/2509.22947) approached from the other side, and worth knowing before building more parts. |
| Signature suppressed but ablated probe holds up | The clearest possible warning against reading signatures as function. Report it as such; it is a result about method, not about the module. |
| Signature appears but ablated probe collapses | Heads carrying the signature without the load. Also a method result, and an argument that the behavioral endpoint should be primary in everything that follows. |
| Evaluation-time substitution does not hurt | Void. The dependence is on activation, not content. |
| Quality drops | Diagnose before concluding. `scrambled` bounds the connector's cost: if `scrambled` also drops, it is the connector; if only `copy` drops, it is the module's format or the content itself. |

Every outcome is informative, which is what the Phase 1 design lacked.

### Budget discipline

Two numbers sank the earlier pilot and both have to be fixed here.

**Tokens per parameter.** Phase 1 trained 1,048,576 tokens against 4,882,944 parameters at
`l6_w256` -- 0.21 tokens per parameter. No induction phase change is visible in that
regime, and neither is anything else. Either the models shrink or the token budget grows
by orders of magnitude; the phase change is early, so small models with a real token
budget is the cheap direction.

**The control must be a control.** Phase 1's parameter-matched arm added 11.0% capacity at
`l2_w64`, 2.0% at `l4_w128` and 0.44% at `l6_w256` -- weakening exactly as the backbone
grew more capable. It was statistically indistinguishable from plain at the two larger
sizes, so the verdict rested on an inert comparison. Match compute, and report the curve
rather than a single point.

**Statistical power.** Phase 1's non-inferiority margin on accuracy was 0.01 against a
measured confidence interval of [-0.0444, +0.0599] on three seeds. That test could not
pass regardless of the truth. Either widen the margin to something the design can resolve
or raise the seed count, and decide which before running.

**Check the baseline before fixing a threshold against it.** Phase 1e was gated on a
+0.05 accuracy gain over a baseline of 0.7591 whose true value was 0.997396 -- the gap was
a formatting artifact, and every one of its 183 failures sat at one position. One
stratification table would have shown it. Stratify first, then predeclare.

## 6. Where this sits relative to the residual-capacity work

Two tracks, deliberately kept apart.

**Track A, standard parts.** A frozen reusable module plus a small per-backbone connector.
Primary question: *can supplied computation displace relearned computation?* First
experiment: copy and induction, above.

**Track B, native capacity.** A widened residual substrate with state-dependent read and
write, and later an injected conditional memory. Primary question: *does giving a backbone
several persistent residual channels improve its ability to integrate auxiliary capacity?*
This is where the earlier "capacity before injection" hypothesis belongs, and it is
already under way -- symmetric GR conversion passed its real-recipient gate bitwise
(`scratch/gr_retrofit/REPORT.md`), with the retention pilot as its next step.

**The copy experiment must not ride the GR route.** Its interface is deliberately thinner
-- exact external structure, small connector, backbone -- and its job is to find out
whether externally supplied exact computation can displace endogenous computation at all.
Adding four residual branches would confound that with a substrate change, and if the
answer is no, the substrate work would have been premature scaffolding for a result that
never came.

There is also an open question on the Track B side that the published work does not
settle. Qwen's PLE results are measured inside an architecture where every sublayer
already carries a four-branch Gated Residual, with the n-gram memory injected into those
branches through hidden-state-conditioned gating. Their ablations show the memory helps at
many layer positions and that one layer suffices, but those are conditional on the rest of
the recipe. The factorial that would settle it --

| | no PLE | PLE |
| --- | --- | --- |
| single residual | ? | ? |
| GR residual | ? | ? |

-- is not in the paper as far as we have found. So there is no clean evidence that GR is
necessary for PLE, and none that the same PLE would work as well without it. That is a
Track B question and it should not be smuggled into Track A.

If Track A succeeds, Track B gets a much cleaner motivation: outsourcing works, so the
question becomes what capacity and what interface a richer outsourced representation
needs. The fixed-width code interface in section 2 is the natural meeting point of the two
tracks.

## 7. Order of work

1. The copy-module experiment above, at a token budget where the phase change is visible.
2. If it passes, the same frozen module into a second backbone of a different size. That
   is the reuse claim, and it is the one nobody has shown.
3. Canonicalization codes, starting from the normalization floor so the learned
   contribution is separable.
4. Only then the compute-amortization curve, which needs several backbones to mean
   anything.
