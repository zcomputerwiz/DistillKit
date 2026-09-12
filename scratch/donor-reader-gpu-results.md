# Donor-reader transplant: initialisation and frozen rho sweep (GPU)

The CPU preflight is in the sibling `donor-reader-results.md`. This is the part it
deferred: four initialised arms, 34 evaluations, no training.

## How it was run

The worktree `.venv` carries `torch 2.14.0+cpu`, which is why GPU work was deferred.
The main checkout's `.venv` (`torch 2.11.0+cu128`, two visible GPUs) imports the
worktree's `distillkit` when invoked from this directory — verified by printing
`distillkit.donor_reader.__file__` — and `tests/test_donor_reader.py` passes 16/16
under it, so the arms were built and scored with worktree code on CUDA.

`main.py` gained `--initialise-only`, which saves the model straight after
`load_student_model` and returns. The doc asks the sweep to grade an arm *before* any
optimizer step, so the thing graded has to be what training would have started from:
same config parsing, same reference loading, same trainability contract. It is saved
before sharding, because a sharded save is a different code path.

Initialised trainable parameter counts match the design exactly:

| arm | streams | collapse | trainable |
| --- | ---: | --- | --- |
| c1_c1 | 2 | equal_mean | `rho.weight` 1 |
| donor_c1 | 2 | equal_mean | `rho.weight` 1 |
| c1_donor | 4 | mixer | `mixer.weight` 10,240 + `rho.weight` 1 |
| donor_donor | 4 | mixer | `mixer.weight` 10,240 + `rho.weight` 1 |

The mixer initialises to 0.25 per stream — an equal mean, not the least-squares weights
fitted in the preflight. Nothing in these results is a fitted collapse.

## rho = 0 is an exact identity

`residual_max` 0.0 and `enabled_minus_bypassed_logits_max` 0.0 on the c1_c1 arm. The
nonzero frozen reader tensors are loaded and the model is still bit-identical to stock.

384 screen documents, 147,918 assistant content tokens, stock content NLL 0.467856.
Every arm's `bypassed` mode is the same untouched stock student, which the scorer
asserts across all 34 files before comparing anything.

## The sweep

Cost is `enabled - bypassed`; negative is better. Content is the grade.

| arm | rho | content | layout | assistant |
| --- | ---: | ---: | ---: | ---: |
| c1_c1 | +0.3 | +0.000113 [−0.000018, +0.000245] | −0.000757 | +0.000065 |
| c1_c1 | +1.0 | +0.000052 [−0.000100, +0.000208] | −0.005508 | −0.000255 |
| c1_c1 | +3.0 | +0.000563 [+0.000299, +0.000837] | −0.029356 | −0.001089 |
| c1_c1 | +10.0 | +0.004712 [+0.004002, +0.005441] | −0.102205 | −0.001193 |
| donor_c1 | +1.0 | +0.000006 [−0.000132, +0.000148] | −0.000020 | +0.000004 |
| donor_c1 | +10.0 | +0.000079 [−0.000248, +0.000402] | −0.005346 | −0.000220 |
| c1_donor | +0.1 | +0.000351 [+0.000117, +0.000589] | −0.028024 | −0.001216 |
| c1_donor | +0.3 | +0.002402 [+0.001858, +0.002963] | −0.097273 | −0.003103 |
| c1_donor | +1.0 | +0.013031 [+0.011600, +0.014531] | −0.148161 | +0.004129 |
| donor_donor | +1.0 | +0.005616 [+0.004704, +0.006571] | −0.014681 | +0.004495 |
| donor_donor | +3.0 | +0.205537 [+0.189629, +0.222414] | −0.146044 | +0.186119 |

Negative rho is in the full table (`python scratch/score_transplant.py`) and reverses
the sign of the layout effect on every arm, which is what says the written direction is
meaningful rather than a norm artefact: c1_donor goes +0.021545 at rho = −0.3 against
−0.097273 at +0.3.

## rho is not comparable across arms; matched effect is

The readers differ in output norm, so one arm's rho = 0.3 writes as hard as another's
rho = 10. Comparing at matched *layout benefit* removes that:

| arm | content cost at layout −0.03 | at layout −0.10 |
| --- | ---: | ---: |
| c1_c1 | +0.000600 | +0.004586 |
| donor_c1 | **never reached** | **never reached** |
| **c1_donor** | **+0.000410** | **+0.002972** |
| donor_donor | +0.028931 | +0.135462 |

Two findings, and they are not the ones a "transplant the donor reader" hypothesis
wants.

**The donor value projection does not transfer.** `donor_c1` cannot produce even
−0.03 of layout benefit at any rho up to 10, at any price. `donor_donor` reaches the
targets only by paying 48x and 46x what `c1_donor` pays. C1's trained value projection
is the necessary factor, and it is the one thing here that was learned from this
student's own distillation.

**The donor temporal filters do transfer, modestly, but only on top of C1's value.**
Given C1's value projection, the donor's four filter banks are about 1.5x more efficient
than C1's own two (+0.000410 against +0.000600, +0.002972 against +0.004586). That is a
real interaction — `c1_c1` and `c1_donor` are the same value representation, so the gap
is the filters — but 1.5x, not the order of magnitude the raw rho columns suggest.

## The decision metric says no

**No arm at any rho improves content.** Every content figure that is negative is smaller
than 0.0001 with a confidence interval spanning zero; every content figure whose interval
excludes zero is positive. What the arms buy is layout, at the same 30:1-ish ratio the
trained C1 module already showed, and by rho = 1.0 (c1_donor) or 3.0 (donor_donor) the
overall assistant NLL is worse than stock because content dominates by token count.

This reproduces, from a frozen untrained reader with one hand-set scalar, exactly the
behaviour diagnosed in the trained C1 module: a device that predicts newlines and think
tags and charges content for it. The transplant did not change the character of the
mechanism; it relocated it.

## Note on the test suite

`tests/test_concurrent_training.py::test_trainer_short_accumulation_window_matches_serial[True]`
failed once during a full-suite run that overlapped a sweep on the same GPUs, and passes
in isolation and in its own file. Neither it nor `distillkit/concurrent_training.py` is
touched by this branch. 545 of 546 passed; that one is contention, not a regression.
