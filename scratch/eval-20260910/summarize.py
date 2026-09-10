"""Render the completed screen's machine-readable paired report for review."""
import json
from pathlib import Path

root = Path(__file__).parent
report = json.loads((root / "report.json").read_text(encoding="utf-8"))
rows = report["rows"]
names = ["student-hf", "gr-stage1-1m", "ple-stage1-1m", "lr-sweep-1e3"]
metrics = [("nll", "nll"), ("mmlu", "acc"), ("arc", "acc"),
           ("arc", "acc_token_norm"), ("arc", "acc_char_norm")]


def cell(name, comparison, task, metric):
    row = next(r for r in rows if (r["checkpoint"], r["comparison"], r["task"], r["metric"])
               == (name, comparison, task, metric))
    factor, decimals = (1, 4) if task == "nll" else (100, 1)
    value = f'{factor * row["estimate"]:.{decimals}f}'
    interval = ", ".join(f"{factor * x:.{decimals}f}" for x in row["ci95"])
    return f"{value} [{interval}]"


lines = ["# Independent screen, 2026-09-10", "",
         "Fresh inference-only runs: 128 documents / 59,794 causal targets, 128 MMLU and",
         "128 ARC-Challenge questions. Text windows are at most 512 tokens. All 1,602",
         "eligible source documents are absent from both cache manifests by ID.",
         "The separately hashed confirmation split (128 items per task) remains unscored.",
         "", "Values are estimate [95% paired percentile bootstrap CI], 10,000 draws.",
         "NLL is nats/token; accuracy is percent and accuracy deltas are percentage points.",
         "Positive NLL deltas hurt; positive accuracy deltas help. MMLU normalization",
         "does not change predictions on this sample; ARC's three variants are retained.", ""]
header = ["| Checkpoint | Comparison | NLL | MMLU | ARC raw | ARC token norm | ARC char norm |",
          "|---|---|---|---|---|---|---|"]
lines += ["## Absolute arms", ""] + header
for name in names:
    modes = ["enabled", "bypassed", "pre_retrofit"]
    if name == "gr-stage1-1m":
        modes += ["full_bypass"]
    for mode in modes:
        lines.append("| " + " | ".join([name, mode] + [cell(name, mode, *m) for m in metrics]) + " |")
lines += ["", "## Paired differences", ""] + header
for name in names:
    for mode in ["enabled - bypassed", "enabled - pre_retrofit", "bypassed - pre_retrofit"]:
        lines.append("| " + " | ".join([name, mode] + [cell(name, mode, *m) for m in metrics]) + " |")
lines += ["", "## Screen verdicts", "",
          "- student-hf: indistinguishable by construction; it has no sidecar.",
          "- gr-stage1-1m: hurting text NLL; raw ARC improves against flag bypass only,",
          "  with no consistent gain across normalizations or against the stock student.",
          "- ple-stage1-1m: hurting text NLL; benchmark accuracy is indistinguishable.",
          "- lr-sweep-1e3: hurting text NLL; benchmark accuracy is indistinguishable.",
          "", "The whole-window NLL damage is dominated by prompt/template prediction.",
          "Assistant-span enabled-minus-bypassed NLL is also worse: GR +0.0224",
          "[0.0199, 0.0250], PLE +0.0306 [0.0223, 0.0383], LR +0.0705",
          "[0.0621, 0.0788]. These span checks use 121 documents with assistant tokens;",
          "seven windows never reach an assistant span. They are a diagnostic on the",
          "same forwards, not a separate confirmation test.",
          "", "## Stage-1 parity checks", "", "```json",
          json.dumps(report["stage1_self_checks"], indent=2), "```", "",
          "GR's False flag removes the table projection but retains the trained gated-residual",
          "branches. Its complete-adapter bypass and PLE's False flag are the correct",
          "frozen-backbone checks against student-hf. Training semantics were not changed.", "",
          "## GR +0.0000 root cause", "",
          "The old sketch loaded examples/_lr_sweep_base.yml, which forced variant=ple.",
          "load_student_model replaced the saved gated_residual variant. Reproducing that",
          "load discards all seven GR adapter tensors and initializes six PLE tensors; the",
          "new PLE value projection and convolution have zero norm, making the adapter inert.",
          "Thus +0.0000 measured a newly initialized PLE, not the trained GR. The fresh tiny",
          "correct-load probe verifies all seven tensors exactly and changes a logit by",
          "3.3046875. Current W_side_proj norm is 2.100469 (the task cited 2.077). See",
          "legacy-loading-audit.json and tiny-gr.json.", "",
          "## Scope and validation", "",
          "This measures next-token prediction on cache-ID-excluded conversations and zero-shot",
          "multiple-choice likelihood. It does not use teacher logits, projection losses,",
          "generation, an optimizer, or training. Whole-document NLL includes template/prompt",
          "tokens; the JSON report also breaks it down by role. It is not out-of-domain",
          "Uncheatable PPL, a full subject-macro MMLU score, proof of pretraining or benchmark",
          "decontamination, or a measurement of generated reasoning. The small screen's CIs",
          "do not account for repeated LR selection or checkpoint seed variation. A zero",
          "accuracy-delta interval from no observed flips does not establish equivalence.", "",
          "The tokenizer emits its existing regex warning. Saved tokenization is preserved",
          "for training comparability; no tokenizer or training-path fix was made.", "",
          "451 tests passed with both GPUs visible. Tiny plumbing passed before full runs.",
          "No training implementation was modified. The package evaluator is maintained here",
          "because safe checkpoint selection needs a tested loading and scoring contract.", "",
          "Rerun from the repository root:", "", "```powershell",
          "powershell -File scratch/run_independent_screen.ps1 -OutputDirectory scratch/eval-screen",
          "```", "", "Completed job durations:", ""]
for name in ["tiny-gr"] + names:
    result = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
    assert result["complete"]
    lines.append(f'- {name}: {result["elapsed_seconds"]:.1f} seconds on {result["device"]}.')
lines += ["", "Normalization disagreement counts:", "", "```json",
          json.dumps(report["normalization_disagreements"], indent=2), "```", ""]
(root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:52]))
