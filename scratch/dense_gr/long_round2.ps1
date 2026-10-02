# Long-context round 2: round 1 plus the verified frontier data -- long code documents with
# their checked questions (all tokens scored: long-range code modelling and the answers) and
# the tool-call conversations (assistant turns only) -- from the long1-u50 blend (round 1
# blended half-way with its base; it kept most of the long-context gain and the best math).
# Round 1 notes, still true of this recipe:
# (SmolDataEnvs: Claude Code, OpenCode, Codex, mini-swe-agent; assistant turns only) mixed
# with the replay captures, all recaptured with int8 weights and bf16 activations
# (capture_w8.ps1). Recipe from the research pass (DISTILLATION.md, 2026-10-01):
#  - one forward per document up to 32K, every layer checkpointed; no sparse stage, so the
#    CSA2 indexer keeps its selection (pass-key is already 100% to 32K) and memory is linear
#  - DeltaNet decay gates at a tenth of the rate (SpectralShift): they hold the long memory
#  - replay keeps short-context skills (ProLong keeps ~40% short data)
# Then the long-context probe, the held-out QA answers and a blend screen against the base;
# the finalists' benchmark suite is a separate, explicit step after choosing a blend.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$base = "$root\scratch\dense_gr\merges-long1\u50"
# The agent capture was stopped at a third and its finished shards salvaged (-a); the rest
# was captured again with documents overlapped across the cards (-b).
$agentA = "..\teacher-cache-agent-smol-a"; $agentB = "..\teacher-cache-agent-smol-b"
$code = "..\teacher-cache-r8-code-short-w8"
$qaA = "..\teacher-cache-frontier-qa-a"; $qaB = "..\teacher-cache-frontier-qa-b"; $tools = "..\teacher-cache-frontier-tools"
$caches = @($agentA, $agentB, $qaA, $qaB, $tools, $code, "..\teacher-cache-curriculum-v4-w8", "..\teacher-cache-thinking-w8",
            "..\teacher-cache-think-first-w8", "..\teacher-cache-expand-code-w8", "..\teacher-cache-general-pilot-w8")
foreach ($cache in $caches) {
    if (-not (Test-Path "$D\$($cache.Substring(3))\manifest.json")) { "missing capture $cache; stopping"; exit 1 }
}
# Judged exclusions, rebuilt now so they take every Codex verdict written by the start:
# Codex's verdict decides wherever it has one (it overrules the frontier judge either way,
# and its own "wrong" excludes on its own); the frontier judge decides the rest.
function Check([string]$what) { if ($LASTEXITCODE -ne 0) { "$what failed (exit $LASTEXITCODE); stopping"; exit 1 } }
if (Test-Path "$root\scratch\dense_gr\checkpoints-2b-long-r2") { "checkpoints-2b-long-r2 exists; refusing to mix runs"; exit 1 }
$F = "$C\frontier"
& $py scratch\frontier\judge_docs.py collect --responses "$F\judge-responses.jsonl" "$F\judge-batched-responses.jsonl" `
    --second "$F\codex-crosscheck\judge-codex.jsonl" "$F\codex-missing\judge-missing.jsonl" `
    "$F\codex-crosscheck2\judge-codex.jsonl" --output "$C\exclude-judged.json"
Check "judged exclusions"
# Plus the QA documents whose source code spells chat markup (capture_inputs.py): it
# tokenizes to real turn and tool tokens inside the user's document.
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-master-v2.json" "$C\exclude-judged.json" "$C\frontier-qa-markup.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-long-r2.json"
Check "exclusion list"
"=== train $(Get-Date -Format HH:mm)"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
    "--inherit", "--tensor-parallel", "--embedding-on", "away", "--checkpoint-layers", "--teacher-cache")
foreach ($c2 in $caches) { $argv.Add($c2) }
# Codex review (codex-review-r2/REVIEW.md): scored uniformly, QA would be 48% of the scored
# tokens but under 1% of it answers. The answers (structural spans from capture_inputs.py,
# with their closing <|im_end|>) weigh 16 to the document's 1, about an eighth of a QA
# document's loss and ~3.5% of all weight; the budget counts weight. With the repeats the
# plan is 24.9% agents / 28.7% QA / 1.9% tools / 44.6% replay by weight, ~20.7M forward
# tokens for the 8M (Codex's seed-22 simulation, codex-review-r2b). Padding to the block
# multiple keeps the final turns flooring cut (28% of tool turns) and short replay documents.
foreach ($a in @("--assistant-only-caches", $agentA, $agentB, $tools, "--ce-only-caches", $code,
                 "--repeat", "$agentA=3", "$agentB=3", "$tools=4", "$code=2", "..\teacher-cache-curriculum-v4-w8=2",
                 "..\teacher-cache-thinking-w8=2", "..\teacher-cache-think-first-w8=2",
                 "..\teacher-cache-expand-code-w8=2", "..\teacher-cache-general-pilot-w8=2", "--pad-to-block",
                 "--answer-spans", "$C\frontier-qa.jsonl", "--answer-weight", "16",
                 "--lr-scale", "linear_attn\.(A_log|dt_bias|in_proj_a)=0.1", "--strip-effort-nonthinking",
                 "--exclude-documents", "..\capture-data\exclude-long-r2.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "32768", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "32768", "--accumulate", "2",
                 "--tokens", "8000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-windows", "64", "--evaluate-every", "100", "--report-every", "25",
                 "--save-every", "100", "--seed", "22",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-long-r2",
                 "--output", "scratch\dense_gr\train-2b-long-r2.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Tee-Object -FilePath "$root\scratch\dense_gr\long-r2-train.log" |
    Where-Object { $_ -match 'learning rates|plan:|stripped|held-out|loss .*->|Traceback|Error|spilled|SystemExit' }
Check "training"
$tuned = "$root\scratch\dense_gr\checkpoints-2b-long-r2\smoke-r1-1-gr-s22-csa2"
if (-not (Test-Path "$tuned\config.json")) { "training produced no checkpoint; stopping"; exit 1 }

"=== long-context probe $(Get-Date -Format HH:mm)"
& $py scratch\dense_gr\long_context_probe.py --arm "long1-u50=$base" --arm "long2=$tuned" `
    --output scratch\csa2-eval\long-context-long2.json 2>&1 | Where-Object { $_ -match '^==|^  ' -and $_ -notmatch 'warn|torch.nn' }
Check "long-context probe"
"=== held-out QA answers $(Get-Date -Format HH:mm)"
& $py scratch\frontier\qa_answer_eval.py --arm "long1-u50=$base" --arm "long2=$tuned" `
    --exclude "$C\exclude-long-r2.json" --output scratch\csa2-eval\qa-answers-long2.json
Check "QA answer eval"
"=== blend screen against the base $(Get-Date -Format HH:mm)"
& .\scratch\dense_gr\merge_search.ps1 -Tuned $tuned -Tag long2 -Base $base
Check "blend screen"
"=== done $(Get-Date -Format HH:mm)"
