# Long-context round 3: round 2 with its fault fixed. Round 2 distilled 15M tokens of long
# code from inside a user turn, where the chat teacher expects the person to stop typing at
# every line break (top-1 0.40, a false <|im_end|> at 47%; framing_check.py), and its
# long-code NLL got worse with distance (32K: 0.527 -> 0.728) while its QA answers improved
# by 0.45 nats. Now (capture_round3.ps1):
#  - the long code documents as raw text, where the teacher is a calibrated code model
#    (top-1 0.88), trained on its KL alone: the teacher's expectations, not the repos'
#  - the QA conversations with every checked question (qa-code + qa-more, ~17.7K), scored
#    on the answers only; the document is context
# Agents, tools, replay and the long1-u50 base as round 2. Round 1 notes, still true:
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
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
$base = "$root\scratch\dense_gr\merges-long1\u50"
# The agent capture was stopped at a third and its finished shards salvaged (-a); the rest
# was captured again with documents overlapped across the cards (-b).
$agentA = "..\teacher-cache-agent-smol-a"; $agentB = "..\teacher-cache-agent-smol-b"
$code = "..\teacher-cache-r8-code-short-w8"
$qa = "..\teacher-cache-frontier-qa2"; $raw = "..\teacher-cache-frontier-code-raw"; $tools = "..\teacher-cache-frontier-tools"
$caches = @($agentA, $agentB, $qa, $raw, $tools, $code, "..\teacher-cache-curriculum-v4-w8", "..\teacher-cache-thinking-w8",
            "..\teacher-cache-think-first-w8", "..\teacher-cache-expand-code-w8", "..\teacher-cache-general-pilot-w8")
foreach ($cache in $caches) {
    if (-not (Test-Path "$D\$($cache.Substring(3))\manifest.json")) { "missing capture $cache; stopping"; exit 1 }
}
# Judged exclusions, rebuilt now so they take every Codex verdict written by the start:
# Codex's verdict decides wherever it has one (it overrules the frontier judge either way,
# and its own "wrong" excludes on its own); the frontier judge decides the rest.
function Check([string]$what) { if ($LASTEXITCODE -ne 0) { "$what failed (exit $LASTEXITCODE); stopping"; exit 1 } }
if (Test-Path "$root\scratch\dense_gr\checkpoints-2b-long-r3") { "checkpoints-2b-long-r3 exists; refusing to mix runs"; exit 1 }
$F = "$C\frontier"
& $py scratch\frontier\judge_docs.py collect --responses "$F\judge-responses.jsonl" "$F\judge-batched-responses.jsonl" `
    --second "$F\codex-crosscheck\judge-codex.jsonl" "$F\codex-missing\judge-missing.jsonl" `
    "$F\codex-crosscheck2\judge-codex.jsonl" "$F\codex-crosscheck3\judge-codex.jsonl" --output "$C\exclude-judged.json"
Check "judged exclusions"
# Plus the QA documents whose source code spells chat markup (capture_inputs.py): it
# tokenizes to real turn and tool tokens inside the user's document.
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-master-v2.json" "$C\exclude-judged.json" "$C\frontier-qa2-markup.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-long-r3.json"
Check "exclusion list"
"=== train $(Get-Date -Format HH:mm)"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
    "--inherit", "--tensor-parallel", "--embedding-on", "away", "--checkpoint-layers", "--teacher-cache")
foreach ($c2 in $caches) { $argv.Add($c2) }
# The QA answers (structural spans with their closing <|im_end|>) weigh 8 each (3x round 2's answers at half the per-token weight: ~5% of all weight),
# the budget counting weight; the raw code is KL alone. Padding to the block multiple keeps
# the final turns flooring cut (28% of tool turns) and short replay documents.
foreach ($a in @("--assistant-only-caches", $agentA, $agentB, $tools, $qa, "--kl-only-caches", $raw,
                 "--ce-only-caches", $code,
                 "--repeat", "$agentA=3", "$agentB=3", "$tools=4", "$code=2", "..\teacher-cache-curriculum-v4-w8=2",
                 "..\teacher-cache-thinking-w8=2", "..\teacher-cache-think-first-w8=2",
                 "..\teacher-cache-expand-code-w8=2", "..\teacher-cache-general-pilot-w8=2", "--pad-to-block",
                 "--answer-spans", "$C\frontier-qa2.jsonl", "--answer-weight", "8",
                 "--lr-scale", "linear_attn\.(A_log|dt_bias|in_proj_a)=0.1", "--strip-effort-nonthinking",
                 "--exclude-documents", "..\capture-data\exclude-long-r3.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "32768", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "32768", "--accumulate", "2",
                 "--tokens", "8000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-windows", "64", "--evaluate-every", "100", "--report-every", "25",
                 "--save-every", "100", "--seed", "23",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-long-r3",
                 "--output", "scratch\dense_gr\train-2b-long-r3.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Tee-Object -FilePath "$root\scratch\dense_gr\long-r3-train.log" |
    Where-Object { $_ -match 'learning rates|plan:|stripped|held-out|loss .*->|Traceback|Error|spilled|SystemExit' }
Check "training"
$tuned = "$root\scratch\dense_gr\checkpoints-2b-long-r3\smoke-r1-1-gr-s23-csa2"
if (-not (Test-Path "$tuned\config.json")) { "training produced no checkpoint; stopping"; exit 1 }

"=== long-context probe $(Get-Date -Format HH:mm)"
& $py scratch\dense_gr\long_context_probe.py --arm "long1-u50=$base" --arm "long3=$tuned" `
    --output scratch\csa2-eval\long-context-long3.json 2>&1 | Where-Object { $_ -match '^==|^  ' -and $_ -notmatch 'warn|torch.nn' }
Check "long-context probe"
"=== held-out QA answers $(Get-Date -Format HH:mm)"
& $py scratch\frontier\qa_answer_eval.py --arm "long1-u50=$base" --arm "long3=$tuned" `
    --exclude "$C\exclude-long-r3.json" --output scratch\csa2-eval\qa-answers-long3.json
Check "QA answer eval"
"=== blend screen against the base $(Get-Date -Format HH:mm)"
& .\scratch\dense_gr\merge_search.ps1 -Tuned $tuned -Tag long3 -Base $base
Check "blend screen"
"=== done $(Get-Date -Format HH:mm)"
