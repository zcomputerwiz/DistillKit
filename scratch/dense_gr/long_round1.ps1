# Long-context round 1: the student distilled on Qwen3.8-27B's own agent traces at up to 32K
# (SmolDataEnvs: Claude Code, OpenCode, Codex, mini-swe-agent; assistant turns only) mixed
# with the replay captures, all recaptured with int8 weights and bf16 activations
# (capture_w8.ps1). Recipe from the research pass (DISTILLATION.md, 2026-10-01):
#  - one forward per document up to 32K, every layer checkpointed; no sparse stage, so the
#    CSA2 indexer keeps its selection (pass-key is already 100% to 32K) and memory is linear
#  - DeltaNet decay gates at a tenth of the rate (SpectralShift): they hold the long memory
#  - replay keeps short-context skills (ProLong keeps ~40% short data)
# Then the long-context probe, a blend screen against the base, and the full suite.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$base = "$root\scratch\dense_gr\merges-r6r8b\u50"
# The agent capture was stopped at a third and its finished shards salvaged (-a); the rest
# was captured again with documents overlapped across the cards (-b).
$agentA = "..\teacher-cache-agent-smol-a"; $agentB = "..\teacher-cache-agent-smol-b"
$code = "..\teacher-cache-r8-code-short-w8"
$caches = @($agentA, $agentB, $code, "..\teacher-cache-curriculum-v4-w8", "..\teacher-cache-thinking-w8",
            "..\teacher-cache-think-first-w8", "..\teacher-cache-expand-code-w8", "..\teacher-cache-general-pilot-w8")
foreach ($cache in $caches) {
    if (-not (Test-Path "$D\$($cache.Substring(3))\manifest.json")) { "missing capture $cache; stopping"; exit 1 }
}
# Answers both judges call wrong (frontier judge + Codex; first-judge-only where Codex has
# not looked yet), rebuilt now so it takes every Codex verdict written by the start.
$F = "$C\frontier"
& $py scratch\frontier\judge_docs.py collect --responses "$F\judge-responses.jsonl" "$F\judge-batched-responses.jsonl" `
    --second "$F\codex-crosscheck\judge-codex.jsonl" "$F\codex-missing\judge-missing.jsonl" `
    "$F\codex-crosscheck2\judge-codex.jsonl" --output "$C\exclude-judged.json"
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-master-v2.json" "$C\exclude-judged.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-long-r1.json"
"=== train $(Get-Date -Format HH:mm)"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
    "--inherit", "--tensor-parallel", "--embedding-on", "away", "--checkpoint-layers", "--teacher-cache")
foreach ($c2 in $caches) { $argv.Add($c2) }
foreach ($a in @("--assistant-only-caches", $agentA, $agentB, "--ce-only-caches", $code,
                 "--lr-scale", "linear_attn\.(A_log|dt_bias|in_proj_a)=0.1", "--strip-effort-nonthinking",
                 "--exclude-documents", "..\capture-data\exclude-long-r1.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "32768", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "32768", "--accumulate", "2",
                 "--tokens", "6000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-windows", "64", "--evaluate-every", "200", "--report-every", "25",
                 "--save-every", "200", "--seed", "21",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-long-r1",
                 "--output", "scratch\dense_gr\train-2b-long-r1.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Where-Object { $_ -match 'learning rates|plan:|stripped|held-out|loss .*->|Traceback|Error|spilled|SystemExit' }
$tuned = "$root\scratch\dense_gr\checkpoints-2b-long-r1\smoke-r1-1-gr-s21-csa2"
if (-not (Test-Path "$tuned\config.json")) { "training produced no checkpoint; stopping"; exit 1 }

"=== long-context probe $(Get-Date -Format HH:mm)"
& $py scratch\dense_gr\long_context_probe.py --arm "long1=$tuned" --output scratch\csa2-eval\long-context-long1.json 2>&1 |
    Where-Object { $_ -match '^==|^  ' -and $_ -notmatch 'warn|torch.nn' }
"=== blend screen against the base $(Get-Date -Format HH:mm)"
& .\scratch\dense_gr\merge_search.ps1 -Tuned $tuned -Tag long1 -Base $base
"=== done $(Get-Date -Format HH:mm)"
