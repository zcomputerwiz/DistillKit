# End-to-end step time of the head losses, separate (old) against shared (--shared-head-loss):
# long round 4's recipe and exclusions, same seed so both see the same batches, 40 optimizer
# steps of which the first 10 (warm-up, first-step shapes) are not timed. Each run saves its
# end-of-run checkpoint under checkpoints-bench-head-*; they are throwaway.
#   powershell -File head_bench.ps1
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
$base = "$root\scratch\dense_gr\merges-long1\u50"
$agentA = "..\teacher-cache-agent-smol-a"; $agentB = "..\teacher-cache-agent-smol-b"
$code = "..\teacher-cache-r8-code-short-w8"
$qa = "..\teacher-cache-frontier-qa2"; $raw = "..\teacher-cache-frontier-code-raw"; $tools = "..\teacher-cache-frontier-tools"
$traces = "..\teacher-cache-teacher-math-gen"
$loops = @("..\teacher-cache-onpolicy-r6-loop", "..\teacher-cache-loop-check")
$caches = @($agentA, $agentB, $qa, $raw, $tools, $code, "..\teacher-cache-curriculum-v4-w8", "..\teacher-cache-thinking-w8",
            "..\teacher-cache-think-first-w8", "..\teacher-cache-expand-code-w8", "..\teacher-cache-general-pilot-w8",
            $traces) + $loops
foreach ($variant in "old", "shared") {
    $out = "scratch\dense_gr\checkpoints-bench-head-$variant"
    if (Test-Path $out) { "$out exists; move it aside first"; exit 1 }
    "=== $variant $(Get-Date -Format HH:mm)"
    $argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
        "--inherit", "--tensor-parallel", "--embedding-on", "away", "--checkpoint-layers", "--teacher-cache")
    foreach ($c2 in $caches) { $argv.Add($c2) }
    $argv.Add("--unlikelihood-caches"); foreach ($l in $loops) { $argv.Add($l) }
    foreach ($a in @("--assistant-only-caches", $agentA, $agentB, $tools, $qa, "--kl-only-caches", $raw,
                     "--ce-only-caches", $code,
                     "--repeat", "$agentA=3", "$agentB=3", "$tools=4", "$code=2", "..\teacher-cache-curriculum-v4-w8=2",
                     "..\teacher-cache-thinking-w8=2", "..\teacher-cache-think-first-w8=2",
                     "..\teacher-cache-expand-code-w8=2", "..\teacher-cache-general-pilot-w8=2",
                     "$traces=2", "..\teacher-cache-onpolicy-r6-loop=2", "..\teacher-cache-loop-check=4", "--pad-to-block",
                     "--answer-spans", "$C\frontier-qa2.jsonl", "--answer-weight", "8",
                     "--lr-scale", "linear_attn\.(A_log|dt_bias|in_proj_a)=0.1", "--strip-effort-nonthinking",
                     "--exclude-documents", "..\capture-data\exclude-long-r4.json", "--suppress-hedges",
                     "--teacher-weight", "0.5", "--teacher-max-length", "32768", "--kl-chunk", "64",
                     "--min-answer-tokens", "2", "--micro-tokens", "32768", "--accumulate", "2",
                     "--tokens", "8000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                     "--max-steps", "40", "--benchmark-warmup-steps", "10", "--report-every", "10", "--seed", "24",
                     "--checkpoints", $out, "--output", "scratch\csa2-eval\head-bench-$variant.json")) { $argv.Add($a) }
    if ($variant -eq "shared") { $argv.Add("--shared-head-loss") }
    & $py $argv 2>&1 | Tee-Object -FilePath "$root\scratch\dense_gr\head-bench-$variant.log" |
        Where-Object { $_ -match '^step|loss .*->|Traceback|Error|spilled' }
    if ($LASTEXITCODE -ne 0) { "$variant run failed (exit $LASTEXITCODE); stopping"; exit 1 }
}
"=== done $(Get-Date -Format HH:mm)"
