# The context-KL arm (--context-kl, Codex-reviewed): round 5's recipe to step 100 of its
# 10M schedule like control_arms.ps1, plus teacher KL alone on assistant-only documents'
# context, then the role-split ledger and the screen's proxies against the same base.
#   powershell -File context_arm.ps1 -Weight 0.05 -Every 8 [-Tag context] [-Extra "--lr-depth-ramp","0.1","1.0"]
param([double]$Weight = 0.05, [int]$Every = 8, [string]$Tag = "context", [string[]]$Extra = @())
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"; $C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
function Check([string]$what) { if ($LASTEXITCODE -ne 0) { "$what failed (exit $LASTEXITCODE); stopping"; exit 1 } }
$base = "$root\scratch\dense_gr\merges-long1\u50"
$arms = "$root\scratch\dense_gr\control-arms"
$out = "$root\scratch\dense_gr\checkpoints-ctl-$Tag"
if (Test-Path $out) { "$out exists; move it aside first; stopping"; exit 1 }
$agentA = "..\teacher-cache-agent-smol-a"; $agentB = "..\teacher-cache-agent-smol-b"
$code = "..\teacher-cache-r8-code-short-w8"
$qa = "..\teacher-cache-frontier-qa2"; $raw = "..\teacher-cache-frontier-code-raw"; $tools = "..\teacher-cache-frontier-tools"
$traces = "..\teacher-cache-teacher-math-gen"
$loops = @("..\teacher-cache-onpolicy-r6-loop", "..\teacher-cache-loop-check")
$nothink = "..\teacher-cache-teacher-nothink-math"; $tcode = "..\teacher-cache-teacher-code"
$caches = @($agentA, $agentB, $qa, $raw, $tools, $code, "..\teacher-cache-curriculum-v4-w8", "..\teacher-cache-thinking-w8",
            "..\teacher-cache-think-first-w8", "..\teacher-cache-expand-code-w8", "..\teacher-cache-general-pilot-w8",
            $traces, $nothink, $tcode) + $loops
"=== train $Tag (context KL $Weight every $Every) $(Get-Date -Format HH:mm)"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
    "--inherit", "--tensor-parallel", "--embedding-on", "away", "--checkpoint-layers", "--teacher-cache")
foreach ($c2 in $caches) { $argv.Add($c2) }
$argv.Add("--unlikelihood-caches"); foreach ($l in $loops) { $argv.Add($l) }
foreach ($a in @("--assistant-only-caches", $agentA, $agentB, $tools, $qa, "--kl-only-caches", $raw,
                 "--ce-only-caches", $code,
                 "--repeat", "$agentA=3", "$agentB=3", "$tools=4", "$code=2", "..\teacher-cache-curriculum-v4-w8=2",
                 "..\teacher-cache-thinking-w8=2", "..\teacher-cache-think-first-w8=2",
                 "..\teacher-cache-expand-code-w8=2", "..\teacher-cache-general-pilot-w8=2",
                 "$traces=2", "..\teacher-cache-onpolicy-r6-loop=2", "..\teacher-cache-loop-check=4",
                 "$nothink=4", "$tcode=8", "--pad-to-block", "--shared-head-loss", "--head-chunk", "512",
                 "--answer-spans", "$C\frontier-qa2.jsonl", "--answer-weight", "8",
                 "--lr-scale", "linear_attn\.(A_log|dt_bias|in_proj_a)=0.1", "--strip-effort-nonthinking",
                 "--exclude-documents", "$C\exclude-long-r5.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "32768", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "32768", "--accumulate", "2",
                 "--tokens", "10000000", "--max-steps", "100", "--warmup", "50", "--decay-fraction", "0.5",
                 "--decay-floor", "0.05", "--evaluate-windows", "64", "--evaluate-every", "100",
                 "--report-every", "25", "--seed", "25", "--checkpoints", $out,
                 "--context-kl", "$Weight", "--context-every", "$Every",
                 "--output", "scratch\dense_gr\train-ctl-$Tag.json") + $Extra) { $argv.Add($a) }
& $py $argv 2>&1 | Tee-Object -FilePath "$root\scratch\dense_gr\ctl-$Tag-train.log" |
    Where-Object { $_ -match 'learning rates|plan:|context|held-out|loss .*->|Traceback|Error|spilled|SystemExit' }
Check "training ($Tag)"
$tuned = "$out\smoke-r1-1-gr-s25-csa2"
if (-not (Test-Path "$tuned\config.json")) { "training $Tag produced no checkpoint; stopping"; exit 1 }
"=== evaluate $Tag $(Get-Date -Format HH:mm)"
$jobs = @()
$jobs += Start-Job -ArgumentList $py, $root, $arms, $base, $Tag, $tuned -ScriptBlock {
    param($py, $root, $arms, $base, $tag, $tuned)
    $env:CUDA_VISIBLE_DEVICES = "0"; $env:PYTHONPATH = $root; $env:PYTHONIOENCODING = "utf-8"
    Set-Location $root
    & $py scratch\dense_gr\atlas.py nll --arm "base=$base" --arm "$tag=$tuned" --output-dir "$arms\ledger-$tag" *> "$arms\ledger-$tag.log"
    if ($LASTEXITCODE -ne 0) { throw "ledger failed (exit $LASTEXITCODE)" }
}
$jobs += Start-Job -ArgumentList $py, $root, $arms, $Tag, $tuned -ScriptBlock {
    param($py, $root, $arms, $tag, $tuned)
    $env:CUDA_VISIBLE_DEVICES = "1"; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu1"
    $env:PYTHONPATH = $root; $env:PYTHONIOENCODING = "utf-8"
    Set-Location $root
    & $py scratch\dense_gr\merge_proxy.py "$tag=$tuned" --output "$arms\proxy.json" *>> "$arms\proxy.log"
    if ($LASTEXITCODE -ne 0) { throw "merge_proxy failed (exit $LASTEXITCODE)" }
}
if ($jobs.Count -ne 2) { "started $($jobs.Count) of 2 jobs; stopping"; $jobs | Stop-Job; exit 1 }
$jobs | Wait-Job -Timeout 7200 | Out-Null
$jobs | Receive-Job
if ($jobs | Where-Object { $_.State -ne "Completed" }) { $jobs | Stop-Job; "evaluation failed or timed out; see $arms\*.log"; exit 1 }
"=== done $(Get-Date -Format HH:mm)"
