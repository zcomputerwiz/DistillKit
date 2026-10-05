# The cheap causal controls the targeted-training review put first (codex-review-targeted):
# round 5's recipe, seed and 10M-token schedule, each stopped at step 100 (~2.3M targets,
# the first state round 5 itself saved), so every arm is the same prefix of the same
# schedule as round 5 (A, exported from its own step-100 state):
#   nonew  no new teacher data (teacher-code, teacher-nothink-math left out): what the new
#          sets themselves cost or buy
#   ramp   the body's learning rate ramped by depth, 0.1 at the embedding to 1.0 at the top
#          (--lr-depth-ramp, round 8c's lever, unused in the long rounds)
#   flat   the ramp's mean rate everywhere (x0.55): depth allocation against smaller steps
# Then every arm and the base on the role-split ledger (atlas.py nll) and the blend
# screen's proxies (merge_proxy.py), one card each.
#   powershell -File control_arms.ps1
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"; $C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
function Check([string]$what) { if ($LASTEXITCODE -ne 0) { "$what failed (exit $LASTEXITCODE); stopping"; exit 1 } }
$base = "$root\scratch\dense_gr\merges-long1\u50"
$r5 = "$root\scratch\dense_gr\checkpoints-2b-long-r5"
$arms = "$root\scratch\dense_gr\control-arms"
New-Item -ItemType Directory -Force $arms | Out-Null
if (-not (Test-Path "$arms\long5-step100\config.json")) {
    "=== export round 5 at step 100 $(Get-Date -Format HH:mm)"
    & $py scratch\dense_gr\export_state.py --state "$r5\state-step-00000100" --like "$r5\smoke-r1-1-gr-s25-csa2" `
        --output "$arms\long5-step100" *> "$arms\export.log"
    Check "export"
}
$agentA = "..\teacher-cache-agent-smol-a"; $agentB = "..\teacher-cache-agent-smol-b"
$code = "..\teacher-cache-r8-code-short-w8"
$qa = "..\teacher-cache-frontier-qa2"; $raw = "..\teacher-cache-frontier-code-raw"; $tools = "..\teacher-cache-frontier-tools"
$traces = "..\teacher-cache-teacher-math-gen"
$loops = @("..\teacher-cache-onpolicy-r6-loop", "..\teacher-cache-loop-check")
$nothink = "..\teacher-cache-teacher-nothink-math"; $tcode = "..\teacher-cache-teacher-code"
$old = @($agentA, $agentB, $qa, $raw, $tools, $code, "..\teacher-cache-curriculum-v4-w8", "..\teacher-cache-thinking-w8",
         "..\teacher-cache-think-first-w8", "..\teacher-cache-expand-code-w8", "..\teacher-cache-general-pilot-w8", $traces) + $loops

function Arm([string]$tag, [bool]$nonew, [string[]]$extra) {
    $out = "$root\scratch\dense_gr\checkpoints-ctl-$tag"
    if (Test-Path "$out\smoke-r1-1-gr-s25-csa2\config.json") { "skip $tag (trained)"; return }
    if (Test-Path $out) { "partial $out exists; move it aside first; stopping"; exit 1 }
    $caches = if ($nonew) { $old } else { $old + @($nothink, $tcode) }
    $exclude = "$C\exclude-long-r5.json"
    if ($nonew) {
        # The round's list trimmed to the captures this arm reads (the trainer refuses ids
        # in none of them).
        $exclude = "$C\exclude-ctl-nonew.json"
        & $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-long-r5.json" --caches ($caches | ForEach-Object { $_ }) --output $exclude
        Check "exclusion list ($tag)"
    }
    "=== train $tag $(Get-Date -Format HH:mm)"
    $argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
        "--inherit", "--tensor-parallel", "--embedding-on", "away", "--checkpoint-layers", "--teacher-cache")
    foreach ($c2 in $caches) { $argv.Add($c2) }
    $argv.Add("--unlikelihood-caches"); foreach ($l in $loops) { $argv.Add($l) }
    $repeat = @("$agentA=3", "$agentB=3", "$tools=4", "$code=2", "..\teacher-cache-curriculum-v4-w8=2",
                "..\teacher-cache-thinking-w8=2", "..\teacher-cache-think-first-w8=2",
                "..\teacher-cache-expand-code-w8=2", "..\teacher-cache-general-pilot-w8=2",
                "$traces=2", "..\teacher-cache-onpolicy-r6-loop=2", "..\teacher-cache-loop-check=4")
    if (-not $nonew) { $repeat += @("$nothink=4", "$tcode=8") }
    foreach ($a in @("--assistant-only-caches", $agentA, $agentB, $tools, $qa, "--kl-only-caches", $raw,
                     "--ce-only-caches", $code, "--repeat") + $repeat + @(
                     "--pad-to-block", "--shared-head-loss", "--head-chunk", "512",
                     "--answer-spans", "$C\frontier-qa2.jsonl", "--answer-weight", "8",
                     "--lr-scale", "linear_attn\.(A_log|dt_bias|in_proj_a)=0.1", "--strip-effort-nonthinking",
                     "--exclude-documents", $exclude, "--suppress-hedges",
                     "--teacher-weight", "0.5", "--teacher-max-length", "32768", "--kl-chunk", "64",
                     "--min-answer-tokens", "2", "--micro-tokens", "32768", "--accumulate", "2",
                     "--tokens", "10000000", "--max-steps", "100", "--warmup", "50", "--decay-fraction", "0.5",
                     "--decay-floor", "0.05", "--evaluate-windows", "64", "--evaluate-every", "100",
                     "--report-every", "25", "--seed", "25", "--checkpoints", $out,
                     "--output", "scratch\dense_gr\train-ctl-$tag.json") + $extra) { $argv.Add($a) }
    & $py $argv 2>&1 | Tee-Object -FilePath "$root\scratch\dense_gr\ctl-$tag-train.log" |
        Where-Object { $_ -match 'learning rates|plan:|prefix:|held-out|loss .*->|Traceback|Error|spilled|SystemExit' }
    Check "training ($tag)"
    if (-not (Test-Path "$out\smoke-r1-1-gr-s25-csa2\config.json")) { "training $tag produced no checkpoint; stopping"; exit 1 }
}
Arm "nonew" $true @()
Arm "ramp" $false @("--lr-depth-ramp", "0.1", "1.0")
Arm "flat" $false @("--lr-depth-ramp", "0.55", "0.55")

"=== evaluate $(Get-Date -Format HH:mm)"
$named = @(@("long5-step100", "$arms\long5-step100"), @("nonew", "$root\scratch\dense_gr\checkpoints-ctl-nonew\smoke-r1-1-gr-s25-csa2"),
           @("ramp", "$root\scratch\dense_gr\checkpoints-ctl-ramp\smoke-r1-1-gr-s25-csa2"),
           @("flat", "$root\scratch\dense_gr\checkpoints-ctl-flat\smoke-r1-1-gr-s25-csa2"))
$ledger = @("--arm", "base=$base") + ($named | ForEach-Object { @("--arm", "$($_[0])=$($_[1])") })
$proxy = @("base=$base") + ($named | ForEach-Object { "$($_[0])=$($_[1])" })
$jobs = @(
    Start-Job -ArgumentList $py, $root, $arms, (($ledger) -join ";") -ScriptBlock {
        param($py, $root, $arms, $ledger)
        $env:CUDA_VISIBLE_DEVICES = "0"; $env:PYTHONPATH = $root; $env:PYTHONIOENCODING = "utf-8"
        Set-Location $root
        & $py scratch\dense_gr\atlas.py nll ($ledger -split ";") --output-dir "$arms\ledger" *> "$arms\ledger.log"
        if ($LASTEXITCODE -ne 0) { throw "ledger failed (exit $LASTEXITCODE)" }
    },
    Start-Job -ArgumentList $py, $root, $arms, (($proxy) -join ";") -ScriptBlock {
        param($py, $root, $arms, $proxy)
        $env:CUDA_VISIBLE_DEVICES = "1"; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu1"
        $env:PYTHONPATH = $root; $env:PYTHONIOENCODING = "utf-8"
        Set-Location $root
        foreach ($arm in $proxy -split ";") {
            & $py scratch\dense_gr\merge_proxy.py $arm --output "$arms\proxy.json" *>> "$arms\proxy.log"
            if ($LASTEXITCODE -ne 0) { throw "merge_proxy failed for $arm (exit $LASTEXITCODE)" }
        }
    })
$jobs | Wait-Job -Timeout 10800 | Out-Null
$jobs | Receive-Job
if ($jobs | Where-Object { $_.State -ne "Completed" }) { $jobs | Stop-Job; "evaluation failed or timed out; see $arms\*.log"; exit 1 }
"=== done $(Get-Date -Format HH:mm)"
