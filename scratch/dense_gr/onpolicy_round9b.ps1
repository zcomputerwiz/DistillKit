# Round 9b: round 9 again with the loop detector fixed and a lighter touch. Round 9's rows
# were 72% mid-sentence decisions, most from enumerations the 16-gram rule took for loops
# ("Second crate: base 3x4, height 6. Total 12." line after line); loop_start now needs a
# whole period copied verbatim. FTPO at weight 1.0 also moved held-out NLL four times as far
# in 165 steps as round 8c's whole run, and math truncations rose: weight 0.3, stopping at
# chosen_win 0.25, held-out every 100 steps. Reuses round 9's prompts and greedy rollouts.
#   powershell -File onpolicy_round9b.ps1 -Base <checkpoint>
param([Parameter(Mandatory)][string]$Base)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$base = $Base

"=== prompts $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\prompts-r9.jsonl")) {
    & $py scratch\dense_gr\rollout_prompts.py --checkpoint $base --exclude "$C\exclude-master-v2.json" `
        --keep-effort --seed 9 --gsm8k 7000 --math 7000 --corpus "$C\thinking-code-math.jsonl=4000" `
        "$C\think-first-5m.jsonl=3000" "$C\expand-code.jsonl=3000" "$C\math-curriculum-v4.jsonl=1000" `
        --output "$C\prompts-r9.jsonl" 2>&1 | Where-Object { $_ -match 'wrote|Error|Traceback' }
}

"=== greedy rollouts $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\onpolicy-r9.jsonl")) {
    $jobs = foreach ($gpu in 0, 1) {
        if (Test-Path "$C\onpolicy-r9-g$gpu.jsonl") { continue }
        Start-Job -ArgumentList $gpu, $py, $root, $base, $C -ScriptBlock {
            param($gpu, $py, $root, $base, $C)
            $env:CUDA_VISIBLE_DEVICES = "$gpu"; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"
            $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
            Set-Location $root
            & $py "$root\scratch\dense_gr\onpolicy_rollouts.py" --checkpoint $base `
                --inputs "$C\prompts-r9.jsonl" "$C\code-prompts-r8.jsonl" --count 100000 --new 1024 `
                --shard "$gpu/2" --seed 0 --greedy --output "$C\onpolicy-r9-g$gpu.jsonl" *> "$C\onpolicy-r9-g$gpu.log"
        }
    }
    if ($jobs) { $jobs | Wait-Job | Receive-Job }
    if (-not ((Test-Path "$C\onpolicy-r9-g0.jsonl") -and (Test-Path "$C\onpolicy-r9-g1.jsonl"))) { "rollouts FAILED"; exit 1 }
    # Joined as bytes: PowerShell 5.1 would re-encode the text
    & $py -c "import sys; open(sys.argv[3], 'wb').write(open(sys.argv[1], 'rb').read() + open(sys.argv[2], 'rb').read())" `
        "$C\onpolicy-r9-g0.jsonl" "$C\onpolicy-r9-g1.jsonl" "$C\onpolicy-r9.jsonl"
    Get-Content "$C\onpolicy-r9-g0.log" | Select-String "rollouts" | Select-Object -Last 1
}

"=== FTPO rows $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\ftpo-r9b.jsonl")) {
    $env:CUDA_VISIBLE_DEVICES = "0"
    & $py scratch\dense_gr\ftpo_rows.py --reference $base --exclude "$C\exclude-master-v2.json" `
        --inputs "$C\onpolicy-r9.jsonl" "$C\onpolicy-r8.jsonl" "$C\onpolicy-r6.jsonl" "$C\onpolicy-r5.jsonl" `
        --output "$C\ftpo-r9b.jsonl" 2>&1 | Where-Object { $_ -notmatch 'Warning|warn|scaled_dot|^\s*$|Triggered|NOTE' }
    Remove-Item Env:\CUDA_VISIBLE_DEVICES
    if (-not (Test-Path "$C\ftpo-r9b.jsonl")) { "FTPO rows FAILED"; exit 1 }
}

$cache = "teacher-cache-r8-code-short"
$caches = @("..\$cache", "..\teacher-cache-curriculum-v4", "..\teacher-cache-thinking", "..\teacher-cache-think-first",
            "..\teacher-cache-expand-code", "..\teacher-cache-general-pilot")
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-master-v2.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-onpolicy-r9b.json"
"=== train $(Get-Date -Format HH:mm)"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
    "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
foreach ($c2 in $caches) { $argv.Add($c2) }
foreach ($a in @("--ce-only-caches", "..\$cache", "--lr-depth-ramp", "0.1", "1.0",
                 "--pairs", "$C\ftpo-r9b.jsonl", "--pairs-per-step", "4", "--pair-weight", "0.3",
                 "--stop-chosen-win", "0.25", "--strip-effort-nonthinking",
                 "--exclude-documents", "..\capture-data\exclude-onpolicy-r9b.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "1536", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "3072", "--accumulate", "2",
                 "--tokens", "3000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-windows", "64", "--evaluate-every", "100", "--report-every", "25",
                 "--save-every", "400", "--seed", "18",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-onpolicy-r9b",
                 "--output", "scratch\dense_gr\train-2b-onpolicy-r9b.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Where-Object { $_ -match 'learning rates|pairs:|plan:|held-out|ftpo|chosen_win|loss .*->|Traceback|Error|spilled|SystemExit' }
$r9 = "$root\scratch\dense_gr\checkpoints-2b-onpolicy-r9b\smoke-r1-1-gr-s18-csa2"
if (-not (Test-Path "$r9\config.json")) { "training produced no checkpoint; stopping"; exit 1 }

"=== blend screen against the base $(Get-Date -Format HH:mm)"
& .\scratch\dense_gr\merge_search.ps1 -Tuned $r9 -Tag r9b -Base $base
"=== done $(Get-Date -Format HH:mm)"
