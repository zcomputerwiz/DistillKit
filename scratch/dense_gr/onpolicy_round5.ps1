# On-policy round 5, from the 0-to-0.7 blend of the thinking pass and round 2, with the
# teacher's chat template: identical to the student's except that it injects the xhigh
# effort text in thinking mode, as every training corpus was rendered. Stripping that text
# from training was what raised held-out code NLL in the rounds (ablate_continuation.ps1:
# plain 0.773 -> 0.776, strip 0.760 -> 0.829), so nothing is stripped: prompts carry it
# in thinking mode (rollout_prompts --keep-effort), curriculum v4 is v3 rendered at the
# template's default effort, and the student is served with the same template. Rollouts
# are sorted as in round 2 (clean -> CE + KL, looping -> unlikelihood), then the result
# is blended back toward the base and screened.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $code = "$root\scratch\downstream\code_bench"; $math = "$root\scratch\downstream\math_bench"
$C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$think = "$root\scratch\dense_gr\merges-r2\ramp0-70-tt"

"=== curriculum v4 $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\math-curriculum-v4.jsonl")) {
    & $py scratch\dense_gr\math_curriculum.py --tokens 1500000 --reasoning-effort xhigh --output "$C\math-curriculum-v4.jsonl" 2>$null
}

"=== prompts $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\prompts-r5.jsonl")) {
    & $py scratch\dense_gr\rollout_prompts.py --checkpoint $think --exclude "$C\exclude-broken-tools.json" `
        --keep-effort --corpus "$C\thinking-code-math.jsonl=1200" "$C\think-first-5m.jsonl=800" "$C\expand-code.jsonl=800" `
        "$C\math-curriculum-v4.jsonl=500" --output "$C\prompts-r5.jsonl" 2>&1 | Where-Object { $_ -match 'wrote|Error|Traceback' }
}

"=== rollouts $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\onpolicy-r5.jsonl")) {
    $jobs = foreach ($shard in "0", "1") {
        Start-Job -ArgumentList $shard, $py, $root, $think, $C -ScriptBlock {
            param($shard, $py, $root, $think, $C)
            $env:CUDA_VISIBLE_DEVICES = $shard; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$shard"
            $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
            Set-Location $root
            $argv = [System.Collections.Generic.List[string]]@("$root\scratch\dense_gr\onpolicy_rollouts.py",
                "--checkpoint", $think, "--inputs", "$C\prompts-r5.jsonl", "--count", "4000",
                "--new", "1024", "--shard", "$shard/2", "--seed", $shard,
                "--output", "$C\onpolicy-r5-$shard.jsonl")
            & $py $argv *> "$C\onpolicy-r5-$shard.log"
        }
    }
    $jobs | Wait-Job | Receive-Job
    # Join as bytes: PowerShell 5.1 reads these as ANSI and writes a BOM the capture rejects
    & $py -c "import sys; open(sys.argv[1], 'wb').write(b''.join(open(p, 'rb').read() for p in sys.argv[2:]))" `
        "$C\onpolicy-r5.jsonl" "$C\onpolicy-r5-0.jsonl" "$C\onpolicy-r5-1.jsonl"
    Get-Content "$C\onpolicy-r5-0.log", "$C\onpolicy-r5-1.log" | Select-String "rollouts" | Select-Object -Last 2
}

"=== classify $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\onpolicy-r5-clean.jsonl")) {
    & $py scratch\dense_gr\classify_rollouts.py "$C\onpolicy-r5.jsonl" --clean "$C\onpolicy-r5-clean.jsonl" `
        --looping "$C\onpolicy-r5-loop.jsonl" 2>$null
}

"=== capture $(Get-Date -Format HH:mm)"
$env:CUDA_VISIBLE_DEVICES = "0,1"
foreach ($spec in @("math-curriculum-v4|teacher-cache-curriculum-v4", "onpolicy-r5-clean|teacher-cache-onpolicy-r5-clean", "onpolicy-r5-loop|teacher-cache-onpolicy-r5-loop")) {
    $jsonl, $cache = $spec -split '\|'
    if (Test-Path "$D\$cache\manifest.json") { continue }
    if (Test-Path "$D\$cache") { Move-Item "$D\$cache" "$D\$cache.failed-$(Get-Date -Format HHmmss)" }
    & $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl "$C\$jsonl.jsonl" --output "$D\$cache" --sequence-length 4096 `
        --top-k 64 --shard-tokens 65536 --int8 --device-map auto *> "$C\capture-$jsonl.log"
    if (Test-Path "$D\$cache\manifest.json") { "captured $cache $(Get-Date -Format HH:mm)" } else { "capture of $cache FAILED"; exit 1 }
}
Remove-Item Env:\CUDA_VISIBLE_DEVICES

"=== train $(Get-Date -Format HH:mm)"
$caches = @("..\teacher-cache-onpolicy-r5-clean", "..\teacher-cache-onpolicy-r5-loop", "..\teacher-cache-curriculum-v4",
            "..\teacher-cache-thinking", "..\teacher-cache-think-first", "..\teacher-cache-expand-code",
            "..\teacher-cache-general-pilot")
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-broken-tools.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-onpolicy-r5.json"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $think,
    "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
foreach ($cache in $caches) { $argv.Add($cache) }
# 1536 = the rollouts' 512 prompt + 1024 new tokens, so a clean rollout keeps its ending
foreach ($a in @("--unlikelihood-caches", "..\teacher-cache-onpolicy-r5-loop",
                 "--exclude-documents", "..\capture-data\exclude-onpolicy-r5.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "1536", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "3072", "--accumulate", "2",
                 "--tokens", "3000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-every", "400", "--evaluate-windows", "64", "--report-every", "50",
                 "--save-every", "400", "--seed", "11",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-onpolicy-r5",
                 "--output", "scratch\dense_gr\train-2b-onpolicy-r5.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Where-Object { $_ -match 'on-policy|stripped|suppress|plan:|excluded|held-out|loss .*->|wrote|Traceback|Error|spilled' }
$r5 = "$root\scratch\dense_gr\checkpoints-2b-onpolicy-r5\smoke-r1-1-gr-s11-csa2"
if (-not (Test-Path "$r5\config.json")) { "training produced no checkpoint; stopping"; exit 1 }


"=== blend screen against the base $(Get-Date -Format HH:mm)"
& .\scratch\dense_gr\merge_search.ps1 -Tuned $r5 -Tag r5 -Base $think
"=== done $(Get-Date -Format HH:mm)"