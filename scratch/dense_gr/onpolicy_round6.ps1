# On-policy round 6: brevity. The student thinks at 2-2.4x the source's length whatever
# the reasoning effort (effort_ab.ps1), so the length is learned -- plausibly from KL toward
# a teacher that puts little mass on closing a thought at any one position. From the
# round-5b blend: four rollouts per served-format prompt (round 5's prompts), the shortest
# acceptable one per prompt (correct where checkable, else finished and loop-free) trained
# on cross entropy alone, looping ones under unlikelihood, replay with the effort text
# stripped only where replies do not think; then blended back toward the base and screened.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$base = "$root\scratch\dense_gr\merges-r5b\ramp0-70"

"=== rollouts, 4 per prompt $(Get-Date -Format HH:mm)"
foreach ($pass in @(@("0", "1"), @("2", "3"))) {
    $jobs = foreach ($gpu in "0", "1") {
        $seed = $pass[[int]$gpu]
        if (Test-Path "$C\onpolicy-r6-s$seed.jsonl") { continue }
        Start-Job -ArgumentList $gpu, $seed, $py, $root, $base, $C -ScriptBlock {
            param($gpu, $seed, $py, $root, $base, $C)
            $env:CUDA_VISIBLE_DEVICES = $gpu; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"
            $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
            Set-Location $root
            $argv = [System.Collections.Generic.List[string]]@("$root\scratch\dense_gr\onpolicy_rollouts.py",
                "--checkpoint", $base, "--inputs", "$C\prompts-r5.jsonl", "--count", "100000",
                "--new", "1024", "--seed", $seed, "--output", "$C\onpolicy-r6-s$seed.jsonl")
            & $py $argv *> "$C\onpolicy-r6-s$seed.log"
        }
    }
    if ($jobs) { $jobs | Wait-Job | Receive-Job }
    Get-Content "$C\onpolicy-r6-s$($pass[0]).log", "$C\onpolicy-r6-s$($pass[1]).log" | Select-String "rollouts" | Select-Object -Last 2
}
if (-not (Test-Path "$C\onpolicy-r6.jsonl")) {
    # Join as bytes: PowerShell 5.1 reads these as ANSI and writes a BOM the capture rejects
    & $py -c "import sys; open(sys.argv[1], 'wb').write(b''.join(open(p, 'rb').read() for p in sys.argv[2:]))" `
        "$C\onpolicy-r6.jsonl" "$C\onpolicy-r6-s0.jsonl" "$C\onpolicy-r6-s1.jsonl" "$C\onpolicy-r6-s2.jsonl" "$C\onpolicy-r6-s3.jsonl"
}

"=== select $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\onpolicy-r6-short.jsonl")) {
    & $py scratch\dense_gr\select_shortest.py "$C\onpolicy-r6.jsonl" --chosen "$C\onpolicy-r6-short.jsonl" `
        --looping "$C\onpolicy-r6-loop.jsonl" 2>$null
}

"=== capture $(Get-Date -Format HH:mm)"
$env:CUDA_VISIBLE_DEVICES = "0,1"
foreach ($spec in @("onpolicy-r6-short|teacher-cache-onpolicy-r6-short", "onpolicy-r6-loop|teacher-cache-onpolicy-r6-loop")) {
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
$caches = @("..\teacher-cache-onpolicy-r6-short", "..\teacher-cache-onpolicy-r6-loop", "..\teacher-cache-curriculum-v4",
            "..\teacher-cache-thinking", "..\teacher-cache-think-first", "..\teacher-cache-expand-code",
            "..\teacher-cache-general-pilot")
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-broken-tools.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-onpolicy-r6.json"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
    "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
foreach ($cache in $caches) { $argv.Add($cache) }
foreach ($a in @("--ce-only-caches", "..\teacher-cache-onpolicy-r6-short",
                 "--unlikelihood-caches", "..\teacher-cache-onpolicy-r6-loop", "--strip-effort-nonthinking",
                 "--exclude-documents", "..\capture-data\exclude-onpolicy-r6.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "1536", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "3072", "--accumulate", "2",
                 "--tokens", "3000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-every", "400", "--evaluate-windows", "64", "--report-every", "50",
                 "--save-every", "400", "--seed", "13",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-onpolicy-r6",
                 "--output", "scratch\dense_gr\train-2b-onpolicy-r6.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Where-Object { $_ -match 'on-policy|stripped|suppress|plan:|excluded|held-out|loss .*->|wrote|Traceback|Error|spilled' }
$r6 = "$root\scratch\dense_gr\checkpoints-2b-onpolicy-r6\smoke-r1-1-gr-s13-csa2"
if (-not (Test-Path "$r6\config.json")) { "training produced no checkpoint; stopping"; exit 1 }

"=== blend screen against the base $(Get-Date -Format HH:mm)"
& .\scratch\dense_gr\merge_search.ps1 -Tuned $r6 -Tag r6 -Base $base
"=== done $(Get-Date -Format HH:mm)"
