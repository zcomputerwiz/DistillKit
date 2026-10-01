# Round 8c: round 8b exactly -- same base, data, pairs and seed -- with the body's learning
# rate ramped by depth (--lr-depth-ramp 0.1 1.0). Blends of round 8b showed the loop fix
# carried by the deep layers but non-thinking code needing the layers coordinated across
# depth: a ramp during training lets the shallow layers move less while every layer keeps
# adapting to the others. Screened against the same base, with round 8b's numbers beside.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$base = "$root\scratch\dense_gr\merges-r67\r7ramp0-70"

"=== capture passing code $(Get-Date -Format HH:mm)"
$cache = "teacher-cache-r8-code-short"
if (-not (Test-Path "$D\$cache\manifest.json")) {
    $env:CUDA_VISIBLE_DEVICES = "0,1"
    if (Test-Path "$D\$cache") { Move-Item "$D\$cache" "$D\$cache.failed-$(Get-Date -Format HHmmss)" }
    & $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl "$C\onpolicy-r8-code-short.jsonl" --output "$D\$cache" --sequence-length 4096 `
        --top-k 64 --shard-tokens 65536 --int8 --device-map auto *> "$C\capture-r8-code-short.log"
    Remove-Item Env:\CUDA_VISIBLE_DEVICES
    if (Test-Path "$D\$cache\manifest.json") { "captured $cache $(Get-Date -Format HH:mm)" } else { "capture FAILED"; exit 1 }
}

$caches = @("..\$cache", "..\teacher-cache-curriculum-v4", "..\teacher-cache-thinking", "..\teacher-cache-think-first",
            "..\teacher-cache-expand-code", "..\teacher-cache-general-pilot")
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-broken-tools.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-onpolicy-r8c.json"
"=== train $(Get-Date -Format HH:mm)"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
    "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
foreach ($c2 in $caches) { $argv.Add($c2) }
foreach ($a in @("--ce-only-caches", "..\$cache", "--lr-depth-ramp", "0.1", "1.0",
                 "--pairs", "$C\pairs-r8b-ref.jsonl", "--pairs-per-step", "2", "--pair-weight", "0.5",
                 "--dpo-beta", "0.1", "--pair-sft-weight", "0.2", "--strip-effort-nonthinking",
                 "--exclude-documents", "..\capture-data\exclude-onpolicy-r8c.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "1536", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "3072", "--accumulate", "2",
                 "--tokens", "3000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-windows", "64", "--evaluate-every", "400", "--report-every", "50",
                 "--save-every", "400", "--seed", "16",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-onpolicy-r8c",
                 "--output", "scratch\dense_gr\train-2b-onpolicy-r8c.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Where-Object { $_ -match 'learning rates|pairs:|stripped|plan:|held-out|loss .*->|Traceback|Error|spilled|SystemExit' }
$r8c = "$root\scratch\dense_gr\checkpoints-2b-onpolicy-r8c\smoke-r1-1-gr-s16-csa2"
if (-not (Test-Path "$r8c\config.json")) { "training produced no checkpoint; stopping"; exit 1 }

"=== blend screen against the base $(Get-Date -Format HH:mm)"
& .\scratch\dense_gr\merge_search.ps1 -Tuned $r8c -Tag r8c -Base $base
"=== done $(Get-Date -Format HH:mm)"
