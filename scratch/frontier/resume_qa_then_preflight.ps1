# After the 2026-10-02 interruption: the frontier QA documents the first capture did not
# finish (salvage_capture.py -> frontier-qa-rest.jsonl), then the split-prefill preflight.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $C = "$D\capture-data"; $cache = "teacher-cache-frontier-qa-b"
$env:PYTHONPATH = "$PWD"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:CUDA_VISIBLE_DEVICES = "0,1"
if (-not (Test-Path "$D\$cache\manifest.json")) {
    if (Test-Path "$D\$cache") { Move-Item "$D\$cache" "$D\$cache.failed-$(Get-Date -Format HHmmss)" }
    "=== capture $cache $(Get-Date -Format HH:mm)"
    & $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl "$C\frontier-qa-rest.jsonl" --output "$D\$cache" --sequence-length 32768 --top-k 64 `
        --shard-tokens 65536 --no-int8 --weight-only-int8 --attn-implementation flash_attention_2 `
        --prefill-chunk 8192 *> "$C\capture-$cache.log"
}
if (Test-Path "$D\$cache\manifest.json") { "captured $cache $(Get-Date -Format HH:mm)" } else { "capture FAILED: $cache" }
"=== split preflight $(Get-Date -Format HH:mm)"
& $py scratch\dense_gr\split_profile.py --checkpoint scratch\dense_gr\merges-long1\u50 --splits 11 15 19 `
    --tails 1 128 512 --output scratch\csa2-eval\split-profile-u50.json *> scratch\dense_gr\split-profile-u50.log
& $py scratch\dense_gr\split_profile.py --checkpoint scratch\dense_gr\merges-long1\u50 --splits 15 19 `
    --tails 1 128 512 --anchor-every 256 --output scratch\csa2-eval\split-profile-u50-anchors.json *> scratch\dense_gr\split-profile-u50-anchors.log
"=== done $(Get-Date -Format HH:mm)"
