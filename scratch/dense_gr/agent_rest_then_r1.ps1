# After the replay recaptures (capture_w8.ps1, pid in $args[0]): the agent traces the first
# capture did not finish (salvage_capture.py), with documents overlapped across the cards,
# then long round 1.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $C = "$D\capture-data"; $cache = "teacher-cache-agent-smol-b"
$env:PYTHONPATH = "$PWD"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:CUDA_VISIBLE_DEVICES = "0,1"
while (Get-Process -Id $args[0] -ErrorAction SilentlyContinue) { Start-Sleep 60 }
# Any replay that was stopped to pick up the batched capture is taken again (finished ones skip).
& .\scratch\dense_gr\capture_w8.ps1
if (-not (Test-Path "$D\$cache\manifest.json")) {
    if (Test-Path "$D\$cache") { Move-Item "$D\$cache" "$D\$cache.failed-$(Get-Date -Format HHmmss)" }
    "=== capture $cache $(Get-Date -Format HH:mm)"
    & $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl "$C\agent-smol-rest.jsonl" --output "$D\$cache" --sequence-length 32768 --top-k 64 `
        --shard-tokens 65536 --no-int8 --weight-only-int8 --attn-implementation flash_attention_2 `
        --prefill-chunk 8192 --overlap 2 *> "$C\capture-$cache.log"
}
if (-not (Test-Path "$D\$cache\manifest.json")) { "capture FAILED: $cache"; exit 1 }
"captured $cache $(Get-Date -Format HH:mm)"
& .\scratch\dense_gr\long_round1.ps1 *> .\scratch\dense_gr\long-r1.log
