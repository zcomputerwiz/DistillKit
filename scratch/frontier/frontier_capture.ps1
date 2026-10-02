# The verified frontier data (capture_inputs.py) through the int8-weight teacher: tool
# conversations, then the long-document questions at up to 32K.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $C = "$D\capture-data"
$env:PYTHONPATH = "$PWD"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:CUDA_VISIBLE_DEVICES = "0,1"
foreach ($job in @(@("frontier-tools", 2048), @("frontier-qa", 32768))) {
    $name, $length = $job; $cache = "teacher-cache-$name"
    if (Test-Path "$D\$cache\manifest.json") { "skip $cache (exists)"; continue }
    if (Test-Path "$D\$cache") { Move-Item "$D\$cache" "$D\$cache.failed-$(Get-Date -Format HHmmss)" }
    "=== capture $cache $(Get-Date -Format HH:mm)"
    & $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl "$C\$name.jsonl" --output "$D\$cache" --sequence-length $length --top-k 64 `
        --shard-tokens ([Math]::Max(65536, $length)) --no-int8 --weight-only-int8 `
        --attn-implementation flash_attention_2 --prefill-chunk 8192 *> "$C\capture-$cache.log"
    if (Test-Path "$D\$cache\manifest.json") { "captured $cache $(Get-Date -Format HH:mm)" } else { "capture FAILED: $cache" }
}
