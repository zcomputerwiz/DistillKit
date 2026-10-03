# Long round 3's frontier captures through the int8-weight teacher, at up to 32K:
# the long code documents as raw text (KL only; framing_check.py) and every checked
# question per document (qa-code + qa-more) as one conversation (answers only).
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $C = "$D\capture-data"
$env:PYTHONPATH = "$PWD"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:CUDA_VISIBLE_DEVICES = "0,1"
$env:PYTHONIOENCODING = "utf-8"
foreach ($name in "frontier-code-raw", "frontier-qa2") {
    $cache = "teacher-cache-$name"
    if (Test-Path "$D\$cache\manifest.json") { "skip $cache (exists)"; continue }
    if (Test-Path "$D\$cache") { "partial $D\$cache exists; salvage or move it aside first"; exit 1 }
    if (-not (Test-Path "$C\$name.jsonl")) { "missing $C\$name.jsonl"; exit 1 }
    "=== capture $cache $(Get-Date -Format HH:mm)"
    & $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl "$C\$name.jsonl" --output "$D\$cache" --sequence-length 32768 --top-k 64 `
        --shard-tokens 65536 --no-int8 --weight-only-int8 `
        --attn-implementation flash_attention_2 --prefill-chunk 8192 *> "$C\capture-$cache.log"
    if (Test-Path "$D\$cache\manifest.json") { "captured $cache $(Get-Date -Format HH:mm)" } else { "capture FAILED: $cache"; exit 1 }
}
"=== done $(Get-Date -Format HH:mm)"
