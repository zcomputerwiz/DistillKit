# framing_check.py end to end: capture the three framings through the int8-weight teacher,
# then score the teacher on the documents' own tokens in each.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $C = "$D\capture-data"; $cache = "$D\teacher-cache-framing-check"
$env:PYTHONPATH = "$PWD"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:CUDA_VISIBLE_DEVICES = "0,1"
$env:PYTHONIOENCODING = "utf-8"
if (-not (Test-Path "$cache\manifest.json")) {
    if (Test-Path $cache) { "partial $cache exists; move it aside first"; exit 1 }
    "=== capture $(Get-Date -Format HH:mm)"
    & $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl "$C\framing-check.jsonl" --output $cache --sequence-length 32768 --top-k 64 `
        --shard-tokens 65536 --no-int8 --weight-only-int8 `
        --attn-implementation flash_attention_2 --prefill-chunk 8192 *> "$C\capture-framing-check.log"
    if (-not (Test-Path "$cache\manifest.json")) { "capture FAILED; see $C\capture-framing-check.log"; exit 1 }
}
"=== score $(Get-Date -Format HH:mm)"
& $py scratch\frontier\framing_check.py score --cache $cache --output scratch\csa2-eval\framing-check.json 2>&1 |
    Where-Object { $_ -notmatch 'W1002|Redirects' }
"=== done $(Get-Date -Format HH:mm)"
