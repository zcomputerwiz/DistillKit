# The teacher's own graded-correct, finished, loop-free thinking traces (teacher_generate.py)
# through the int8-weight teacher for KL targets: the student trains on them with cross
# entropy and KL alike, since the text is the teacher's own.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $C = "$D\capture-data"; $cache = "$D\teacher-cache-teacher-math-gen"
$env:PYTHONPATH = "$PWD"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:CUDA_VISIBLE_DEVICES = "0,1"
$env:PYTHONIOENCODING = "utf-8"
if (Test-Path "$cache\manifest.json") { "exists: $cache"; exit 0 }
if (Test-Path $cache) { "partial $cache exists; salvage or move it aside first"; exit 1 }
"=== capture $(Get-Date -Format HH:mm)"
& $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
    --input-jsonl "$C\teacher-math-gen.jsonl" --output $cache --sequence-length 8192 --top-k 64 `
    --shard-tokens 65536 --no-int8 --weight-only-int8 `
    --attn-implementation flash_attention_2 --prefill-chunk 8192 *> "$C\capture-teacher-math-gen.log"
if (Test-Path "$cache\manifest.json") { "captured $(Get-Date -Format HH:mm)" } else { "capture FAILED"; exit 1 }
