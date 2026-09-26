Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$id = (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'no_thinking.ps1' }).ProcessId
if ($id) { while (Get-Process -Id $id -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 } }
$env:PYTHONPATH = "$PWD"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:CUDA_VISIBLE_DEVICES = "0,1"
$T = "D:\DeepThought\Projects\HybridModel\teacher-hf"; $D = "D:\DeepThought\Projects\HybridModel"
"=== capture $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$D\teacher-cache-think-first\manifest.json")) {
    & .venv\Scripts\python.exe -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl "$D\capture-data\think-first-5m.jsonl" --output "$D\teacher-cache-think-first" `
        --sequence-length 4096 --top-k 64 --shard-tokens 65536 --int8 --device-map auto *> "$D\capture-data\capture-think-first.log"
}
"=== train both $(Get-Date -Format HH:mm)"
$control = "scratch\dense_gr\checkpoints-2b-borrow-control\smoke-r1-1-gr-s1-csa2"
& .venv\Scripts\python.exe scratch\dense_gr\smoke_train.py --init-from $control --inherit --sparse-stage `
    --tensor-parallel --embedding-on away --teacher-cache ..\teacher-cache-5m ..\teacher-cache-expand-code `
    ..\teacher-cache-expand-chat ..\teacher-cache-general-pilot ..\teacher-cache-general-scale `
    ..\teacher-cache-dehedged ..\teacher-cache-think-first `
    --exclude-documents ..\capture-data\exclude-for-think-first.json --suppress-hedges `
    --teacher-weight 0.5 --teacher-max-length 1024 --kl-chunk 64 --min-answer-tokens 2 --micro-tokens 3072 `
    --accumulate 2 --tokens 3000000 --warmup 50 --decay-fraction 0.3 --decay-floor 0.1 --evaluate-every 400 `
    --evaluate-windows 64 --report-every 50 --save-every 400 --seed 3 `
    --checkpoints scratch\dense_gr\checkpoints-2b-hedge-both --output scratch\dense_gr\train-2b-hedge-both.json 2>&1 |
    Where-Object { $_ -match 'suppress|plan:|held-out|loss .*->|wrote|Traceback|Error|spilled' }
"=== done $(Get-Date -Format HH:mm)"