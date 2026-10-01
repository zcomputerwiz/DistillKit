# Captures with the int8-weight, bf16-activation teacher (--weight-only-int8): the SmolDataEnvs
# agent traces at up to 32K (prepare_agent_traces.py), then every replay capture again from
# its own token ids, since LLM.int8's targets depended on each document's length.
# Each capture is skipped when its manifest exists; a failed one is moved aside.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $C = "$D\capture-data"
$env:PYTHONPATH = "$PWD"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:CUDA_VISIBLE_DEVICES = "0,1"

function Capture([string]$source, [string]$cache, [int]$length) {
    if (Test-Path "$D\$cache\manifest.json") { "skip $cache (exists)"; return }
    if (Test-Path "$D\$cache") { Move-Item "$D\$cache" "$D\$cache.failed-$(Get-Date -Format HHmmss)" }
    "=== capture $cache $(Get-Date -Format HH:mm)"
    & $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl $source --output "$D\$cache" --sequence-length $length --top-k 64 `
        --shard-tokens ([Math]::Max(65536, $length)) --no-int8 --weight-only-int8 `
        --attn-implementation flash_attention_2 --prefill-chunk 8192 *> "$C\capture-$cache.log"
    if (Test-Path "$D\$cache\manifest.json") { "captured $cache $(Get-Date -Format HH:mm)" } else { "capture FAILED: $cache" }
}

# The agent traces were stopped at a third, salvaged and finished separately
# (salvage_capture.py, agent_rest_then_r1.ps1).
foreach ($name in "curriculum-v4", "thinking", "think-first", "expand-code", "general-pilot", "r8-code-short") {
    $source = "$C\recapture-$name.jsonl"
    if (-not (Test-Path $source)) { & $py scratch\dense_gr\export_capture_inputs.py "$D\teacher-cache-$name" $source }
    Capture $source "teacher-cache-$name-w8" 4096
}
"=== done $(Get-Date -Format HH:mm)"
