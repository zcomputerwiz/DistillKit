# Long round 5's new data, end to end, stopping at the first failure:
#  1. on one llama-server: the teacher's own non-thinking answers to 2,500 kept math problems
#     (teacher_generate.py --nothink; nothink_pilot.py chose these over its thinking-mode
#     answers with the thought dropped), then its code for 1,500 KodCode problems
#  2. the code's pytest suites in the sandbox (verify_code.py); capture input from both
#  3. both captured through the int8-weight teacher
#   powershell -File round5_data.ps1
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $D = "D:\DeepThought\Projects\HybridModel"; $C = "$D\capture-data"
$env:PYTHONPATH = "$PWD"; $env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$gguf = "C:\Users\Owner\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\4ca720788d1e01f1bff70c033e0d0028fd02e502\Qwen3.8-27B-UD-Q8_K_XL.gguf"
if (-not (Test-Path "$C\teacher-code-verified.jsonl")) {
    "=== server $(Get-Date -Format HH:mm)"
    $server = Start-Process -PassThru -WindowStyle Hidden -FilePath "$D\llama-bin\b11205\llama-server.exe" `
        -ArgumentList "-m `"$gguf`" -ngl 99 -sm tensor -fa on -c 131072 -kvu -np 32 --port 8090 --host 127.0.0.1 --no-webui" `
        -RedirectStandardError "$C\teacher-server.log"
    $t = 0
    do { Start-Sleep 5; $t += 5; try { $h = Invoke-RestMethod http://127.0.0.1:8090/health -ErrorAction Stop } catch { $h = $null } }
    until ($h.status -eq 'ok' -or $server.HasExited -or $t -gt 400)
    if ($h.status -ne 'ok') { "server did not come up"; if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }; exit 1 }
    "=== non-thinking math $(Get-Date -Format HH:mm)"
    & $py scratch\dense_gr\teacher_generate.py --nothink --problems "$C\teacher-nothink-problems.jsonl" --workers 32 `
        --max-tokens 4096 --output "$C\teacher-gen-nothink.jsonl" 2>&1 | Where-Object { $_ -match 'tok/s|traces|correct|skipped|Traceback|Error' } | Select-Object -Last 8
    if ($LASTEXITCODE -ne 0) { "non-thinking generation failed (exit $LASTEXITCODE)"; Stop-Process -Id $server.Id -Force; exit 1 }
    "=== teacher code $(Get-Date -Format HH:mm)"
    & $py scratch\dense_gr\teacher_code.py generate --prompts "$C\code-prompts-teacher.jsonl" `
        --output "$C\teacher-code-rollouts.jsonl" 2>&1 | Where-Object { $_ -match 'tok/s|skipped|Traceback|Error' } | Select-Object -Last 5
    $generated = $LASTEXITCODE
    Stop-Process -Id $server.Id -Force; Start-Sleep 10
    if ($generated -ne 0) { "code generation failed (exit $generated)"; exit 1 }
    "=== sandbox $(Get-Date -Format HH:mm)"
    & $py scratch\downstream\code_bench\verify_code.py --prompts "$C\code-prompts-teacher.jsonl" `
        --rollouts "$C\teacher-code-rollouts.jsonl" --output "$C\teacher-code-verified.jsonl"
    if ($LASTEXITCODE -ne 0) { "verification failed (exit $LASTEXITCODE)"; exit 1 }
}
& $py scratch\dense_gr\teacher_code.py inputs
if ($LASTEXITCODE -ne 0) { "code inputs failed"; exit 1 }
& $py scratch\dense_gr\nothink_inputs.py inputs --output "$C\teacher-nothink-math.jsonl"
if ($LASTEXITCODE -ne 0) { "non-thinking inputs failed"; exit 1 }
$env:CUDA_VISIBLE_DEVICES = "0,1"
$jobs = @(@("teacher-code", "teacher-code.jsonl"), @("teacher-nothink-math", "teacher-nothink-math.jsonl"))
foreach ($job in $jobs) {
    $cache = "$D\teacher-cache-$($job[0])"
    if (Test-Path "$cache\manifest.json") { "skip $cache (exists)"; continue }
    if (Test-Path $cache) { "partial $cache exists; move it aside first"; exit 1 }
    "=== capture $($job[0]) $(Get-Date -Format HH:mm)"
    & $py -m distillkit.sample_transformers --model "$D\teacher-hf" --tokenizer-json "$D\teacher-hf\tokenizer.json" `
        --input-jsonl "$C\$($job[1])" --output $cache --sequence-length 8192 --top-k 64 --shard-tokens 65536 `
        --no-int8 --weight-only-int8 --attn-implementation flash_attention_2 --prefill-chunk 8192 *> "$C\capture-$($job[0]).log"
    if (-not (Test-Path "$cache\manifest.json")) { "capture failed: $($job[0])"; exit 1 }
}
"=== done $(Get-Date -Format HH:mm)"
