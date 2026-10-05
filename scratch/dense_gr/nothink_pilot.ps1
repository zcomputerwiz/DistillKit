# nothink_pilot.py end to end: the teacher's native non-thinking answers for the pilot's
# problems (llama-server), then both answer sets through the int8-weight teacher, then the
# comparison. Stops at the first failure.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $D = "D:\DeepThought\Projects\HybridModel"; $C = "$D\capture-data"
$env:PYTHONPATH = "$PWD"; $env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$gguf = "C:\Users\Owner\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\4ca720788d1e01f1bff70c033e0d0028fd02e502\Qwen3.8-27B-UD-Q8_K_XL.gguf"
"=== server $(Get-Date -Format HH:mm)"
$server = Start-Process -PassThru -WindowStyle Hidden -FilePath "$D\llama-bin\b11205\llama-server.exe" `
    -ArgumentList "-m `"$gguf`" -ngl 99 -sm tensor -fa on -c 131072 -kvu -np 32 --port 8090 --host 127.0.0.1 --no-webui" `
    -RedirectStandardError "$C\teacher-server.log"
$t = 0
do { Start-Sleep 5; $t += 5; try { $h = Invoke-RestMethod http://127.0.0.1:8090/health -ErrorAction Stop } catch { $h = $null } }
until ($h.status -eq 'ok' -or $server.HasExited -or $t -gt 400)
if ($h.status -ne 'ok') { "server did not come up"; if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }; exit 1 }
"=== native non-thinking answers $(Get-Date -Format HH:mm)"
& $py scratch\dense_gr\teacher_generate.py --nothink --problems "$C\nothink-pilot-problems.jsonl" --workers 32 `
    --max-tokens 4096 --output "$C\nothink-pilot-native.jsonl" 2>&1 | Where-Object { $_ -match 'tok/s|traces|correct|Traceback|Error' }
$generated = $LASTEXITCODE
Stop-Process -Id $server.Id -Force; Start-Sleep 10
if ($generated -ne 0) { "generation failed (exit $generated)"; exit 1 }
"=== inputs $(Get-Date -Format HH:mm)"
& $py scratch\dense_gr\nothink_pilot.py inputs
if ($LASTEXITCODE -ne 0) { "inputs failed"; exit 1 }
"=== capture $(Get-Date -Format HH:mm)"
$cache = "$D\teacher-cache-nothink-pilot"
if (Test-Path $cache) { "$cache exists; move it aside first"; exit 1 }
$env:CUDA_VISIBLE_DEVICES = "0,1"
& $py -m distillkit.sample_transformers --model "$D\teacher-hf" --tokenizer-json "$D\teacher-hf\tokenizer.json" `
    --input-jsonl "$C\nothink-pilot.jsonl" --output $cache --sequence-length 8192 --top-k 64 --shard-tokens 65536 `
    --no-int8 --weight-only-int8 --attn-implementation flash_attention_2 --prefill-chunk 8192 *> "$C\capture-nothink-pilot.log"
if (-not (Test-Path "$cache\manifest.json")) { "capture failed; see $C\capture-nothink-pilot.log"; exit 1 }
"=== score $(Get-Date -Format HH:mm)"
& $py scratch\dense_gr\nothink_pilot.py score
"=== done $(Get-Date -Format HH:mm)"
