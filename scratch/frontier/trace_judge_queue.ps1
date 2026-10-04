# Judge the teacher's thinking traces as teacher_generate.py writes them: build requests
# for the traces not yet judged, run them, wait for more. Sleeps past the free tier's daily
# reset when the quota runs out (frontier_batch exit 3). Ends when the generator has exited
# and every eligible trace has a verdict, then collects the kept list.
#   powershell -File trace_judge_queue.ps1 -Generator <pid of teacher_generate.py>
param([int]$Generator = 0)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $F = "..\capture-data\frontier"
$env:PYTHONIOENCODING = "utf-8"
$responses = "$F\trace-judge-responses.jsonl"
if (-not (Test-Path $responses)) { New-Item -ItemType File $responses | Out-Null }
$after = 0  # passes after generation ended: traces whose verdict never parses stop the loop at 3
function NextReset { $u = (Get-Date).ToUniversalTime().Date.AddDays(1).AddMinutes(1); $u.ToLocalTime() }
while ($true) {
    $generating = $Generator -and (Get-Process -Id $Generator -ErrorAction SilentlyContinue)
    $built = & $py scratch\frontier\trace_judge.py build --traces ..\capture-data\teacher-gen-math.jsonl `
        --output "$F\trace-judge-requests.jsonl" --done $responses
    "$(Get-Date -Format 'MM-dd HH:mm') $built"
    if ($LASTEXITCODE -ne 0) { "build failed"; exit 1 }
    if (-not $generating) { $after += 1 }
    if ($built -match ' in 0 requests' -or $after -gt 3) {
        if (-not $generating) { break }
    } else {
        & $py scratch\frontier\frontier_batch.py --input "$F\trace-judge-requests.jsonl" --output $responses --workers 12 2>&1 |
            Select-Object -Last 2
        if ($LASTEXITCODE -eq 3) {
            $wake = NextReset; "quota exhausted; sleeping until $wake"
            while ((Get-Date) -lt $wake) { Start-Sleep 60 }
            continue
        }
        if ($LASTEXITCODE -ne 0) { "frontier_batch failed (exit $LASTEXITCODE)"; exit 1 }
    }
    if ($generating) { Start-Sleep 1200 }
}
& $py scratch\frontier\trace_judge.py collect --responses $responses --traces ..\capture-data\teacher-gen-math.jsonl --output ..\capture-data\teacher-gen-keep.json
"=== done $(Get-Date -Format 'MM-dd HH:mm')"
