# The frontier jobs across OpenRouter's daily quota (1,000 free requests, reset at 00:00 UTC):
# each job resumes where it stopped; when the quota runs out (exit 3) the queue sleeps until
# just after the next reset. Stops once everything is answered or the model's access ends.
# Then verifies each job's output.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $F = "..\capture-data\frontier"
$jobs = @(@("qa-code2-requests.jsonl", "qa-code2-responses.jsonl"),
          @("tools2-requests.jsonl", "tools2-responses.jsonl"),
          @("judge-batched-requests.jsonl", "judge-batched-responses.jsonl"))
$end = Get-Date "2026-10-05 23:59"
function NextReset { $u = (Get-Date).ToUniversalTime().Date.AddDays(1).AddMinutes(1); $u.ToLocalTime() }
while ((Get-Date) -lt $end) {
    $exhausted = $false
    foreach ($job in $jobs) {
        "=== $($job[0]) $(Get-Date -Format 'MM-dd HH:mm')"
        & $py scratch\frontier\frontier_batch.py --input "$F\$($job[0])" --output "$F\$($job[1])" --workers 12 2>&1 |
            Select-Object -Last 3
        if ($LASTEXITCODE -eq 3) { $exhausted = $true; break }
    }
    if (-not $exhausted) {
        # A pass with no quota stop: retry once more for transient errors, then finish.
        $left = 0
        foreach ($job in $jobs) {
            $answered = @(Get-Content "$F\$($job[1])" | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { -not $_.error } | ForEach-Object { $_.id }) | Sort-Object -Unique
            $left += (Get-Content "$F\$($job[0])" | Measure-Object -Line).Lines - $answered.Count
        }
        if ($left -le 0) { break }
        "$left requests still unanswered; one more pass"
        continue
    }
    $wake = NextReset
    "quota exhausted; sleeping until $wake"
    while ((Get-Date) -lt $wake) { Start-Sleep 60 }
}
"=== verify $(Get-Date -Format 'MM-dd HH:mm')"
& $py scratch\frontier\qa_verify.py --docs ..\capture-data\long-docs-code.jsonl `
    --responses "$F\qa-code-responses.jsonl" "$F\qa-code2-responses.jsonl" --output "$F\qa-code.jsonl"
foreach ($pair in @(@("tools-responses.jsonl", "tools.jsonl"), @("tools2-responses.jsonl", "tools2.jsonl"))) {
    & $py scratch\frontier\tool_tasks.py verify --responses "$F\$($pair[0])" --output "$F\$($pair[1])"
}
& $py scratch\frontier\judge_docs.py collect --responses "$F\judge-responses.jsonl" "$F\judge-batched-responses.jsonl" `
    --second "$F\codex-crosscheck\judge-codex.jsonl" "$F\codex-missing\judge-missing.jsonl" "$F\codex-crosscheck2\judge-codex.jsonl" --output ..\capture-data\exclude-judged.json
"=== done $(Get-Date -Format 'MM-dd HH:mm')"
