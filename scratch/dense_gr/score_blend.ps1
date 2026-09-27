# Sandbox-score and summarize one checkpoint's generations already on disk (for a run whose
# own scoring step found nothing under its tag).
#   powershell -File score_blend.ps1 -Tag <name>
param([Parameter(Mandatory)][string]$Tag)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"
$code = "$root\scratch\downstream\code_bench"; $math = "$root\scratch\downstream\math_bench"
$env:PYTHONPATH = $root
foreach ($d in Get-ChildItem $code -Directory -Filter "$Tag-*-2k-sampled-s*") {
    $bench = if ($d.Name -match "humaneval") { "humaneval" } else { "mbpp" }
    powershell -NoProfile -ExecutionPolicy Bypass -File "$code\run_docker.ps1" $d.FullName $bench *> "$($d.FullName)\sandbox.log"
}
foreach ($bench in "humaneval", "mbpp") {
    "== thinking, sampled, $bench, 3 seeds"
    & $py "$code\sampled_compare.py" "source=$code\source-$bench-2k-sampled" "think=$code\think-$bench-2k-sampled" `
        "control=$code\control-$bench-2k-sampled" "onpolicy2=$code\onpolicy2-$bench-2k-sampled" "$Tag=$code\$Tag-$bench-2k-sampled"
}
"== looping (MBPP+ seed 0)"
& $py "$code\no_code_audit.py" "$code\think-mbpp-2k-sampled-s0" "$code\onpolicy2-mbpp-2k-sampled-s0" "$code\$Tag-mbpp-2k-sampled-s0"
& $py -c @"
import json, os
m = r'$math'
for bench in ('gsm8k', 'math500'):
    for mode in ('think-sampled', 'nothink-greedy'):
        for name in ('source', 'think', 'control', 'onpolicy2', r'$Tag'):
            p = os.path.join(m, '%s-%s-%s' % (name, bench, mode), 'results.json')
            if os.path.exists(p):
                s = json.load(open(p))['summary']
                print('%-8s %-15s %-15s accuracy %.1f%%  no answer %d  truncated %d  mean %d tok' % (bench, mode, name, 100 * s['accuracy'], s['no_answer'], s['truncated'], s['mean_tokens']))
"@
"=== done $(Get-Date -Format HH:mm)"
