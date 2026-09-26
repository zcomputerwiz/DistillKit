# The source through llama-server (two instances), then scoring, math and the arithmetic probe.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"
$code = "$root\scratch\downstream\code_bench"; $math = "$root\scratch\downstream\math_bench"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$source = "$root\..\student-2b-hf"
$think = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"

# Sampled code runs still missing, one lane per server.
$lanes = @{ "http://127.0.0.1:8081" = @("humaneval|0", "humaneval|1", "humaneval|2");
            "http://127.0.0.1:8082" = @("mbpp|1", "mbpp|2") }
$jobs = foreach ($url in $lanes.Keys) {
    Start-Job -ArgumentList $url, ($lanes[$url] -join ';'), $py, $code, $root, $source -ScriptBlock {
        param($url, $items, $py, $code, $root, $source)
        $env:PYTHONPATH = $root; Set-Location $root
        foreach ($item in ($items -split ';')) {
            $bench, $seed = $item -split '\|'
            $d = "$code\source-$bench-2k-sampled-s$seed"
            if (Test-Path "$d\completions.jsonl") { continue }
            $argv = [System.Collections.Generic.List[string]]@("$code\generate.py", "--checkpoint", $source,
                "--bench", $bench, "--max-new-tokens", "2048", "--sample", "--seed", $seed,
                "--server", $url, "--output", $d)
            & $py $argv *> "$code\source-$bench-2k-sampled-s$seed.server.log"
        }
    }
}
# Math for the thinking-pass model on the compiled loop, alongside (GPU 0 has room).
$mathjob = Start-Job -ArgumentList $py, $math, $root, $think -ScriptBlock {
    param($py, $math, $root, $think)
    $env:CUDA_VISIBLE_DEVICES = "0"; $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
    Set-Location $root
    foreach ($mode in "think-sampled", "nothink-greedy") { foreach ($bench in "gsm8k", "math500") {
        $argv = [System.Collections.Generic.List[string]]@("$math\run_math.py", "--checkpoint", $think,
            "--bench", $bench, "--compiled", "--output", "$math\think-$bench-$mode")
        if ($mode -eq "think-sampled") { $argv.Add("--sample") } else { $argv.Add("--no-thinking") }
        & $py $argv *> "$math\think-$bench-$mode.log" } }
}
$jobs | Wait-Job | Receive-Job
"=== source code generation done $(Get-Date -Format HH:mm)"

# Source math through both servers in parallel.
$mjobs = foreach ($spec in @("http://127.0.0.1:8081|think-sampled", "http://127.0.0.1:8082|nothink-greedy")) {
    Start-Job -ArgumentList $spec, $py, $math, $root, $source -ScriptBlock {
        param($spec, $py, $math, $root, $source)
        $url, $mode = $spec -split '\|'
        $env:PYTHONPATH = $root; Set-Location $root
        foreach ($bench in "gsm8k", "math500") {
            $argv = [System.Collections.Generic.List[string]]@("$math\run_math.py", "--checkpoint", $source,
                "--bench", $bench, "--server", $url, "--output", "$math\source-$bench-$mode")
            if ($mode -eq "think-sampled") { $argv.Add("--sample") } else { $argv.Add("--no-thinking") }
            & $py $argv *> "$math\source-$bench-$mode.log"
        }
    }
}
# Meanwhile the sandbox scores every sampled code run (already-scored ones are skipped).
foreach ($d in Get-ChildItem $code -Directory -Filter "*-2k-sampled-s*") {
    $bench = if ($d.Name -match "humaneval") { "humaneval" } else { "mbpp" }
    powershell -NoProfile -ExecutionPolicy Bypass -File "$code\run_docker.ps1" $d.FullName $bench *> "$($d.FullName)\sandbox.log"
}
foreach ($bench in "humaneval", "mbpp") {
    "== thinking, sampled (T 0.6, top-p 0.95, top-k 20), $bench, 3 seeds"
    & $py "$code\sampled_compare.py" "source=$code\source-$bench-2k-sampled" "think=$code\think-$bench-2k-sampled"
}
@($mjobs) + @($mathjob) | Wait-Job | Receive-Job
& $py -c @"
import json, os
from math import comb
out = r'$math'
for bench in ('gsm8k', 'math500'):
    for mode in ('think-sampled', 'nothink-greedy'):
        rows = {}
        for name in ('source', 'think'):
            path = os.path.join(out, '%s-%s-%s' % (name, bench, mode), 'results.json')
            if os.path.exists(path):
                rows[name] = json.load(open(path))
        if len(rows) < 2:
            print(bench, mode, 'missing', sorted(rows)); continue
        s, t = rows['source'], rows['think']
        a = {r['id']: r['correct'] for r in s['records']}; b = {r['id']: r['correct'] for r in t['records']}
        up = sum(b[k] and not a[k] for k in a); down = sum(a[k] and not b[k] for k in a); n = up + down
        p = min(1.0, 2 * sum(comb(n, k) for k in range(min(up, down) + 1)) / 2 ** n) if n else 1.0
        fmt = lambda r: '%.1f%% (no answer %d, truncated %d, mean %d tok)' % (100 * r['summary']['accuracy'], r['summary']['no_answer'], r['summary']['truncated'], r['summary']['mean_tokens'])
        print('%-8s %-15s source %s | think %s | +%d -%d p %.3f' % (bench, mode, fmt(s), fmt(t), up, down, p))
"@
"=== arithmetic probe"
$env:CUDA_VISIBLE_DEVICES = "0"
& $py "$math\arithmetic_probe.py" "source=$source" "control=$root\scratch\dense_gr\checkpoints-2b-borrow-control\smoke-r1-1-gr-s1-csa2" `
    "think=$think" 2>&1 | Where-Object { $_ -match '^\s*[-+*/] |op digits|overall|Error|Traceback' }
"=== done $(Get-Date -Format HH:mm)"
