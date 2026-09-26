# GSM8K and MATH-500 for the source and the thinking-pass model, after the sampled code runs.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $out = "$root\scratch\downstream\math_bench"
$log = "$root\scratch\downstream\code_bench\sampled-eval.log"
while (-not (Select-String -Path $log -Pattern 'generation done' -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 30 }
$think = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"
$source = "$root\..\student-2b-hf"
$jobs = foreach ($spec in @("0|think|$think|compiled", "1|source|$source|hf")) {
    Start-Job -ArgumentList $spec, $py, $out, $root -ScriptBlock {
        param($spec, $py, $out, $root)
        $gpu, $name, $ck, $how = $spec -split '\|'
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        foreach ($mode in "think-sampled", "nothink-greedy") {
            foreach ($bench in "gsm8k", "math500") {
                $argv = [System.Collections.Generic.List[string]]@("$out\run_math.py", "--checkpoint", $ck,
                    "--bench", $bench, "--output", "$out\$name-$bench-$mode")
                if ($how -eq "compiled") { $argv.Add("--compiled") }
                if ($mode -eq "think-sampled") { $argv.Add("--sample") } else { $argv.Add("--no-thinking") }
                & $py $argv *> "$out\$name-$bench-$mode.log"
            }
        }
    }
}
$jobs | Wait-Job | Receive-Job
& $py -c @"
import json, os
out = r'$out'
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
        up = sum(b[k] and not a[k] for k in a); down = sum(a[k] and not b[k] for k in a)
        from math import comb
        n = up + down; p = min(1.0, 2 * sum(comb(n, k) for k in range(min(up, down) + 1)) / 2 ** n) if n else 1.0
        fmt = lambda r: '%.1f%% (no answer %d, truncated %d, mean %d tok)' % (100 * r['summary']['accuracy'], r['summary']['no_answer'], r['summary']['truncated'], r['summary']['mean_tokens'])
        print('%-8s %-15s source %s | think %s | +%d -%d p %.3f' % (bench, mode, fmt(s), fmt(t), up, down, p))
"@
"=== done $(Get-Date -Format HH:mm)"
