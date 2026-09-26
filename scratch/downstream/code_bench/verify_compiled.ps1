# CompiledGreedy against HF generate on the same prompts: agreement and speed.
param([int]$WaitFor = 0)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$env:PYTHONPATH = "$PWD"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$py = ".venv\Scripts\python.exe"
if ($WaitFor) { while (Get-Process -Id $WaitFor -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 } }
$ck = "scratch\dense_gr\checkpoints-2b-finish\smoke-r1-1-gr-s2-csa2"
$out = "scratch\downstream\code_bench"
foreach ($mode in @("eager", "compiled")) {
    Remove-Item -Recurse -Force "$out\verify-$mode" -ErrorAction SilentlyContinue
    $flag = if ($mode -eq "compiled") { @("--compiled") } else { @() }
    & $py "$out\generate.py" --checkpoint $ck --bench humaneval --limit 32 --batch-size 32 `
        --max-new-tokens 256 --output "$out\verify-$mode" @flag 2>&1 | Select-String 'wrote|Traceback|Error'
}
& $py -c @"
import json
def load(m):
    rows = [json.loads(l) for l in open(r'$out\verify-%s\completions.jsonl' % m, encoding='utf-8')]
    manifest = json.load(open(r'$out\verify-%s\manifest.json' % m))
    return rows, manifest
e, em = load('eager'); c, cm = load('compiled')
same = sum(a['raw'] == b['raw'] for a, b in zip(e, c))
prefix = []
for a, b in zip(e, c):
    n = 0
    for x, y in zip(a['raw'], b['raw']):
        if x != y: break
        n += 1
    prefix.append(n / max(len(a['raw']), 1))
print('identical completions %d / %d; mean shared prefix %.0f%%' % (same, len(e), 100 * sum(prefix) / len(prefix)))
print('eager %.0f s (%.0f tok/s)   compiled %.0f s (%.0f tok/s)' % (em['elapsed_seconds'], em['tokens_per_second'], cm['elapsed_seconds'], cm['tokens_per_second']))
"@
"verify done $(Get-Date -Format HH:mm)"
