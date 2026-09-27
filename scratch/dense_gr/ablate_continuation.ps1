# Why does held-out code NLL rise when training continues from the thinking pass? The
# thinking pass itself lowered it (0.787 -> 0.773); the replay-only control raised it
# (0.777 -> 0.831) and differs from the thinking pass's recipe in three ways at once.
# Training only -- the held-out NLL is reported by the trainer -- from the thinking pass:
#   plain   the thinking pass's recipe, continued (re-warm alone)
#   strip   plain + --strip-effort-prompt
#   capgen  plain + general-pilot + a 1536 cap (no stripping)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $C = "D:\DeepThought\Projects\HybridModel\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$think = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"
foreach ($arm in "plain", "strip", "capgen") {
    $caches = @("..\teacher-cache-think-first", "..\teacher-cache-thinking", "..\teacher-cache-expand-code")
    $cap = "1024"
    if ($arm -eq "capgen") { $caches += "..\teacher-cache-general-pilot"; $cap = "1536" }
    & $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-broken-tools.json" `
        --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-ablate-$arm.json" | Out-Null
    $argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $think,
        "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
    foreach ($c in $caches) { $argv.Add($c) }
    foreach ($a in @("--exclude-documents", "..\capture-data\exclude-ablate-$arm.json", "--suppress-hedges",
                     "--teacher-weight", "0.5", "--teacher-max-length", $cap, "--kl-chunk", "64",
                     "--min-answer-tokens", "2", "--micro-tokens", "3072", "--accumulate", "2",
                     "--tokens", "3000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                     "--evaluate-every", "100", "--evaluate-windows", "64", "--report-every", "50",
                     "--save-every", "100000", "--seed", "10",
                     "--checkpoints", "scratch\dense_gr\checkpoints-ablate-$arm",
                     "--output", "scratch\dense_gr\train-ablate-$arm.json")) { $argv.Add($a) }
    if ($arm -eq "strip") { $argv.Add("--strip-effort-prompt") }
    "=== $arm $(Get-Date -Format HH:mm)"
    & $py $argv 2>&1 | Where-Object { $_ -match 'stripped|plan:|held-out|loss .*->|Traceback|Error|spilled' }
}
& $py -c @"
import json
for arm in ('plain', 'strip', 'capgen'):
    h = json.load(open(r'scratch\dense_gr\train-ablate-%s.json' % arm))['history']
    code = [(r['step'], v) for r in h for k, v in (r.get('heldout_by_source') or {}).items() if 'expand-code' in k]
    print('%-7s expand-code held-out: %s' % (arm, '  '.join('%d:%.4f' % s for s in code)))
"@
"=== done $(Get-Date -Format HH:mm)"
