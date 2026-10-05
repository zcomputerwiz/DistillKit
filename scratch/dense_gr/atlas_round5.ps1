# Phase 1 of TARGETED_TRAINING.md on round 5, in the order the Codex review set: first what
# regressed and by role (the ledger), then which parameter families carry it (reverts).
#   GPU 0: the role-split ledger -- base (long1-u50), round 4 u50, round 5 and its blends
#   GPU 1: each parameter family of round 5 put back to the base, on the same ledger
#   powershell -File atlas_round5.ps1
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
$base = "$root\scratch\dense_gr\merges-long1\u50"
$tuned = "$root\scratch\dense_gr\checkpoints-2b-long-r5\smoke-r1-1-gr-s25-csa2"
$prev = "$root\scratch\dense_gr\merges-long4\u50"
$m = "$root\scratch\dense_gr\merges-long5"
$out = "$root\scratch\csa2-eval\atlas\long5"
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($p in $base, $tuned, $prev, "$m\u50", "$m\ramp0-70") {
    if (-not (Test-Path "$p\config.json")) { "missing checkpoint $p; stopping"; exit 1 }
}
"=== atlas $(Get-Date -Format HH:mm)"
$jobs = @(
    Start-Job -ArgumentList 0, $py, $root, $out, $base, $tuned, $prev, $m -ScriptBlock {
        param($gpu, $py, $root, $out, $base, $tuned, $prev, $m)
        $env:CUDA_VISIBLE_DEVICES = "$gpu"; $env:PYTHONPATH = $root; $env:PYTHONIOENCODING = "utf-8"
        Set-Location $root
        & $py scratch\dense_gr\atlas.py nll --arm "base=$base" --arm "long4-u50=$prev" --arm "long5=$tuned" `
            --arm "long5-u50=$m\u50" --arm "long5-ramp0-70=$m\ramp0-70" --output-dir $out *> "$out\nll.log"
        if ($LASTEXITCODE -ne 0) { throw "ledger failed (exit $LASTEXITCODE)" }
    },
    Start-Job -ArgumentList 1, $py, $root, $out, $base, $tuned -ScriptBlock {
        param($gpu, $py, $root, $out, $base, $tuned)
        $env:CUDA_VISIBLE_DEVICES = "$gpu"; $env:PYTHONPATH = $root; $env:PYTHONIOENCODING = "utf-8"
        Set-Location $root
        & $py scratch\dense_gr\atlas.py revert --base $base --tuned $tuned --output-dir $out *> "$out\revert.log"
        if ($LASTEXITCODE -ne 0) { throw "reverts failed (exit $LASTEXITCODE)" }
    })
$jobs | Wait-Job -Timeout 14400 | Out-Null
$jobs | Receive-Job
if ($jobs | Where-Object { $_.State -ne "Completed" }) {
    $jobs | Stop-Job; "atlas failed or timed out; see $out\*.log"; exit 1
}
"=== done $(Get-Date -Format HH:mm)"
