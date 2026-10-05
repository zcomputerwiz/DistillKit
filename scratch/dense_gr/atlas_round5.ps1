# Phase 1 of TARGETED_TRAINING.md on round 5: the atlas of the base (long1-u50) and the
# change map from it to round 5's tuned checkpoint, one process per GPU.
#   GPU 0: unit importance (base, then tuned), then the logit lens of both
#   GPU 1: the change map (base -> tuned), then per-domain loss of base, round 4 u50, round 5
#   powershell -File atlas_round5.ps1
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
$base = "$root\scratch\dense_gr\merges-long1\u50"
$tuned = "$root\scratch\dense_gr\checkpoints-2b-long-r5\smoke-r1-1-gr-s25-csa2"
$prev = "$root\scratch\dense_gr\merges-long4\u50"
$out = "$root\scratch\csa2-eval\atlas\long5"
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($p in $base, $tuned, $prev) { if (-not (Test-Path "$p\config.json")) { "missing checkpoint $p; stopping"; exit 1 } }
"=== atlas $(Get-Date -Format HH:mm)"
$jobs = @(
    Start-Job -ArgumentList 0, $py, $root, $out, $base, $tuned -ScriptBlock {
        param($gpu, $py, $root, $out, $base, $tuned)
        $env:CUDA_VISIBLE_DEVICES = "$gpu"; $env:PYTHONPATH = $root; $env:PYTHONIOENCODING = "utf-8"
        Set-Location $root
        & $py scratch\dense_gr\atlas.py importance --arm "base=$base" --arm "tuned=$tuned" --output-dir $out *> "$out\importance.log"
        if ($LASTEXITCODE -ne 0) { throw "importance failed (exit $LASTEXITCODE)" }
        & $py scratch\dense_gr\atlas.py lens --arm "base=$base" --arm "tuned=$tuned" --output-dir $out *> "$out\lens.log"
        if ($LASTEXITCODE -ne 0) { throw "lens failed (exit $LASTEXITCODE)" }
    },
    Start-Job -ArgumentList 1, $py, $root, $out, $base, $tuned, $prev -ScriptBlock {
        param($gpu, $py, $root, $out, $base, $tuned, $prev)
        $env:CUDA_VISIBLE_DEVICES = "$gpu"; $env:PYTHONPATH = $root; $env:PYTHONIOENCODING = "utf-8"
        Set-Location $root
        & $py scratch\dense_gr\atlas.py change --base $base --tuned $tuned --output-dir $out *> "$out\change.log"
        if ($LASTEXITCODE -ne 0) { throw "change map failed (exit $LASTEXITCODE)" }
        & $py scratch\dense_gr\atlas.py nll --arm "base=$base" --arm "long4-u50=$prev" --arm "long5=$tuned" --output-dir $out *> "$out\nll.log"
        if ($LASTEXITCODE -ne 0) { throw "nll failed (exit $LASTEXITCODE)" }
    })
$jobs | Wait-Job -Timeout 14400 | Out-Null
$jobs | Receive-Job
if ($jobs | Where-Object { $_.State -ne "Completed" }) {
    $jobs | Stop-Job; "atlas failed or timed out; see $out\*.log"; exit 1
}
"=== done $(Get-Date -Format HH:mm)"
