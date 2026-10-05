# A round's checks after training, runnable on their own (rerunning a round's script refuses
# an existing checkpoint directory): the blend screen against the base, then the loop gate
# on the fresh bank for the tuned run and two blends.
#   powershell -File finish_round.ps1 -Tag long4 -Tuned <checkpoint> -Base <checkpoint>
param([Parameter(Mandatory)][string]$Tag, [Parameter(Mandatory)][string]$Tuned,
      [Parameter(Mandatory)][string]$Base, [string]$BaseName = "long1-u50")
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
"=== blend screen against the base $(Get-Date -Format HH:mm)"
& .\scratch\dense_gr\merge_search.ps1 -Tuned $Tuned -Tag $Tag -Base $Base
if ($LASTEXITCODE -ne 0) { "blend screen failed (exit $LASTEXITCODE); stopping"; exit 1 }
# The loop gate (math_truncation.py, loop_gate.py) on the fresh bank -- MATH test outside
# MATH-500, which no repair input carries -- at 4,096 new tokens. The base runs too unless
# its fresh-bank result is already saved. One process per GPU, two at a time.
"=== loop test $(Get-Date -Format HH:mm)"
$m = "$root\scratch\dense_gr\merges-$Tag"
$tests = @(@($Tag, $Tuned), @("$Tag-u50", "$m\u50"), @("$Tag-ramp0-70", "$m\ramp0-70"))
$baseResult = "scratch\csa2-eval\math-truncation-fresh-$BaseName.json"
if (-not (Test-Path $baseResult)) { $tests = @(, @($BaseName, $Base)) + $tests }
for ($i = 0; $i -lt $tests.Count; $i += 2) {
    $jobs = foreach ($j in $i..([Math]::Min($i + 1, $tests.Count - 1))) {
        $name, $path = $tests[$j]
        Start-Job -ArgumentList ($j % 2), $py, $root, $name, $path -ScriptBlock {
            param($gpu, $py, $root, $name, $path)
            # A compile cache per GPU, as merge_search keeps: two processes compiling into one
            # cache raced (a kernel file read half-written) and the survivor recompiled into a spill.
            $env:CUDA_VISIBLE_DEVICES = "$gpu"; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"
            $env:PYTHONPATH = $root; $env:PYTHONIOENCODING = "utf-8"
            Set-Location $root
            & $py scratch\dense_gr\math_truncation.py --arm "$name=$path" --bank fresh `
                --output "scratch\csa2-eval\math-truncation-fresh-$name.json" *> "scratch\dense_gr\mt-$name.log"
            if ($LASTEXITCODE -ne 0) { throw "loop test $name failed (exit $LASTEXITCODE)" }
        }
    }
    # Bounded (an arm takes well under an hour): a worker hung in a CUDA wait or a compile
    # would otherwise hold the round forever.
    $jobs | Wait-Job -Timeout 7200 | Out-Null
    $jobs | Receive-Job
    if ($jobs | Where-Object { $_.State -ne "Completed" }) {
        $jobs | Stop-Job; "loop test failed or timed out; see scratch\dense_gr\mt-*.log"; exit 1
    }
}
& $py scratch\dense_gr\loop_gate.py --base $baseResult `
    --candidates ($tests | Where-Object { $_[0] -ne $BaseName } | ForEach-Object { "scratch\csa2-eval\math-truncation-fresh-$($_[0]).json" }) `
    --output "scratch\csa2-eval\loop-gate-$Tag.json"
if ($LASTEXITCODE -ne 0) { "loop gate: no candidate passes"; exit 1 }
"=== done $(Get-Date -Format HH:mm)"
