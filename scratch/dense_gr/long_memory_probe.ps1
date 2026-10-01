# How long a training sequence fits on the two cards: three steps of the real trainer per
# length on a 32K capture of Claude Code traces, KL + CE as in the rounds, tensor parallel,
# every layer checkpointed, no sparse stage (its recorded attention is quadratic; the token
# path CSA2 trains on otherwise keeps memory linear). Reports peak and spill per length.
#   powershell -File long_memory_probe.ps1 -Cache <capture> [-Lengths 8192,16384,32768]
param([Parameter(Mandatory)][string]$Cache, [int[]]$Lengths = @(8192, 16384, 32768),
      [string]$Extra = "", [string]$Tag = "")
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $env:PYTHONPATH = "$PWD"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$base = "$PWD\scratch\dense_gr\merges-r6r8b\u50"
foreach ($length in $Lengths) {
    "=== $length $(Get-Date -Format HH:mm)"
    $argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
        "--inherit", "--tensor-parallel", "--embedding-on", "away", "--checkpoint-layers",
        "--teacher-cache", $Cache, "--teacher-weight", "0.5", "--teacher-max-length", "$length",
        "--micro-tokens", "$length", "--accumulate", "1", "--kl-chunk", "64",
        "--tokens", "100000000", "--max-steps", "3", "--warmup", "1", "--report-every", "1",
        "--checkpoints", "scratch\dense_gr\checkpoints-memory-probe-$length$Tag",
        "--output", "scratch\dense_gr\memory-probe-$length$Tag.json")
    foreach ($a in ($Extra -split ' ' | Where-Object { $_ })) { $argv.Add($a) }
    & $py $argv 2>&1 | Where-Object { $_ -match 'plan:|^step|loss .*->|peak|spill|out of memory|OutOfMemory|Traceback|Error' -and $_ -notmatch 'Warning' }
}
