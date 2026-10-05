# Blends of a base (default: the thinking pass) with a tuned round, screened on train-split and held-out
# proxies (merge_proxy.py) -- never the benchmarks, which only the finalists see.
#   powershell -File merge_search.ps1 -Tuned <checkpoint> -Tag <name> [-Base <checkpoint>]
param([Parameter(Mandatory)][string]$Tuned, [Parameter(Mandatory)][string]$Tag, [string]$Base = "")
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
$think = if ($Base) { $Base } else { "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2" }
$out = "$root\scratch\dense_gr\merges-$Tag"
New-Item -ItemType Directory -Force $out | Out-Null
# name | uniform alpha, or shallow,deep ramp [| extra merge_weights flag]
# qk: the tuned run with its attention layers' query/key maps from the base (QK-Restore,
# arXiv 2606.11052) -- long-range routing kept, the rest of the update whole.
$specs = @("u30|0.3", "u50|0.5", "u70|0.7", "ramp0-100|0,1", "ramp25-100|0.25,1", "ramp50-100|0.5,1", "ramp0-70|0,0.7",
           "qk|1|--restore-qk")
$arms = [System.Collections.Generic.List[string]]@("think=$think", "tuned=$Tuned")
# A merge already on disk is reused only if it was made from these two checkpoints.
$same = { param($x, $y) [IO.Path]::GetFullPath($x).TrimEnd('\') -ieq [IO.Path]::GetFullPath($y).TrimEnd('\') }
foreach ($spec in $specs) {
    $name, $a, $flag = $spec -split '\|'
    if (Test-Path "$out\$name\model.safetensors") {
        $made = Get-Content "$out\$name\merge.json" -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
        if (-not $made -or -not (& $same $made.base $think) -or -not (& $same $made.tuned $Tuned)) {
            "stale merge $out\$name (made from other checkpoints); move it aside or use another -Tag"; exit 1
        }
    } else {
        $argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\merge_weights.py", "--base", $think,
            "--tuned", $Tuned, "--output", "$out\$name")
        if ($a -match ',') { $s, $d = $a -split ','; $argv.Add("--shallow"); $argv.Add($s); $argv.Add("--deep"); $argv.Add($d) }
        else { $argv.Add("--alpha"); $argv.Add($a) }
        if ($flag) { $argv.Add($flag) }
        & $py $argv
        if ($LASTEXITCODE -ne 0) { "merge $name failed (exit $LASTEXITCODE)"; exit 1 }
    }
    $arms.Add("$name=$out\$name")
}
"=== screening $($arms.Count) arms $(Get-Date -Format HH:mm)"
$half = [Math]::Ceiling($arms.Count / 2)
$jobs = foreach ($gpu in 0, 1) {
    $mine = if ($gpu -eq 0) { $arms[0..($half - 1)] } else { $arms[$half..($arms.Count - 1)] }
    # One ";"-joined string: an array argument arrives in the job flattened into one string
    Start-Job -ArgumentList $gpu, $py, $root, $out, ($mine -join ";") -ScriptBlock {
        param($gpu, $py, $root, $out, $mine)
        $env:CUDA_VISIBLE_DEVICES = "$gpu"; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"
        $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
        Set-Location $root
        # One process per arm: the compiled generator's CUDA-graph pools outlive the model,
        # and five arms in one process spilled ~10 GB a card into system memory (long3).
        # merge_proxy adds each arm to the same results file.
        foreach ($arm in $mine -split ";") {
            & $py scratch\dense_gr\merge_proxy.py $arm --output "$out\proxy-gpu$gpu.json" *>> "$out\proxy-gpu$gpu.log"
            if ($LASTEXITCODE -ne 0) { throw "merge_proxy on gpu $gpu failed for $arm (exit $LASTEXITCODE)" }
        }
    }
}
# Bounded (five arms a card take about an hour): a worker hung in a CUDA wait or a compile
# would otherwise hold the round forever.
$jobs | Wait-Job -Timeout 10800 | Out-Null
$jobs | Receive-Job
if ($jobs | Where-Object { $_.State -ne "Completed" }) {
    $jobs | Stop-Job; "blend screen failed or timed out; see $out\proxy-gpu*.log"; exit 1
}
Get-Content "$out\proxy-gpu0.log", "$out\proxy-gpu1.log" | Select-String "code nll|Traceback|Error"
"=== done $(Get-Date -Format HH:mm)"
exit 0
