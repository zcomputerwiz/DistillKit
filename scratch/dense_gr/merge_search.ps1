# Blends of the thinking pass (base) with a tuned round, screened on train-split and held-out
# proxies (merge_proxy.py) -- never the benchmarks, which only the finalists see.
#   powershell -File merge_search.ps1 <tuned checkpoint> <tag>
param([Parameter(Mandatory)][string]$Tuned, [Parameter(Mandatory)][string]$Tag)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$think = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"
$out = "$root\scratch\dense_gr\merges-$Tag"
New-Item -ItemType Directory -Force $out | Out-Null
# name | uniform alpha, or shallow,deep ramp
$specs = @("u30|0.3", "u50|0.5", "u70|0.7", "ramp0-100|0,1", "ramp25-100|0.25,1", "ramp50-100|0.5,1", "ramp0-70|0,0.7")
$arms = [System.Collections.Generic.List[string]]@("think=$think", "tuned=$Tuned")
foreach ($spec in $specs) {
    $name, $a = $spec -split '\|'
    if (-not (Test-Path "$out\$name\model.safetensors")) {
        $argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\merge_weights.py", "--base", $think,
            "--tuned", $Tuned, "--output", "$out\$name")
        if ($a -match ',') { $s, $d = $a -split ','; $argv.Add("--shallow"); $argv.Add($s); $argv.Add("--deep"); $argv.Add($d) }
        else { $argv.Add("--alpha"); $argv.Add($a) }
        & $py $argv
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
        $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        $argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\merge_proxy.py")
        foreach ($arm in $mine -split ";") { $argv.Add($arm) }
        $argv.Add("--output"); $argv.Add("$out\proxy-gpu$gpu.json")
        & $py $argv *> "$out\proxy-gpu$gpu.log"
    }
}
$jobs | Wait-Job | Receive-Job
Get-Content "$out\proxy-gpu0.log", "$out\proxy-gpu1.log" | Select-String "code nll|Traceback|Error"
"=== done $(Get-Date -Format HH:mm)"
