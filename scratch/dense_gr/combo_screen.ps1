# Task arithmetic: rounds 6 (shortest-of-four, cross entropy) and 7 (DPO on own loops) were
# trained from the same base, so their updates add: base + ramp(0, 0.7) x r6 + a x r7.
# Round 6's blend is the reference arm; screened with merge_proxy.py.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$d = "$root\scratch\dense_gr"
$base = "$d\merges-r5b\ramp0-70"
$r6 = "$d\checkpoints-2b-onpolicy-r6\smoke-r1-1-gr-s13-csa2"
$r7 = "$d\checkpoints-2b-onpolicy-r7\smoke-r1-1-gr-s14-csa2"
$out = "$d\merges-r67"
New-Item -ItemType Directory -Force $out | Out-Null
$specs = @("r7u30|0.3:0.3", "r7u50|0.5:0.5", "r7ramp0-70|0:0.7", "r7u70|0.7:0.7")
$arms = [System.Collections.Generic.List[string]]@("r6blend=$d\merges-r6\ramp0-70")
foreach ($spec in $specs) {
    $name, $ramp = $spec -split '\|'
    if (-not (Test-Path "$out\$name\model.safetensors")) {
        & $py "$d\merge_weights.py" --base $base --tuned $r6 --shallow 0 --deep 0.7 --also "${r7}:$ramp" --output "$out\$name"
    }
    $arms.Add("$name=$out\$name")
}
"=== screening $(Get-Date -Format HH:mm)"
$half = [Math]::Ceiling($arms.Count / 2)
$jobs = foreach ($gpu in 0, 1) {
    $mine = if ($gpu -eq 0) { $arms[0..($half - 1)] } else { $arms[$half..($arms.Count - 1)] }
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
Get-Content "$out\proxy-gpu0.log", "$out\proxy-gpu1.log" | Select-String "code nll|Traceback" | ForEach-Object { $_.Line -replace '\s+', ' ' }
"=== done $(Get-Date -Format HH:mm)"
