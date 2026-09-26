# Thinking mode under Qwen's recommended sampling: 3 seeds x 2 benchmarks per model.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $out = "$root\scratch\downstream\code_bench"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$think = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"
$source = "$root\..\student-2b-hf"
$jobs = foreach ($spec in @("0|think|$think|compiled", "1|source|$source|hf")) {
    Start-Job -ArgumentList $spec, $py, $out, $root -ScriptBlock {
        param($spec, $py, $out, $root)
        $gpu, $name, $ck, $how = $spec -split '\|'
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        foreach ($seed in 0, 1, 2) {
            foreach ($bench in "mbpp", "humaneval") {
                $argv = [System.Collections.Generic.List[string]]@("$out\generate.py", "--checkpoint", $ck,
                    "--bench", $bench, "--max-new-tokens", "2048", "--batch-size", "64", "--sample",
                    "--seed", "$seed", "--output", "$out\$name-$bench-2k-sampled-s$seed")
                if ($how -eq "compiled") { $argv.Add("--compiled") }
                & $py $argv *> "$out\$name-$bench-2k-sampled-s$seed.log"
            }
        }
    }
}
$jobs | Wait-Job | Receive-Job
"=== generation done $(Get-Date -Format HH:mm)"
foreach ($name in "think", "source") { foreach ($seed in 0, 1, 2) { foreach ($bench in "humaneval", "mbpp") {
    $d = "$out\$name-$bench-2k-sampled-s$seed"
    if (Test-Path "$d\completions.jsonl") {
        powershell -NoProfile -ExecutionPolicy Bypass -File "$out\run_docker.ps1" $d $bench *> "$d\sandbox.log"
    } else { "missing $d" } } } }
foreach ($bench in "humaneval", "mbpp") {
    "== thinking, sampled (T 0.6, top-p 0.95, top-k 20), $bench, 3 seeds"
    & $py "$out\sampled_compare.py" "source=$out\source-$bench-2k-sampled" "think=$out\think-$bench-2k-sampled"
}
foreach ($name in "source", "think") { foreach ($bench in "humaneval", "mbpp") {
    $m = Get-Content "$out\$name-$bench-2k-sampled-s0\manifest.json" -ErrorAction SilentlyContinue | ConvertFrom-Json
    if ($m) { "{0,-24} mean tokens {1:N0}  truncated {2}" -f "$name-$bench-s0", $m.mean_generated_tokens, $m.truncations } } }
"=== done $(Get-Date -Format HH:mm)"
