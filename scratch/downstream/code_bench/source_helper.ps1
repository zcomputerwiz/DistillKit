Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$env:CUDA_VISIBLE_DEVICES = "0"; $env:PYTHONPATH = "$PWD"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$out = "$PWD\scratch\downstream\code_bench"
foreach ($run in @("humaneval|2", "mbpp|2", "humaneval|1")) {
    $bench, $seed = $run -split '\|'
    $d = "$out\source-$bench-2k-sampled-s$seed"
    if (Test-Path "$d\completions.jsonl") { continue }
    if (Select-String -Path "$out\source-$bench-2k-sampled-s$seed.log" -Pattern "$bench \d+/" -Quiet -ErrorAction SilentlyContinue) { continue }
    $argv = [System.Collections.Generic.List[string]]@("$out\generate.py", "--checkpoint", "$PWD\..\student-2b-hf",
        "--bench", $bench, "--max-new-tokens", "2048", "--batch-size", "64", "--sample", "--seed", $seed,
        "--output", $d)
    & "$PWD\.venv\Scripts\python.exe" $argv *> "$out\source-$bench-2k-sampled-s$seed.gpu0.log"
}