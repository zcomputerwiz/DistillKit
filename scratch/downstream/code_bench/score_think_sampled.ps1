Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$out = "$PWD\scratch\downstream\code_bench"
foreach ($d in Get-ChildItem $out -Directory -Filter "think-*-2k-sampled-s*") {
    $bench = if ($d.Name -match "humaneval") { "humaneval" } else { "mbpp" }
    powershell -NoProfile -ExecutionPolicy Bypass -File "$out\run_docker.ps1" $d.FullName $bench *> "$($d.FullName)\sandbox.log"
}