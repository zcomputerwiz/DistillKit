Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $out = "$PWD\scratch\downstream\code_bench"
$jobs = foreach ($spec in @(@("0", "nofix"), @("1", "fix"))) {
    Start-Job -ArgumentList $spec[0], $spec[1], $py, $out, "$PWD" -ScriptBlock {
        param($gpu, $arm, $py, $out, $root)
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        $ck = "$root\scratch\dense_gr\checkpoints-2b-hedge-$arm\smoke-r1-1-gr-s3-csa2"
        foreach ($bench in "mbpp", "humaneval") {
            $argv = [System.Collections.Generic.List[string]]@("$out\generate.py", "--checkpoint", $ck,
                "--bench", $bench, "--max-new-tokens", "2048", "--batch-size", "64", "--compiled",
                "--output", "$out\$arm-$bench-2k")
            & $py $argv 2>&1 | Select-String 'wrote|Traceback|Error'
        }
    }
}
$jobs | Wait-Job | Receive-Job
foreach ($arm in "nofix", "fix") {
    foreach ($bench in "humaneval", "mbpp") {
        powershell -NoProfile -ExecutionPolicy Bypass -File "$out\run_docker.ps1" "$out\$arm-$bench-2k" $bench *> "$out\$arm-$bench-2k\sandbox.log"
    }
}
& $py "$out\compare.py" "source=$out\source-humaneval-2k" "nofix=$out\nofix-humaneval-2k" "fix=$out\fix-humaneval-2k" "control=$out\control-humaneval-2k"
& $py "$out\compare.py" "source=$out\source-mbpp-2k" "nofix=$out\nofix-mbpp-2k" "fix=$out\fix-mbpp-2k" "control=$out\control-mbpp-2k"
& $py "$out\no_code_audit.py" "$out\nofix-mbpp-2k" "$out\fix-mbpp-2k"
"done $(Get-Date -Format HH:mm)"