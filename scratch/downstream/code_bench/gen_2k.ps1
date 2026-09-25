param([string]$Waits)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
foreach ($id in ($Waits -split ',')) { if ($id) { while (Get-Process -Id ([int]$id) -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 } } }
$py = "$PWD\.venv\Scripts\python.exe"
$out = "$PWD\scratch\downstream\code_bench"
$models = @{ source = "$PWD\..\student-2b-hf"; finish = "$PWD\scratch\dense_gr\checkpoints-2b-finish\smoke-r1-1-gr-s2-csa2"; control = "$PWD\scratch\dense_gr\checkpoints-2b-borrow-control\smoke-r1-1-gr-s1-csa2" }
$plan = @{ "0" = @(@("finish","mbpp"), @("finish","humaneval"), @("control","mbpp")); "1" = @(@("source","mbpp"), @("source","humaneval"), @("control","humaneval")) }
$jobs = foreach ($gpu in $plan.Keys) {
    Start-Job -ArgumentList $gpu, ($plan[$gpu] | ForEach-Object { $_ -join ':' }), $py, $out, $models, "$PWD" -ScriptBlock {
        param($gpu, $items, $py, $out, $models, $root)
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        foreach ($item in $items) {
            $m, $b = $item -split ':'
            & $py "$out\generate.py" --checkpoint $models[$m] --bench $b --max-new-tokens 2048 --batch-size 16 `
                --output "$out\$m-$b-2k" 2>&1 | Select-String 'wrote|Traceback|Error'
        }
    }
}
$jobs | Wait-Job | Receive-Job
"done $(Get-Date -Format HH:mm)"