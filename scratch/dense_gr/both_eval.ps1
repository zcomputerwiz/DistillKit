# Evaluate the "both" arm (originals + think-first remix + hedge fixes) against fix and source.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $out = "$root\scratch\downstream\code_bench"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$both = "$root\scratch\dense_gr\checkpoints-2b-hedge-both\smoke-r1-1-gr-s3-csa2"
$fix = "$root\scratch\dense_gr\checkpoints-2b-hedge-fix\smoke-r1-1-gr-s3-csa2"
$nofix = "$root\scratch\dense_gr\checkpoints-2b-hedge-nofix\smoke-r1-1-gr-s3-csa2"

# Code generation: one list of runs per GPU, every argument list built explicitly.
$runs = @{
    "0" = [System.Collections.Generic.List[string]]@("both|$both|think", "both|$both|nothink")
    "1" = [System.Collections.Generic.List[string]]@("nofix|$nofix|nothink")
}
$jobs = foreach ($gpu in $runs.Keys) {
    Start-Job -ArgumentList $gpu, ($runs[$gpu] -join ';'), $py, $out, $root -ScriptBlock {
        param($gpu, $items, $py, $out, $root)
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        foreach ($item in ($items -split ';')) {
            $name, $ck, $mode = $item -split '\|'
            foreach ($bench in "mbpp", "humaneval") {
                $suffix = if ($mode -eq "nothink") { "-nt" } else { "" }
                $argv = [System.Collections.Generic.List[string]]@("$out\generate.py", "--checkpoint", $ck,
                    "--bench", $bench, "--max-new-tokens", "2048", "--batch-size", "64", "--compiled",
                    "--output", "$out\$name-$bench-2k$suffix")
                if ($mode -eq "nothink") { $argv.Add("--no-thinking") }
                & $py $argv *> "$out\$name-$bench-2k$suffix.log"
            }
        }
    }
}
$jobs | Wait-Job | Receive-Job
"=== generation done $(Get-Date -Format HH:mm)"

# Everything else on the GPUs while the sandbox scores on the CPU.
$env:CUDA_VISIBLE_DEVICES = "0"
"=== hedge propensity"
& $py scratch\dense_gr\hedge_propensity.py "source=..\student-2b-hf" "fix=$fix" "both=$both" 2>&1 |
    Where-Object { $_ -match 'P\(hedge|documents|Error|Traceback' }
Remove-Item Env:\CUDA_VISIBLE_DEVICES
"=== WikiText"
& $py scratch\dense_gr\general_nll.py --arm "source=..\student-2b-hf" --arm "fix=$fix" --arm "both=$both" `
    --reference fix --output scratch\dense_gr\general-nll-both.json 2>&1 |
    Where-Object { $_ -match 'arm |^\S+\s+[0-9.]+\s+[+-]|Error|Traceback' }
foreach ($t in @('nll', 'mmlu')) {
    & $py -m distillkit.independent_eval evaluate --bundle scratch\csa2-eval\q512-bundle.json `
        --checkpoint $both --split screen --tasks $t --max-seconds 550 `
        --output "scratch\csa2-eval\hedge-both.$t.json" 2>&1 | Where-Object { $_ -match 'Traceback|Error' }
}
$s = "C:\Users\Owner\AppData\Local\Temp\claude\D--DeepThought-Projects-HybridModel\2bb2dc81-93cd-457f-b0c5-90fa2d5ae146\scratchpad"
"=== MMLU"
& $py "$s\mmlu_table.py" "source=scratch/csa2-eval/q512-stock.mmlu.json" "fix=scratch/csa2-eval/hedge-fix.mmlu.json" `
    "both=scratch/csa2-eval/hedge-both.mmlu.json"
"=== in-domain NLL"
& $py "$s\nll_table.py" "source=scratch/csa2-eval/q512-stock.nll.json" "fix=scratch/csa2-eval/hedge-fix.nll.json" `
    "both=scratch/csa2-eval/hedge-both.nll.json"

foreach ($d in "both-mbpp-2k", "both-humaneval-2k", "both-mbpp-2k-nt", "both-humaneval-2k-nt",
               "nofix-mbpp-2k-nt", "nofix-humaneval-2k-nt") {
    $bench = if ($d -match "humaneval") { "humaneval" } else { "mbpp" }
    if (Test-Path "$out\$d\completions.jsonl") {
        powershell -NoProfile -ExecutionPolicy Bypass -File "$out\run_docker.ps1" "$out\$d" $bench *> "$out\$d\sandbox.log"
    } else { "missing $d" }
}
foreach ($bench in "humaneval", "mbpp") {
    "== thinking, $bench"
    & $py "$out\compare.py" "source=$out\source-$bench-2k" "fix=$out\fix-$bench-2k" "both=$out\both-$bench-2k"
    "== no thinking, $bench"
    & $py "$out\compare.py" "source=$out\source-$bench-2k-nt" "nofix=$out\nofix-$bench-2k-nt" "fix=$out\fix-$bench-2k-nt" "both=$out\both-$bench-2k-nt"
}
& $py "$out\no_code_audit.py" "$out\source-mbpp-2k" "$out\fix-mbpp-2k" "$out\both-mbpp-2k"
foreach ($d in "source-mbpp-2k", "fix-mbpp-2k", "both-mbpp-2k", "source-humaneval-2k", "fix-humaneval-2k", "both-humaneval-2k") {
    $m = Get-Content "$out\$d\manifest.json" -ErrorAction SilentlyContinue | ConvertFrom-Json
    if ($m) { "{0,-22} mean tokens {1:N0}  truncated {2}" -f $d, $m.mean_generated_tokens, $m.truncations }
}
"=== done $(Get-Date -Format HH:mm)"
