# The full benchmark suite for one checkpoint, against the source, the thinking pass, the
# replay control and round 2: sampled thinking code (3 seeds), math in both modes, the
# arithmetic probe, hedge rate, WikiText, MMLU, sandbox scoring, loop audit.
#   powershell -File eval_checkpoint.ps1 -Checkpoint <dir> -Tag <name>
param([Parameter(Mandatory)][string]$Checkpoint, [Parameter(Mandatory)][string]$Tag)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$code = "$root\scratch\downstream\code_bench"; $math = "$root\scratch\downstream\math_bench"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$think = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"
$r2 = $Checkpoint
"=== code, sampled thinking, 3 seeds $(Get-Date -Format HH:mm)"
$jobs = foreach ($gpu in "0", "1") {
    Start-Job -ArgumentList $gpu, $py, $code, $root, $r2, $Tag -ScriptBlock {
        param($gpu, $py, $code, $root, $r2, $Tag)  # a job sees only what it is passed
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"
        $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        $work = if ($gpu -eq "0") { @("mbpp|0", "mbpp|1", "mbpp|2") } else { @("humaneval|0", "humaneval|1", "humaneval|2") }
        foreach ($item in $work) {
            $bench, $seed = $item -split '\|'
            $argv = [System.Collections.Generic.List[string]]@("$code\generate.py", "--checkpoint", $r2,
                "--bench", $bench, "--max-new-tokens", "2048", "--batch-size", "64", "--sample", "--seed", $seed,
                "--compiled", "--output", "$code\$Tag-$bench-2k-sampled-s$seed")
            & $py $argv *> "$code\$Tag-$bench-2k-sampled-s$seed.log"
        }
    }
}
$jobs | Wait-Job | Receive-Job
"=== math $(Get-Date -Format HH:mm)"
$jobs = foreach ($spec in @("0|gsm8k", "1|math500")) {
    Start-Job -ArgumentList $spec, $py, $math, $root, $r2, $Tag -ScriptBlock {
        param($spec, $py, $math, $root, $r2, $Tag)
        $gpu, $bench = $spec -split '\|'
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"
        $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        foreach ($mode in "think-sampled", "nothink-greedy") {
            $argv = [System.Collections.Generic.List[string]]@("$math\run_math.py", "--checkpoint", $r2,
                "--bench", $bench, "--compiled", "--output", "$math\$Tag-$bench-$mode")
            if ($mode -eq "think-sampled") { $argv.Add("--sample") } else { $argv.Add("--no-thinking") }
            & $py $argv *> "$math\$Tag-$bench-$mode.log"
        }
    }
}
$jobs | Wait-Job | Receive-Job
$env:CUDA_VISIBLE_DEVICES = "0"
"=== arithmetic probe"
& $py "$math\arithmetic_probe.py" "source=$D\student-2b-hf" "think=$think" "onpolicy2=$root\scratch\dense_gr\checkpoints-2b-onpolicy-r2\smoke-r1-1-gr-s6-csa2" "$Tag=$r2" 2>&1 |
    Where-Object { $_ -match '^\s*[-+*/] |op digits|overall|Error|Traceback' }
"=== hedge propensity"
& $py scratch\dense_gr\hedge_propensity.py "source=..\student-2b-hf" "think=$think" "onpolicy2=$root\scratch\dense_gr\checkpoints-2b-onpolicy-r2\smoke-r1-1-gr-s6-csa2" "$Tag=$r2" 2>&1 |
    Where-Object { $_ -match 'P\(hedge|Error|Traceback' }
Remove-Item Env:\CUDA_VISIBLE_DEVICES
"=== WikiText"
& $py scratch\dense_gr\general_nll.py --arm "source=..\student-2b-hf" --arm "think=$think" --arm "onpolicy2=$root\scratch\dense_gr\checkpoints-2b-onpolicy-r2\smoke-r1-1-gr-s6-csa2" --arm "$Tag=$r2" `
    --reference think --output scratch\dense_gr\general-nll-$Tag.json 2>&1 |
    Where-Object { $_ -match 'arm |^\S+\s+[0-9.]+\s+[+-]|Error|Traceback' }
& $py -m distillkit.independent_eval evaluate --bundle scratch\csa2-eval\q512-bundle.json --checkpoint $r2 `
    --split screen --tasks mmlu --max-seconds 550 --output scratch\csa2-eval\$Tag.mmlu.json 2>&1 |
    Where-Object { $_ -match 'Traceback|Error' }
$s = "C:\Users\Owner\AppData\Local\Temp\claude\D--DeepThought-Projects-HybridModel\2bb2dc81-93cd-457f-b0c5-90fa2d5ae146\scratchpad"
"=== MMLU"
& $py "$s\mmlu_table.py" "source=scratch/csa2-eval/q512-stock.mmlu.json" "think=scratch/csa2-eval/thinking-pass.mmlu.json" "onpolicy2=scratch/csa2-eval/onpolicy-r2.mmlu.json" `
    "$Tag=scratch/csa2-eval/$Tag.mmlu.json"
"=== sandbox $(Get-Date -Format HH:mm)"
foreach ($d in Get-ChildItem $code -Directory -Filter "$Tag-*-2k-sampled-s*") {
    $bench = if ($d.Name -match "humaneval") { "humaneval" } else { "mbpp" }
    powershell -NoProfile -ExecutionPolicy Bypass -File "$code\run_docker.ps1" $d.FullName $bench *> "$($d.FullName)\sandbox.log"
}
foreach ($bench in "humaneval", "mbpp") {
    "== thinking, sampled, $bench, 3 seeds"
    & $py "$code\sampled_compare.py" "source=$code\source-$bench-2k-sampled" "think=$code\think-$bench-2k-sampled" "control=$code\control-$bench-2k-sampled" "onpolicy2=$code\onpolicy2-$bench-2k-sampled" `
        "$Tag=$code\$Tag-$bench-2k-sampled"
}
"== looping (MBPP+ seed 0)"
& $py "$code\no_code_audit.py" "$code\source-mbpp-2k-sampled-s0" "$code\think-mbpp-2k-sampled-s0" "$code\onpolicy2-mbpp-2k-sampled-s0" "$code\$Tag-mbpp-2k-sampled-s0"
& $py -c @"
import json, os
m = r'$math'
for bench in ('gsm8k', 'math500'):
    for mode in ('think-sampled', 'nothink-greedy'):
        for name in ('source', 'think', 'control', 'onpolicy2', r'$Tag'):
            p = os.path.join(m, '%s-%s-%s' % (name, bench, mode), 'results.json')
            if os.path.exists(p):
                s = json.load(open(p))['summary']
                print('%-8s %-15s %-9s accuracy %.1f%%  no answer %d  truncated %d  mean %d tok' % (bench, mode, name, 100 * s['accuracy'], s['no_answer'], s['truncated'], s['mean_tokens']))
"@
"=== done $(Get-Date -Format HH:mm)"
