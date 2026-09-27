# On-policy round 4. The control (replay only, same schedule) showed what the rounds did:
# rollouts + unlikelihood cut truncations ~40% and the curriculum fixed non-thinking answer
# formats, while HumanEval+ fell with code's share of the mix -- 29% in the thinking pass,
# 16% in the rounds, and round 3 (fewer code rollouts) lost most. So: round 2's rollouts
# (all clean ones, loops under unlikelihood), curriculum v3, and expand-code planned twice.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $code = "$root\scratch\downstream\code_bench"; $math = "$root\scratch\downstream\math_bench"
$C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$think = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"

"=== train $(Get-Date -Format HH:mm)"
$caches = @("..\teacher-cache-onpolicy-r2-clean", "..\teacher-cache-onpolicy-r2-loop", "..\teacher-cache-curriculum-v3",
            "..\teacher-cache-thinking", "..\teacher-cache-think-first", "..\teacher-cache-expand-code",
            "..\teacher-cache-general-pilot")
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-broken-tools.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-onpolicy-r3.json"
& $py -c "import json, sys; json.dump(sorted(set(json.load(open(sys.argv[1]))) | set(json.load(open(sys.argv[2])))), open(sys.argv[3], 'w'), indent=1)" `
    "$C\exclude-onpolicy-r3.json" "$C\exclude-onpolicy-r3.json" "$C\exclude-r4.json"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $think,
    "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
foreach ($c in $caches) { $argv.Add($c) }
# 1536 = the rollouts' 512 prompt + 1024 new tokens, so a clean rollout keeps its ending
foreach ($a in @("--unlikelihood-caches", "..\teacher-cache-onpolicy-r2-loop", "--repeat", "..\teacher-cache-expand-code=2", "--strip-effort-prompt",
                 "--exclude-documents", "..\capture-data\exclude-r4.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "1536", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "3072", "--accumulate", "2",
                 "--tokens", "3000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-every", "400", "--evaluate-windows", "64", "--report-every", "50",
                 "--save-every", "400", "--seed", "9",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-onpolicy-r4",
                 "--output", "scratch\dense_gr\train-2b-onpolicy-r4.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Where-Object { $_ -match 'on-policy|stripped|suppress|plan:|excluded|held-out|loss .*->|wrote|Traceback|Error|spilled' }
$r2 = "$root\scratch\dense_gr\checkpoints-2b-onpolicy-r4\smoke-r1-1-gr-s9-csa2"
if (-not (Test-Path "$r2\config.json")) { "training produced no checkpoint; stopping"; exit 1 }

"=== code, sampled thinking, 3 seeds $(Get-Date -Format HH:mm)"
$jobs = foreach ($gpu in "0", "1") {
    Start-Job -ArgumentList $gpu, $py, $code, $root, $r2 -ScriptBlock {
        param($gpu, $py, $code, $root, $r2)
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"
        $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        $work = if ($gpu -eq "0") { @("mbpp|0", "mbpp|1", "mbpp|2") } else { @("humaneval|0", "humaneval|1", "humaneval|2") }
        foreach ($item in $work) {
            $bench, $seed = $item -split '\|'
            $argv = [System.Collections.Generic.List[string]]@("$code\generate.py", "--checkpoint", $r2,
                "--bench", $bench, "--max-new-tokens", "2048", "--batch-size", "64", "--sample", "--seed", $seed,
                "--compiled", "--output", "$code\onpolicy4-$bench-2k-sampled-s$seed")
            & $py $argv *> "$code\onpolicy4-$bench-2k-sampled-s$seed.log"
        }
    }
}
$jobs | Wait-Job | Receive-Job
"=== math $(Get-Date -Format HH:mm)"
$jobs = foreach ($spec in @("0|gsm8k", "1|math500")) {
    Start-Job -ArgumentList $spec, $py, $math, $root, $r2 -ScriptBlock {
        param($spec, $py, $math, $root, $r2)
        $gpu, $bench = $spec -split '\|'
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"
        $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        foreach ($mode in "think-sampled", "nothink-greedy") {
            $argv = [System.Collections.Generic.List[string]]@("$math\run_math.py", "--checkpoint", $r2,
                "--bench", $bench, "--compiled", "--output", "$math\onpolicy4-$bench-$mode")
            if ($mode -eq "think-sampled") { $argv.Add("--sample") } else { $argv.Add("--no-thinking") }
            & $py $argv *> "$math\onpolicy4-$bench-$mode.log"
        }
    }
}
$jobs | Wait-Job | Receive-Job
$env:CUDA_VISIBLE_DEVICES = "0"
"=== arithmetic probe"
& $py "$math\arithmetic_probe.py" "source=$D\student-2b-hf" "think=$think" "onpolicy2=$root\scratch\dense_gr\checkpoints-2b-onpolicy-r2\smoke-r1-1-gr-s6-csa2" "onpolicy4=$r2" 2>&1 |
    Where-Object { $_ -match '^\s*[-+*/] |op digits|overall|Error|Traceback' }
"=== hedge propensity"
& $py scratch\dense_gr\hedge_propensity.py "source=..\student-2b-hf" "think=$think" "onpolicy2=$root\scratch\dense_gr\checkpoints-2b-onpolicy-r2\smoke-r1-1-gr-s6-csa2" "onpolicy4=$r2" 2>&1 |
    Where-Object { $_ -match 'P\(hedge|Error|Traceback' }
Remove-Item Env:\CUDA_VISIBLE_DEVICES
"=== WikiText"
& $py scratch\dense_gr\general_nll.py --arm "source=..\student-2b-hf" --arm "think=$think" --arm "onpolicy2=$root\scratch\dense_gr\checkpoints-2b-onpolicy-r2\smoke-r1-1-gr-s6-csa2" --arm "onpolicy4=$r2" `
    --reference think --output scratch\dense_gr\general-nll-onpolicy-r4.json 2>&1 |
    Where-Object { $_ -match 'arm |^\S+\s+[0-9.]+\s+[+-]|Error|Traceback' }
& $py -m distillkit.independent_eval evaluate --bundle scratch\csa2-eval\q512-bundle.json --checkpoint $r2 `
    --split screen --tasks mmlu --max-seconds 550 --output scratch\csa2-eval\onpolicy-r4.mmlu.json 2>&1 |
    Where-Object { $_ -match 'Traceback|Error' }
$s = "C:\Users\Owner\AppData\Local\Temp\claude\D--DeepThought-Projects-HybridModel\2bb2dc81-93cd-457f-b0c5-90fa2d5ae146\scratchpad"
"=== MMLU"
& $py "$s\mmlu_table.py" "source=scratch/csa2-eval/q512-stock.mmlu.json" "think=scratch/csa2-eval/thinking-pass.mmlu.json" "onpolicy2=scratch/csa2-eval/onpolicy-r2.mmlu.json" `
    "onpolicy4=scratch/csa2-eval/onpolicy-r4.mmlu.json"
"=== sandbox $(Get-Date -Format HH:mm)"
foreach ($d in Get-ChildItem $code -Directory -Filter "onpolicy4-*-2k-sampled-s*") {
    $bench = if ($d.Name -match "humaneval") { "humaneval" } else { "mbpp" }
    powershell -NoProfile -ExecutionPolicy Bypass -File "$code\run_docker.ps1" $d.FullName $bench *> "$($d.FullName)\sandbox.log"
}
foreach ($bench in "humaneval", "mbpp") {
    "== thinking, sampled, $bench, 3 seeds"
    & $py "$code\sampled_compare.py" "source=$code\source-$bench-2k-sampled" "think=$code\think-$bench-2k-sampled" "control=$code\control-$bench-2k-sampled" "onpolicy2=$code\onpolicy2-$bench-2k-sampled" `
        "onpolicy4=$code\onpolicy4-$bench-2k-sampled"
}
"== looping (MBPP+ seed 0)"
& $py "$code\no_code_audit.py" "$code\source-mbpp-2k-sampled-s0" "$code\think-mbpp-2k-sampled-s0" "$code\onpolicy2-mbpp-2k-sampled-s0" "$code\onpolicy4-mbpp-2k-sampled-s0"
& $py -c @"
import json, os
m = r'$math'
for bench in ('gsm8k', 'math500'):
    for mode in ('think-sampled', 'nothink-greedy'):
        for name in ('source', 'think', 'control', 'onpolicy2', 'onpolicy4'):
            p = os.path.join(m, '%s-%s-%s' % (name, bench, mode), 'results.json')
            if os.path.exists(p):
                s = json.load(open(p))['summary']
                print('%-8s %-15s %-9s accuracy %.1f%%  no answer %d  truncated %d  mean %d tok' % (bench, mode, name, 100 * s['accuracy'], s['no_answer'], s['truncated'], s['mean_tokens']))
"@
"=== done $(Get-Date -Format HH:mm)"
