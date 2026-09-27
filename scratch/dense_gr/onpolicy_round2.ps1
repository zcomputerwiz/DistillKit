# On-policy round 2, from the thinking pass. Round 1 trained KL on every rollout and the
# teacher endorses loops it is shown, so this one sorts rollouts first:
#   clean (finished, no loop, correct where checkable) -> ordinary documents, CE + KL
#   looping -> KL-only with unlikelihood on the repeats
#   the rest -> dropped
# Prompts are rendered as served (no injected xhigh effort text, 40% non-thinking, half
# with verifiable GSM8K/MATH train answers), and the replayed captures have the effort text
# stripped from the student's side. Parallel jobs get their own Inductor cache.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $code = "$root\scratch\downstream\code_bench"; $math = "$root\scratch\downstream\math_bench"
$C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$think = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"

"=== prompts $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\prompts-r2.jsonl")) {
    & $py scratch\dense_gr\rollout_prompts.py --checkpoint $think --exclude "$C\exclude-broken-tools.json" `
        --corpus "$C\thinking-code-math.jsonl=1200" "$C\think-first-5m.jsonl=800" "$C\expand-code.jsonl=800" `
        "$C\math-curriculum-v2.jsonl=500" --output "$C\prompts-r2.jsonl" 2>&1 | Where-Object { $_ -match 'wrote|Error|Traceback' }
}

"=== rollouts $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\onpolicy-r2.jsonl")) {
    $jobs = foreach ($shard in "0", "1") {
        Start-Job -ArgumentList $shard, $py, $root, $think, $C -ScriptBlock {
            param($shard, $py, $root, $think, $C)
            $env:CUDA_VISIBLE_DEVICES = $shard; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$shard"
            $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
            Set-Location $root
            $argv = [System.Collections.Generic.List[string]]@("$root\scratch\dense_gr\onpolicy_rollouts.py",
                "--checkpoint", $think, "--inputs", "$C\prompts-r2.jsonl", "--count", "4000",
                "--new", "1024", "--shard", "$shard/2", "--seed", $shard,
                "--output", "$C\onpolicy-r2-$shard.jsonl")
            & $py $argv *> "$C\onpolicy-r2-$shard.log"
        }
    }
    $jobs | Wait-Job | Receive-Job
    # Join as bytes: PowerShell 5.1 reads these as ANSI and writes a BOM the capture rejects
    & $py -c "import sys; open(sys.argv[1], 'wb').write(b''.join(open(p, 'rb').read() for p in sys.argv[2:]))" `
        "$C\onpolicy-r2.jsonl" "$C\onpolicy-r2-0.jsonl" "$C\onpolicy-r2-1.jsonl"
    Get-Content "$C\onpolicy-r2-0.log", "$C\onpolicy-r2-1.log" | Select-String "rollouts" | Select-Object -Last 2
}

"=== classify $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\onpolicy-r2-clean.jsonl")) {
    & $py scratch\dense_gr\classify_rollouts.py "$C\onpolicy-r2.jsonl" --clean "$C\onpolicy-r2-clean.jsonl" `
        --looping "$C\onpolicy-r2-loop.jsonl" 2>$null
}

"=== capture $(Get-Date -Format HH:mm)"
$env:CUDA_VISIBLE_DEVICES = "0,1"
foreach ($spec in @("onpolicy-r2-clean|teacher-cache-onpolicy-r2-clean", "onpolicy-r2-loop|teacher-cache-onpolicy-r2-loop")) {
    $jsonl, $cache = $spec -split '\|'
    if (Test-Path "$D\$cache\manifest.json") { continue }
    if (Test-Path "$D\$cache") { Move-Item "$D\$cache" "$D\$cache.failed-$(Get-Date -Format HHmmss)" }
    & $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl "$C\$jsonl.jsonl" --output "$D\$cache" --sequence-length 4096 `
        --top-k 64 --shard-tokens 65536 --int8 --device-map auto *> "$C\capture-$jsonl.log"
    if (Test-Path "$D\$cache\manifest.json") { "captured $cache $(Get-Date -Format HH:mm)" } else { "capture of $cache FAILED"; exit 1 }
}
Remove-Item Env:\CUDA_VISIBLE_DEVICES

"=== train $(Get-Date -Format HH:mm)"
$caches = @("..\teacher-cache-onpolicy-r2-clean", "..\teacher-cache-onpolicy-r2-loop", "..\teacher-cache-curriculum-v2",
            "..\teacher-cache-thinking", "..\teacher-cache-think-first", "..\teacher-cache-expand-code",
            "..\teacher-cache-general-pilot")
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-broken-tools.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-onpolicy-r2.json"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $think,
    "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
foreach ($c in $caches) { $argv.Add($c) }
# 1536 = the rollouts' 512 prompt + 1024 new tokens, so a clean rollout keeps its ending
foreach ($a in @("--unlikelihood-caches", "..\teacher-cache-onpolicy-r2-loop", "--strip-effort-prompt",
                 "--exclude-documents", "..\capture-data\exclude-onpolicy-r2.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "1536", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "3072", "--accumulate", "2",
                 "--tokens", "3000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-every", "400", "--evaluate-windows", "64", "--report-every", "50",
                 "--save-every", "400", "--seed", "6",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-onpolicy-r2",
                 "--output", "scratch\dense_gr\train-2b-onpolicy-r2.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Where-Object { $_ -match 'on-policy|stripped|suppress|plan:|excluded|held-out|loss .*->|wrote|Traceback|Error|spilled' }
$r2 = "$root\scratch\dense_gr\checkpoints-2b-onpolicy-r2\smoke-r1-1-gr-s6-csa2"
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
                "--compiled", "--output", "$code\onpolicy2-$bench-2k-sampled-s$seed")
            & $py $argv *> "$code\onpolicy2-$bench-2k-sampled-s$seed.log"
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
                "--bench", $bench, "--compiled", "--output", "$math\onpolicy2-$bench-$mode")
            if ($mode -eq "think-sampled") { $argv.Add("--sample") } else { $argv.Add("--no-thinking") }
            & $py $argv *> "$math\onpolicy2-$bench-$mode.log"
        }
    }
}
$jobs | Wait-Job | Receive-Job
$env:CUDA_VISIBLE_DEVICES = "0"
"=== arithmetic probe"
& $py "$math\arithmetic_probe.py" "source=$D\student-2b-hf" "think=$think" "onpolicy2=$r2" 2>&1 |
    Where-Object { $_ -match '^\s*[-+*/] |op digits|overall|Error|Traceback' }
"=== hedge propensity"
& $py scratch\dense_gr\hedge_propensity.py "source=..\student-2b-hf" "think=$think" "onpolicy2=$r2" 2>&1 |
    Where-Object { $_ -match 'P\(hedge|Error|Traceback' }
Remove-Item Env:\CUDA_VISIBLE_DEVICES
"=== WikiText"
& $py scratch\dense_gr\general_nll.py --arm "source=..\student-2b-hf" --arm "think=$think" --arm "onpolicy2=$r2" `
    --reference think --output scratch\dense_gr\general-nll-onpolicy-r2.json 2>&1 |
    Where-Object { $_ -match 'arm |^\S+\s+[0-9.]+\s+[+-]|Error|Traceback' }
& $py -m distillkit.independent_eval evaluate --bundle scratch\csa2-eval\q512-bundle.json --checkpoint $r2 `
    --split screen --tasks mmlu --max-seconds 550 --output scratch\csa2-eval\onpolicy-r2.mmlu.json 2>&1 |
    Where-Object { $_ -match 'Traceback|Error' }
$s = "C:\Users\Owner\AppData\Local\Temp\claude\D--DeepThought-Projects-HybridModel\2bb2dc81-93cd-457f-b0c5-90fa2d5ae146\scratchpad"
"=== MMLU"
& $py "$s\mmlu_table.py" "source=scratch/csa2-eval/q512-stock.mmlu.json" "think=scratch/csa2-eval/thinking-pass.mmlu.json" `
    "onpolicy2=scratch/csa2-eval/onpolicy-r2.mmlu.json"
"=== sandbox $(Get-Date -Format HH:mm)"
foreach ($d in Get-ChildItem $code -Directory -Filter "onpolicy2-*-2k-sampled-s*") {
    $bench = if ($d.Name -match "humaneval") { "humaneval" } else { "mbpp" }
    powershell -NoProfile -ExecutionPolicy Bypass -File "$code\run_docker.ps1" $d.FullName $bench *> "$($d.FullName)\sandbox.log"
}
foreach ($bench in "humaneval", "mbpp") {
    "== thinking, sampled, $bench, 3 seeds"
    & $py "$code\sampled_compare.py" "source=$code\source-$bench-2k-sampled" "think=$code\think-$bench-2k-sampled" `
        "onpolicy2=$code\onpolicy2-$bench-2k-sampled"
}
"== looping (MBPP+ seed 0)"
& $py "$code\no_code_audit.py" "$code\source-mbpp-2k-sampled-s0" "$code\think-mbpp-2k-sampled-s0" "$code\onpolicy2-mbpp-2k-sampled-s0"
& $py -c @"
import json, os
m = r'$math'
for bench in ('gsm8k', 'math500'):
    for mode in ('think-sampled', 'nothink-greedy'):
        for name in ('source', 'think', 'onpolicy2'):
            p = os.path.join(m, '%s-%s-%s' % (name, bench, mode), 'results.json')
            if os.path.exists(p):
                s = json.load(open(p))['summary']
                print('%-8s %-15s %-9s accuracy %.1f%%  no answer %d  truncated %d  mean %d tok' % (bench, mode, name, 100 * s['accuracy'], s['no_answer'], s['truncated'], s['mean_tokens']))
"@
"=== done $(Get-Date -Format HH:mm)"
