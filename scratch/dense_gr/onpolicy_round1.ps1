# On-policy round 1: student rollouts, teacher capture, KL-only training with curriculum and
# replay, then the full evaluation. Starts when the math/probe queue finishes. Parallel jobs
# get their own Inductor cache: two processes compiling the same graph into one cache corrupt it.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $code = "$root\scratch\downstream\code_bench"; $math = "$root\scratch\downstream\math_bench"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$think = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"
while (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'server_queue.ps1' }) { Start-Sleep -Seconds 30 }
"=== evaluation queue finished $(Get-Date -Format HH:mm); stopping llama-servers"
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force

"=== rollouts $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$D\capture-data\onpolicy-r1.jsonl")) {
    $jobs = foreach ($shard in "0", "1") {
        Start-Job -ArgumentList $shard, $py, $root, $think, $D -ScriptBlock {
            param($shard, $py, $root, $think, $D)
            $env:CUDA_VISIBLE_DEVICES = $shard; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$shard"; $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
            Set-Location $root
            $argv = [System.Collections.Generic.List[string]]@("$root\scratch\dense_gr\onpolicy_rollouts.py",
                "--checkpoint", $think, "--inputs", "$D\capture-data\thinking-code-math.jsonl",
                "$D\capture-data\think-first-5m.jsonl", "$D\capture-data\math-curriculum.jsonl",
                "$D\capture-data\expand-code.jsonl", "--exclude", "$D\capture-data\exclude-broken-tools.json",
                "--count", "1500", "--shard", "$shard/2", "--seed", $shard,
                "--output", "$D\capture-data\onpolicy-r1-$shard.jsonl")
            & $py $argv *> "$D\capture-data\onpolicy-r1-$shard.log"
        }
    }
    $jobs | Wait-Job | Receive-Job
    Get-Content "$D\capture-data\onpolicy-r1-0.jsonl", "$D\capture-data\onpolicy-r1-1.jsonl" |
        Set-Content "$D\capture-data\onpolicy-r1.jsonl" -Encoding utf8
    Get-Content "$D\capture-data\onpolicy-r1-0.log", "$D\capture-data\onpolicy-r1-1.log" | Select-String "rollouts" | Select-Object -Last 2
}

"=== capture $(Get-Date -Format HH:mm)"
$env:CUDA_VISIBLE_DEVICES = "0,1"
foreach ($spec in @("onpolicy-r1|teacher-cache-onpolicy-r1", "math-curriculum|teacher-cache-curriculum")) {
    $jsonl, $cache = $spec -split '\|'
    if (Test-Path "$D\$cache\manifest.json") { continue }
    & $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl "$D\capture-data\$jsonl.jsonl" --output "$D\$cache" --sequence-length 4096 `
        --top-k 64 --shard-tokens 65536 --int8 --device-map auto *> "$D\capture-data\capture-$jsonl.log"
    "captured $cache $(Get-Date -Format HH:mm)"
}
Remove-Item Env:\CUDA_VISIBLE_DEVICES

"=== train $(Get-Date -Format HH:mm)"
$caches = @("..\teacher-cache-onpolicy-r1", "..\teacher-cache-curriculum", "..\teacher-cache-thinking",
            "..\teacher-cache-think-first", "..\teacher-cache-expand-code", "..\teacher-cache-general-pilot")
& $py scratch\dense_gr\exclusion_for.py --master "$D\capture-data\exclude-broken-tools.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$D\capture-data\exclude-onpolicy-r1.json"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $think,
    "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
foreach ($c in $caches) { $argv.Add($c) }
foreach ($a in @("--kl-only-caches", "..\teacher-cache-onpolicy-r1", "--exclude-documents",
                 "..\capture-data\exclude-onpolicy-r1.json", "--suppress-hedges", "--teacher-weight", "0.5",
                 "--teacher-max-length", "1024", "--kl-chunk", "64", "--min-answer-tokens", "2",
                 "--micro-tokens", "3072", "--accumulate", "2", "--tokens", "3000000", "--warmup", "50",
                 "--decay-fraction", "0.5", "--decay-floor", "0.05", "--evaluate-every", "400",
                 "--evaluate-windows", "64", "--report-every", "50", "--save-every", "400", "--seed", "5",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-onpolicy-r1",
                 "--output", "scratch\dense_gr\train-2b-onpolicy-r1.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Where-Object { $_ -match 'on-policy|suppress|plan:|excluded|held-out|loss .*->|wrote|Traceback|Error|spilled' }
$r1 = "$root\scratch\dense_gr\checkpoints-2b-onpolicy-r1\smoke-r1-1-gr-s5-csa2"
if (-not (Test-Path "$r1\config.json")) { "training produced no checkpoint; stopping"; exit 1 }

"=== code, sampled thinking, 3 seeds $(Get-Date -Format HH:mm)"
$jobs = foreach ($gpu in "0", "1") {
    Start-Job -ArgumentList $gpu, $py, $code, $root, $r1 -ScriptBlock {
        param($gpu, $py, $code, $root, $r1)
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"; $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        $work = if ($gpu -eq "0") { @("mbpp|0", "mbpp|1", "mbpp|2") } else { @("humaneval|0", "humaneval|1", "humaneval|2") }
        foreach ($item in $work) {
            $bench, $seed = $item -split '\|'
            $argv = [System.Collections.Generic.List[string]]@("$code\generate.py", "--checkpoint", $r1,
                "--bench", $bench, "--max-new-tokens", "2048", "--batch-size", "64", "--sample", "--seed", $seed,
                "--compiled", "--output", "$code\onpolicy-$bench-2k-sampled-s$seed")
            & $py $argv *> "$code\onpolicy-$bench-2k-sampled-s$seed.log"
        }
    }
}
$jobs | Wait-Job | Receive-Job
"=== math $(Get-Date -Format HH:mm)"
$jobs = foreach ($spec in @("0|gsm8k", "1|math500")) {
    Start-Job -ArgumentList $spec, $py, $math, $root, $r1 -ScriptBlock {
        param($spec, $py, $math, $root, $r1)
        $gpu, $bench = $spec -split '\|'
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"; $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        $argv = [System.Collections.Generic.List[string]]@("$math\run_math.py", "--checkpoint", $r1, "--bench", $bench,
            "--compiled", "--sample", "--output", "$math\onpolicy-$bench-think-sampled")
        & $py $argv *> "$math\onpolicy-$bench-think-sampled.log"
    }
}
$jobs | Wait-Job | Receive-Job
$env:CUDA_VISIBLE_DEVICES = "0"
"=== arithmetic probe"
& $py "$math\arithmetic_probe.py" "source=$D\student-2b-hf" "think=$think" "onpolicy=$r1" 2>&1 |
    Where-Object { $_ -match '^\s*[-+*/] |op digits|overall|Error|Traceback' }
"=== hedge propensity"
& $py scratch\dense_gr\hedge_propensity.py "source=..\student-2b-hf" "think=$think" "onpolicy=$r1" 2>&1 |
    Where-Object { $_ -match 'P\(hedge|Error|Traceback' }
Remove-Item Env:\CUDA_VISIBLE_DEVICES
"=== WikiText"
& $py scratch\dense_gr\general_nll.py --arm "source=..\student-2b-hf" --arm "think=$think" --arm "onpolicy=$r1" `
    --reference think --output scratch\dense_gr\general-nll-onpolicy-r1.json 2>&1 |
    Where-Object { $_ -match 'arm |^\S+\s+[0-9.]+\s+[+-]|Error|Traceback' }
& $py -m distillkit.independent_eval evaluate --bundle scratch\csa2-eval\q512-bundle.json --checkpoint $r1 `
    --split screen --tasks mmlu --max-seconds 550 --output scratch\csa2-eval\onpolicy-r1.mmlu.json 2>&1 |
    Where-Object { $_ -match 'Traceback|Error' }
$s = "C:\Users\Owner\AppData\Local\Temp\claude\D--DeepThought-Projects-HybridModel\2bb2dc81-93cd-457f-b0c5-90fa2d5ae146\scratchpad"
"=== MMLU"
& $py "$s\mmlu_table.py" "source=scratch/csa2-eval/q512-stock.mmlu.json" "think=scratch/csa2-eval/thinking-pass.mmlu.json" `
    "onpolicy=scratch/csa2-eval/onpolicy-r1.mmlu.json"
"=== sandbox $(Get-Date -Format HH:mm)"
foreach ($d in Get-ChildItem $code -Directory -Filter "onpolicy-*-2k-sampled-s*") {
    $bench = if ($d.Name -match "humaneval") { "humaneval" } else { "mbpp" }
    powershell -NoProfile -ExecutionPolicy Bypass -File "$code\run_docker.ps1" $d.FullName $bench *> "$($d.FullName)\sandbox.log"
}
foreach ($bench in "humaneval", "mbpp") {
    "== thinking, sampled, $bench, 3 seeds"
    & $py "$code\sampled_compare.py" "source=$code\source-$bench-2k-sampled" "think=$code\think-$bench-2k-sampled" `
        "onpolicy=$code\onpolicy-$bench-2k-sampled"
}
"== looping (MBPP+ seed 0)"
& $py "$code\no_code_audit.py" "$code\source-mbpp-2k-sampled-s0" "$code\think-mbpp-2k-sampled-s0" "$code\onpolicy-mbpp-2k-sampled-s0"
& $py -c @"
import json, os
m = r'$math'
for bench in ('gsm8k', 'math500'):
    for name in ('source', 'think', 'onpolicy'):
        p = os.path.join(m, '%s-%s-think-sampled' % (name, bench), 'results.json')
        if os.path.exists(p):
            s = json.load(open(p))['summary']
            print('%-8s %-9s accuracy %.1f%%  no answer %d  truncated %d  mean %d tok' % (bench, name, 100 * s['accuracy'], s['no_answer'], s['truncated'], s['mean_tokens']))
"@
"=== done $(Get-Date -Format HH:mm)"
