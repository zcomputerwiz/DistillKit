# On-policy round 3: round 2's rollouts, re-sorted. Round 2 put cross entropy on every clean
# rollout, verified or not, and lost 5 points of HumanEval+ while hedging doubled: the
# student learned its own unchecked code. Here only rollouts verified correct (GSM8K/MATH
# train answers) train at all -- cross entropy and KL -- plus the looping ones under
# unlikelihood. Curriculum v3: "only the answer" replies are bare (v2 showed working and
# the model learned to restate problems), and subtraction includes negative results.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $code = "$root\scratch\downstream\code_bench"; $math = "$root\scratch\downstream\math_bench"
$C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$think = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"

"=== curriculum v3 $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\math-curriculum-v3.jsonl")) {
    & $py scratch\dense_gr\math_curriculum.py --tokens 1500000 --output "$C\math-curriculum-v3.jsonl" 2>$null
}
$env:CUDA_VISIBLE_DEVICES = "0,1"
$cache = "teacher-cache-curriculum-v3"
if (-not (Test-Path "$D\$cache\manifest.json")) {
    if (Test-Path "$D\$cache") { Move-Item "$D\$cache" "$D\$cache.failed-$(Get-Date -Format HHmmss)" }
    & $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl "$C\math-curriculum-v3.jsonl" --output "$D\$cache" --sequence-length 4096 `
        --top-k 64 --shard-tokens 65536 --int8 --device-map auto *> "$C\capture-math-curriculum-v3.log"
    if (Test-Path "$D\$cache\manifest.json") { "captured $cache $(Get-Date -Format HH:mm)" } else { "capture of $cache FAILED"; exit 1 }
}
Remove-Item Env:\CUDA_VISIBLE_DEVICES
& $py scratch\dense_gr\unverified_ids.py "$C\onpolicy-r2-clean.jsonl" --output "$C\drop-r2.json"
"=== train $(Get-Date -Format HH:mm)"
$caches = @("..\teacher-cache-onpolicy-r2-clean", "..\teacher-cache-onpolicy-r2-loop", "..\teacher-cache-curriculum-v3",
            "..\teacher-cache-thinking", "..\teacher-cache-think-first", "..\teacher-cache-expand-code",
            "..\teacher-cache-general-pilot")
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-broken-tools.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-onpolicy-r3.json"
& $py -c "import json, sys; json.dump(sorted(set(json.load(open(sys.argv[1]))) | set(json.load(open(sys.argv[2])))), open(sys.argv[3], 'w'), indent=1)" `
    "$C\exclude-onpolicy-r3.json" "$C\drop-r2.json" "$C\exclude-and-drop-r3.json"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $think,
    "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
foreach ($cache in $caches) { $argv.Add($cache) }
# 1536 = the rollouts' 512 prompt + 1024 new tokens, so a clean rollout keeps its ending
foreach ($a in @("--unlikelihood-caches", "..\teacher-cache-onpolicy-r2-loop", "--strip-effort-prompt",
                 "--exclude-documents", "..\capture-data\exclude-and-drop-r3.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "1536", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "3072", "--accumulate", "2",
                 "--tokens", "3000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-every", "400", "--evaluate-windows", "64", "--report-every", "50",
                 "--save-every", "400", "--seed", "7",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-onpolicy-r3",
                 "--output", "scratch\dense_gr\train-2b-onpolicy-r3.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Where-Object { $_ -match 'on-policy|stripped|suppress|plan:|excluded|held-out|loss .*->|wrote|Traceback|Error|spilled' }
$r2 = "$root\scratch\dense_gr\checkpoints-2b-onpolicy-r3\smoke-r1-1-gr-s7-csa2"
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
                "--compiled", "--output", "$code\onpolicy3-$bench-2k-sampled-s$seed")
            & $py $argv *> "$code\onpolicy3-$bench-2k-sampled-s$seed.log"
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
                "--bench", $bench, "--compiled", "--output", "$math\onpolicy3-$bench-$mode")
            if ($mode -eq "think-sampled") { $argv.Add("--sample") } else { $argv.Add("--no-thinking") }
            & $py $argv *> "$math\onpolicy3-$bench-$mode.log"
        }
    }
}
$jobs | Wait-Job | Receive-Job
$env:CUDA_VISIBLE_DEVICES = "0"
"=== arithmetic probe"
& $py "$math\arithmetic_probe.py" "source=$D\student-2b-hf" "think=$think" "onpolicy2=$root\scratch\dense_gr\checkpoints-2b-onpolicy-r2\smoke-r1-1-gr-s6-csa2" "onpolicy3=$r2" 2>&1 |
    Where-Object { $_ -match '^\s*[-+*/] |op digits|overall|Error|Traceback' }
"=== hedge propensity"
& $py scratch\dense_gr\hedge_propensity.py "source=..\student-2b-hf" "think=$think" "onpolicy2=$root\scratch\dense_gr\checkpoints-2b-onpolicy-r2\smoke-r1-1-gr-s6-csa2" "onpolicy3=$r2" 2>&1 |
    Where-Object { $_ -match 'P\(hedge|Error|Traceback' }
Remove-Item Env:\CUDA_VISIBLE_DEVICES
"=== WikiText"
& $py scratch\dense_gr\general_nll.py --arm "source=..\student-2b-hf" --arm "think=$think" --arm "onpolicy2=$root\scratch\dense_gr\checkpoints-2b-onpolicy-r2\smoke-r1-1-gr-s6-csa2" --arm "onpolicy3=$r2" `
    --reference think --output scratch\dense_gr\general-nll-onpolicy-r3.json 2>&1 |
    Where-Object { $_ -match 'arm |^\S+\s+[0-9.]+\s+[+-]|Error|Traceback' }
& $py -m distillkit.independent_eval evaluate --bundle scratch\csa2-eval\q512-bundle.json --checkpoint $r2 `
    --split screen --tasks mmlu --max-seconds 550 --output scratch\csa2-eval\onpolicy-r3.mmlu.json 2>&1 |
    Where-Object { $_ -match 'Traceback|Error' }
$s = "C:\Users\Owner\AppData\Local\Temp\claude\D--DeepThought-Projects-HybridModel\2bb2dc81-93cd-457f-b0c5-90fa2d5ae146\scratchpad"
"=== MMLU"
& $py "$s\mmlu_table.py" "source=scratch/csa2-eval/q512-stock.mmlu.json" "think=scratch/csa2-eval/thinking-pass.mmlu.json" "onpolicy2=scratch/csa2-eval/onpolicy-r2.mmlu.json" `
    "onpolicy3=scratch/csa2-eval/onpolicy-r3.mmlu.json"
"=== sandbox $(Get-Date -Format HH:mm)"
foreach ($d in Get-ChildItem $code -Directory -Filter "onpolicy3-*-2k-sampled-s*") {
    $bench = if ($d.Name -match "humaneval") { "humaneval" } else { "mbpp" }
    powershell -NoProfile -ExecutionPolicy Bypass -File "$code\run_docker.ps1" $d.FullName $bench *> "$($d.FullName)\sandbox.log"
}
foreach ($bench in "humaneval", "mbpp") {
    "== thinking, sampled, $bench, 3 seeds"
    & $py "$code\sampled_compare.py" "source=$code\source-$bench-2k-sampled" "think=$code\think-$bench-2k-sampled" "onpolicy2=$code\onpolicy2-$bench-2k-sampled" `
        "onpolicy3=$code\onpolicy3-$bench-2k-sampled"
}
"== looping (MBPP+ seed 0)"
& $py "$code\no_code_audit.py" "$code\source-mbpp-2k-sampled-s0" "$code\think-mbpp-2k-sampled-s0" "$code\onpolicy2-mbpp-2k-sampled-s0" "$code\onpolicy3-mbpp-2k-sampled-s0"
& $py -c @"
import json, os
m = r'$math'
for bench in ('gsm8k', 'math500'):
    for mode in ('think-sampled', 'nothink-greedy'):
        for name in ('source', 'think', 'onpolicy2', 'onpolicy3'):
            p = os.path.join(m, '%s-%s-%s' % (name, bench, mode), 'results.json')
            if os.path.exists(p):
                s = json.load(open(p))['summary']
                print('%-8s %-15s %-9s accuracy %.1f%%  no answer %d  truncated %d  mean %d tok' % (bench, mode, name, 100 * s['accuracy'], s['no_answer'], s['truncated'], s['mean_tokens']))
"@
"=== done $(Get-Date -Format HH:mm)"
