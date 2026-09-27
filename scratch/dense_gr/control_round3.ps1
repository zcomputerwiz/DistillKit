# Control for the on-policy rounds: the same continuation from the thinking pass -- same
# schedule, cap, stripping, hedge suppression, 3M tokens -- on the replayed captures alone,
# no rollouts and no curriculum. Every round lost HumanEval+ against the thinking pass,
# and held-out code NLL rose in each; this says whether continuing to train does that by
# itself, or what the rounds added does.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$code = "$root\scratch\downstream\code_bench"; $math = "$root\scratch\downstream\math_bench"
$C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$think = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"
"=== train $(Get-Date -Format HH:mm)"
$caches = @("..\teacher-cache-thinking", "..\teacher-cache-think-first", "..\teacher-cache-expand-code",
            "..\teacher-cache-general-pilot")
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-broken-tools.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-control-r3.json"
& $py -c "import json, sys; json.dump(sorted(set(json.load(open(sys.argv[1]))) | set(json.load(open(sys.argv[2])))), open(sys.argv[3], 'w'), indent=1)" `
    "$C\exclude-control-r3.json" "$C\exclude-control-r3.json" "$C\exclude-control-r3b.json"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $think,
    "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
foreach ($cache in $caches) { $argv.Add($cache) }
# 1536 = the rollouts' 512 prompt + 1024 new tokens, so a clean rollout keeps its ending
foreach ($a in @("--strip-effort-prompt",
                 "--exclude-documents", "..\capture-data\exclude-control-r3b.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "1536", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "3072", "--accumulate", "2",
                 "--tokens", "3000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-every", "400", "--evaluate-windows", "64", "--report-every", "50",
                 "--save-every", "400", "--seed", "8",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-control-r3",
                 "--output", "scratch\dense_gr\train-2b-control-r3.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Where-Object { $_ -match 'on-policy|stripped|suppress|plan:|excluded|held-out|loss .*->|wrote|Traceback|Error|spilled' }
$r2 = "$root\scratch\dense_gr\checkpoints-2b-control-r3\smoke-r1-1-gr-s8-csa2"
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
                "--compiled", "--output", "$code\control-$bench-2k-sampled-s$seed")
            & $py $argv *> "$code\control-$bench-2k-sampled-s$seed.log"
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
                "--bench", $bench, "--compiled", "--output", "$math\control-$bench-$mode")
            if ($mode -eq "think-sampled") { $argv.Add("--sample") } else { $argv.Add("--no-thinking") }
            & $py $argv *> "$math\control-$bench-$mode.log"
        }
    }
}
$jobs | Wait-Job | Receive-Job
$env:CUDA_VISIBLE_DEVICES = "0"
foreach ($bench in "humaneval", "mbpp") {
    foreach ($d in Get-ChildItem $code -Directory -Filter "control-$bench-2k-sampled-s*") {
        powershell -NoProfile -ExecutionPolicy Bypass -File "$code\run_docker.ps1" $d.FullName $bench *> "$($d.FullName)\sandbox.log"
    }
    "== thinking, sampled, $bench, 3 seeds"
    & $py "$code\sampled_compare.py" "source=$code\source-$bench-2k-sampled" "think=$code\think-$bench-2k-sampled" `
        "onpolicy2=$code\onpolicy2-$bench-2k-sampled" "onpolicy3=$code\onpolicy3-$bench-2k-sampled" "control=$code\control-$bench-2k-sampled"
}
& $py -c @"
import json, os
m = r'$math'
for bench in ('gsm8k', 'math500'):
    for mode in ('think-sampled', 'nothink-greedy'):
        for name in ('source', 'think', 'onpolicy2', 'onpolicy3', 'control'):
            p = os.path.join(m, '%s-%s-%s' % (name, bench, mode), 'results.json')
            if os.path.exists(p):
                s = json.load(open(p))['summary']
                print('%-8s %-15s %-9s accuracy %.1f%%  no answer %d  truncated %d  mean %d tok' % (bench, mode, name, 100 * s['accuracy'], s['no_answer'], s['truncated'], s['mean_tokens']))
"@
"=== done $(Get-Date -Format HH:mm)"