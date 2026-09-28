# Thinking-mode math at the teacher template's reasoning efforts, for one checkpoint.
# xhigh (the default) asks the model to "consider plausible alternatives"; low asks it to
# keep thinking brief; medium sends no system prompt. Round 5b's blend thinks at ~2.4x the
# source's length on GSM8K and truncates more on MATH-500.
#   powershell -File effort_ab.ps1 -Checkpoint <dir> -Tag <name>
param([Parameter(Mandatory)][string]$Checkpoint, [Parameter(Mandatory)][string]$Tag)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $math = "$root\scratch\downstream\math_bench"
$jobs = foreach ($spec in @("0|gsm8k", "1|math500")) {
    Start-Job -ArgumentList $spec, $py, $math, $root, $Checkpoint, $Tag -ScriptBlock {
        param($spec, $py, $math, $root, $Checkpoint, $Tag)
        $gpu, $bench = $spec -split '\|'
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"
        $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        foreach ($effort in "low", "medium") {
            $argv = [System.Collections.Generic.List[string]]@("$math\run_math.py", "--checkpoint", $Checkpoint,
                "--bench", $bench, "--compiled", "--sample", "--reasoning-effort", $effort,
                "--output", "$math\$Tag-$bench-think-sampled-$effort")
            & $py $argv *> "$math\$Tag-$bench-think-sampled-$effort.log"
        }
    }
}
$jobs | Wait-Job | Receive-Job
& $py -c @"
import json, os
m = r'$math'
for bench in ('gsm8k', 'math500'):
    for suffix in ('', '-low', '-medium'):
        p = os.path.join(m, r'$Tag-%s-think-sampled%s' % (bench, suffix), 'results.json')
        if os.path.exists(p):
            s = json.load(open(p))['summary']
            print('%-8s %-7s accuracy %.1f%%  no answer %d  truncated %d  mean %d tok' % (bench, suffix.strip('-') or 'xhigh', 100 * s['accuracy'], s['no_answer'], s['truncated'], s['mean_tokens']))
"@
