# Two cheap checks on the thinking pass, GSM8K thinking sampled: Qwen's presence penalty
# against loops (GPU 0), and the teacher template's xhigh system prompt that every thinking
# corpus carried in training (GPU 1). Baseline: think-gsm8k-think-sampled.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $math = "$root\scratch\downstream\math_bench"
$think = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"
$xhigh = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
$jobs = foreach ($spec in @("0|presence", "1|xhigh")) {
    Start-Job -ArgumentList $spec, $py, $math, $root, $think, $xhigh -ScriptBlock {
        param($spec, $py, $math, $root, $think, $xhigh)
        $gpu, $arm = $spec -split '\|'
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"
        $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        $argv = [System.Collections.Generic.List[string]]@("$math\run_math.py", "--checkpoint", $think,
            "--bench", "gsm8k", "--compiled", "--sample", "--output", "$math\think-gsm8k-think-sampled-$arm")
        if ($arm -eq "presence") { $argv.Add("--presence"); $argv.Add("1.5") } else { $argv.Add("--system"); $argv.Add($xhigh) }
        & $py $argv *> "$math\think-gsm8k-think-sampled-$arm.log"
    }
}
$jobs | Wait-Job | Receive-Job
& $py -c @"
import json, os
m = r'$math'
for arm in ('', '-presence', '-xhigh'):
    p = os.path.join(m, 'think-gsm8k-think-sampled%s' % arm, 'results.json')
    if os.path.exists(p):
        s = json.load(open(p))['summary']
        print('%-10s accuracy %.1f%%  no answer %d  truncated %d  mean %d tok' % (arm or 'baseline', 100 * s['accuracy'], s['no_answer'], s['truncated'], s['mean_tokens']))
"@
"=== done $(Get-Date -Format HH:mm)"
