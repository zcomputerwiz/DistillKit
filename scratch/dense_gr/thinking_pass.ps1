# Thinking-weighted finishing pass from the "both" checkpoint, then the full evaluation.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $out = "$root\scratch\downstream\code_bench"
$T = "D:\DeepThought\Projects\HybridModel\teacher-hf"; $D = "D:\DeepThought\Projects\HybridModel"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'thinking_corpus.py' }) { Start-Sleep -Seconds 30 }

"=== capture $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$D\teacher-cache-thinking\manifest.json")) {
    $env:CUDA_VISIBLE_DEVICES = "0,1"
    & $py -m distillkit.sample_transformers --model $T --tokenizer-json "$T\tokenizer.json" `
        --input-jsonl "$D\capture-data\thinking-code-math.jsonl" --output "$D\teacher-cache-thinking" `
        --sequence-length 4096 --top-k 64 --shard-tokens 65536 --int8 --device-map auto *> "$D\capture-data\capture-thinking.log"
    Remove-Item Env:\CUDA_VISIBLE_DEVICES
}
# Only ids that exist in the captures this run reads may be excluded: the contaminated
# think-first copies.
& $py -c @"
import json
ids = [r for r in json.load(open(r'$D\capture-data\exclude-for-think-first.json')) if r.endswith(':tf')]
json.dump(ids, open(r'$D\capture-data\exclude-thinking-pass.json', 'w'), indent=1)
print('excluding', len(ids), 'contaminated think-first copies')
"@
"=== train $(Get-Date -Format HH:mm)"
$both = "scratch\dense_gr\checkpoints-2b-hedge-both\smoke-r1-1-gr-s3-csa2"
& $py scratch\dense_gr\smoke_train.py --init-from $both --inherit --sparse-stage --tensor-parallel `
    --embedding-on away --teacher-cache ..\teacher-cache-think-first ..\teacher-cache-thinking ..\teacher-cache-expand-code `
    --exclude-documents ..\capture-data\exclude-thinking-pass.json --suppress-hedges --teacher-weight 0.5 `
    --teacher-max-length 1024 --kl-chunk 64 --min-answer-tokens 2 --micro-tokens 3072 --accumulate 2 `
    --tokens 3000000 --warmup 50 --decay-fraction 0.5 --decay-floor 0.05 --evaluate-every 400 `
    --evaluate-windows 64 --report-every 50 --save-every 400 --seed 4 `
    --checkpoints scratch\dense_gr\checkpoints-2b-thinking-pass --output scratch\dense_gr\train-2b-thinking-pass.json 2>&1 |
    Where-Object { $_ -match 'suppress|plan:|excluded|held-out|loss .*->|wrote|Traceback|Error|spilled' }
$ck = "$root\scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2"

"=== code generation $(Get-Date -Format HH:mm)"
$jobs = foreach ($spec in @("0|think", "1|nothink")) {
    Start-Job -ArgumentList $spec, $py, $out, $root, $ck -ScriptBlock {
        param($spec, $py, $out, $root, $ck)
        $gpu, $mode = $spec -split '\|'
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        foreach ($bench in "mbpp", "humaneval") {
            $suffix = if ($mode -eq "nothink") { "-nt" } else { "" }
            $argv = [System.Collections.Generic.List[string]]@("$out\generate.py", "--checkpoint", $ck,
                "--bench", $bench, "--max-new-tokens", "2048", "--batch-size", "64", "--compiled",
                "--output", "$out\think-$bench-2k$suffix")
            if ($mode -eq "nothink") { $argv.Add("--no-thinking") }
            & $py $argv *> "$out\think-$bench-2k$suffix.log"
        }
    }
}
$jobs | Wait-Job | Receive-Job

$env:CUDA_VISIBLE_DEVICES = "0"
"=== hedge propensity"
& $py scratch\dense_gr\hedge_propensity.py "source=..\student-2b-hf" "both=$root\scratch\dense_gr\checkpoints-2b-hedge-both\smoke-r1-1-gr-s3-csa2" "think=$ck" 2>&1 |
    Where-Object { $_ -match 'P\(hedge|Error|Traceback' }
Remove-Item Env:\CUDA_VISIBLE_DEVICES
"=== WikiText"
& $py scratch\dense_gr\general_nll.py --arm "source=..\student-2b-hf" --arm "both=$both" --arm "think=$ck" `
    --reference both --output scratch\dense_gr\general-nll-thinking-pass.json 2>&1 |
    Where-Object { $_ -match 'arm |^\S+\s+[0-9.]+\s+[+-]|Error|Traceback' }
foreach ($t in @('nll', 'mmlu')) {
    & $py -m distillkit.independent_eval evaluate --bundle scratch\csa2-eval\q512-bundle.json `
        --checkpoint $ck --split screen --tasks $t --max-seconds 550 `
        --output "scratch\csa2-eval\thinking-pass.$t.json" 2>&1 | Where-Object { $_ -match 'Traceback|Error' }
}
$s = "C:\Users\Owner\AppData\Local\Temp\claude\D--DeepThought-Projects-HybridModel\2bb2dc81-93cd-457f-b0c5-90fa2d5ae146\scratchpad"
"=== MMLU"
& $py "$s\mmlu_table.py" "source=scratch/csa2-eval/q512-stock.mmlu.json" "both=scratch/csa2-eval/hedge-both.mmlu.json" `
    "think=scratch/csa2-eval/thinking-pass.mmlu.json"
"=== in-domain NLL"
& $py "$s\nll_table.py" "source=scratch/csa2-eval/q512-stock.nll.json" "both=scratch/csa2-eval/hedge-both.nll.json" `
    "think=scratch/csa2-eval/thinking-pass.nll.json"
foreach ($d in "think-mbpp-2k", "think-humaneval-2k", "think-mbpp-2k-nt", "think-humaneval-2k-nt") {
    $bench = if ($d -match "humaneval") { "humaneval" } else { "mbpp" }
    if (Test-Path "$out\$d\completions.jsonl") {
        powershell -NoProfile -ExecutionPolicy Bypass -File "$out\run_docker.ps1" "$out\$d" $bench *> "$out\$d\sandbox.log"
    } else { "missing $d" }
}
foreach ($bench in "humaneval", "mbpp") {
    "== thinking, $bench"
    & $py "$out\compare.py" "source=$out\source-$bench-2k" "both=$out\both-$bench-2k" "think=$out\think-$bench-2k"
    "== no thinking, $bench"
    & $py "$out\compare.py" "source=$out\source-$bench-2k-nt" "both=$out\both-$bench-2k-nt" "think=$out\think-$bench-2k-nt"
}
& $py "$out\no_code_audit.py" "$out\source-humaneval-2k" "$out\both-humaneval-2k" "$out\think-humaneval-2k"
foreach ($d in "source-humaneval-2k", "both-humaneval-2k", "think-humaneval-2k", "source-mbpp-2k", "both-mbpp-2k", "think-mbpp-2k") {
    $m = Get-Content "$out\$d\manifest.json" -ErrorAction SilentlyContinue | ConvertFrom-Json
    if ($m) { "{0,-22} mean tokens {1:N0}  truncated {2}" -f $d, $m.mean_generated_tokens, $m.truncations }
}
"=== done $(Get-Date -Format HH:mm)"
