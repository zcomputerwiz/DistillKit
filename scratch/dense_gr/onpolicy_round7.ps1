# Round 7: DPO on the current model's own failures, as an arm beside round 6 from the same
# base. Pairs are the shortest acceptable rollout of a prompt (chosen) over its looping or
# truncated rollouts (rejected), all sampled from the base at served settings -- round 6's
# four per prompt plus one greedy pass, which loops more -- so every negative is one the
# model plausibly produces. Reference log-probs from the base; DPO plus cross entropy on
# the chosen side, mixed into replay; unlikelihood off, to isolate the preference term.
# Starts after round 6. Then blended back toward the base and screened.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$base = "$root\scratch\dense_gr\merges-r5b\ramp0-70"

"=== greedy rollouts $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\greedy-r7.jsonl")) {
    $jobs = foreach ($shard in "0", "1") {
        Start-Job -ArgumentList $shard, $py, $root, $base, $C -ScriptBlock {
            param($shard, $py, $root, $base, $C)
            $env:CUDA_VISIBLE_DEVICES = $shard; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$shard"
            $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
            Set-Location $root
            $argv = [System.Collections.Generic.List[string]]@("$root\scratch\dense_gr\onpolicy_rollouts.py",
                "--checkpoint", $base, "--inputs", "$C\prompts-r5.jsonl", "--count", "100000", "--greedy",
                "--new", "1024", "--shard", "$shard/2", "--output", "$C\greedy-r7-$shard.jsonl")
            & $py $argv *> "$C\greedy-r7-$shard.log"
        }
    }
    $jobs | Wait-Job | Receive-Job
    & $py -c "import sys; open(sys.argv[1], 'wb').write(b''.join(open(p, 'rb').read() for p in sys.argv[2:]))" `
        "$C\greedy-r7.jsonl" "$C\greedy-r7-0.jsonl" "$C\greedy-r7-1.jsonl"
}

"=== pairs $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\pairs-r7.jsonl")) {
    & $py scratch\dense_gr\build_pairs.py "$C\onpolicy-r6.jsonl" "$C\greedy-r7.jsonl" --output "$C\pairs-r7.jsonl" 2>$null
}
if (-not (Test-Path "$C\pairs-r7-ref.jsonl")) {
    $env:CUDA_VISIBLE_DEVICES = "0"
    & $py scratch\dense_gr\ref_logprobs.py --reference $base --pairs "$C\pairs-r7.jsonl" --output "$C\pairs-r7-ref.jsonl" 2>&1 | Where-Object { $_ -match 'pairs with|Traceback|Error' }
    Remove-Item Env:\CUDA_VISIBLE_DEVICES
}

$caches = @("..\teacher-cache-curriculum-v4", "..\teacher-cache-thinking", "..\teacher-cache-think-first",
            "..\teacher-cache-expand-code", "..\teacher-cache-general-pilot")
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-broken-tools.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-onpolicy-r7.json"
function Train([string]$tag, [string[]]$extra) {
    $argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
        "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
    foreach ($cache in $caches) { $argv.Add($cache) }
    foreach ($a in @("--pairs", "$C\pairs-r7-ref.jsonl", "--pairs-per-step", "2", "--pair-weight", "0.5",
                     "--dpo-beta", "0.1", "--pair-sft-weight", "0.2", "--strip-effort-nonthinking",
                     "--exclude-documents", "..\capture-data\exclude-onpolicy-r7.json", "--suppress-hedges",
                     "--teacher-weight", "0.5", "--teacher-max-length", "1536", "--kl-chunk", "64",
                     "--min-answer-tokens", "2", "--micro-tokens", "3072", "--accumulate", "2",
                     "--tokens", "3000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                     "--evaluate-windows", "64", "--seed", "14",
                     "--checkpoints", "scratch\dense_gr\checkpoints-2b-onpolicy-$tag",
                     "--output", "scratch\dense_gr\train-2b-onpolicy-$tag.json")) { $argv.Add($a) }
    foreach ($a in $extra) { $argv.Add($a) }
    & $py $argv 2>&1 | Where-Object { $_ -match 'pairs:|no pairs|stripped|plan:|held-out|loss .*->|step |Traceback|Error|spilled|SystemExit' }
}

"=== smoke: 12 steps with pairs $(Get-Date -Format HH:mm)"
Train "r7-smoke" @("--max-steps", "12", "--evaluate-every", "100000", "--report-every", "4", "--save-every", "100000")
if (-not (Test-Path "$root\scratch\dense_gr\train-2b-onpolicy-r7-smoke.json")) { "smoke run failed; stopping"; exit 1 }
& $py -c "import json; h = json.load(open(r'scratch\dense_gr\train-2b-onpolicy-r7-smoke.json'))['history']; print([{k: round(r[k], 4) for k in ('step', 'loss', 'dpo', 'dpo_margin', 'chosen_logp') if k in r} for r in h])"

"=== train $(Get-Date -Format HH:mm)"
Train "r7" @("--evaluate-every", "400", "--report-every", "50", "--save-every", "400")
$r7 = "$root\scratch\dense_gr\checkpoints-2b-onpolicy-r7\smoke-r1-1-gr-s14-csa2"
if (-not (Test-Path "$r7\config.json")) { "training produced no checkpoint; stopping"; exit 1 }

"=== blend screen against the base $(Get-Date -Format HH:mm)"
& .\scratch\dense_gr\merge_search.ps1 -Tuned $r7 -Tag r7 -Base $base
"=== done $(Get-Date -Format HH:mm)"
