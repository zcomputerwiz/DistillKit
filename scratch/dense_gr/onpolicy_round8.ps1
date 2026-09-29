# Round 8: DPO with verified code. Round 7's pairs (math and general prompts) fixed loops
# and length, but a code answer only had to finish, and code paid (non-thinking HumanEval+
# 44.5% -> 40.9%). Here KodCode problems with pytest suites join the served-format prompts:
# code rollouts run against their tests in the sandbox (verify_code.py: no network, no
# mounts), and a pair's chosen side must pass while rejected sides loop, fail their tests
# or run out of tokens. From the round-6+7 combination (the MATH-500 leader), three samples
# per prompt (two sampled, one greedy), DPO plus chosen-side CE mixed into replay; then
# blended back toward the base and screened.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$C = "$D\capture-data"; $code = "$root\scratch\downstream\code_bench"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$base = "$root\scratch\dense_gr\merges-r67\r7ramp0-70"

"=== code prompts $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\code-prompts-r8.jsonl")) {
    & $py scratch\dense_gr\code_prompts.py --checkpoint $base --count 2500 --output "$C\code-prompts-r8.jsonl" 2>&1 |
        Where-Object { $_ -match 'wrote|Traceback|Error' }
}

function Rollouts([string]$tag, [string]$shard, [string]$seed, [string[]]$extra, [string]$gpu) {
    Start-Job -ArgumentList $tag, $shard, $seed, $extra, $gpu, $py, $root, $base, $C -ScriptBlock {
        param($tag, $shard, $seed, $extra, $gpu, $py, $root, $base, $C)
        $env:CUDA_VISIBLE_DEVICES = $gpu; $env:TORCHINDUCTOR_CACHE_DIR += "-gpu$gpu"
        $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
        Set-Location $root
        $argv = [System.Collections.Generic.List[string]]@("$root\scratch\dense_gr\onpolicy_rollouts.py",
            "--checkpoint", $base, "--inputs", "$C\prompts-r5.jsonl", "$C\code-prompts-r8.jsonl",
            "--count", "100000", "--new", "1024", "--shard", $shard, "--seed", $seed,
            "--output", "$C\onpolicy-r8-$tag.jsonl")
        foreach ($a in $extra) { $argv.Add($a) }
        & $py $argv *> "$C\onpolicy-r8-$tag.log"
    }
}
"=== rollouts $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\onpolicy-r8.jsonl")) {
    $jobs = @()
    if (-not (Test-Path "$C\onpolicy-r8-s0.jsonl")) { $jobs += Rollouts "s0" "0/1" "0" @() "0" }
    if (-not (Test-Path "$C\onpolicy-r8-s1.jsonl")) { $jobs += Rollouts "s1" "0/1" "1" @() "1" }
    if ($jobs) { $jobs | Wait-Job | Receive-Job }
    $jobs = @()
    if (-not (Test-Path "$C\onpolicy-r8-g0.jsonl")) { $jobs += Rollouts "g0" "0/2" "0" @("--greedy") "0" }
    if (-not (Test-Path "$C\onpolicy-r8-g1.jsonl")) { $jobs += Rollouts "g1" "1/2" "0" @("--greedy") "1" }
    if ($jobs) { $jobs | Wait-Job | Receive-Job }
    Get-Content "$C\onpolicy-r8-s0.log", "$C\onpolicy-r8-g0.log" | Select-String "rollouts" | Select-Object -Last 2
    & $py -c "import sys; open(sys.argv[1], 'wb').write(b''.join(open(p, 'rb').read() for p in sys.argv[2:]))" `
        "$C\onpolicy-r8.jsonl" "$C\onpolicy-r8-s0.jsonl" "$C\onpolicy-r8-s1.jsonl" "$C\onpolicy-r8-g0.jsonl" "$C\onpolicy-r8-g1.jsonl"
}

"=== verify code in the sandbox $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\onpolicy-r8-verified.jsonl")) {
    & $py "$code\verify_code.py" --prompts "$C\code-prompts-r8.jsonl" --rollouts "$C\onpolicy-r8.jsonl" `
        --output "$C\onpolicy-r8-verified.jsonl" 2>&1 | Where-Object { $_ -match 'verified|Traceback|Error' }
}

"=== pairs $(Get-Date -Format HH:mm)"
if (-not (Test-Path "$C\pairs-r8.jsonl")) {
    & $py scratch\dense_gr\build_pairs.py "$C\onpolicy-r8-verified.jsonl" --output "$C\pairs-r8.jsonl" 2>&1 |
        Where-Object { $_ -match 'pairs|Traceback|Error' }
}
if (-not (Test-Path "$C\pairs-r8-ref.jsonl")) {
    $env:CUDA_VISIBLE_DEVICES = "0"
    & $py scratch\dense_gr\ref_logprobs.py --reference $base --pairs "$C\pairs-r8.jsonl" --output "$C\pairs-r8-ref.jsonl" 2>&1 |
        Where-Object { $_ -match 'pairs with|Traceback|Error' }
    Remove-Item Env:\CUDA_VISIBLE_DEVICES
}

$caches = @("..\teacher-cache-curriculum-v4", "..\teacher-cache-thinking", "..\teacher-cache-think-first",
            "..\teacher-cache-expand-code", "..\teacher-cache-general-pilot")
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-broken-tools.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-onpolicy-r8.json"
function Train([string]$tag, [string[]]$extra) {
    $argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
        "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
    foreach ($cache in $caches) { $argv.Add($cache) }
    foreach ($a in @("--pairs", "$C\pairs-r8-ref.jsonl", "--pairs-per-step", "3", "--pair-weight", "0.5",
                     "--dpo-beta", "0.1", "--pair-sft-weight", "0.2", "--strip-effort-nonthinking",
                     "--exclude-documents", "..\capture-data\exclude-onpolicy-r8.json", "--suppress-hedges",
                     "--teacher-weight", "0.5", "--teacher-max-length", "1536", "--kl-chunk", "64",
                     "--min-answer-tokens", "2", "--micro-tokens", "3072", "--accumulate", "2",
                     "--tokens", "3000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                     "--evaluate-windows", "64", "--seed", "15",
                     "--checkpoints", "scratch\dense_gr\checkpoints-2b-onpolicy-$tag",
                     "--output", "scratch\dense_gr\train-2b-onpolicy-$tag.json")) { $argv.Add($a) }
    foreach ($a in $extra) { $argv.Add($a) }
    & $py $argv 2>&1 | Where-Object { $_ -match 'pairs:|no pairs|stripped|plan:|held-out|loss .*->|Traceback|Error|spilled|SystemExit' }
}

"=== train $(Get-Date -Format HH:mm)"
Train "r8" @("--evaluate-every", "400", "--report-every", "50", "--save-every", "400")
$r8 = "$root\scratch\dense_gr\checkpoints-2b-onpolicy-r8\smoke-r1-1-gr-s15-csa2"
if (-not (Test-Path "$r8\config.json")) { "training produced no checkpoint; stopping"; exit 1 }

"=== blend screen against the base $(Get-Date -Format HH:mm)"
& .\scratch\dense_gr\merge_search.ps1 -Tuned $r8 -Tag r8 -Base $base
"=== done $(Get-Date -Format HH:mm)"
