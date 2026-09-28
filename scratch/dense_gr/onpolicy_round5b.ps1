# Round 5b: round 5's rollouts and captures, retrained with the effort text stripped only
# where the reply does not think (--strip-effort-nonthinking). The 5m, expand-code and
# expand-chat captures were rendered in thinking mode around non-thinking replies, so round
# 5 learned non-thinking code with a system prompt serving never sends; the screen's
# served-format code NLL rose 0.765 -> 0.848 while the as-captured one fell. Then blended
# back toward the base and screened with the served-format proxy.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$T = "$D\teacher-hf"; $code = "$root\scratch\downstream\code_bench"; $math = "$root\scratch\downstream\math_bench"
$C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$think = "$root\scratch\dense_gr\merges-r2\ramp0-70-tt"

"=== train $(Get-Date -Format HH:mm)"
$caches = @("..\teacher-cache-onpolicy-r5-clean", "..\teacher-cache-onpolicy-r5-loop", "..\teacher-cache-curriculum-v4",
            "..\teacher-cache-thinking", "..\teacher-cache-think-first", "..\teacher-cache-expand-code",
            "..\teacher-cache-general-pilot")
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-broken-tools.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-onpolicy-r5.json"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $think,
    "--inherit", "--sparse-stage", "--tensor-parallel", "--embedding-on", "away", "--teacher-cache")
foreach ($cache in $caches) { $argv.Add($cache) }
# 1536 = the rollouts' 512 prompt + 1024 new tokens, so a clean rollout keeps its ending
foreach ($a in @("--unlikelihood-caches", "..\teacher-cache-onpolicy-r5-loop", "--strip-effort-nonthinking",
                 "--exclude-documents", "..\capture-data\exclude-onpolicy-r5.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "1536", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "3072", "--accumulate", "2",
                 "--tokens", "3000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-every", "400", "--evaluate-windows", "64", "--report-every", "50",
                 "--save-every", "400", "--seed", "12",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-onpolicy-r5b",
                 "--output", "scratch\dense_gr\train-2b-onpolicy-r5b.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Where-Object { $_ -match 'on-policy|stripped|suppress|plan:|excluded|held-out|loss .*->|wrote|Traceback|Error|spilled' }
$r5 = "$root\scratch\dense_gr\checkpoints-2b-onpolicy-r5b\smoke-r1-1-gr-s12-csa2"
if (-not (Test-Path "$r5\config.json")) { "training produced no checkpoint; stopping"; exit 1 }


"=== blend screen against the base $(Get-Date -Format HH:mm)"
& .\scratch\dense_gr\merge_search.ps1 -Tuned $r5 -Tag r5b -Base $think
"=== done $(Get-Date -Format HH:mm)"