# Two arms from the control checkpoint, identical but for the hedge fixes:
#   nofix  the five captures as they are
#   fix    the 210 de-hedged documents in place of their originals, and --suppress-hedges
param([int]$WaitFor = 0)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$env:PYTHONPATH = "$PWD"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$py = ".venv\Scripts\python.exe"
if ($WaitFor) { while (Get-Process -Id $WaitFor -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 } }
$control = "scratch\dense_gr\checkpoints-2b-borrow-control\smoke-r1-1-gr-s1-csa2"
$caches = @("..\teacher-cache-5m", "..\teacher-cache-expand-code", "..\teacher-cache-expand-chat",
            "..\teacher-cache-general-pilot", "..\teacher-cache-general-scale")
$arms = [ordered]@{
    "nofix" = @{ caches = $caches; exclude = "..\capture-data\run5m-contamination.json"; extra = @() }
    "fix"   = @{ caches = $caches + @("..\teacher-cache-dehedged");
                 exclude = "..\capture-data\exclude-contamination-and-dehedged.json";
                 extra = @("--suppress-hedges") }
}
foreach ($name in $arms.Keys) {
    $out = "scratch\dense_gr\train-2b-hedge-$name.json"
    if (Test-Path $out) { "skip $name"; continue }
    "=== train $name $(Get-Date -Format HH:mm)"
    $a = $arms[$name]
    & $py scratch\dense_gr\smoke_train.py --init-from $control --inherit --sparse-stage `
        --tensor-parallel --embedding-on away --teacher-cache $a.caches `
        --exclude-documents $a.exclude --teacher-weight 0.5 --teacher-max-length 1024 `
        --kl-chunk 64 --min-answer-tokens 2 --micro-tokens 3072 --accumulate 2 `
        --tokens 3000000 --warmup 50 --decay-fraction 0.3 --decay-floor 0.1 `
        --evaluate-every 400 --evaluate-windows 64 --report-every 50 --save-every 400 --seed 3 `
        --checkpoints "scratch\dense_gr\checkpoints-2b-hedge-$name" --output $out @($a.extra) 2>&1 |
        Where-Object { $_ -match 'suppress|held-out|loss .*->|wrote|Traceback|Error|spilled' }
}
$ck = @{}
foreach ($name in $arms.Keys) { $ck[$name] = "scratch\dense_gr\checkpoints-2b-hedge-$name\smoke-r1-1-gr-s3-csa2" }
"=== hedge propensity $(Get-Date -Format HH:mm)"
$env:CUDA_VISIBLE_DEVICES = "0"
& $py scratch\dense_gr\hedge_propensity.py "source=..\student-2b-hf" "control=$control" `
    "nofix=$($ck.nofix)" "fix=$($ck.fix)" 2>&1 | Where-Object { $_ -match 'P\(hedge|documents|Error|Traceback' }
Remove-Item Env:\CUDA_VISIBLE_DEVICES
"=== WikiText"
& $py scratch\dense_gr\general_nll.py --arm "source=..\student-2b-hf" --arm "nofix=$($ck.nofix)" `
    --arm "fix=$($ck.fix)" --reference nofix --output scratch\dense_gr\general-nll-hedge.json 2>&1 |
    Where-Object { $_ -match 'arm |^\S+\s+[0-9.]+\s+[+-]|Error|Traceback' }
foreach ($name in $arms.Keys) {
    foreach ($t in @('nll', 'mmlu')) {
        & $py -m distillkit.independent_eval evaluate --bundle scratch\csa2-eval\q512-bundle.json `
            --checkpoint $ck[$name] --split screen --tasks $t --max-seconds 550 `
            --output ("scratch\csa2-eval\hedge-{0}.{1}.json" -f $name, $t) 2>&1 | Where-Object { $_ -match 'Traceback|Error' }
    }
}
$s = "C:\Users\Owner\AppData\Local\Temp\claude\D--DeepThought-Projects-HybridModel\2bb2dc81-93cd-457f-b0c5-90fa2d5ae146\scratchpad"
"=== MMLU"
& $py "$s\mmlu_table.py" "source=scratch/csa2-eval/q512-stock.mmlu.json" "control=scratch/csa2-eval/borrow-control.mmlu.json" `
    "nofix=scratch/csa2-eval/hedge-nofix.mmlu.json" "fix=scratch/csa2-eval/hedge-fix.mmlu.json"
"=== in-domain NLL"
& $py "$s\nll_table.py" "source=scratch/csa2-eval/q512-stock.nll.json" "nofix=scratch/csa2-eval/hedge-nofix.nll.json" `
    "fix=scratch/csa2-eval/hedge-fix.nll.json"
"=== done $(Get-Date -Format HH:mm)"
