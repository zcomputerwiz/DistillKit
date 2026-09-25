# Three arms from the scale checkpoint, identical budget, data order and settings:
# the all-Full control, and two Reuse patterns picked by borrow_profile.py.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$env:PYTHONPATH = "$PWD"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$py = ".venv\Scripts\python.exe"
$arms = [ordered]@{
    "control" = "scratch\dense_gr\checkpoints-2b-scale\smoke-r1-1-gr-s0-csa2"
    "ffuffu"  = "scratch\dense_gr\checkpoints-2b-borrow\ffuffu"
    "fufufu"  = "scratch\dense_gr\checkpoints-2b-borrow\fufufu"
}
foreach ($name in $arms.Keys) {
    $out = "scratch\dense_gr\train-2b-borrow-$name.json"
    if (Test-Path $out) { "skip $name (done)"; continue }
    "=== train $name $(Get-Date -Format HH:mm)"
    & $py scratch\dense_gr\smoke_train.py --init-from $arms[$name] --inherit --sparse-stage `
        --tensor-parallel --embedding-on away `
        --teacher-cache ..\teacher-cache-5m ..\teacher-cache-expand-code ..\teacher-cache-expand-chat `
            ..\teacher-cache-general-pilot ..\teacher-cache-general-scale `
        --exclude-documents ..\capture-data\run5m-contamination.json --teacher-weight 0.5 `
        --teacher-max-length 1024 --kl-chunk 64 --min-answer-tokens 2 --micro-tokens 3072 `
        --accumulate 2 --tokens 3000000 --warmup 50 --decay-fraction 0.3 --decay-floor 0.1 `
        --evaluate-every 400 --evaluate-windows 64 --report-every 50 --save-every 400 --seed 1 `
        --checkpoints "scratch\dense_gr\checkpoints-2b-borrow-$name" --output $out 2>&1 |
        Where-Object { $_ -match 'held-out|held [0-9]|loss .*->|wrote|Traceback|Error|spilled' }
}
$ck = @{}
foreach ($name in $arms.Keys) { $ck[$name] = "scratch\dense_gr\checkpoints-2b-borrow-$name\smoke-r1-1-gr-s0-csa2" }
"=== WikiText $(Get-Date -Format HH:mm)"
& $py scratch\dense_gr\general_nll.py --arm "source=..\student-2b-hf" --arm "control=$($ck.control)" `
    --arm "ffuffu=$($ck.ffuffu)" --arm "fufufu=$($ck.fufufu)" --reference control `
    --output scratch\dense_gr\general-nll-borrow.json 2>&1 |
    Where-Object { $_ -match 'arm |^\S+\s+[0-9.]+\s+[+-]|Error|Traceback' }
foreach ($name in $arms.Keys) {
    foreach ($t in @('nll', 'mmlu')) {
        & $py -m distillkit.independent_eval evaluate --bundle scratch\csa2-eval\q512-bundle.json `
            --checkpoint $ck[$name] --split screen --tasks $t --max-seconds 550 `
            --output "scratch\csa2-eval\borrow-$name.$t.json" 2>&1 | Where-Object { $_ -match 'Traceback|Error' }
    }
}
$s = "C:\Users\Owner\AppData\Local\Temp\claude\D--DeepThought-Projects-HybridModel\2bb2dc81-93cd-457f-b0c5-90fa2d5ae146\scratchpad"
"=== MMLU"
& $py "$s\mmlu_table.py" "source=scratch/csa2-eval/q512-stock.mmlu.json" "control=scratch/csa2-eval/borrow-control.mmlu.json" `
    "ffuffu=scratch/csa2-eval/borrow-ffuffu.mmlu.json" "fufufu=scratch/csa2-eval/borrow-fufufu.mmlu.json"
foreach ($name in @('ffuffu', 'fufufu')) {
    & $py -m distillkit.independent_eval report --reference scratch\csa2-eval\borrow-control.mmlu.json `
        --results "scratch\csa2-eval\borrow-$name.mmlu.json" --bootstrap 10000 `
        --output "scratch\csa2-eval\report-borrow-$name-vs-control.json" 2>&1 | Out-Null
}
& $py -c "import json; [print('MMLU  %-7s - control %+.4f  [%+.4f, %+.4f]' % (r, x['estimate'], *x['ci95'])) for r in ('ffuffu','fufufu') for x in json.load(open('scratch/csa2-eval/report-borrow-%s-vs-control.json' % r))['rows'] if x.get('task')=='mmlu' and x.get('metric')=='acc' and x.get('comparison')=='enabled - pre_retrofit']"
"=== in-domain NLL"
& $py "$s\nll_table.py" "source=scratch/csa2-eval/q512-stock.nll.json" "control=scratch/csa2-eval/borrow-control.nll.json" `
    "ffuffu=scratch/csa2-eval/borrow-ffuffu.nll.json" "fufufu=scratch/csa2-eval/borrow-fufufu.nll.json"
"=== done $(Get-Date -Format HH:mm)"
