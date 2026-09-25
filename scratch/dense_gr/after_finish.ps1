# Queue after the finishing pass: the own-rotary FUFUFU arm, scoring, code generation.
param([int]$WaitFor = 0)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$env:PYTHONPATH = "$PWD"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$py = ".venv\Scripts\python.exe"
$s = "C:\Users\Owner\AppData\Local\Temp\claude\D--DeepThought-Projects-HybridModel\2bb2dc81-93cd-457f-b0c5-90fa2d5ae146\scratchpad"
if ($WaitFor) { while (Get-Process -Id $WaitFor -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 } }
$finish = "scratch\dense_gr\checkpoints-2b-finish\smoke-r1-1-gr-s2-csa2"
$ownrot = "scratch\dense_gr\checkpoints-2b-borrow-fufufu-ownrot\smoke-r1-1-gr-s1-csa2"
$control = "scratch\dense_gr\checkpoints-2b-borrow-control\smoke-r1-1-gr-s1-csa2"

"=== train fufufu-ownrot $(Get-Date -Format HH:mm)"
if (-not (Test-Path "scratch\dense_gr\train-2b-borrow-fufufu-ownrot.json")) {
    & $py scratch\dense_gr\smoke_train.py --init-from scratch\dense_gr\checkpoints-2b-borrow\fufufu-ownrot `
        --inherit --sparse-stage --tensor-parallel --embedding-on away `
        --teacher-cache ..\teacher-cache-5m ..\teacher-cache-expand-code ..\teacher-cache-expand-chat `
            ..\teacher-cache-general-pilot ..\teacher-cache-general-scale `
        --exclude-documents ..\capture-data\run5m-contamination.json --teacher-weight 0.5 `
        --teacher-max-length 1024 --kl-chunk 64 --min-answer-tokens 2 --micro-tokens 3072 `
        --accumulate 2 --tokens 3000000 --warmup 50 --decay-fraction 0.3 --decay-floor 0.1 `
        --evaluate-every 400 --evaluate-windows 64 --report-every 50 --save-every 400 --seed 1 `
        --checkpoints scratch\dense_gr\checkpoints-2b-borrow-fufufu-ownrot `
        --output scratch\dense_gr\train-2b-borrow-fufufu-ownrot.json 2>&1 |
        Where-Object { $_ -match 'held-out|loss .*->|wrote|Traceback|Error|spilled' }
}

"=== WikiText $(Get-Date -Format HH:mm)"
& $py scratch\dense_gr\general_nll.py --arm "source=..\student-2b-hf" --arm "control=$control" `
    --arm "finish=$finish" --arm "fufufu-ownrot=$ownrot" --reference control `
    --output scratch\dense_gr\general-nll-finish.json 2>&1 |
    Where-Object { $_ -match 'arm |^\S+\s+[0-9.]+\s+[+-]|Error|Traceback' }
foreach ($arm in @(@{n='finish';c=$finish}, @{n='fufufu-ownrot';c=$ownrot})) {
    foreach ($t in @('nll', 'mmlu')) {
        & $py -m distillkit.independent_eval evaluate --bundle scratch\csa2-eval\q512-bundle.json `
            --checkpoint $arm.c --split screen --tasks $t --max-seconds 550 `
            --output ("scratch\csa2-eval\{0}.{1}.json" -f $arm.n, $t) 2>&1 | Where-Object { $_ -match 'Traceback|Error' }
    }
}
"=== MMLU"
& $py "$s\mmlu_table.py" "source=scratch/csa2-eval/q512-stock.mmlu.json" "control=scratch/csa2-eval/borrow-control.mmlu.json" `
    "finish=scratch/csa2-eval/finish.mmlu.json" "fufufu=scratch/csa2-eval/borrow-fufufu.mmlu.json" `
    "fufufu-ownrot=scratch/csa2-eval/fufufu-ownrot.mmlu.json"
foreach ($pair in @(@('finish','q512-stock','source'), @('finish','borrow-control','control'), @('fufufu-ownrot','borrow-control','control'))) {
    & $py -m distillkit.independent_eval report --reference ("scratch\csa2-eval\{0}.mmlu.json" -f $pair[1]) `
        --results ("scratch\csa2-eval\{0}.mmlu.json" -f $pair[0]) --bootstrap 10000 `
        --output ("scratch\csa2-eval\report-{0}-vs-{1}.json" -f $pair[0], $pair[2]) 2>&1 | Out-Null
}
& $py -c "import json; [print('MMLU  %-13s - %-7s %+.4f  [%+.4f, %+.4f]' % (a, b, x['estimate'], *x['ci95'])) for a, b in (('finish','source'),('finish','control'),('fufufu-ownrot','control')) for x in json.load(open('scratch/csa2-eval/report-%s-vs-%s.json' % (a, b)))['rows'] if x.get('task')=='mmlu' and x.get('metric')=='acc' and x.get('comparison')=='enabled - pre_retrofit']"
"=== in-domain NLL"
& $py "$s\nll_table.py" "source=scratch/csa2-eval/q512-stock.nll.json" "control=scratch/csa2-eval/borrow-control.nll.json" `
    "finish=scratch/csa2-eval/finish.nll.json" "fufufu=scratch/csa2-eval/borrow-fufufu.nll.json" `
    "fufufu-ownrot=scratch/csa2-eval/fufufu-ownrot.nll.json"

"=== code generation $(Get-Date -Format HH:mm)"
$out = "scratch\downstream\code_bench"
# A short smoke first: batched, left-padded generation on the CSA2 model.
$env:CUDA_VISIBLE_DEVICES = "0"
& $py "$out\generate.py" --checkpoint $finish --bench humaneval --limit 4 --max-new-tokens 64 `
    --output "$out\smoke-finish" 2>&1 | Where-Object { $_ -match 'wrote|Traceback|Error' }
$jobs = @()
foreach ($spec in @(@{gpu='0'; ck=$finish; name='finish'}, @{gpu='1'; ck='..\student-2b-hf'; name='source'})) {
    $cmd = "`$env:CUDA_VISIBLE_DEVICES='$($spec.gpu)'; `$env:PYTHONPATH='$PWD'; Set-Location '$PWD'; " +
           "foreach (`$b in 'mbpp','humaneval') { & '$py' '$out\generate.py' --checkpoint '$($spec.ck)' --bench `$b --output ('$out\$($spec.name)-' + `$b) 2>&1 | Select-String 'wrote|Traceback|Error' }"
    $jobs += Start-Job -ScriptBlock ([scriptblock]::Create($cmd))
}
$jobs | Wait-Job | Receive-Job
"=== done $(Get-Date -Format HH:mm)"
