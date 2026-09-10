param(
    [string]$OutputDirectory = 'scratch/eval-screen',
    [int]$Documents = 128,
    [int]$Questions = 128,
    [string]$Device = 'cuda:0'
)
$ErrorActionPreference = 'Stop'
$evalPython = '.venv/Scripts/python.exe'
$evalTable = (Get-Content examples/_lr_sweep_base.yml | Select-String '^  table_path: ').Line.Substring(14).Trim()
$env:HF_DATASETS_CACHE = "$PWD/scratch/eval-datasets-cache"

function Invoke-Evaluation {
    & $evalPython -m distillkit.independent_eval @args
    if ($LASTEXITCODE -ne 0) { throw "Evaluator failed with exit code $LASTEXITCODE" }
}

Invoke-Evaluation prepare --tokenizer ../student-hf --documents ../capture-data/heldout.jsonl --manifests ../teacher-cache-1m/manifest.json ../teacher-cache-5m/manifest.json --docs $Documents --questions $Questions --document-tokens 512 --min-assistant-tokens 64 --output "$OutputDirectory/bundle.json"
# A tiny run checks all three scoring paths before committing the screen's budget.
Invoke-Evaluation evaluate --bundle "$OutputDirectory/bundle.json" --checkpoint ../runs/gr-stage1-1m --table $evalTable --device $Device --limit 2 --output "$OutputDirectory/tiny-gr.json"
Invoke-Evaluation evaluate --bundle "$OutputDirectory/bundle.json" --checkpoint ../student-hf --device $Device --output "$OutputDirectory/student-hf.json"
foreach ($evalName in @('gr-stage1-1m', 'ple-stage1-1m', 'lr-sweep-1e3')) {
    Invoke-Evaluation evaluate --bundle "$OutputDirectory/bundle.json" --checkpoint "../runs/$evalName" --table $evalTable --device $Device --output "$OutputDirectory/$evalName.json"
}
Invoke-Evaluation report --reference "$OutputDirectory/student-hf.json" --results "$OutputDirectory/student-hf.json" "$OutputDirectory/gr-stage1-1m.json" "$OutputDirectory/ple-stage1-1m.json" "$OutputDirectory/lr-sweep-1e3.json" --stage1-checkpoints gr-stage1-1m ple-stage1-1m --output "$OutputDirectory/report.json"
