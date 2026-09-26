Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
while (Get-Process -Id 20228,10048 -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
$env:PYTHONPATH = "$PWD"; $env:CUDA_VISIBLE_DEVICES = "0,1"
Remove-Item ..\capture-data\rewrite-hedges-pilot.jsonl -ErrorAction SilentlyContinue
& .venv\Scripts\python.exe scratch\dense_gr\rewrite_hedges.py --input ..\capture-data\run5m.jsonl `
    --output ..\capture-data\rewrite-hedges-pilot.jsonl --limit 10 --batch-size 8 *> ..\capture-data\rewrite-pilot.log
"pilot done $(Get-Date -Format HH:mm)"