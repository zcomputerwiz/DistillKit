Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$id = (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'math_queue.ps1' }).ProcessId
if ($id) { while (Get-Process -Id $id -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 } }
$env:PYTHONPATH = "$PWD"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:CUDA_VISIBLE_DEVICES = "0"
& .venv\Scripts\python.exe scratch\downstream\math_bench\arithmetic_probe.py "source=..\student-2b-hf" `
    "control=scratch\dense_gr\checkpoints-2b-borrow-control\smoke-r1-1-gr-s1-csa2" `
    "think=scratch\dense_gr\checkpoints-2b-thinking-pass\smoke-r1-1-gr-s4-csa2" 2>&1 |
    Where-Object { $_ -match '^\s*[-+*/] |op digits|overall|Error|Traceback' }