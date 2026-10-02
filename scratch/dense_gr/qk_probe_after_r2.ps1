# After long round 2: the long-context probe for its QK-Restore arm (merge_search's "qk":
# the tuned run with the base's attention query/key maps) and its half blend. The blend
# screen's proxies are all short, and long-range recall is what QK-Restore is for. Round
# 2's own probe already scored the base and the tuned run on the same fixed documents.
#   powershell -File qk_probe_after_r2.ps1 [-WaitFor <pid of long_round2.ps1>]
param([int]$WaitFor = 0)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
if ($WaitFor) { while (Get-Process -Id $WaitFor -ErrorAction SilentlyContinue) { Start-Sleep 60 } }
$m = "$root\scratch\dense_gr\merges-long2"
$arms = foreach ($name in "qk", "u50") { if (Test-Path "$m\$name\model.safetensors") { "--arm"; "long2-$name=$m\$name" } }
if (-not $arms) { "no round-2 blends in $m; stopping"; exit 1 }
"=== long-context probe $(Get-Date -Format HH:mm)"
& $py scratch\dense_gr\long_context_probe.py $arms --output scratch\csa2-eval\long-context-long2-qk.json 2>&1 |
    Where-Object { $_ -match '^==|^  |Traceback|Error' -and $_ -notmatch 'warn|torch.nn' }
if ($LASTEXITCODE -ne 0) { "probe failed (exit $LASTEXITCODE)"; exit 1 }
"=== done $(Get-Date -Format HH:mm)"
