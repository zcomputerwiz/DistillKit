Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
powershell -NoProfile -ExecutionPolicy Bypass -File scratch\downstream\code_bench\verify_compiled.ps1 -WaitFor 25084 *> scratch\downstream\code_bench\verify-compiled.log
powershell -NoProfile -ExecutionPolicy Bypass -File scratch\dense_gr\hedge_arms.ps1 *> scratch\dense_gr\hedge-arms.log