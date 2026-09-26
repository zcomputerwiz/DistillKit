param([string]$Waits)
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $out = "$PWD\scratch\downstream\code_bench"; $root = "$PWD"
$plan = @{
  "1" = @(@("fix", "$root\scratch\dense_gr\checkpoints-2b-hedge-fix\smoke-r1-1-gr-s3-csa2", "compiled"),
          @("source", "$root\..\student-2b-hf", "hf"))
  "0" = @(@("nofix", "$root\scratch\dense_gr\checkpoints-2b-hedge-nofix\smoke-r1-1-gr-s3-csa2", "compiled"))
}
$jobs = foreach ($gpu in $plan.Keys) {
  Start-Job -ArgumentList $gpu, ($plan[$gpu] | ForEach-Object { $_ -join '|' }), $py, $out, $root, $Waits -ScriptBlock {
    param($gpu, $items, $py, $out, $root, $waits)
    if ($gpu -eq "0") { foreach ($id in ($waits -split ',')) { if ($id) { while (Get-Process -Id ([int]$id) -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 } } } }
    $env:CUDA_VISIBLE_DEVICES = $gpu; $env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
    Set-Location $root
    foreach ($item in $items) {
      $name, $ck, $how = $item -split '\|'
      foreach ($bench in "mbpp", "humaneval") {
        $argv = [System.Collections.Generic.List[string]]@("$out\generate.py", "--checkpoint", $ck, "--bench", $bench,
          "--max-new-tokens", "2048", "--batch-size", "64", "--no-thinking", "--output", "$out\$name-$bench-2k-nt")
        if ($how -eq "compiled") { $argv.Add("--compiled") }
        & $py $argv *> "$out\$name-$bench-2k-nt.log"
      }
    }
  }
}
$jobs | Wait-Job | Receive-Job
foreach ($name in "source", "nofix", "fix") { foreach ($bench in "humaneval", "mbpp") {
  $d = "$out\$name-$bench-2k-nt"
  if (Test-Path "$d\completions.jsonl") { powershell -NoProfile -ExecutionPolicy Bypass -File "$out\run_docker.ps1" $d $bench *> "$d\sandbox.log" } } }
if (Test-Path "$out\nofix-mbpp-2k\completions.jsonl") { powershell -NoProfile -ExecutionPolicy Bypass -File "$out\run_docker.ps1" "$out\nofix-mbpp-2k" mbpp *> "$out\nofix-mbpp-2k\sandbox.log" }
"== thinking, MBPP+"
& $py "$out\compare.py" "source=$out\source-mbpp-2k" "nofix=$out\nofix-mbpp-2k" "fix=$out\fix-mbpp-2k"
foreach ($bench in "humaneval", "mbpp") {
  "== no thinking, $bench"
  & $py "$out\compare.py" "source=$out\source-$bench-2k-nt" "nofix=$out\nofix-$bench-2k-nt" "fix=$out\fix-$bench-2k-nt"
}
foreach ($name in "source", "nofix", "fix") { foreach ($bench in "humaneval", "mbpp") {
  $m = Get-Content "$out\$name-$bench-2k-nt\manifest.json" -ErrorAction SilentlyContinue | ConvertFrom-Json
  if ($m) { "{0,-22} mean tokens {1:N0}  truncated {2}" -f "$name-$bench-nt", $m.mean_generated_tokens, $m.truncations } } }
"done $(Get-Date -Format HH:mm)"