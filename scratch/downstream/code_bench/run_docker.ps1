# Execute one completions directory in the sandbox: no network, no host mounts.
# Files go in and out with `docker cp`; the container is removed afterwards.
#   powershell -File scratch\downstream\code_bench\run_docker.ps1 <completions dir> <mbpp|humaneval>
param([Parameter(Mandatory)][string]$Dir, [Parameter(Mandatory)][ValidateSet("mbpp", "humaneval")][string]$Bench)
$ErrorActionPreference = "Stop"
$root = "D:\DeepThought\Projects\HybridModel\DistillKit"
$samples = Join-Path $Dir "samples.jsonl"
& "$root\.venv\Scripts\python.exe" -c @"
import json, sys
src, dst, bench = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, encoding='utf-8') as f, open(dst, 'w', encoding='utf-8') as out:
    for line in f:
        r = json.loads(line)
        tid = r['task_id'] if bench == 'humaneval' else 'Mbpp/%s' % r['task_id']
        out.write(json.dumps({'task_id': tid, 'solution': r['code']}) + '\n')
"@ (Join-Path $Dir "completions.jsonl") $samples $Bench
$name = "code-bench-" + [guid]::NewGuid().ToString("N").Substring(0, 8)
docker create --name $name --network none --memory 8g --pids-limit 512 --cpus 8 `
    code-bench-sandbox evalplus.evaluate --dataset $Bench --samples /home/runner/samples.jsonl --parallel 8 | Out-Null
try {
    docker cp $samples "${name}:/home/runner/samples.jsonl"
    docker start -a $name
    docker cp "${name}:/home/runner/samples_eval_results.json" (Join-Path $Dir "eval_results.json")
} finally {
    docker rm -f $name | Out-Null
}
