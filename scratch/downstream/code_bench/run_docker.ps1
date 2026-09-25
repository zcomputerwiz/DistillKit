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
with open(src, encoding='utf-8-sig') as f, open(dst, 'w', encoding='utf-8') as out:
    for line in f:
        r = json.loads(line)
        tid = str(r['task_id'])
        if bench == 'mbpp' and not tid.startswith('Mbpp/'):
            tid = 'Mbpp/' + tid
        out.write(json.dumps({'task_id': tid, 'solution': r['code']}) + '\n')
"@ (Join-Path $Dir "completions.jsonl") $samples $Bench
$name = "code-bench-" + [guid]::NewGuid().ToString("N").Substring(0, 8)
# robust_eval.py rather than `evalplus.evaluate`: one process per sample, so a test that gets
# its worker killed fails that sample instead of hanging the pool. The Docker VM here has
# 3.9 GiB; each test is capped at 512 MiB and the whole run at two hours.
docker create --name $name --network none --memory 3700m --pids-limit 512 --cpus 8 `
    -e EVALPLUS_MAX_MEMORY_BYTES=536870912 code-bench-sandbox `
    timeout 7200 python robust_eval.py run $Bench /home/runner/samples.jsonl /home/runner/results.json 3 | Out-Null
try {
    docker cp $samples "${name}:/home/runner/samples.jsonl"
    docker start -a $name
    docker cp "${name}:/home/runner/results.json" (Join-Path $Dir "eval_results.json")
} finally {
    docker rm -f $name | Out-Null
}
