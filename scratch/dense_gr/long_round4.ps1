# Long-context round 4: round 3 plus a fix for the thinking loops it brought. Round 3 (raw
# code under KL, QA answers only) gained long-code NLL at every length and its QA answers
# by 0.51 nats, but its long MATH thinking loops: 31 of 256 answers unfinished and looping
# at 4,096 tokens against the base's 5 (math_truncation.py). The teacher cannot be the
# signal there -- given a loop it continues it, 85% at the first repeat, 97% by the sixth
# (loop_teacher.py) -- so round 4 adds:
#  - the teacher's own thinking on 9,212 MATH/GSM8K train problems (teacher_generate.py
#    via llama.cpp), graded correct, finished and loop-free, then ranked by a frontier judge
#    (trace_judge.py; Codex spot check agreed with 85% of its keeps); only the kept ones
#    train, with cross entropy and KL alike, since the text is the teacher's own
#  - looping rollouts under unlikelihood (--unlikelihood-caches): the earlier on-policy
#    loop captures and round 3's own loops (loop-check), which cut truncations ~40% in the
#    September rounds
# Then the loop test joins the probe, QA eval and blend screen as a gate.
# Round 3's notes: round 2 distilled long code from inside a user turn, where the chat
# teacher expects the person to stop typing at every line break (framing_check.py); round 3
# moved it to raw text under KL alone and scored the QA conversations on answers only.
# Round 1 notes, still true:
# (SmolDataEnvs: Claude Code, OpenCode, Codex, mini-swe-agent; assistant turns only) mixed
# with the replay captures, all recaptured with int8 weights and bf16 activations
# (capture_w8.ps1). Recipe from the research pass (DISTILLATION.md, 2026-10-01):
#  - one forward per document up to 32K, every layer checkpointed; no sparse stage, so the
#    CSA2 indexer keeps its selection (pass-key is already 100% to 32K) and memory is linear
#  - DeltaNet decay gates at a tenth of the rate (SpectralShift): they hold the long memory
#  - replay keeps short-context skills (ProLong keeps ~40% short data)
# Then the long-context probe, the held-out QA answers and a blend screen against the base;
# the finalists' benchmark suite is a separate, explicit step after choosing a blend.
Set-Location "D:\DeepThought\Projects\HybridModel\DistillKit"
$py = "$PWD\.venv\Scripts\python.exe"; $root = "$PWD"; $D = "D:\DeepThought\Projects\HybridModel"
$C = "$D\capture-data"
$env:PYTHONPATH = $root; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
$base = "$root\scratch\dense_gr\merges-long1\u50"
# The agent capture was stopped at a third and its finished shards salvaged (-a); the rest
# was captured again with documents overlapped across the cards (-b).
$agentA = "..\teacher-cache-agent-smol-a"; $agentB = "..\teacher-cache-agent-smol-b"
$code = "..\teacher-cache-r8-code-short-w8"
$qa = "..\teacher-cache-frontier-qa2"; $raw = "..\teacher-cache-frontier-code-raw"; $tools = "..\teacher-cache-frontier-tools"
$traces = "..\teacher-cache-teacher-math-gen"
# r5-loop is left out: it shares 103 document ids with r6-loop, with different rollouts
# (Codex review r4), and r6 is the larger and later set.
$loops = @("..\teacher-cache-onpolicy-r6-loop", "..\teacher-cache-loop-check")
$caches = @($agentA, $agentB, $qa, $raw, $tools, $code, "..\teacher-cache-curriculum-v4-w8", "..\teacher-cache-thinking-w8",
            "..\teacher-cache-think-first-w8", "..\teacher-cache-expand-code-w8", "..\teacher-cache-general-pilot-w8",
            $traces) + $loops
foreach ($cache in $caches) {
    if (-not (Test-Path "$D\$($cache.Substring(3))\manifest.json")) { "missing capture $cache; stopping"; exit 1 }
}
# Judged exclusions, rebuilt now so they take every Codex verdict written by the start:
# Codex's verdict decides wherever it has one (it overrules the frontier judge either way,
# and its own "wrong" excludes on its own); the frontier judge decides the rest.
function Check([string]$what) { if ($LASTEXITCODE -ne 0) { "$what failed (exit $LASTEXITCODE); stopping"; exit 1 } }
if (Test-Path "$root\scratch\dense_gr\checkpoints-2b-long-r4") { "checkpoints-2b-long-r4 exists; refusing to mix runs"; exit 1 }
$F = "$C\frontier"
& $py scratch\frontier\judge_docs.py collect --responses "$F\judge-responses.jsonl" "$F\judge-batched-responses.jsonl" `
    --second "$F\codex-crosscheck\judge-codex.jsonl" "$F\codex-missing\judge-missing.jsonl" `
    "$F\codex-crosscheck2\judge-codex.jsonl" "$F\codex-crosscheck3\judge-codex.jsonl" --output "$C\exclude-judged.json"
Check "judged exclusions"
# Plus the QA documents whose source code spells chat markup (capture_inputs.py): it
# tokenizes to real turn and tool tokens inside the user's document.
# And the teacher traces the frontier judge did not keep (or never judged): the ranked index
# and its exclusion list, rebuilt from every verdict so far.
& $py scratch\frontier\trace_judge.py collect --responses "$F\trace-judge-responses.jsonl" `
    --traces "$C\teacher-gen-math.jsonl" --output "$C\teacher-gen-keep.json" --exclusions "$C\exclude-teacher-gen.json"
Check "trace keep list"
# And looping rollouts where the unlikelihood detector marks nothing (the thought closed
# before the loop, or the repeats vary): they would train KL over the whole loop.
& $py scratch\dense_gr\loop_negatives.py --caches ($loops | ForEach-Object { $_ }) --output "$C\exclude-zero-negative-loops.json"
Check "zero-negative loops"
# And documents carrying a problem of the loop gate's fresh bank (MATH repeats some problems
# across its splits): 43 at review time, in r6-loop, the traces and the thinking replay.
& $py scratch\dense_gr\fresh_bank_overlap.py --caches ($loops + @($traces, "..\teacher-cache-thinking-w8",
    "..\teacher-cache-think-first-w8", "..\teacher-cache-curriculum-v4-w8") | ForEach-Object { $_ }) `
    --output "$C\exclude-fresh-bank-overlap.json"
Check "fresh-bank overlap"
& $py scratch\dense_gr\exclusion_for.py --master "$C\exclude-master-v2.json" "$C\exclude-judged.json" "$C\frontier-qa2-markup.json" `
    "$C\exclude-teacher-gen.json" "$C\exclude-zero-negative-loops.json" "$C\exclude-fresh-bank-overlap.json" `
    --caches ($caches | ForEach-Object { $_ }) --output "$C\exclude-long-r4.json"
Check "exclusion list"
"=== train $(Get-Date -Format HH:mm)"
$argv = [System.Collections.Generic.List[string]]@("scratch\dense_gr\smoke_train.py", "--init-from", $base,
    "--inherit", "--tensor-parallel", "--embedding-on", "away", "--checkpoint-layers", "--teacher-cache")
foreach ($c2 in $caches) { $argv.Add($c2) }
# The QA answers (structural spans with their closing <|im_end|>) weigh 8 each (3x round 2's answers at half the per-token weight: ~5% of all weight),
# the budget counting weight; the raw code is KL alone. Padding to the block multiple keeps
# the final turns flooring cut (28% of tool turns) and short replay documents. The teacher's
# own traces are ordinary documents (cross entropy and KL); the looping rollouts are KL-only
# with unlikelihood, and no KL, on tokens repeating an earlier n-gram inside the thought.
$argv.Add("--unlikelihood-caches"); foreach ($l in $loops) { $argv.Add($l) }
foreach ($a in @("--assistant-only-caches", $agentA, $agentB, $tools, $qa, "--kl-only-caches", $raw,
                 "--ce-only-caches", $code,
                 "--repeat", "$agentA=3", "$agentB=3", "$tools=4", "$code=2", "..\teacher-cache-curriculum-v4-w8=2",
                 "..\teacher-cache-thinking-w8=2", "..\teacher-cache-think-first-w8=2",
                 "..\teacher-cache-expand-code-w8=2", "..\teacher-cache-general-pilot-w8=2",
                 # The repair data repeated so 8M tokens sees enough of it (Codex's seed-24
                 # simulation saw 574 traces and 10 of round 3's loops once without).
                 "$traces=2", "..\teacher-cache-onpolicy-r6-loop=2", "..\teacher-cache-loop-check=4", "--pad-to-block",
                 "--answer-spans", "$C\frontier-qa2.jsonl", "--answer-weight", "8",
                 "--lr-scale", "linear_attn\.(A_log|dt_bias|in_proj_a)=0.1", "--strip-effort-nonthinking",
                 "--exclude-documents", "..\capture-data\exclude-long-r4.json", "--suppress-hedges",
                 "--teacher-weight", "0.5", "--teacher-max-length", "32768", "--kl-chunk", "64",
                 "--min-answer-tokens", "2", "--micro-tokens", "32768", "--accumulate", "2",
                 "--tokens", "8000000", "--warmup", "50", "--decay-fraction", "0.5", "--decay-floor", "0.05",
                 "--evaluate-windows", "64", "--evaluate-every", "100", "--report-every", "25",
                 "--save-every", "100", "--seed", "24",
                 "--checkpoints", "scratch\dense_gr\checkpoints-2b-long-r4",
                 "--output", "scratch\dense_gr\train-2b-long-r4.json")) { $argv.Add($a) }
& $py $argv 2>&1 | Tee-Object -FilePath "$root\scratch\dense_gr\long-r4-train.log" |
    Where-Object { $_ -match 'learning rates|plan:|stripped|held-out|loss .*->|Traceback|Error|spilled|SystemExit' }
Check "training"
$tuned = "$root\scratch\dense_gr\checkpoints-2b-long-r4\smoke-r1-1-gr-s24-csa2"
if (-not (Test-Path "$tuned\config.json")) { "training produced no checkpoint; stopping"; exit 1 }

$long3 = "$root\scratch\dense_gr\checkpoints-2b-long-r3\smoke-r1-1-gr-s23-csa2"
"=== long-context probe $(Get-Date -Format HH:mm)"
# Round 3 in the same run: the probe's code documents are llama.cpp's live source tree.
& $py scratch\dense_gr\long_context_probe.py --arm "long1-u50=$base" --arm "long3=$long3" --arm "long4=$tuned" `
    --output scratch\csa2-eval\long-context-long4.json 2>&1 | Where-Object { $_ -match '^==|^  ' -and $_ -notmatch 'warn|torch.nn' }
Check "long-context probe"
"=== held-out QA answers $(Get-Date -Format HH:mm)"
# The original eight questions per document (comparable with round 2), then every checked
# question as round 3 trained them.
foreach ($q in "first", "all") {
    & $py scratch\frontier\qa_answer_eval.py --arm "long1-u50=$base" --arm "long3=$long3" `
        --arm "long4=$tuned" --questions $q --exclude "$C\exclude-long-r4.json" --output "scratch\csa2-eval\qa-answers-long4-$q.json"
    Check "QA answer eval ($q)"
}
# The blend screen and the loop gate (finish_round.ps1, also runnable on its own).
& .\scratch\dense_gr\finish_round.ps1 -Tag long4 -Tuned $tuned -Base $base
Check "blend screen and loop gate"
"=== done $(Get-Date -Format HH:mm)"
