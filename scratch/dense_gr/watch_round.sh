#!/bin/bash
# Watch a long_round*.ps1 run; return as soon as it ends or anything goes wrong:
# the round log says it stopped or failed, the round's PowerShell exits (0 only when its log
# ends on "=== done"), the trainer log shows a traceback, the GPUs spill into system memory,
# a GPU idles while training, or both GPUs idle for [quiet checks] (default 60, 30 minutes)
# in a GPU phase -- not before the log's first "===" line (exclusion lists), nor in
# "=== sandbox" (Docker tests) or "=== blend screen" (merges): those run on the CPU.
# Nested scripts end on their own lines (merge_search: "=== blend screen done", finish_round:
# "=== round checks done"), so "=== done" is the watched script's own; [done line] for another.
#   bash watch_round.sh <round pid> <round log> <train log> [max seconds] [quiet checks] [done line]
pid=$1; round=$2; train=$3; limit=${4:-7000}; quiet_limit=${5:-60}; done_line=${6:-=== done}
# PowerShell's logs are UTF-16 with a byte-order mark; anything else is read as is (iconv
# from UTF-16 "succeeds" on an even-length UTF-8 file, as nonsense).
read16() {
    local bom; bom=$(head -c2 "$1" 2>/dev/null | od -An -tx1 | tr -d ' \n')
    if [ "$bom" = "fffe" ] || [ "$bom" = "feff" ]; then iconv -f UTF-16 -t UTF-8 "$1" 2>/dev/null; else cat "$1" 2>/dev/null; fi
}
# The wrappers' own failure lines, not a summary's JSON key ("failed": 12).
failure='stopping|(^|[^"])failed($|[^"])|refusing|no candidate passes|did not come up|move it aside'
failed() { if read16 "$round" | grep -qE "$failure"; then echo "ROUND REPORTS A FAILURE"; read16 "$round" | tail -12; exit 1; fi; }
start=$(date +%s); idle=0; quiet=0
while true; do
    failed
    if ! tasklist //FI "PID eq $pid" | grep -q powershell; then
        failed  # a failure line written just before the exit
        if read16 "$round" | grep -a "^===" | tail -1 | grep -qF "$done_line"; then
            echo "ROUND EXITED $(date +%H:%M)"; read16 "$round" | grep -vE "held-out  teacher|^\s*$" | tail -25; exit 0
        fi
        echo "ROUND EXITED WITHOUT $done_line $(date +%H:%M)"; read16 "$round" | tail -12; exit 1
    fi
    if read16 "$train" | grep -qE "^Traceback"; then
        echo "TRAINER TRACEBACK"; read16 "$train" | grep -A8 "^Traceback" | tail -12; exit 1
    fi
    spill=$(powershell -NoProfile -Command "(Get-Counter '\GPU Adapter Memory(*)\Shared Usage').CounterSamples | ? { \$_.InstanceName -like '*1262e*' } | % { [int](\$_.CookedValue/1MB) } | Measure-Object -Maximum | % Maximum")
    if [ "${spill:-0}" -gt 2048 ]; then echo "SPILL ${spill} MB $(date +%H:%M)"; exit 1; fi
    # Idle GPUs only count while training: the merges after it run on the CPU, and the
    # loop test's odd arm leaves one GPU free (finish_round logs never train at all).
    if read16 "$round" | grep -q "=== train" && ! read16 "$round" | grep -q "=== long-context probe"; then
        u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | head -1)
        if [ "${u:-0}" -lt 5 ]; then idle=$((idle + 1)); else idle=0; fi
        if [ $idle -ge 10 ]; then echo "A GPU IDLE 5 MIN DURING TRAINING"; read16 "$train" | tail -3; exit 1; fi
    fi
    phase=$(read16 "$round" | grep -a "^===" | tail -1)
    top=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    if [ -z "$phase" ] || echo "$phase" | grep -qE "^=== (sandbox|blend screen)" || [ "${top:-0}" -ge 5 ]; then
        quiet=0
    else
        quiet=$((quiet + 1))
    fi
    if [ $quiet -ge "$quiet_limit" ]; then echo "BOTH GPUS IDLE $((quiet / 2)) MIN IN $phase"; read16 "$round" | tail -5; exit 1; fi
    if [ $(( $(date +%s) - start )) -gt "$limit" ]; then
        echo "STILL RUNNING $(date +%H:%M)"; read16 "$train" | grep -E "^step" | tail -1; exit 2
    fi
    sleep 30
done
