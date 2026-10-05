#!/bin/bash
# Watch a long_round*.ps1 run; return as soon as it ends or anything goes wrong:
# the round's PowerShell exits, the round log says it stopped or failed, the trainer log
# shows a traceback, the GPUs spill into system memory, or (while training) a GPU idles.
# In any phase, both GPUs idle for [quiet checks] (default 60, 30 minutes) is a hung worker.
#   bash watch_round.sh <round pid> <round log> <train log> [max seconds] [quiet checks]
pid=$1; round=$2; train=$3; limit=${4:-7000}; quiet_limit=${5:-60}
read16() { iconv -f UTF-16 -t UTF-8 "$1" 2>/dev/null || cat "$1"; }
start=$(date +%s); idle=0; quiet=0
while true; do
    if ! tasklist //FI "PID eq $pid" | grep -q powershell; then
        echo "ROUND EXITED $(date +%H:%M)"; read16 "$round" | grep -vE "held-out  teacher|^\s*$" | tail -25; exit 0
    fi
    if read16 "$round" | grep -qE "stopping|failed|refusing|no candidate passes"; then
        echo "ROUND REPORTS A FAILURE"; read16 "$round" | tail -12; exit 1
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
    # Any phase: the CPU stretches (merges, exclusion lists) end well inside the limit.
    top=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    if [ "${top:-0}" -lt 5 ]; then quiet=$((quiet + 1)); else quiet=0; fi
    if [ $quiet -ge "$quiet_limit" ]; then echo "BOTH GPUS IDLE $((quiet / 2)) MIN $(date +%H:%M)"; read16 "$round" | tail -5; exit 1; fi
    if [ $(( $(date +%s) - start )) -gt "$limit" ]; then
        echo "STILL RUNNING $(date +%H:%M)"; read16 "$train" | grep -E "^step" | tail -1; exit 2
    fi
    sleep 30
done
