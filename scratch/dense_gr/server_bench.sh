#!/bin/bash
# Each llama-server configuration: start it, single-stream and 16-stream decode speed, stop it.
G='C:\Users\Owner\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\4ca720788d1e01f1bff70c033e0d0028fd02e502\Qwen3.8-27B-UD-Q8_K_XL.gguf'
BIN=/d/DeepThought/Projects/HybridModel/llama-bin/b11205/llama-server.exe
PY=/d/DeepThought/Projects/HybridModel/DistillKit/.venv/Scripts/python.exe
BENCH=/d/DeepThought/Projects/HybridModel/DistillKit/scratch/dense_gr/server_bench.py
run() {
  name=$1; shift
  "$BIN" -m "$G" -ngl 99 -fa on -c 65536 -np 16 --port 8090 --host 127.0.0.1 --no-webui "$@" > /d/DeepThought/Projects/HybridModel/capture-data/bench-$name.log 2>&1 &
  until curl -s 127.0.0.1:8090/health | grep -q '"ok"'; do sleep 3; if ! tasklist | grep -q llama-server; then echo "$name: failed to start"; tail -3 /d/DeepThought/Projects/HybridModel/capture-data/bench-$name.log; return; fi; done
  echo "== $name ($*)"
  $PY $BENCH 1 256 2>/dev/null
  $PY $BENCH 16 256 2>/dev/null
  taskkill //IM llama-server.exe //F > /dev/null 2>&1; sleep 5
}
run layer -sm layer
run tensor -sm tensor
run layer-mtp -sm layer --spec-type draft-mtp
run tensor-mtp -sm tensor --spec-type draft-mtp
