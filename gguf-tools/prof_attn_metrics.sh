#!/bin/bash
# Hardware-counter profile of the prefill attention kernel, which is now the
# largest single cost (1652 ms of 5160 at a 4096-token prefill, 32%) after the
# MoE work.  --gpu-metrics-devices needs root on GB10 (ERR_NVGPUCTRPERM as a
# normal user); nsys does NOT replay kernels, so this stays memory-safe on a
# loaded model.
#
# The question this answers, BEFORE any kernel is written: is our attention
# compute-bound (tensor cores would help, as Entrpi's HMMA kernel suggests) or
# bandwidth/latency-bound on the sparse KV gather (they would not)?  Getting
# this backwards is exactly what produced the 2026-07-22 MoE NO-GO.
set -u
M=/srv/models/v5mx4-0731-mmqaligned.gguf
P=/home/claude/pulsar-work/speed-bench/promessi_sposi.txt
OUT=/home/claude/ab-results
export TMPDIR=/home/claude/nsys-tmp
export PULSAR_LOCK_FILE=/home/claude/pulsar-claude.lock
mkdir -p "$TMPDIR" "$OUT"

sync; echo 3 > /proc/sys/vm/drop_caches; sleep 3

nsys profile \
  --gpu-metrics-devices=all \
  --trace=cuda \
  --sample=none \
  --force-overwrite true \
  -o "$OUT/attn-metrics" \
  /home/claude/pulsar-mmq/pulsar-bench -m "$M" \
    --prompt-file "$P" \
    --ctx-start 4096 --ctx-max 4096 --gen-tokens 0 \
    --prefill-chunk 4096 \
  > "$OUT/attn-metrics.log" 2>&1
echo "nsys rc=$?"
echo "ATTN_METRICS_DONE"
