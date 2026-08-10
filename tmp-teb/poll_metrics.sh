#!/bin/bash
# Snapshot pulsar /metrics every 2 min during the aging run; correlate later.
OUT=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb/aging-metrics.log
while true; do
  echo "=== $(date +%H:%M:%S)" >> "$OUT"
  curl -s -m 5 http://sparky.defense.lan:8099/metrics 2>/dev/null | \
    grep -E "pool|bank|admission|evict|spec|latency|churn|resident" | head -40 >> "$OUT"
  fails=0
  while ! curl -s -m 10 http://sparky.defense.lan:8099/version > /dev/null 2>&1; do
    fails=$((fails + 1))
    if [ $fails -ge 5 ]; then echo "server down 5x, poller exiting" >> "$OUT"; exit 0; fi
    sleep 30
  done
  sleep 120
done
