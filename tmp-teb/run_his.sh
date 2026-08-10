#!/bin/bash
set -u
BASE=http://sparky.defense.lan:8098
OUT=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
MODEL="${1:?pass his served model id}"
for seed in 42 43 44; do
  echo "=== HIS seed $seed $(date +%H:%M:%S)"
  tool-eval-bench run --hardmode --seed $seed \
      --base-url $BASE --model "$MODEL" \
      --json-file $OUT/his-seed$seed.json --no-live \
      > $OUT/his-seed$seed.log 2> $OUT/his-seed$seed.progress
  echo "rc=$?"
done
for c in 1 2 4; do
  echo "=== HIS perf c$c $(date +%H:%M:%S)"
  tool-eval-bench bench --perf-only --pp 2048 --tg 128 --concurrency $c \
      --base-url $BASE --model "$MODEL" \
      --json-file $OUT/his-perf-c$c.json --no-live \
      > $OUT/his-perf-c$c.log 2>&1
  echo "rc=$?"
done
echo "HIS_ALL_DONE $(date +%H:%M:%S)"
