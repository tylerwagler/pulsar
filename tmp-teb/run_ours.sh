#!/bin/bash
set -u
BASE=http://sparky.defense.lan:8099
OUT=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
for seed in 42 43 44; do
  echo "=== OURS seed $seed $(date +%H:%M:%S)"
  tool-eval-bench run --hardmode --seed $seed \
      --base-url $BASE --model deepseek-v4-flash \
      --json-file $OUT/ours-seed$seed.json --no-live \
      > $OUT/ours-seed$seed.log 2> $OUT/ours-seed$seed.progress
  echo "rc=$?"
done
for c in 1 2 4; do
  echo "=== OURS perf c$c $(date +%H:%M:%S)"
  tool-eval-bench bench --perf-only --pp 2048 --tg 128 --concurrency $c \
      --base-url $BASE --model deepseek-v4-flash \
      --json-file $OUT/ours-perf-c$c.json --no-live \
      > $OUT/ours-perf-c$c.log 2>&1
  echo "rc=$?"
done
echo "OURS_ALL_DONE $(date +%H:%M:%S)"
