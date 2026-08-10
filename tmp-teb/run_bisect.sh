#!/bin/bash
set -u
BASE=http://sparky.defense.lan:8099
OUT=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
echo "=== BISECT seed43 cold-cache $(date +%H:%M:%S)"
tool-eval-bench run --hardmode --seed 43 \
    --base-url $BASE --model deepseek-v4-flash \
    --json-file $OUT/bisect-s43-cold.json --no-live \
    > $OUT/bisect-s43-cold.log 2> $OUT/bisect-s43-cold.progress
echo "rc=$?"
echo "=== BISECT seed43 warm-cache $(date +%H:%M:%S)"
tool-eval-bench run --hardmode --seed 43 \
    --base-url $BASE --model deepseek-v4-flash \
    --json-file $OUT/bisect-s43-warm.json --no-live \
    > $OUT/bisect-s43-warm.log 2> $OUT/bisect-s43-warm.progress
echo "rc=$?"
echo "BISECT_DONE $(date +%H:%M:%S)"
