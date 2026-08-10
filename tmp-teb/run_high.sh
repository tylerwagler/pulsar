#!/bin/bash
# Level-matched head-to-head: ours pinned + reasoning_effort=high, full
# 84-scenario suite, vs his default-HIGH 85/85/85.
set -u
D=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
BASE=http://sparky.defense.lan:8099
echo "=== pinned HIGH full suite $(date +%H:%M:%S)"
tool-eval-bench run --hardmode --seed 42 \
    --backend-kwargs '{"reasoning_effort": "high"}' \
    --base-url $BASE --model deepseek-v4-flash \
    --json-file $D/pin-high-s42.json --no-live \
    > $D/pin-high-s42.log 2> $D/pin-high-s42.progress
echo "rc=$?"
echo "HIGH_DONE $(date +%H:%M:%S)"
