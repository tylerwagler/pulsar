#!/bin/bash
set -u
D=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
tool-eval-bench run --hardmode --seed 42 \
    --base-url http://sparky.defense.lan:8099 --model deepseek-v4-flash \
    --json-file $D/confirm-full-s42.json --no-live \
    > $D/confirm-full-s42.log 2> $D/confirm-full-s42.progress
echo "rc=$?"
echo "CONFIRM_DONE $(date +%H:%M:%S)"
