#!/bin/bash
set -u
D=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
echo "=== pressure 0.75 @ 393216 $(date +%H:%M:%S)"
tool-eval-bench run --short --seed 42 \
    --context-pressure 0.75 --context-size 393216 \
    --base-url http://sparky.defense.lan:8099 --model deepseek-v4-flash \
    --json-file $D/pressure-75.json --no-live \
    > $D/pressure-75.log 2> $D/pressure-75.progress
echo "rc=$?"
echo "=== control: pressure 0 $(date +%H:%M:%S)"
tool-eval-bench run --short --seed 42 \
    --base-url http://sparky.defense.lan:8099 --model deepseek-v4-flash \
    --json-file $D/pressure-0.json --no-live \
    > $D/pressure-0.log 2> $D/pressure-0.progress
echo "rc=$?"
echo "PRESSURE_DONE $(date +%H:%M:%S)"
