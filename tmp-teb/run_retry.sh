#!/bin/bash
set -u
D=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
tool-eval-bench run --hardmode --seed 42 \
    --scenarios TC-14 TC-38 TC-43 TC-45 TC-51 TC-57 TC-58 TC-60 TC-62 TC-66 \
                TC-69 TC-71 TC-72 TC-73 TC-74 TC-75 TC-76 TC-80 TC-82 TC-83 TC-84 \
    --base-url http://sparky.defense.lan:8099 --model deepseek-v4-flash \
    --json-file $D/retry-fails.json --no-live \
    > $D/retry-fails.log 2> $D/retry-fails.progress
echo "rc=$?"
echo "RETRY_DONE $(date +%H:%M:%S)"
