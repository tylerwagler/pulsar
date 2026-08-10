#!/bin/bash
# Stage 2: seed sensitivity of Hard Mode P on a fresh server per seed
# (defaults). Stage 3 variants rebooted by the driver with --no-kv-disk or
# flags-off to kill the suspected coupling vector.
set -u
BASE=http://sparky.defense.lan:8099
OUT=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
tag="${1:?tag}"; seed="${2:?seed}"
tool-eval-bench run --hardmode-only --seed $seed \
    --base-url $BASE --model deepseek-v4-flash \
    --json-file $OUT/hm-$tag-s$seed.json --no-live \
    > $OUT/hm-$tag-s$seed.log 2> $OUT/hm-$tag-s$seed.progress
echo "hm-$tag-s$seed rc=$?"
