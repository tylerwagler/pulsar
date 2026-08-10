#!/bin/bash
# Age a fresh server with one full-suite run, then measure Hard Mode on the
# aged instance. Fresh-server hardmode-only baseline: P=17/30 (8 identical
# passes). If P < 17 here, accumulated server state degrades quality.
set -u
D=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
BASE=http://sparky.defense.lan:8099
bash $D/srv_cycle.sh "" ""
echo "=== aging run: full suite seed 42 $(date +%H:%M:%S)"
tool-eval-bench run --hardmode --seed 42 \
    --base-url $BASE --model deepseek-v4-flash \
    --json-file $D/aging-full-s42.json --no-live \
    > $D/aging-full-s42.log 2> $D/aging-full-s42.progress
echo "full rc=$?"
echo "=== aged-server hardmode probe $(date +%H:%M:%S)"
tool-eval-bench run --hardmode-only --seed 43 \
    --base-url $BASE --model deepseek-v4-flash \
    --json-file $D/hm-AGED-s43.json --no-live \
    > $D/hm-AGED-s43.log 2> $D/hm-AGED-s43.progress
echo "aged rc=$?"
echo "AGING_DONE $(date +%H:%M:%S)"
