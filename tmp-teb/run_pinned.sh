#!/bin/bash
# EVAL PIN acceptance: hardmode-only fresh, then age with a full suite, then
# hardmode-only again. PASS = both hardmode scores byte-identical categories.
set -u
D=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
BASE=http://sparky.defense.lan:8099
bash $D/srv_cycle.sh "--no-kv-disk" "PULSAR_EVAL_PIN=1"
run() {
  tool-eval-bench run $2 --seed $3 \
      --base-url $BASE --model deepseek-v4-flash \
      --json-file $D/$1.json --no-live > $D/$1.log 2> $D/$1.progress
  echo "$1 rc=$?"
}
echo "=== pinned hardmode fresh $(date +%H:%M:%S)"
run pin-hm-fresh --hardmode-only 43
echo "=== pinned full suite (aging) $(date +%H:%M:%S)"
run pin-full-s42 --hardmode 42
echo "=== pinned hardmode aged $(date +%H:%M:%S)"
run pin-hm-aged --hardmode-only 43
echo "PINNED_DONE $(date +%H:%M:%S)"
