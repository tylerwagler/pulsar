#!/bin/bash
# Hard-Mode-only bisect ladder for the greedy nondeterminism / quality drop.
# Stage 1: cold-cache vs warm-cache (same binary, defaults) — is the drift
#          the KV disk cache's continued-prefill numerics path?
# Stage 2: all three speed tiers off, cold cache — do the tiers cost P points?
# Cache is wiped and the server rebooted between stages by the caller loop.
set -u
BASE=http://sparky.defense.lan:8099
OUT=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
run() {
  tool-eval-bench run --hardmode-only --seed 43 \
      --base-url $BASE --model deepseek-v4-flash \
      --json-file $OUT/$1.json --no-live \
      > $OUT/$1.log 2> $OUT/$1.progress
  echo "$1 rc=$?"
}
case "${1:?stage}" in
  cold)  run hm-defaults-cold ;;
  warm)  run hm-defaults-warm ;;
  cold2) run hm-defaults-cold2 ;;
  flagsoff) run hm-flagsoff-cold ;;
esac
