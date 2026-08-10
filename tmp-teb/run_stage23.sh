#!/bin/bash
set -u
D=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
# Stage 2: defaults, fresh server + wiped cache per seed
for s in 42 44; do
  bash $D/srv_cycle.sh "" ""
  bash $D/run_bisect3.sh defaults $s
done
# Stage 3a: disk cache OFF, defaults otherwise, all three seeds
for s in 42 43 44; do
  bash $D/srv_cycle.sh "--no-kv-disk" ""
  bash $D/run_bisect3.sh nodisk $s
done
echo "STAGE23_DONE $(date +%H:%M:%S)"
