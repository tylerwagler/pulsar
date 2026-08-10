#!/bin/bash
D=/home/claude/Projects/pulsar/.claude/worktrees/flashinfer-attn/tmp-teb
T="/tmp/claude-1000/-home-claude-Projects-pulsar--claude-worktrees-mmq-port/996df3fb-1f6a-4e11-8f63-7f10de20b529/tasks/b3kky8184.output"
prev=""
while true; do
  cur=$({ ls "$D"/*.json 2>/dev/null; grep -hE "rc=[0-9]|OURS_ALL_DONE" "$T" 2>/dev/null; } | sort -u)
  comm -13 <(echo "$prev") <(echo "$cur") 2>/dev/null | grep -v "^$"
  prev="$cur"
  grep -hE "Traceback|Connection refused|failed to connect" "$D"/ours-*.log 2>/dev/null | sort -u | head -2
  if grep -q OURS_ALL_DONE "$T" 2>/dev/null; then break; fi
  sleep 60
done
echo "watch: ours driver finished"
