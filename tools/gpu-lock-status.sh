#!/usr/bin/env bash
# gpu-lock-status.sh -- who holds the Spark's GPU lock, who waits, and what is wrong (#43).
#
#   tools/gpu-lock-status.sh            (GPU_LOCK overrides the lock path, default ~/gpu.lock)
#
# Read-only.  Prints the holder and the queue from /proc/locks, then three findings, each an exit-1 anomaly:
#   ORPHANED LOCK   the pid /proc/locks names as the holder is gone, yet the lock is held: a process that
#                   inherited the lock's fd holds it (`flock LOCK cmd` without -o, cmd started a server);
#                   those processes are listed -- ending them frees the GPU.
#   INHERITED FD    a process other than a holder or a waiter has the lock file open (the same leak, still
#                   behind a live holder).
#   OUTSIDE LOCK    a GPU compute process (nvidia-smi) that does not descend from the holder: a job left
#                   running after its lock was released (`flock -o` with a backgrounded server, or a killed
#                   flock / tools/gpu-run.sh), or a job that never took the lock.
# A holder whose parent is gone (reparented to init: its ssh session or script died) is reported, not
# counted: `flock -o` jobs launched with nohup look exactly like that and are fine while they run.
set -u
LOCK=${GPU_LOCK:-$HOME/gpu.lock}
[ -e "$LOCK" ] || { echo "no lock file $LOCK"; exit 0; }
ino=$(stat -c %i "$LOCK")
dev=$(stat -c %D "$LOCK")
dev=$(printf '%02x:%02x' $((16#$dev >> 8)) $((16#$dev & 255)))   # /proc/locks spells MAJ:MIN in hex
bad=0
desc() {   # pid -> "pid ppid etime args", or "pid (gone)"
  ps -o pid=,ppid=,etime=,args= -p "$1" 2>/dev/null | sed 's/^ *//' | cut -c1-170 || true
}
alive() { [ -d "/proc/$1" ]; }

holders=() waiters=()
while read -r line; do
  set -- $line
  if [ "$2" = "->" ]; then waiters+=("$6"); else holders+=("$5"); fi
done < <(grep -E "FLOCK +ADVISORY +WRITE +[0-9]+ +$dev:$ino " /proc/locks)

# every process with the lock file open: a flock(1) that took the lock or waits for it, a tools/gpu-run.sh
# wrapper (it opens the lock as its own fd 9 and lets a short-lived `flock 9` child take it -- /proc/locks then
# names that child, gone, as the holder: the wrapper is the holder), or a process that INHERITED the fd
openers=()
for p in /proc/[0-9]*; do
  for f in "$p"/fd/*; do
    [ "$(readlink "$f" 2>/dev/null)" = "$LOCK" ] && { openers+=("${p#/proc/}"); break; }
  done
done
is_wrapper() { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q 'gpu-run\.sh'; }
waits_via_child() { local c; for c in $(pgrep -P "$1"); do case " ${waiters[*]} " in *" $c "*) return 0 ;; esac; done; return 1; }

echo "GPU lock $LOCK (inode $ino): ${#holders[@]} holder(s), ${#waiters[@]} waiting"
live_holders=()
show_holder() {
  live_holders+=("$1")
  echo "  HOLD  $(desc "$1")"
  [ "$(ps -o ppid= -p "$1" | tr -d ' ')" = 1 ] &&
    echo "        (parent gone: reparented to init -- fine for a nohup'd job, check it is still wanted)"
  for c in $(pgrep -P "$1"); do echo "        job $(desc "$c")"; done
}
wrappers_holding=()
for o in "${openers[@]}"; do is_wrapper "$o" && ! waits_via_child "$o" && wrappers_holding+=("$o"); done
for h in "${holders[@]}"; do
  if alive "$h"; then
    show_holder "$h"
  elif [ ${#wrappers_holding[@]} -gt 0 ]; then
    for w in "${wrappers_holding[@]}"; do show_holder "$w"; done
  else
    echo "  HOLD  $h (gone) -- ORPHANED LOCK: the lock lives on in an inherited fd"
    bad=1
  fi
done
for w in "${waiters[@]}"; do
  pw=$(ps -o ppid= -p "$w" 2>/dev/null | tr -d ' ')
  if [ -n "$pw" ] && is_wrapper "$pw"; then echo "  WAIT  $(desc "$pw")"; else echo "  WAIT  $(desc "$w")"; fi
done

# an opener that is no holder, no waiter and no wrapper inherited the fd
listed=" ${holders[*]} ${waiters[*]} "
for o in "${openers[@]}"; do
  case "$listed" in *" $o "*) continue ;; esac
  is_wrapper "$o" && continue
  echo "  INHERITED FD  $(desc "$o")  -- holds the lock's fd; it keeps the lock past its parent"
  bad=1
done

# GPU users that do not descend from a live holder
descends() {   # pid ancestor
  local p=$1 n=0
  while [ "$p" -gt 1 ] && [ $n -lt 64 ]; do
    [ "$p" = "$2" ] && return 0
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$p" ] || return 1
    n=$((n + 1))
  done
  return 1
}
if command -v nvidia-smi > /dev/null; then
  while IFS=, read -r gpid mem name; do
    gpid=$(echo "$gpid" | tr -d ' ')
    [ -n "$gpid" ] || continue
    inside=0
    for h in "${live_holders[@]}"; do descends "$gpid" "$h" && inside=1; done
    if [ $inside = 1 ]; then
      echo "  GPU   $gpid ${mem# } (under the holder)"
    else
      echo "  OUTSIDE LOCK  GPU process $(desc "$gpid") using${mem} -- not under the lock's holder"
      bad=1
    fi
  done < <(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null)
fi
[ -e "$HOME/BATTERY_RUNNING" ] && echo "  battery marker present: $HOME/BATTERY_RUNNING"
[ $bad = 0 ] && echo "clean" || echo "ANOMALY (see above)"
exit $bad
