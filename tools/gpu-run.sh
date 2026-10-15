#!/usr/bin/env bash
# gpu-run.sh -- run ONE job under the Spark's GPU lock and leave nothing behind (#43).
#
#   tools/gpu-run.sh [-t SECS] [-w SECS] [--battery] -- CMD [ARG...]
#
# What it guards against (both have bitten):
#   - `flock LOCK cmd` (no -o) hands the lock's fd to cmd, and every process cmd starts inherits it: a
#     pulsar-server started in the background keeps the GPU LOCKED after cmd and flock are gone, and
#     /proc/locks names a dead pid as the holder.
#   - `flock -o LOCK cmd` closes the fd for cmd, so nothing inherits it -- but a server cmd leaves running,
#     or a cmd orphaned by a killed flock / a dropped ssh session, keeps the GPU BUSY with the lock free,
#     and the next job loads beside it (the GB10 server-swap garbage, or an OOM).
# So this wrapper holds the lock itself (fd 9, closed for the job: nothing inherits it), runs the job in a
# session of its own, tags every process of it (GPU_RUN_TOKEN in its environment, which a daemonizing child
# keeps), and on ANY exit -- the job's end, -t, Ctrl-C, SIGTERM, SIGHUP -- kills what is left of the job
# before the lock is released.  Only a SIGKILL of this wrapper escapes; tools/gpu-lock-status.sh then shows
# the job's processes as GPU users outside the lock.
#
#   -t SECS    kill the job after SECS (the battery's `timeout 4000`)
#   -w SECS    give up (exit 9) if the lock is not ours within SECS; default: wait
#   --battery  this job IS the battery: touch ~/BATTERY_RUNNING while it runs (removed on exit).  Without it
#              the wrapper refuses (exit 9) while that marker exists, before waiting and again once the lock
#              is ours -- nothing else runs on the box during a battery (the accounting gate measures it).
# GPU_LOCK overrides the lock path (default ~/gpu.lock).  The job's exit code is returned; 9 = not run.
set -u
LOCK=${GPU_LOCK:-$HOME/gpu.lock}
MARKER=$HOME/BATTERY_RUNNING
tmo=0 wait_s="" battery=0
while [ $# -gt 0 ]; do
  case "$1" in
    -t) tmo=$2; shift 2 ;;
    -w) wait_s=$2; shift 2 ;;
    --battery) battery=1; shift ;;
    --) shift; break ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: $0 [-t SECS] [-w SECS] [--battery] -- CMD [ARG...]" >&2; exit 2; }
say() { echo "gpu-run[$$]: $*" >&2; }
battery_blocks() { [ "$battery" = 0 ] && [ -e "$MARKER" ]; }

battery_blocks && { say "a battery is running ($MARKER) -- not queuing"; exit 9; }
exec 9>>"$LOCK" || { say "cannot open $LOCK"; exit 9; }
if ! flock ${wait_s:+-w "$wait_s"} 9; then say "the lock was not ours within ${wait_s}s"; exit 9; fi
if battery_blocks; then say "a battery started while we waited ($MARKER) -- stepping aside"; exit 9; fi
[ "$battery" = 1 ] && touch "$MARKER"

token="gpu-run-$$-$(date +%s%N)"
job=""
# every live process carrying this run's token (the job, its children, anything that daemonized away)
tagged() {
  local p
  for p in /proc/[0-9]*; do
    [ "${p#/proc/}" = "$$" ] && continue
    [ -O "$p" ] || continue   # the job runs as us; another user's environ is unreadable anyway
    grep -qzxF "GPU_RUN_TOKEN=$token" "$p/environ" 2>/dev/null && echo "${p#/proc/}"
  done
}
reap() {
  trap '' INT TERM HUP
  local left p n=0
  left=$(tagged)
  if [ -n "$left" ]; then
    say "the job left $(echo "$left" | wc -w) process(es) running -- terminating:"
    for p in $left; do say "  $(ps -o pid=,etimes=,args= -p "$p" 2>/dev/null | cut -c1-160)"; done
    [ -n "$job" ] && kill -TERM -- "-$job" 2>/dev/null
    kill -TERM $left 2>/dev/null
    while [ $n -lt 20 ] && [ -n "$(tagged)" ]; do sleep 0.5; n=$((n + 1)); done
    left=$(tagged)
    [ -n "$left" ] && { say "  still alive after 10 s, SIGKILL: $left"; kill -KILL $left 2>/dev/null; }
  fi
  [ "$battery" = 1 ] && rm -f "$MARKER"
}
trap 'reap; exit 130' INT
trap 'reap; exit 143' TERM
trap 'reap; exit 129' HUP

# The job: its own session (setsid -- one process group to signal), fd 9 closed (it never holds the lock),
# tagged.  `timeout` inside the session, so -t also ends the whole group.
if [ "$tmo" != 0 ]; then
  GPU_RUN_TOKEN=$token setsid -w timeout --kill-after=30 "$tmo" "$@" 9>&- &
else
  GPU_RUN_TOKEN=$token setsid -w "$@" 9>&- &
fi
job=$!
wait "$job"
rc=$?
reap
exit $rc
