#!/bin/bash
# One fmtcmp GPU job on Sparky for an EXL3 arm (exllamav3's own forward; pulsar does not run Nemotron yet):
#
#   gpuq submit --class short --timeout 20m --mem 90G --label fmtcmp --wait -- \
#       ~/fmtcmp/tools/sparky_exl3_job.sh capture PACK ~/fmtcmp/caps/exl3-k4 [BUDGET_S]
#   gpuq submit ... -- ~/fmtcmp/tools/sparky_exl3_job.sh agentic PACK ~/fmtcmp/teb-k4 [BUDGET_S] -- <tool-eval-bench args>
#
# capture: capture_exl3.py over the frozen suite, --selfcheck on the first job of a pack (the chunk plumbing
#          proof), resumable per document; it stops starting documents 60 s before the budget.
# agentic: exl3_openai_shim.py on port 8301 + tool-eval-bench (pinned b60e2f0) with the same args as the NVFP4 arm;
#          pass --resume RUN_ID to continue.
# PY defaults to the exllamav3 fork's venv on Sparky (~/inkling/.venv; the fork at ~/inkling/exllamav3).
set -euo pipefail
MODE=${1:?mode: capture|agentic}
PACK=${2:?EXL3 pack dir}
OUT=${3:?output dir}
BUDGET=${4:-1080}
shift 4 || shift $#
[ "${1:-}" = "--" ] && shift
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$HOME/inkling/.venv/bin/python}
EXL3=${EXL3:-$HOME/inkling/exllamav3}
SUITE=${SUITE:-/mnt/models/fmtcmp/suite-v1}
PORT=${PORT:-8301}
T0=$(date +%s)
[ "$PORT" = 8000 ] && { echo "refusing port 8000 (production)"; exit 2; }
mkdir -p "$OUT" ~/fmtcmp/logs
export PYTHONPATH=$EXL3${PYTHONPATH:+:$PYTHONPATH}
echo "[fmtcmp-exl3] $(date -Is) $MODE pack=$PACK budget=${BUDGET}s"

if [ "$MODE" = capture ]; then
  SC=()
  [ -e "$OUT/.selfcheck-pass" ] || SC=(--selfcheck)
  rc=0
  timeout --signal=TERM $(( BUDGET - 20 )) "$PY" "$HERE/capture_exl3.py" --model "$PACK" --suite "$SUITE" \
    --out "$OUT" --deadline $(( T0 + BUDGET - 60 )) "${SC[@]}" || rc=$?
  if [ "$rc" = 0 ] && [ ${#SC[@]} -gt 0 ]; then touch "$OUT/.selfcheck-pass"; fi
  exit "$rc"
elif [ "$MODE" = agentic ]; then
  LOG=~/fmtcmp/logs/exl3-shim-$$.log
  "$PY" "$HERE/exl3_openai_shim.py" --model "$PACK" --port "$PORT" > "$LOG" 2>&1 &
  SHIM=$!
  trap 'kill $SHIM 2>/dev/null || true; wait $SHIM 2>/dev/null || true' EXIT
  until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
    kill -0 $SHIM 2>/dev/null || { tail -30 "$LOG"; exit 3; }
    [ $(( $(date +%s) - T0 )) -lt $(( BUDGET - 120 )) ] || { echo "shim not up in budget"; exit 4; }
    sleep 5
  done
  LEFT=$(( BUDGET - ($(date +%s) - T0) - 30 ))
  echo "[fmtcmp-exl3] shim up after $(( $(date +%s) - T0 ))s; client budget ${LEFT}s"
  cd "$OUT"
  rc=0
  timeout --signal=INT "$LEFT" ~/fmtcmp/venv-teb/bin/tool-eval-bench run --base-url "http://127.0.0.1:$PORT" \
    --backend unknown --no-live "$@" || rc=$?
  [ "$rc" = 124 ] && rc=0
  exit "$rc"
else
  echo "unknown mode $MODE"; exit 2
fi
