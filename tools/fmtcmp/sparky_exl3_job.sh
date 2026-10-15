#!/bin/bash
# One fmtcmp GPU job on Sparky for an EXL3 arm (exllamav3's own forward; pulsar does not run Nemotron yet):
#
#   gpuq submit --class short --timeout 20m --mem 90G --label fmtcmp --wait -- \
#       ~/fmtcmp/repo/tools/fmtcmp/sparky_exl3_job.sh PACK ~/fmtcmp/caps/exl3-k4 [BUDGET_S]
#
# capture_exl3.py over the frozen suite (speed.json first, once), --selfcheck on the first job of a pack (the chunk-plumbing proof; a pack
# that fails it is not captured), resumable per document; it stops starting documents 60 s before the budget.
# PY defaults to the exllamav3 fork's venv on Sparky (~/inkling/.venv; the fork at ~/inkling/exllamav3).
set -euo pipefail
PACK=${1:?EXL3 pack dir}
OUT=${2:?output dir}
BUDGET=${3:-1080}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$HOME/inkling/.venv/bin/python}
EXL3=${EXL3:-$HOME/inkling/exllamav3}
SUITE=${SUITE:-/mnt/models/fmtcmp/suite-v1}
T0=$(date +%s)
mkdir -p "$OUT"
export PYTHONPATH=$EXL3${PYTHONPATH:+:$PYTHONPATH}
echo "[fmtcmp-exl3] $(date -Is) pack=$PACK budget=${BUDGET}s"
SC=()
[ -e "$OUT/.selfcheck-pass" ] || SC=(--selfcheck)
rc=0
timeout --signal=TERM $(( BUDGET - 20 )) "$PY" "$HERE/capture_exl3.py" --model "$PACK" --suite "$SUITE" \
  --out "$OUT" --deadline $(( T0 + BUDGET - 60 )) --speed "${SC[@]}" || rc=$?
if [ "$rc" = 0 ] && [ ${#SC[@]} -gt 0 ]; then touch "$OUT/.selfcheck-pass"; fi
exit "$rc"
