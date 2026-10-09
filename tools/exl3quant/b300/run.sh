#!/usr/bin/env bash
# The whole V4.1 EXL3 job on one multi-GPU host, resumable: re-run the same command after a crash or preemption and
# it continues (the forward from its last checkpointed layer, the units not yet done, assembly / grading of what is
# missing).  Uniform K3 trunk + drafter (kmaps/v41-k3-uniform.json).
#
#   V41_ROOT=/nvme/v41 bash run.sh
#
# Stages (each logs to $RUN/logs/<stage>.log, marks $RUN/stage/<stage>.done):
#   forward    v41_stream.py --hessians-only on FORWARD_GPU: the source model's layer-streamed forward over the
#              calibration rows, every block's Hessians + held-out captures (then the drafter's)
#   quant      quant_worker.py, one worker per GPU in WORKER_GPUS, started with the forward: units of a block are
#              claimed as soon as its Hessians land; FORWARD_GPU joins when the forward exits, LATE_GPUS when
#              LATE_MARKER exists (the reference captures' GPUs, once refs/run_refs.sh is done)
#   assemble   units -> exl3-kK/model-<block>.safetensors (+ .sha256), units pruned
#   grade      grade_layer.py on every block (held-out rows), fanned over all GPUs
#   summary    summarize.py -> $RUN/summary.md / summary.json
#
# Knobs (environment):
#   RUN=$V41_ROOT/run  SRC=$V41_ROOT/src/v41  CALIB=bundle/calib/calib-v41-code-v1.safetensors
#   FORWARD_GPU=0  WORKER_GPUS="<every GPU but FORWARD_GPU and LATE_GPUS>"  LATE_GPUS=""  LATE_MARKER=$V41_ROOT/refs/GPUS_FREE
#   DRY RUN: LAYERS=<stop after this many trunk layers>  QUANT_LAYERS=<a-b>  EXPERTS=<first N experts per block>
#            DRAFTER=0  (sparky: WORKER_GPUS="0 0" simulates two GPUs' workers on one)
#   STOP_AFTER=<stage>  stop after that stage (e.g. forward, to test resume)
set -uo pipefail
ROOT=${V41_ROOT:?set V41_ROOT}
HERE=$(cd "$(dirname "$0")" && pwd)
X=$(dirname "$HERE")                                   # tools/exl3quant
PY=${PY:-$ROOT/venv/bin/python}
EXL3=${EXL3:-$ROOT/exllamav3}
CONTAINER=${CONTAINER:-$ROOT/bundle/l279-container}
KMAP=${KMAP:-$X/kmaps/v41-k3-uniform.json}
SRC=${SRC:-$ROOT/src/v41}
CALIB=${CALIB:-$ROOT/bundle/calib/calib-v41-code-v1.safetensors}
RUN=${RUN:-$ROOT/run}
FORWARD_GPU=${FORWARD_GPU:-0}
LATE_GPUS=${LATE_GPUS:-}
LATE_MARKER=${LATE_MARKER:-$ROOT/refs/GPUS_FREE}
DRAFTER=${DRAFTER:-1}
mkdir -p "$RUN/logs" "$RUN/stage" "$RUN/state"
exec > >(tee -a "$RUN/logs/run.log") 2>&1
say() { echo "[run $(date -u +%FT%TZ)] $*"; }
die() { say "FAILED: $*"; exit 1; }
done_() { touch "$RUN/stage/$1.done"; say "stage $1 done"; [ "${STOP_AFTER:-}" = "$1" ] && { say "STOP_AFTER=$1"; exit 0; }; }

ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd' ')
if [ -z "${WORKER_GPUS:-}" ]; then
  WORKER_GPUS=$(for g in $ALL_GPUS; do case " $FORWARD_GPU $LATE_GPUS " in *" $g "*) ;; *) echo -n "$g ";; esac; done)
fi
SEL=()
[ -n "${QUANT_LAYERS:-}" ] && SEL+=(--quant-layers "$QUANT_LAYERS")
[ "$DRAFTER" = 1 ] && SEL+=(--drafter)
[ -n "${EXPERTS:-}" ] && SEL+=(--experts "$EXPERTS")
QW=("$PY" "$X/quant_worker.py")
COMMON=(--ref "$SRC" --run "$RUN" --kmap "$KMAP" --container "$CONTAINER" "${SEL[@]}")

# ---- preflight ------------------------------------------------------------------------------------------------
say "host $(hostname), GPUs [$ALL_GPUS], forward on $FORWARD_GPU, workers on [$WORKER_GPUS], late [$LATE_GPUS]"
[ -f "$SRC/.verified" ] || [ -n "${NO_VERIFY:-}" ] || die "$SRC/.verified missing: run fetch.sh"
[ -f "$CALIB" ] || die "no calibration rows $CALIB"
"$PY" -c "import torch, exllamav3_ext" || die "venv: run bootstrap.sh"
"${QW[@]}" status "${COMMON[@]}" || die "status (K map / container / Hessian layout)"
{ echo "{\"started\": \"$(date -u +%FT%TZ)\", \"host\": \"$(hostname)\", \"src\": \"$SRC\", \"calib\": \"$CALIB\","
  echo " \"calib_sha256\": \"$(sha256sum "$CALIB" | cut -d' ' -f1)\", \"kmap\": \"$KMAP\","
  echo " \"engine\": \"$(cat "$ROOT/bundle/MANIFEST.json" 2>/dev/null | tr -d '\n' | head -c 2000 || echo none)\","
  echo " \"gpus\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | tr -s ' ' | paste -sd';')\"}"; } \
  > "$RUN/logs/run-$(date -u +%Y%m%dT%H%M%S).json"

# ---- forward + quant ------------------------------------------------------------------------------------------
if [ ! -f "$RUN/stage/quant.done" ]; then
  rm -rf "$RUN/units/.claims"                    # every worker of a previous attempt is gone
  rm -f "$RUN/state/forward.failed"
  FWD_PID=
  if [ ! -f "$RUN/state/forward.done" ]; then
    FARGS=(--ref "$SRC" --exllamav3 "$EXL3" --calib "$CALIB" --out "$RUN" --hessians-only --device cuda:0)
    [ -n "${LAYERS:-}" ] && FARGS+=(--layers "$LAYERS")
    [ -n "${QUANT_LAYERS:-}" ] && FARGS+=(--quant-layers "$QUANT_LAYERS")
    [ "$DRAFTER" = 1 ] && FARGS+=(--drafter)
    say "forward: GPU $FORWARD_GPU (log $RUN/logs/forward.log, records $RUN/log.jsonl)"
    ( CUDA_VISIBLE_DEVICES=$FORWARD_GPU "$PY" "$X/v41_stream.py" "${FARGS[@]}" >> "$RUN/logs/forward.log" 2>&1
      rc=$?
      if [ $rc -eq 0 ]; then touch "$RUN/state/forward.done"; else echo $rc > "$RUN/state/forward.failed"; fi ) &
    FWD_PID=$!
  fi
  declare -A WPID=()
  start_worker() {  # gpu tag
    say "worker $2 on GPU $1"
    CUDA_VISIBLE_DEVICES=$1 "${QW[@]}" work "${COMMON[@]}" --exllamav3 "$EXL3" --worker "$2" --wait \
      >> "$RUN/logs/worker-$2.out" 2>&1 &
    WPID[$2]=$!
  }
  i=0; for g in $WORKER_GPUS; do start_worker "$g" "g$g-$i"; i=$((i + 1)); done
  fwd_joined=0; late_joined=0; [ -z "$LATE_GPUS" ] && late_joined=1
  [ -z "$FWD_PID" ] && { start_worker "$FORWARD_GPU" "g$FORWARD_GPU-f"; fwd_joined=1; }
  while :; do
    if [ $fwd_joined = 0 ] && ! kill -0 "$FWD_PID" 2>/dev/null; then
      wait "$FWD_PID"
      if [ -f "$RUN/state/forward.done" ]; then say "forward done"; start_worker "$FORWARD_GPU" "g$FORWARD_GPU-f"
      else say "forward FAILED (rc $(cat "$RUN/state/forward.failed")): tail of its log:"; tail -n 30 "$RUN/logs/forward.log"; fi
      fwd_joined=1
    fi
    if [ $late_joined = 0 ] && [ -f "$LATE_MARKER" ]; then
      for g in $LATE_GPUS; do start_worker "$g" "g$g-late"; done; late_joined=1
    fi
    alive=0; for w in "${!WPID[@]}"; do kill -0 "${WPID[$w]}" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && [ $fwd_joined = 1 ] && break
    "${QW[@]}" status "${COMMON[@]}" > "$RUN/logs/status.json" 2>/dev/null
    sleep 30
  done
  rc_all=0
  for w in "${!WPID[@]}"; do wait "${WPID[$w]}"; rc=$?; [ $rc -ne 0 ] && { say "worker $w exited $rc"; rc_all=1; }; done
  [ -f "$RUN/state/forward.done" ] || die "the forward did not finish; re-run run.sh to resume it"
  "${QW[@]}" status "${COMMON[@]}" | tee "$RUN/logs/status.json"
  [ $rc_all = 0 ] || die "a worker failed (logs/worker-*.out); re-run run.sh: done units are kept, the rest redone"
  done_ quant
fi

# ---- assemble -------------------------------------------------------------------------------------------------
if [ ! -f "$RUN/stage/assemble.done" ]; then
  "${QW[@]}" assemble "${COMMON[@]}" --prune 2>&1 | tee -a "$RUN/logs/assemble.log"
  [ "${PIPESTATUS[0]}" = 0 ] || die "assemble"
  done_ assemble
fi

# ---- grade (every block, held-out rows), fanned over the GPUs ----------------------------------------------------
if [ ! -f "$RUN/stage/grade.done" ]; then
  blocks=$("${QW[@]}" blocks "${COMMON[@]}") || die "block list"
  read -ra GG <<< "$ALL_GPUS"; n=${#GG[@]}; k=0; pids=()
  while read -r b name; do
    [ -f "$RUN/grade-$name.json" ] && continue
    g=${GG[$((k % n))]}; k=$((k + 1))
    ( CUDA_VISIBLE_DEVICES=$g "$PY" "$X/grade_layer.py" --ref "$SRC" --exllamav3 "$EXL3" --run "$RUN" --block "$b" \
        ${EXPERTS:+--experts $EXPERTS} < /dev/null >> "$RUN/logs/grade.log" 2>&1 || echo "$b" >> "$RUN/logs/grade.failed" ) &
    pids+=($!)
    if [ ${#pids[@]} -ge $n ]; then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
  done <<< "$blocks"
  wait
  [ -s "$RUN/logs/grade.failed" ] && { cat "$RUN/logs/grade.failed"; rm -f "$RUN/logs/grade.failed"; die "grading"; }
  done_ grade
fi

# ---- summary --------------------------------------------------------------------------------------------------
"$PY" "$X/summarize.py" --run "$RUN" | tee "$RUN/logs/summary.log"
done_ summary
say "all stages done: $RUN/summary.md; next: pack.sh"
