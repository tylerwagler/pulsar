#!/bin/bash
# One fmtcmp GPU job on Sparky: start NVIDIA's NVFP4 checkpoint in the card's pinned vLLM image on a private port,
# run one client against it inside a time budget, tear the container down.  Meant to run under gpuq:
#
#   gpuq submit --class short --timeout 20m --mem 96G --label fmtcmp --wait -- \
#       ~/fmtcmp/tools/sparky_vllm_job.sh capture   ~/fmtcmp/caps/nvfp4      [BUDGET_S]
#   gpuq submit ... -- ~/fmtcmp/tools/sparky_vllm_job.sh agentic ~/fmtcmp/teb [BUDGET_S] -- <tool-eval-bench args>
#
# capture: tools/fidelity/capture_vllm_prompt_logprobs.py over the frozen suite (resumable per document: run the
#          same job again until it prints "done"); flags from vllm_capture.flags.
# agentic: tool-eval-bench (pinned b60e2f0) against the card's Spark server (vllm_agentic_spark.flags; SPEC=lilicorr
#          default, SPEC=none to serve without the drafter); pass --resume RUN_ID to continue a run.
# The whole job, server start included, stays under BUDGET_S (default 1080 s = 18 min) so gpuq's 20-min timeout never
# has to kill it; the EXIT/TERM trap removes the container either way.  Never port 8000; no scratch in /tmp.
set -euo pipefail

MODE=${1:?mode: capture|agentic}
OUT=${2:?output dir}
BUDGET=${3:-1080}
shift 3 || shift $#
[ "${1:-}" = "--" ] && shift

HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd}  # nightly-3f1bcfd82 (card pin), arm64
MODEL=${MODEL:-/srv/models/nemotron35-super-vl-nvfp4}       # local NVMe copy of /mnt/models/nemotron35-super-vl-nvfp4
DRAFT=${DRAFT:-/mnt/models/nemotron35-super-lilicorr}
SUITE=${SUITE:-/mnt/models/fmtcmp/suite-v1}
PORT=${PORT:-8300}
SPEC=${SPEC:-lilicorr}
GPU_UTIL=${GPU_UTIL:-0.75}
# GB10 is unified memory: the default mmap loader keeps the 70 GB checkpoint in page cache WHILE the weights fill
# the GPU allocation, and the load thrashes (measured 2026-10-15: 260 s per shard, 88 shards).  fastsafetensors
# streams to the device.  A loading detail only: the weights and the numerics are the same.
LOAD_FORMAT=${LOAD_FORMAT:-fastsafetensors}
NAME=fmtcmp-$MODE-$$
T0=$(date +%s)
[ "$PORT" = 8000 ] && { echo "refusing port 8000 (production)"; exit 2; }

mkdir -p "$OUT" ~/fmtcmp/cache/vllm ~/fmtcmp/cache/flashinfer ~/fmtcmp/cache/triton ~/fmtcmp/logs
LOG=~/fmtcmp/logs/$NAME.server.log
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM INT

flags_from() {  # one flag a line; "--flag value with spaces" -> two argv entries
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue;; esac
    if [[ "$line" == *" "* ]]; then ARGS+=("${line%% *}" "${line#* }"); else ARGS+=("$line"); fi
  done < "$1"
}

ARGS=("$MODEL" --host 127.0.0.1 --port "$PORT" --load-format "$LOAD_FORMAT")
if [ "$MODE" = capture ]; then
  flags_from "$HERE/vllm_capture.flags"
  ARGS+=(--gpu-memory-utilization "$GPU_UTIL")
elif [ "$MODE" = agentic ]; then
  flags_from "$HERE/vllm_agentic_spark.flags"
  if [ "$SPEC" = lilicorr ]; then
    ARGS+=(--speculative_config.model "$DRAFT" --speculative_config.num_speculative_tokens 7
           --speculative_config.method dflash)
  elif [ "$SPEC" != none ]; then echo "SPEC must be lilicorr or none"; exit 2; fi
else
  echo "unknown mode $MODE"; exit 2
fi

echo "[fmtcmp] $(date -Is) $MODE image=$IMAGE model=$MODEL port=$PORT budget=${BUDGET}s spec=$SPEC"
echo "[fmtcmp] vllm serve ${ARGS[*]}"
docker run -d --name "$NAME" --gpus all --ipc=host --network host \
  -v /srv/models:/srv/models:ro -v /mnt/models:/mnt/models:ro \
  -v ~/fmtcmp/cache/vllm:/root/.cache/vllm -v ~/fmtcmp/cache/flashinfer:/root/.cache/flashinfer \
  -v ~/fmtcmp/cache/triton:/root/.triton \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  --entrypoint vllm "$IMAGE" serve "${ARGS[@]}" >/dev/null
docker logs -f "$NAME" > "$LOG" 2>&1 &

until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
  if ! docker ps -q -f name="^$NAME\$" | grep -q .; then
    echo "[fmtcmp] server exited during startup; tail of $LOG:"; tail -40 "$LOG"; exit 3
  fi
  if [ $(( $(date +%s) - T0 )) -gt $(( BUDGET - 120 )) ]; then
    echo "[fmtcmp] server not up within budget; tail of $LOG:"; tail -20 "$LOG"; exit 4
  fi
  sleep 5
done
UP=$(( $(date +%s) - T0 ))
LEFT=$(( BUDGET - UP - 30 ))
echo "[fmtcmp] server up after ${UP}s; client budget ${LEFT}s"
curl -s "http://127.0.0.1:$PORT/v1/models" > "$OUT/served-models.json"

rc=0
if [ "$MODE" = capture ]; then
  timeout --signal=TERM "$LEFT" python3 "$HERE/../fidelity/capture_vllm_prompt_logprobs.py" \
    --endpoint "http://127.0.0.1:$PORT" --suite "$SUITE" --tokens-dir "$SUITE/tokens" \
    --topk 64 --out "$OUT" || rc=$?
else
  mkdir -p "$OUT" && cd "$OUT"
  timeout --signal=INT "$LEFT" ~/fmtcmp/venv-teb/bin/tool-eval-bench run \
    --base-url "http://127.0.0.1:$PORT" --backend vllm --no-live "$@" || rc=$?
fi
echo "[fmtcmp] client rc=$rc after $(( $(date +%s) - T0 ))s (124 = budget reached; rerun to resume)"
grep -E "throughput" "$LOG" | tail -3 || true
[ "$rc" = 124 ] && rc=0
exit "$rc"
