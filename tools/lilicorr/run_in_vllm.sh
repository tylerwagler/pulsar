#!/bin/bash
# run_in_vllm.sh SCRIPT [ARGS...] -- run a tools/lilicorr script inside the vLLM image that serves LiLiCorr on the GB10
# host (the serving job's pinned digest), GPU attached, the MODEL_DIRS read-only at their own paths, $WORK (default
# ~/l298h) mounted rw at /work.  MODEL_DIRS (required): the space-separated host directories holding LILICORR_DRAFT
# and NEMOTRON_NVFP4, which are passed into the container.
# The script directory is mounted at /work/tools, so SCRIPT is a name in it.  Submit through gpuq, e.g.
#   gpuq submit --class short --timeout 20m --mem 96G --label l298-head-dump --wait -- \
#     ~/l298h/tools/run_in_vllm.sh vllm_dump.py --out /work/goldens
set -euo pipefail
SCRIPT=${1:?script}; shift
IMAGE=vllm/vllm-openai@sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd
WORK=${WORK:-$HOME/l298h}
MODEL_DIRS=${MODEL_DIRS:?set MODEL_DIRS to the host directories that hold the checkpoints}
mounts=()
for d in $MODEL_DIRS; do mounts+=(-v "$d:$d:ro"); done
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=l298h-$$
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM INT USR1
mkdir -p "$WORK"
docker run --rm --name "$NAME" --gpus all --ipc=host --network host \
  "${mounts[@]}" -v "$WORK":/work -v "$HERE":/work/tools:ro \
  -v ~/fmtcmp/cache/vllm:/root/.cache/vllm -v ~/fmtcmp/cache/flashinfer:/root/.cache/flashinfer \
  -v ~/fmtcmp/cache/triton:/root/.triton \
  -e LILICORR_DRAFT="${LILICORR_DRAFT:-}" -e NEMOTRON_NVFP4="${NEMOTRON_NVFP4:-}" \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e PYTHONUNBUFFERED=1 -w /work \
  --entrypoint python3 "$IMAGE" "/work/tools/$SCRIPT" "$@" &
wait $!
