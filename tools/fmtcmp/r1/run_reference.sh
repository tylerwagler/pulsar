#!/bin/bash
# L296 fmtcmp R1: capture the BF16 reference on any 4x 80 GB-class GPU box (H100 / H200 / A100-80G / B200 ...),
# with Docker (the card's pinned vLLM image) or, where there is no Docker (e.g. inside a rental container), a venv
# with the same vLLM build.  Run from the unpacked kit directory.  See RUNBOOK.md.  Every step is resumable.
#
#   r1/run_reference.sh check      # ~1 min: kit checksums, runtime + vLLM build, GPUs, checkpoint identity
#   r1/run_reference.sh capture    # server up, 1-min smoke vs the shipped NVFP4 capture, then the full capture
#   r1/run_reference.sh agentic    # the BF16 card's 4-GPU server (MTP 3) + tool-eval-bench, same args as Sparky
#   r1/run_reference.sh bundle     # results -> fmtcmp-r1-results-<host>-<date>.tar + .sha256 (bring this back)
#
# RUNTIME=docker (default when docker works) | venv.  venv: r1/run_reference.sh venv  builds $WORK/venv first.
set -euo pipefail

KIT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:?set MODEL=/path/to/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 (the local snapshot dir)}
VLLM_COMMIT=3f1bcfd82bb2e93fcd364fc7a5491718453a77e2                        # the card's pin
VLLM_VERSION=0.32.0rc1.dev190+g3f1bcfd82
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd}  # nightly-3f1bcfd82, multi-arch index
PORT=${PORT:-8300}
TP=${TP:-4}
GPU_UTIL=${GPU_UTIL:-0.85}
MIN_GPU_MIB=${MIN_GPU_MIB:-79000}
WORK=${WORK:-$KIT/work}
SUITE=$KIT/suite-v1
CAPS=$WORK/caps/bf16
SMOKE_DOC=$(cat "$KIT/r1/SMOKE_DOC")
if [ -z "${RUNTIME:-}" ]; then
  if docker info >/dev/null 2>&1; then RUNTIME=docker; else RUNTIME=venv; fi
fi
NAME=fmtcmp-r1-$$
mkdir -p "$WORK/logs" "$WORK/cache"
SRV_PID=

say() { echo "[fmtcmp-r1] $(date -Is) $*" | tee -a "$WORK/logs/run.log"; }
die() { say "FAIL: $*"; exit 1; }
cleanup() {
  if [ "$RUNTIME" = docker ]; then docker rm -f "$NAME" >/dev/null 2>&1 || true
  elif [ -n "$SRV_PID" ]; then kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

flags_from() {  # one flag a line; "--flag value with spaces" -> two argv entries
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue;; esac
    if [[ "$line" == *" "* ]]; then ARGS+=("${line%% *}" "${line#* }"); else ARGS+=("$line"); fi
  done < "$1"
}

vpy() {  # python inside the runtime
  if [ "$RUNTIME" = docker ]; then docker run --rm --entrypoint python3 "$IMAGE" "$@"
  else "$WORK/venv/bin/python" "$@"; fi
}

venv() {
  command -v uv >/dev/null || python3 -m pip install -q uv
  [ -x "$WORK/venv/bin/python" ] || uv venv -q -p 3.12 "$WORK/venv"
  # vLLM publishes a wheel per commit; this is the build inside the card's image.
  VIRTUAL_ENV=$WORK/venv uv pip install -q "vllm==$VLLM_VERSION" \
    --extra-index-url "https://wheels.vllm.ai/$VLLM_COMMIT" --index-strategy unsafe-best-match \
    || die "could not install vLLM $VLLM_VERSION from wheels.vllm.ai/$VLLM_COMMIT"
  say "venv ready: $("$WORK/venv/bin/python" -c 'import vllm,torch;print(vllm.__version__, torch.__version__)')"
}

check() {
  say "check: kit checksums"
  (cd "$KIT" && sha256sum --quiet -c SHA256SUMS) || die "kit checksums (re-copy the kit)"
  say "check: runtime $RUNTIME"
  if [ "$RUNTIME" = docker ]; then
    docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE" >/dev/null || die "cannot pull $IMAGE"
  else
    [ -x "$WORK/venv/bin/python" ] || die "no venv: run  r1/run_reference.sh venv  first"
  fi
  vpy -c "import vllm; print(vllm.__version__)" 2>/dev/null | tail -1 > "$WORK/logs/vllm-version.txt"
  [ "$(cat "$WORK/logs/vllm-version.txt")" = "$VLLM_VERSION" ] || die "vLLM is $(cat "$WORK/logs/vllm-version.txt"), need $VLLM_VERSION"
  say "check: GPUs"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader,nounits \
    | tee "$WORK/logs/gpus.txt"
  local n; n=$(awk -F', ' -v m="$MIN_GPU_MIB" '$3+0 >= m' "$WORK/logs/gpus.txt" | wc -l)
  [ "$n" -ge "$TP" ] || die "need $TP GPUs with >= $MIN_GPU_MIB MiB, found $n"
  awk -F', ' '{ if ($4+0 > 2000) bad=1 } END { exit bad }' "$WORK/logs/gpus.txt" || die "GPUs are busy (> 2 GB used): free them first"
  say "check: checkpoint identity ($MODEL)"
  (cd "$MODEL" && sha256sum --quiet -c "$KIT/r1/bf16-identity.sha256") || die "BF16 config/index/tokenizer differ from the checkpoint the suite was built against"
  local shards; shards=$(ls "$MODEL"/model-*.safetensors | wc -l)
  [ "$shards" = "$(cat "$KIT/r1/bf16-shard-count")" ] || die "shard count $shards != $(cat "$KIT/r1/bf16-shard-count")"
  local bytes; bytes=$(du -sbL "$MODEL"/model-*.safetensors | awk '{s+=$1} END {print s}')
  [ "$bytes" = "$(cat "$KIT/r1/bf16-shard-bytes")" ] || die "shard bytes $bytes != $(cat "$KIT/r1/bf16-shard-bytes") (incomplete download?)"
  local free; free=$(df -PB1 "$WORK" | awk 'NR==2 {print $4}')
  [ "$free" -gt $((2 * 1024 ** 3)) ] || die "need >= 2 GB free under $WORK"
  say "check: PASS"
}

serve() {  # $1 = capture|agentic
  local mpath=$MODEL
  [ "$RUNTIME" = docker ] && mpath=/model
  ARGS=("$mpath" --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP")
  if [ "$1" = capture ]; then
    flags_from "$KIT/tools/fmtcmp/vllm_capture.flags"
    ARGS+=(--gpu-memory-utilization "$GPU_UTIL")
  else
    flags_from "$KIT/r1/vllm_agentic_4gpu.flags"
  fi
  say "serve ($1, $RUNTIME): vllm serve ${ARGS[*]}"
  if [ "$RUNTIME" = docker ]; then
    docker run -d --name "$NAME" --gpus all --ipc=host --network host -v "$MODEL":/model:ro \
      -v "$WORK/cache":/root/.cache -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
      --entrypoint vllm "$IMAGE" serve "${ARGS[@]}" >/dev/null
    docker logs -f "$NAME" > "$WORK/logs/server-$1.log" 2>&1 &
  else
    HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 XDG_CACHE_HOME=$WORK/cache \
      "$WORK/venv/bin/vllm" serve "${ARGS[@]}" > "$WORK/logs/server-$1.log" 2>&1 &
    SRV_PID=$!
  fi
  local t0; t0=$(date +%s)
  until curl -sf "http://127.0.0.1:$PORT/v1/models" > "$WORK/logs/served-models-$1.json"; do
    if [ "$RUNTIME" = docker ]; then
      docker ps -q -f name="^$NAME\$" | grep -q . || { tail -40 "$WORK/logs/server-$1.log"; die "server exited during startup"; }
    else
      kill -0 "$SRV_PID" 2>/dev/null || { tail -40 "$WORK/logs/server-$1.log"; die "server exited during startup"; }
    fi
    [ $(( $(date +%s) - t0 )) -lt 3600 ] || die "server not up after 60 min"
    sleep 10
  done
  say "server up after $(( $(date +%s) - t0 ))s"
}

capture() {
  check
  serve capture
  local cap="python3 $KIT/tools/fidelity/capture_vllm_prompt_logprobs.py --endpoint http://127.0.0.1:$PORT --suite"
  say "smoke: $SMOKE_DOC twice, vs the shipped NVFP4 capture (catches a wrong model / tokenizer / flag in a minute)"
  rm -rf "$WORK/smoke"
  mkdir -p "$WORK/smoke/a" "$WORK/smoke/b"
  python3 "$KIT/r1/one_doc_suite.py" "$SUITE" "$SMOKE_DOC" "$WORK/smoke/suite"
  $cap "$WORK/smoke/suite" --tokens-dir "$WORK/smoke/suite/tokens" --topk 64 --out "$WORK/smoke/a" 2>>"$WORK/logs/run.log"
  $cap "$WORK/smoke/suite" --tokens-dir "$WORK/smoke/suite/tokens" --topk 64 --out "$WORK/smoke/b" 2>>"$WORK/logs/run.log"
  python3 "$KIT/tools/fmtcmp/grade.py" --suite "$SUITE" --only "$SMOKE_DOC" --ref "$WORK/smoke/a" \
    nvfp4="$KIT/selfcheck/nvfp4" bf16-rerun="$WORK/smoke/b" --draws 200 --json "$WORK/smoke/grade.json" \
    | tee "$WORK/logs/smoke.md"
  python3 - "$WORK/smoke/grade.json" <<'EOF' || die "smoke failed (see logs/smoke.md); do not run the full capture"
import json, sys
g = json.load(open(sys.argv[1]))["groups"]["all"]
nv, re = g["nvfp4"], g["bf16-rerun"]
print(f"smoke: BF16 vs NVFP4 KL/token {nv['kl'][0]:.3e} top-1 {100*nv['top1']:.1f}%; "
      f"BF16 rerun KL {re['kl'][0]:.3e} top-1 {100*re['top1']:.1f}% (0 / 100% = deterministic)")
# A wrong model, tokenizer or chat render reads KL >> 1 and top-1 < 60%; a sound BF16 vs NVFP4 is far inside these.
sys.exit(0 if nv["kl"][0] < 0.5 and nv["top1"] > 0.80 else 1)
EOF
  say "smoke PASS; full capture into $CAPS (resumable)"
  mkdir -p "$CAPS"
  $cap "$SUITE" --tokens-dir "$SUITE/tokens" --topk 64 --out "$CAPS" 2>&1 | tee -a "$WORK/logs/capture.log"
  local n; n=$(ls "$CAPS"/*.jsonl | wc -l)
  [ "$n" = "$(python3 -c "import json;print(json.load(open('$SUITE/manifest.json'))['n_docs'])")" ] || die "only $n documents captured; rerun"
  say "capture complete: $n documents"
}

agentic() {
  check
  serve agentic
  local teb="$KIT/tool-eval-bench-b60e2f0" args; args=$(cat "$KIT/r1/AGENTIC_ARGS")
  mkdir -p "$WORK/agentic"
  if [ ! -x "$WORK/teb/.venv/bin/tool-eval-bench" ]; then
    command -v uv >/dev/null || python3 -m pip install -q uv
    rm -rf "$WORK/teb" && cp -r "$teb" "$WORK/teb" && (cd "$WORK/teb" && uv sync --frozen -q -p 3.12)
  fi
  say "tool-eval-bench (pinned b60e2f0): $args"
  # shellcheck disable=SC2086
  (cd "$WORK/agentic" && "$WORK/teb/.venv/bin/tool-eval-bench" run --base-url "http://127.0.0.1:$PORT" \
     --backend vllm --no-live $args) 2>&1 | tee -a "$WORK/logs/agentic.log"
  say "agentic done"
}

bundle() {
  local out="$KIT/fmtcmp-r1-results-$(hostname -s)-$(date +%Y%m%d-%H%M).tar"
  (cd "$WORK" && find caps logs smoke agentic -type f -print0 2>/dev/null | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
  tar -C "$WORK" -cf "$out" MANIFEST.sha256 caps logs smoke $( [ -d "$WORK/agentic" ] && echo agentic )
  sha256sum "$out" > "$out.sha256"
  say "bring back: $out and $out.sha256 ($(du -h "$out" | cut -f1))"
}

case "${1:-}" in
  venv) venv;;
  check) check;;
  capture) capture;;
  agentic) agentic;;
  bundle) bundle;;
  *) echo "usage: MODEL=/path/to/bf16 [RUNTIME=docker|venv] $0 venv|check|capture|agentic|bundle"; exit 2;;
esac
