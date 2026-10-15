#!/bin/bash
# L296 fmtcmp R1 hand-off: capture the BF16 reference on 4x H100 (and, optionally, BF16's agentic score).
# Run from the unpacked kit directory.  See RUNBOOK.md.  Every step is resumable; rerun the same command.
#
#   ./run_reference.sh check      # ~1 min: kit checksums, docker + image, GPUs, checkpoint identity. Starts nothing.
#   ./run_reference.sh capture    # server up (~5-15 min), 1-min smoke vs the shipped NVFP4 capture, full capture
#   ./run_reference.sh agentic    # the card's 4x H100 BF16 server (MTP 3) + tool-eval-bench, same args as Sparky
#   ./run_reference.sh bundle     # results -> fmtcmp-r1-results-<host>-<date>.tar + .sha256 (bring this back)
set -euo pipefail

KIT=$(cd "$(dirname "$0")/.." && pwd)            # kit root: run_reference.sh lives in <kit>/h100/
MODEL=${MODEL:?set MODEL=/path/to/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 (the local snapshot dir)}
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd}  # nightly-3f1bcfd82 (the card's pin; multi-arch index)
UV_IMAGE=${UV_IMAGE:-ghcr.io/astral-sh/uv:python3.12-bookworm-slim}
PORT=${PORT:-8300}
TP=${TP:-4}
GPU_UTIL=${GPU_UTIL:-0.85}
WORK=${WORK:-$KIT/work}
SUITE=$KIT/suite-v1
CAPS=$WORK/caps/bf16
NAME=fmtcmp-r1-$$
SMOKE_DOC=$(cat "$KIT/h100/SMOKE_DOC")
mkdir -p "$WORK/logs" "$WORK/cache/vllm" "$WORK/cache/flashinfer" "$WORK/cache/triton"

say() { echo "[fmtcmp-r1] $(date -Is) $*" | tee -a "$WORK/logs/run.log"; }
die() { say "FAIL: $*"; exit 1; }
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

flags_from() {  # one flag a line; "--flag value with spaces" -> two argv entries
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue;; esac
    if [[ "$line" == *" "* ]]; then ARGS+=("${line%% *}" "${line#* }"); else ARGS+=("$line"); fi
  done < "$1"
}

check() {
  say "check: kit checksums"
  (cd "$KIT" && sha256sum --quiet -c SHA256SUMS) || die "kit checksums (re-copy the kit)"
  say "check: docker + image $IMAGE"
  docker info >/dev/null 2>&1 || die "docker not usable by $(id -un)"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE" >/dev/null || die "cannot pull $IMAGE"
  docker run --rm --entrypoint python3 "$IMAGE" -c "import vllm; print(vllm.__version__)" 2>/dev/null | tail -1 \
    | tee "$WORK/logs/vllm-version.txt" | grep -q '^0.32.0rc1.dev190+g3f1bcfd82' || die "unexpected vLLM build: $(cat "$WORK/logs/vllm-version.txt")"
  say "check: GPUs"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | tee "$WORK/logs/gpus.txt"
  local n; n=$(grep -c 'H100' "$WORK/logs/gpus.txt" || true)
  [ "$n" -ge "$TP" ] || die "need $TP H100s, found $n"
  awk -F', ' '{ if ($4+0 > 2000) bad=1 } END { exit bad }' "$WORK/logs/gpus.txt" || die "GPUs are busy (memory.used > 2 GB): free them first"
  say "check: checkpoint identity ($MODEL)"
  (cd "$MODEL" && sha256sum -c "$KIT/h100/bf16-identity.sha256" --quiet) || die "BF16 config/index/tokenizer differ from the checkpoint the suite was built against"
  local shards; shards=$(ls "$MODEL"/model-*.safetensors | wc -l)
  [ "$shards" = "$(cat "$KIT/h100/bf16-shard-count")" ] || die "shard count $shards != $(cat "$KIT/h100/bf16-shard-count")"
  local bytes; bytes=$(du -sbL "$MODEL"/model-*.safetensors | awk '{s+=$1} END {print s}')
  [ "$bytes" = "$(cat "$KIT/h100/bf16-shard-bytes")" ] || die "shard bytes $bytes != $(cat "$KIT/h100/bf16-shard-bytes") (incomplete download?)"
  local free; free=$(df -PB1 "$WORK" | awk 'NR==2 {print $4}')
  [ "$free" -gt $((2 * 1024 ** 3)) ] || die "need >= 2 GB free under $WORK"
  say "check: PASS"
}

serve() {  # $1 = capture|agentic
  ARGS=(/model --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP")
  if [ "$1" = capture ]; then
    flags_from "$KIT/tools/fmtcmp/vllm_capture.flags"
    ARGS+=(--gpu-memory-utilization "$GPU_UTIL")
  else
    flags_from "$KIT/h100/vllm_agentic_h100.flags"
  fi
  say "serve ($1): vllm serve ${ARGS[*]}"
  docker run -d --name "$NAME" --gpus all --ipc=host --network host \
    -v "$MODEL":/model:ro -v "$WORK/cache/vllm":/root/.cache/vllm \
    -v "$WORK/cache/flashinfer":/root/.cache/flashinfer -v "$WORK/cache/triton":/root/.triton \
    -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
    --entrypoint vllm "$IMAGE" serve "${ARGS[@]}" >/dev/null
  docker logs -f "$NAME" > "$WORK/logs/server-$1.log" 2>&1 &
  local t0; t0=$(date +%s)
  until curl -sf "http://127.0.0.1:$PORT/v1/models" > "$WORK/logs/served-models-$1.json"; do
    docker ps -q -f name="^$NAME\$" | grep -q . || { tail -40 "$WORK/logs/server-$1.log"; die "server exited during startup"; }
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
  python3 "$KIT/h100/one_doc_suite.py" "$SUITE" "$SMOKE_DOC" "$WORK/smoke/suite"
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
  local teb="$KIT/tool-eval-bench-b60e2f0"
  mkdir -p "$WORK/agentic"
  say "tool-eval-bench (pinned b60e2f0) in $UV_IMAGE: $(cat "$KIT/h100/AGENTIC_ARGS")"
  # shellcheck disable=SC2046
  docker run --rm --network host -v "$teb":/teb:ro -v "$WORK/agentic":/work -w /work "$UV_IMAGE" \
    sh -c "cp -r /teb /tmp/teb && cd /tmp/teb && uv sync --frozen -q && cd /work && \
           /tmp/teb/.venv/bin/tool-eval-bench run --base-url http://127.0.0.1:$PORT --backend vllm --no-live \
           $(cat "$KIT/h100/AGENTIC_ARGS")" 2>&1 | tee -a "$WORK/logs/agentic.log"
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
  check) check;;
  capture) capture;;
  agentic) agentic;;
  bundle) bundle;;
  *) echo "usage: MODEL=/path/to/bf16 $0 check|capture|agentic|bundle"; exit 2;;
esac
