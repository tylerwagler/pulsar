#!/usr/bin/env bash
# Plan 90 A1/A3 — pulsar-side teacher-forced capture (the BASELINE row).
#
# RUN THIS ON SPARKY, IN A MAINTENANCE WINDOW. The CLI loads the model itself,
# so pulsar-server must be stopped first ([[ds4-model-load-discipline]]):
# a second 87G load alongside a live server is how the Xid 31 incident happened.
#
# The row this produces is the standing divergence budget every speed lever in
# plan 90 is priced against, so it must be captured on the SHIPPED config —
# do not tune anything for it.
#
# Usage:
#   capture_pulsar_side.sh MODEL.gguf [SUITE_DIR] [OUT_DIR]
set -euo pipefail

MODEL="${1:?usage: capture_pulsar_side.sh MODEL.gguf [SUITE_DIR] [OUT_DIR]}"
SUITE="${2:-/mnt/pve1-models/fidelity/suite-v1}"
OUT="${3:-$HOME/fidelity-caps/pulsar-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"
STRIDE="${KL_STRIDE:-4}"
TOPK="${KL_TOPK:-20}"          # match the reference's cap; higher is wasted
CTX="${KL_CTX:-262144}"

BIN="${PULSAR_BIN:-./pulsar}"
TOOLS="$(cd "$(dirname "$0")" && pwd)"

echo "== preflight"
[ -x "$BIN" ] || { echo "no pulsar CLI at $BIN (build with 'make cuda-spark', NOT plain make)"; exit 1; }
[ -f "$MODEL" ] || { echo "no model at $MODEL"; exit 1; }
[ -d "$SUITE" ] || { echo "no suite at $SUITE"; exit 1; }

if pgrep -x pulsar-server >/dev/null 2>&1; then
    echo "REFUSING: pulsar-server is running. Stop it first — the CLI loads its"
    echo "own copy of the model and two 87G loads will not fit (Xid 31)."
    exit 1
fi

avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
echo "   MemAvailable: ${avail} GiB"
if [ "$avail" -lt 95 ]; then
    echo "   low — run: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
    echo "   (on GB10 this is almost always reclaimable page cache from a prior"
    echo "    mmap'd model, NOT a UVM leak — trust drop_caches)"
    exit 1
fi

mkdir -p "$OUT/klrf" "$OUT/jsonl"
echo "== capturing to $OUT"

for f in "$SUITE"/*.txt; do
    name=$(basename "$f" .txt)
    klrf="$OUT/klrf/$name.klrf"
    js="$OUT/jsonl/$name.jsonl"
    if [ -f "$js" ]; then echo "   skip $name (done)"; continue; fi
    echo "   -- $name"
    "$BIN" -m "$MODEL" --ctx "$CTX" --kl-stride "$STRIDE" \
        --kl-file "$f" --kl-ref-dump "$klrf"
    python3 "$TOOLS/pulsar_kldump_to_jsonl.py" "$klrf" \
        --name "$name" --topk "$TOPK" \
        --tag "pulsar-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
        --out "$js"
    # the .klrf holds FULL-VOCAB f32 logits and is enormous; the jsonl is the
    # cross-rig artifact. Keep klrf only if you want the exact --kl-score lane.
    [ "${KEEP_KLRF:-0}" = "1" ] || rm -f "$klrf" "$klrf.tokens.json"
done

echo
echo "== done. Build the A3 table against the work-rig reference:"
echo "   python3 $TOOLS/fidelity_compare.py --dir \\"
echo "       /mnt/pve1-models/fidelity/suite-v1/ref-DeepSeek-V4-Flash-0731 \\"
echo "       $OUT/jsonl"
echo
echo "   The 'all' row median/p95 IS the standing divergence budget (plan 90 A2)."
