#!/bin/bash
# Assemble the R1 hand-off kit (any 4x 80 GB GPU box) (run on Sparky, after the NVFP4 capture has the smoke document):
#   tools/fmtcmp/r1/pack_kit.sh [OUT_DIR]      -> OUT_DIR/fmtcmp-r1-kit-<date>.tar + .sha256
# Contents: the frozen suite, the capture/grade tools, the R1 scripts, the NVFP4 capture of the smoke document,
# the BF16 checkpoint identity, and SHA256SUMS over all of it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TOOLS=$(cd "$HERE/../.." && pwd)                      # <repo>/tools
OUT=${1:-/mnt/models/fmtcmp/r1-kit}
SUITE=${SUITE:-/mnt/models/fmtcmp/suite-v1}
NVFP4_CAPS=${NVFP4_CAPS:-$HOME/fmtcmp/caps/nvfp4}
BF16=${BF16:-/mnt/models/nemotron35-super-vl-bf16}
SMOKE_DOC=${SMOKE_DOC:-chat-11}                          # a ~1k-token code row: fast, and in every capture

STAMP=$(date +%Y%m%d)
STAGE=$(mktemp -d "$HOME/fmtcmp/kitstage.XXXX")
K=$STAGE/fmtcmp-r1-kit
mkdir -p "$K/tools/fmtcmp" "$K/tools/fidelity" "$K/selfcheck/nvfp4" "$K/r1"
cp "$TOOLS/fidelity/capture_vllm_prompt_logprobs.py" "$TOOLS/fidelity/fidelity_compare.py" "$K/tools/fidelity/"
cp "$TOOLS/fmtcmp/grade.py" "$TOOLS/fmtcmp/vllm_capture.flags" "$K/tools/fmtcmp/"
cp "$HERE/run_reference.sh" "$HERE/one_doc_suite.py" "$HERE/RUNBOOK.md" "$K/r1/"
cp -r "$SUITE" "$K/suite-v1"
[ -f "$NVFP4_CAPS/$SMOKE_DOC.jsonl" ] || { echo "no NVFP4 capture of $SMOKE_DOC yet"; exit 1; }
cp "$NVFP4_CAPS/$SMOKE_DOC.jsonl" "$K/selfcheck/nvfp4/"
echo "$SMOKE_DOC" > "$K/r1/SMOKE_DOC"
(cd "$BF16" && sha256sum config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
   chat_template.jinja generation_config.json) > "$K/r1/bf16-identity.sha256"
ls "$BF16"/model-*.safetensors | wc -l | tr -d ' ' > "$K/r1/bf16-shard-count"
du -sbL "$BF16"/model-*.safetensors | awk '{s+=$1} END {print s}' > "$K/r1/bf16-shard-bytes"
(git -C "$TOOLS/.." rev-parse HEAD 2>/dev/null || echo "${HARNESS_SHA:?not a git checkout: set HARNESS_SHA}") > "$K/r1/HARNESS_SHA"
chmod +x "$K/r1/run_reference.sh"
(cd "$K" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
mkdir -p "$OUT"
tar -C "$STAGE" -cf "$OUT/fmtcmp-r1-kit-$STAMP.tar" fmtcmp-r1-kit
(cd "$OUT" && sha256sum "fmtcmp-r1-kit-$STAMP.tar" > "fmtcmp-r1-kit-$STAMP.tar.sha256")
rm -rf "$STAGE"
echo "kit: $OUT/fmtcmp-r1-kit-$STAMP.tar ($(du -h "$OUT/fmtcmp-r1-kit-$STAMP.tar" | cut -f1)); $(cat "$OUT/fmtcmp-r1-kit-$STAMP.tar.sha256")"
