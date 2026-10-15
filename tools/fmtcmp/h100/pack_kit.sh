#!/bin/bash
# Assemble the R1 hand-off kit (run on Sparky, after the NVFP4 capture has the smoke document):
#   tools/fmtcmp/h100/pack_kit.sh [OUT_DIR]      -> OUT_DIR/fmtcmp-h100-kit-<date>.tar + .sha256
# Contents: the frozen suite, the capture/grade tools, the H100 scripts, the NVFP4 capture of the smoke document,
# tool-eval-bench at the pinned sha, the BF16 checkpoint identity, and SHA256SUMS over all of it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TOOLS=$(cd "$HERE/../.." && pwd)                      # <repo>/tools
OUT=${1:-/mnt/models/fmtcmp/kit}
SUITE=${SUITE:-/mnt/models/fmtcmp/suite-v1}
NVFP4_CAPS=${NVFP4_CAPS:-$HOME/fmtcmp/caps/nvfp4}
TEB=${TEB:-$HOME/fmtcmp/tool-eval-bench-b60e2f0}
BF16=${BF16:-/mnt/models/nemotron35-super-vl-bf16}
SMOKE_DOC=${SMOKE_DOC:-chat-11}                          # a ~1k-token code row: fast, and in every capture
AGENTIC_ARGS=${AGENTIC_ARGS:-$(cat "$HERE/AGENTIC_ARGS")}

STAMP=$(date +%Y%m%d)
STAGE=$(mktemp -d "$HOME/fmtcmp/kitstage.XXXX")
K=$STAGE/fmtcmp-h100-kit
mkdir -p "$K/tools/fmtcmp/h100" "$K/tools/fidelity" "$K/selfcheck/nvfp4" "$K/h100"
cp "$TOOLS/fidelity/capture_vllm_prompt_logprobs.py" "$TOOLS/fidelity/fidelity_compare.py" "$K/tools/fidelity/"
cp "$TOOLS/fmtcmp/grade.py" "$TOOLS/fmtcmp/vllm_capture.flags" "$K/tools/fmtcmp/"
cp "$HERE/run_reference.sh" "$HERE/one_doc_suite.py" "$HERE/vllm_agentic_h100.flags" "$HERE/RUNBOOK.md" "$K/h100/"
cp -r "$SUITE" "$K/suite-v1"
[ -f "$NVFP4_CAPS/$SMOKE_DOC.jsonl" ] || { echo "no NVFP4 capture of $SMOKE_DOC yet"; exit 1; }
cp "$NVFP4_CAPS/$SMOKE_DOC.jsonl" "$K/selfcheck/nvfp4/"
rsync -a --exclude .venv --exclude runs --exclude data/benchmarks.sqlite "$TEB/" "$K/tool-eval-bench-b60e2f0/"
echo "$SMOKE_DOC" > "$K/h100/SMOKE_DOC"
echo "$AGENTIC_ARGS" > "$K/h100/AGENTIC_ARGS"
(cd "$BF16" && sha256sum config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
   chat_template.jinja generation_config.json) > "$K/h100/bf16-identity.sha256"
ls "$BF16"/model-*.safetensors | wc -l | tr -d ' ' > "$K/h100/bf16-shard-count"
du -sbL "$BF16"/model-*.safetensors | awk '{s+=$1} END {print s}' > "$K/h100/bf16-shard-bytes"
(git -C "$TOOLS/.." rev-parse HEAD 2>/dev/null || echo "${HARNESS_SHA:?not a git checkout: set HARNESS_SHA}") > "$K/h100/HARNESS_SHA"
chmod +x "$K/h100/run_reference.sh"
(cd "$K" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
mkdir -p "$OUT"
tar -C "$STAGE" -cf "$OUT/fmtcmp-h100-kit-$STAMP.tar" fmtcmp-h100-kit
(cd "$OUT" && sha256sum "fmtcmp-h100-kit-$STAMP.tar" > "fmtcmp-h100-kit-$STAMP.tar.sha256")
rm -rf "$STAGE"
echo "kit: $OUT/fmtcmp-h100-kit-$STAMP.tar ($(du -h "$OUT/fmtcmp-h100-kit-$STAMP.tar" | cut -f1)); $(cat "$OUT/fmtcmp-h100-kit-$STAMP.tar.sha256")"
