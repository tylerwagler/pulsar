#!/bin/bash
# Full artifact rebuild from source weights -- COLLAPSED pipeline.
#
# Three of the old seven stages are gone because the quantizer now does them
# itself (the staged script and trim_reap.py were deleted with them; git
# history is the record):
#
#   old 3  REAP trim               -> --reap-survivors   (expert trim + router/
#                                     bias permutation during generation)
#   old 5  merge_dspark_gguf.py    -> --dspark-template  (drafter tensors and
#                                     binding KVs emitted into the same file)
#   old 6  repack_iq2_mmq.py       -> a format map naming IQ2_XXS_MMQ, emitted
#                                     as type 43 directly
#
# What is left is: build two templates, run the quantizer once, gate it.
#
# ORDER WAS FORCED, not chosen. REAP had to collapse FIRST: the old trim was a
# post-quantize transplant that re-sliced whole tensors, so it needed block
# geometry for every type in the file -- pointing the old pipeline at an MMQ
# map died with `KeyError: 43`, because trim had to parse a type it never
# emits. With REAP inside the quantizer there is no post-pass left to trip
# over, and the merge collapse then had nothing to sequence against.
#
# Cost: ~92.5 GB written for a 92.5 GB artifact (1.0x, was 4.1x). No full-size
# intermediates at all, so nothing has to be deleted mid-run.
set -euo pipefail

R=${R:-/home/claude/rebuild}
PY=${PY:-/home/claude/.venvs/pulsar-quant/bin/python}
G=${G:-/home/claude/Projects/pulsar/gguf-tools}
HF=${HF:-/mnt/pve1-models/dsv4-flash-0731}
IMATRIX=${IMATRIX:-/mnt/pve1-models/ds4-quant-archive/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat}
# Measured 2026-08-13, not guessed. The 91 IQ2_XXS_MMQ expert stacks ARE the
# run: 91 x 73 s at 16 threads ~= 111 min against a measured 109 min quantize
# stage, so the 345 MXFP8_LT and 38 CUTLASS_MXFP4 tensors are noise beside
# them. That path is CPU-bound and scales -- one 415 MB IQ2 stack took 130.9 s
# at 8 threads, 72.9 s at 16 and 43.7 s at 48 (sublinear but real; a repeat at
# 16 came back 73.0 s, so page cache is not confounding it). 36 leaves the box
# headroom for a concurrent session.
#
# The CUTLASS path does NOT scale: warm 16 and warm 48 both hit 3.2 s on an
# 855 MB stack, and its apparent speedup was entirely page cache. Probe BOTH
# regimes before believing any thread number here.
#
# Staging the source weights on local disk was measured and rejected: local
# reads 1.1 GB/s vs 575 MB/s from the NAS, but the run reads ~145 GB of source
# once, so it saves ~2 min of a ~109 min stage -- and the dominant IQ2 path is
# CPU-bound anyway (its cold and warm times are identical). It also would not
# fit: 145 GB of source plus a 92.5 GB artifact against ~154 GB free.
THREADS=${THREADS:-36}

SURV=$G/reap/reap25-lcb50-survivors.json
MAP=$G/prisma/v5mx4-format-map.json         # the only map: names IQ2_XXS_MMQ, not plain IQ2_XXS

TMPL=$R/template-reap.gguf
DTMPL=$R/dspark-template.gguf
FINAL=$R/v5mx4-collapsed.gguf

step() { echo; echo "=== $* ==="; date -u +%H:%M:%S; }

mkdir -p "$R"

step "1/3 build_main_template (REAP-shaped: expert dim 192, router/bias stay 256)"
"$PY" "$G/build_main_template.py" --hf "$HF" --out "$TMPL" \
    --tokenizer-from-hf --reap-survivors "$SURV"

step "2/3 build_dspark_template (drafter manifest from the checkpoint's mtp.* block)"
"$PY" "$G/build_dspark_template.py" --hf "$HF" --out "$DTMPL"

step "3/3 quantize: main + drafter, pruned and pre-formatted, in ONE pass"
# dspark_type_flags.txt holds exact-name --tensor-type overrides and is a
# QUANTIZER argument, not a template-builder one. Building the drafter with
# default types is what produced the 2026-08-01 'has type bf16, expected f32'
# load failure.
# shellcheck disable=SC2046
"$G/deepseek4-quantize" --hf "$HF" --template "$TMPL" --dspark-template "$DTMPL" \
    --out "$FINAL" --overwrite \
    --format-map "$MAP" --reap-survivors "$SURV" \
    --imatrix "$IMATRIX" --threads "$THREADS" \
    $(cat "$G/dspark_type_flags.txt")

step "gate: audit_artifact_types (fails on any plain twin type)"
"$PY" "$G/audit_artifact_types.py" "$FINAL"

step "gate: classify every difference against the shipped reference"
# Expect IDENTICAL for most, MMQ for the 91 IQ2 stacks, LT for the 25 drafter
# tensors, and zero MISMATCH. A reference is optional -- skip if unset.
if [ -n "${REFERENCE:-}" ]; then
    "$PY" "$G/verify_against_reference.py" "$FINAL" "$REFERENCE"
else
    echo "REFERENCE unset; skipping the differential gate"
fi

step "done"
ls -la "$FINAL"
df -h /home | tail -1
