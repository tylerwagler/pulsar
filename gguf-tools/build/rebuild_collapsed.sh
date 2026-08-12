#!/bin/bash
# Full artifact rebuild from source weights -- COLLAPSED pipeline.
#
# Supersedes rebuild_stages_3_7.sh. Two stages of the old seven are gone
# because the quantizer now does them itself:
#
#   old 3  reap/trim_reap.py     -> --reap-survivors  (expert trim + router/bias
#                                   permutation happen during generation)
#   old 6  repack_iq2_mmq.py     -> a format map naming IQ2_XXS_MMQ, which the
#                                   quantizer emits as type 43 directly
#
# Both collapses were verified byte-exact against the shipped v5mx4 artifact
# before this script replaced the staged one; see docs/ARTIFACT_BUILD.md §Collapse.
#
# ORDER IS LOAD-BEARING: REAP had to collapse FIRST. trim_reap.py is a
# post-quantize transplant that re-slices whole tensors, and it cannot parse a
# type it has no block geometry for -- pointing the old pipeline at an MMQ map
# died with `KeyError: 43` in tbytes(). With REAP inside the quantizer there is
# no post-pass left to trip over, so MMQ became emittable in the same step.
#
# Cost: ~185 GB written for a 92.5 GB artifact (2.0x, was 4.1x), one full-size
# intermediate instead of three, and the peak now fits without deleting anything
# mid-run.
set -euo pipefail

R=${R:-/home/claude/rebuild}
PY=${PY:-/home/claude/.venvs/pulsar-quant/bin/python}
G=${G:-/home/claude/Projects/pulsar/.claude/worktrees/kv-rope-bf16/gguf-tools}
HF=${HF:-/mnt/pve1-models/dsv4-flash-0731}
IMATRIX=${IMATRIX:-/mnt/pve1-models/ds4-quant-archive/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat}
THREADS=${THREADS:-16}

SURV=$G/reap/reap25-lcb50-survivors.json
MAP=$G/prisma/v5mx4-format-map-mmq.json     # MMQ variant: 91 IQ2_XXS -> IQ2_XXS_MMQ

TMPL=$R/template-reap.gguf
COMPACT=$R/compact-mmq.gguf                  # 81.5 GB, 192 experts, type 43
DTMPL=$R/dspark-template.gguf
DRAFT=$R/dspark-drafter.gguf
FINAL=$R/v5mx4-collapsed.gguf

step() { echo; echo "=== $* ==="; date -u +%H:%M:%S; }

mkdir -p "$R"

step "1/5 build_main_template (REAP-shaped: expert dims 192, router/bias stay 256)"
"$PY" "$G/build_main_template.py" --hf "$HF" --out "$TMPL" \
    --tokenizer-from-hf --reap-survivors "$SURV"

step "2/5 quantize: pruned + pre-formatted in ONE pass (was steps 2, 3 and 6)"
"$G/deepseek4-quantize" --hf "$HF" --template "$TMPL" --out "$COMPACT" \
    --format-map "$MAP" --reap-survivors "$SURV" \
    --imatrix "$IMATRIX" --threads "$THREADS" --overwrite

step "3/5 build_dspark_template (drafter manifest from the checkpoint's mtp.* block)"
"$PY" "$G/build_dspark_template.py" --hf "$HF" --out "$DTMPL"

step "4/5 quantize the drafter with the pinned type table"
# dspark_type_flags.txt holds exact-name --tensor-type overrides; they are
# QUANTIZER arguments, not template-builder ones. Building the drafter with
# default types is what produced the 2026-08-01 'has type bf16, expected f32'
# load failure.
# shellcheck disable=SC2046
"$G/deepseek4-quantize" --hf "$HF" --template "$DTMPL" --out "$DRAFT" \
    --overwrite $(cat "$G/dspark_type_flags.txt")

step "5/5 merge drafter into main (main bytes preserved verbatim)"
"$PY" "$G/merge_dspark_gguf.py" "$COMPACT" "$DRAFT" "$FINAL"

step "gate: audit_artifact_types (fails on any plain twin type)"
"$PY" "$G/audit_artifact_types.py" "$FINAL"

step "done"
ls -la "$FINAL"
df -h /home | tail -1
