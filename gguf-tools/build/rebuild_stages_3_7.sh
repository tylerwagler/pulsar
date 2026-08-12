#!/bin/bash
# Stages 3-7 of the artifact rebuild, in the order execution proved they go.
# Halts on the first failure: every stage here feeds the next, and a silent
# continue would produce a plausible-looking wrong artifact.
set -euo pipefail

R=/home/claude/rebuild
PY=/home/claude/.venvs/pulsar-quant/bin/python
G=/home/claude/Projects/pulsar/.claude/worktrees/kv-rope-bf16/gguf-tools
HF=/mnt/pve1-models/dsv4-flash-0731

INTER=$R/intermediate-full256.gguf     # 102 GB, full 256 experts
COMPACT=$R/compact-reap25.gguf         #  86 GB, 192 experts
DTMPL=$R/dspark-template.gguf
DRAFT=$R/dspark-drafter.gguf
MERGED=$R/merged.gguf
FINAL=$R/v5mx4-rebuild.gguf

step() { echo; echo "=== $* ==="; date -u +%H:%M:%S; }

step "3/7 trim_reap: stamp REAP25 survivors (256 -> 192, -1e30 bias sentinels)"
"$PY" "$G/reap/trim_reap.py" --oracle "$INTER" --out "$COMPACT"

# Disk will not hold intermediate + compact + merged + final simultaneously
# (102+86+92.5+92.5 = 373 GB against ~241 free). The intermediate is dead the
# moment the compact verifies, so drop it here rather than at the merge.
step "3b/7 drop the 102 GB intermediate (disk headroom)"
ls -la "$COMPACT"
rm -v "$INTER"
df -h /home | tail -1

step "4/7 build_dspark_template (drafter manifest from the checkpoint's mtp.* block)"
"$PY" "$G/build_dspark_template.py" --hf "$HF" --out "$DTMPL"

step "5/7 quantize the drafter with the pinned type table"
# dspark_type_flags.txt holds exact-name --tensor-type overrides; they are
# QUANTIZER arguments, not template-builder ones. Building the drafter with
# default types is what produced the 2026-08-01 'has type bf16, expected f32'
# load failure.
# shellcheck disable=SC2046
"$G/deepseek4-quantize" --hf "$HF" --template "$DTMPL" --out "$DRAFT" \
    $(cat "$G/dspark_type_flags.txt")

step "6/7 merge drafter into main (main bytes preserved verbatim)"
"$PY" "$G/merge_dspark_gguf.py" "$COMPACT" "$DRAFT" "$MERGED"

# Second deletion point, learned the hard way: the merge inputs are dead once
# MERGED exists, and the repack needs another full-size copy. Keeping them gave
# compact 81.5 + drafter 11 + merged 92.5 + final 92.5 = 277 GB on a 296 GB
# disk -> ENOSPC partway through the final copy. Both are recoverable from
# MERGED via unmerge_dspark_gguf.py, so dropping them here costs nothing.
step "6b/7 drop the merge inputs (disk headroom for the final copy)"
ls -la "$MERGED"
rm -v "$COMPACT" "$DRAFT"
df -h /home | tail -1

step "7/7 repack IQ2_XXS (16) -> IQ2_XXS_MMQ (43)"
"$PY" "$G/repack_iq2_mmq.py" "$MERGED" "$FINAL"

step "GATE: artifact-type audit"
"$PY" "$G/audit_artifact_types.py" "$FINAL" --census

step "DONE"
ls -la "$FINAL"
df -h /home | tail -1
