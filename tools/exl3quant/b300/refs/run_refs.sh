#!/usr/bin/env bash
# The reference captures we owe, on REF_GPUS (MP=2) while run.sh quantizes on the others.  DeepSeek's own inference/
# stack (torch 2.10.0+cu130, tilelang 0.1.8: bootstrap.sh REFS=1), the V4.1 kit's capture script and format.
# Resumable: a stage is skipped once refs/stage/<stage>.done exists.  When it exits (done or failed) it touches
# refs/GPUS_FREE, and run.sh starts quant workers on REF_GPUS.
#
#   V41_ROOT=/nvme/v41 REF_GPUS=6,7 [VEXP=1] bash refs/run_refs.sh
#
# Stages (outputs under refs/out/; see refs/README.md for what each file is and which gate reads it):
#   probe-v41     fp8 GEMM go/no-go on each REF GPU (NaN / nondeterminism killed the B200 rentals): NO-GO stops here
#   convert-v41   convert.py --model-parallel 2 -> refs/v41-mp2 (~445 GB; removed at the end unless KEEP_CONVERTED=1)
#   v41-text      ref_capture_v41.py logits,anchors,dspark,greedy: story at 512,2048,3840,4096,4102,6144,8192,30464,
#                 code at 512,2048,3840 (one depth list, each slug takes the depths it has) -> out/v41
#   v41-short     ref_capture_v41.py logits at 64,128,256 (story, code) -> out/v41-short
#   v41-images    capture_extra.py --mode images -> out/v41-images
#   with VEXP=1 (fetch.sh VEXP=1 first): probe-vexp, convert-vexp (MP=2), vexp-short (capture_extra --mode depths at
#   64,128,256 -> out/vexp-short), vexp-images (-> out/vexp-images)
set -uo pipefail
ROOT=${V41_ROOT:?set V41_ROOT}
HERE=$(cd "$(dirname "$0")" && pwd)
KIT=${KIT:-$ROOT/bundle/refkit}
RPY=${RPY:-$ROOT/venv-ref/bin/python}
REF_GPUS=${REF_GPUS:?REF_GPUS=a,b: the two GPUs the captures use}
MP=2
R=$ROOT/refs
V41=$ROOT/src/v41
VEXP_SRC=$ROOT/src/vexp
V41_REV=2cba9e42aa026125f3ed06c6d98c1db82f7ca027
VEXP_REV=6821d6ad3681a4b137b066b76094fa82ebd0a380
mkdir -p "$R/stage" "$R/out" "$R/logs"
exec > >(tee -a "$R/logs/run_refs.log") 2>&1
say() { echo "[refs $(date -u +%FT%TZ)] $*"; }
trap 'touch "$R/GPUS_FREE"; say "GPUs $REF_GPUS released (refs/GPUS_FREE)"' EXIT
[ "$(echo "$REF_GPUS" | tr ',' '\n' | wc -l)" = $MP ] || { say "REF_GPUS must name $MP GPUs"; exit 1; }
export CUDA_VISIBLE_DEVICES=$REF_GPUS
TORCHRUN=("$(dirname "$RPY")/torchrun" --nproc-per-node "$MP")

stage() {  # name command...: run once, log to refs/logs/<name>.log
  local n=$1; shift
  [ -f "$R/stage/$n.done" ] && { say "$n: done earlier"; return 0; }
  say "$n: start"; local t0; t0=$(date +%s)
  "$@" > "$R/logs/$n.log" 2>&1
  local rc=$?
  if [ $rc = 0 ]; then touch "$R/stage/$n.done"; rm -f "$R/stage/$n.FAILED"; say "$n: done in $(( $(date +%s) - t0 )) s"
  else echo $rc > "$R/stage/$n.FAILED"; say "$n: FAILED rc=$rc -- tail of refs/logs/$n.log:"; tail -n 25 "$R/logs/$n.log"; fi
  return $rc
}
probe() {  # inference dir: each REF GPU
  local g rc=0
  for g in ${REF_GPUS//,/ }; do CUDA_VISIBLE_DEVICES=$g "$RPY" "$HERE/probe_fp8.py" --inference "$1" || rc=1; done
  return $rc
}
convert() {  # src dst [--n-experts N]
  local src=$1 dst=$2; shift 2
  [ -f "$dst/model$((MP - 1))-mp$MP.safetensors" ] && return 0
  (cd "$src/inference" && "$RPY" convert.py --hf-ckpt-path "$src" --save-path "$dst.tmp" --model-parallel $MP "$@") \
    && rm -rf "$dst" && mv "$dst.tmp" "$dst"
}
kit() {  # out phases depths
  mkdir -p "$1"
  (cd "$V41/inference" && PYTHONPATH="$V41/encoding" "${TORCHRUN[@]}" "$KIT/ref_capture_v41.py" --ckpt "$R/v41-mp$MP" \
     --config "$V41/inference/config.json" --prompts "$KIT" --out "$1" --slugs story code --phases "$2" --depths "$3" \
     --revision "$V41_REV") && cp "$KIT/story.tokens.bin" "$KIT/code.tokens.bin" "$1/"   # the gate reads both side by side
}
extra() {  # src mpdir out repo rev mode [args]
  local src=$1 mp=$2 out=$3 repo=$4 rev=$5 mode=$6; shift 6
  mkdir -p "$out"
  (cd "$src/inference" && PYTHONPATH="$src/encoding" "${TORCHRUN[@]}" "$HERE/capture_extra.py" --inference "$src/inference" \
     --ckpt "$mp" --tokenizer "$src" --kit "$KIT" --out "$out" --mode "$mode" --model "$repo" --revision "$rev" "$@")
}

say "REF_GPUS=$REF_GPUS: $(nvidia-smi -i "$REF_GPUS" --query-gpu=name,memory.total --format=csv,noheader | paste -sd';')"
[ -f "$V41/.verified" ] || [ -n "${NO_VERIFY:-}" ] || { say "src/v41 not verified: fetch.sh"; exit 1; }
"$RPY" -c "import tilelang" || { say "no reference venv: bootstrap.sh REFS=1"; exit 1; }

stage probe-v41 probe "$V41/inference" || { say "NO-GO: DeepSeek's fp8 kernels are not sound on this GPU; no V4.1 capture"; exit 1; }
stage convert-v41 convert "$V41" "$R/v41-mp$MP" || exit 1
stage v41-text kit "$R/out/v41" logits,anchors,dspark,greedy 512,2048,3840,4096,4102,6144,8192,30464
stage v41-short kit "$R/out/v41-short" logits 64,128,256
stage v41-images extra "$V41" "$R/v41-mp$MP" "$R/out/v41-images" deepseek-ai/DeepSeek-V4.1-Flash $V41_REV images
[ "${KEEP_CONVERTED:-0}" = 1 ] || { ls "$R"/stage/v41-*.FAILED >/dev/null 2>&1 || rm -rf "$R/v41-mp$MP"; }

if [ "${VEXP:-0}" = 1 ]; then
  [ -f "$VEXP_SRC/.verified" ] || { say "src/vexp not verified: fetch.sh VEXP=1"; exit 1; }
  ne=$("$RPY" -c "import json; print(json.load(open('$VEXP_SRC/inference/config.json'))['n_routed_experts'])")
  stage probe-vexp probe "$VEXP_SRC/inference" || exit 1
  stage convert-vexp convert "$VEXP_SRC" "$R/vexp-mp$MP" --n-experts "$ne" || exit 1
  stage vexp-short extra "$VEXP_SRC" "$R/vexp-mp$MP" "$R/out/vexp-short" deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
    $VEXP_REV depths --slugs story,code --depths 64,128,256
  stage vexp-images extra "$VEXP_SRC" "$R/vexp-mp$MP" "$R/out/vexp-images" deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
    $VEXP_REV images
  [ "${KEEP_CONVERTED:-0}" = 1 ] || { ls "$R"/stage/vexp-*.FAILED >/dev/null 2>&1 || rm -rf "$R/vexp-mp$MP"; }
fi
(cd "$R/out" && find . -type f ! -name SHA256SUMS -printf '%P\n' | sort | xargs -r sha256sum > SHA256SUMS)
failed=$(for f in "$R"/stage/*.FAILED; do [ -e "$f" ] && basename "$f" .FAILED; done | paste -sd' ')
say "captures done${failed:+; FAILED: $failed}; out: $(du -sh "$R/out" | cut -f1)"
[ -z "$failed" ]
