#!/usr/bin/env bash
# What comes back from the rental, with checksums.  Builds $PACK as a tree of hard links (no copy; same
# filesystem) plus MANIFEST.sha256 over every file, then (optionally) pushes it.
#
#   V41_ROOT=/nvme/v41 bash pack.sh                      build $V41_ROOT/pack + MANIFEST.sha256
#   V41_ROOT=/nvme/v41 HESSIANS=1 bash pack.sh           ... including hessians/ (~175 GB more; see README)
#   V41_ROOT=/nvme/v41 bash pack.sh push user@host:/dest  rsync it out (resumable) and verify the far side
#
# Pulling from home instead (the usual way: the rental has the public IP):
#   rsync -a --partial --info=progress2 -e "ssh -p PORT" root@HOST:$V41_ROOT/pack/ /mnt/models/v41-exl3-ours/
#   (cd /mnt/models/v41-exl3-ours && sha256sum -c --quiet MANIFEST.sha256 && echo VERIFIED)
#
# Contents (quantization; under quant/):
#   exl3-k3/model-*.safetensors (+ .sha256), proxy-*.json   ~145 GB, the EXL3 experts (kmap.py assemble reads them)
#   holdout/*.safetensors                                   ~6.6 GB, held-out FFN inputs (re-grading at home)
#   grade-*.json, summary.md, summary.json                  the per-block grades and the table
#   log.jsonl, logs/                                        forward + worker + stage logs (timings for the next run)
#   debug/                                                  exllamav3's Cholesky-failure dumps (if any)
#   kmap.json                                               the K map the run used
#   hessians/*.safetensors (HESSIANS=1 only)                ~175 GB: re-quantize at another K without the forward
# Contents (references; under refs/, if refs/run_refs.sh ran): see refs/README.md
set -euo pipefail
ROOT=${V41_ROOT:?set V41_ROOT}
RUN=${RUN:-$ROOT/run}
PACK=${PACK:-$ROOT/pack}
HERE=$(cd "$(dirname "$0")" && pwd)
KMAP=${KMAP:-$(dirname "$HERE")/kmaps/v41-k3-uniform.json}
say() { echo "[pack $(date -u +%FT%TZ)] $*"; }

if [ "${1:-}" = push ]; then
  DEST=${2:?pack.sh push user@host:/path}
  [ -f "$PACK/MANIFEST.sha256" ] || { say "no $PACK/MANIFEST.sha256: run pack.sh first"; exit 1; }
  rsync -a --partial --info=progress2 "$PACK/" "$DEST/"
  host=${DEST%%:*}; path=${DEST#*:}
  ssh "$host" "cd '$path' && sha256sum -c --quiet MANIFEST.sha256" && say "pushed and VERIFIED at $DEST"
  exit 0
fi

[ -f "$RUN/stage/summary.done" ] || say "WARNING: run.sh has not finished (no stage/summary.done) -- packing what exists"
rm -rf "$PACK"; mkdir -p "$PACK/quant"
link() { [ -e "$1" ] && cp -al "$1" "$2"; return 0; }
for d in exl3-k* holdout logs debug; do for p in "$RUN"/$d; do link "$p" "$PACK/quant/"; done; done
for f in "$RUN"/grade-*.json "$RUN"/summary.md "$RUN"/summary.json "$RUN"/log.jsonl; do link "$f" "$PACK/quant/"; done
cp "$KMAP" "$PACK/quant/kmap.json"
[ "${HESSIANS:-0}" = 1 ] && link "$RUN/hessians" "$PACK/quant/"
[ -d "$ROOT/refs/out" ] && link "$ROOT/refs/out" "$PACK/refs"
cp "$ROOT/bundle/MANIFEST.json" "$PACK/bundle-MANIFEST.json"
# every shard's own .sha256 was written when it was assembled: check them before the manifest vouches for them
for s in "$PACK"/quant/exl3-k*/model-*.safetensors; do
  [ -e "$s" ] || continue
  (cd "$(dirname "$s")" && sha256sum --quiet -c "$(basename "$s").sha256") || { say "FAILED: $s"; exit 1; }
done
say "hashing $(du -sh --apparent-size "$PACK" | cut -f1) for MANIFEST.sha256"
(cd "$PACK" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 -P 16 -n 8 sha256sum | sort -k2 > MANIFEST.sha256)
say "$PACK: $(wc -l < "$PACK/MANIFEST.sha256") files, $(du -sh --apparent-size "$PACK" | cut -f1)"
