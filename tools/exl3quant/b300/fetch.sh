#!/usr/bin/env bash
# The source checkpoints onto local NVMe, pinned and verified file by file against the Hub's sha256 (LFS oid).
#
#   V41_ROOT=/nvme/v41 bash fetch.sh            V4.1-Flash @ 2cba9e42 (~476 GB) -> $V41_ROOT/src/v41
#   V41_ROOT=/nvme/v41 VEXP=1 bash fetch.sh     also V4-Flash-Vision-Exp @ 6821d6ad (~157 GB) -> src/vexp (refs only)
#
# Resumable (hf download skips complete files); a model is ready iff src/<name>/.verified exists.  HF_TOKEN in the
# environment raises the Hub's rate limits (public repos; Xet-backed repos are throttled per IP -- a slow transfer
# is upstream shaping).  The calibration rows are NOT fetched: they ship in the bundle
# (bundle/calib/calib-v41-code-v1.safetensors, built by calib.py at home from /mnt/models/reap-corpus + local
# source trees that are not on this host).
set -euo pipefail
ROOT=${V41_ROOT:?set V41_ROOT}
PY=$ROOT/venv/bin/python
mkdir -p "$ROOT/src" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/fetch.log") 2>&1
say() { echo "[fetch $(date -u +%FT%TZ)] $*"; }

fetch() {  # name repo revision min_free_gb
  local name=$1 repo=$2 rev=$3 need=$4 dir=$ROOT/src/$1
  if [ -f "$dir/.verified" ]; then say "$name: verified ($(cat "$dir/.verified"))"; return; fi
  local free; free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
  [ "$free" -ge "$need" ] || { say "FAILED: $name needs ~${need} GB free under $ROOT, $free GB"; exit 1; }
  say "$name: hf download $repo @ $rev"
  t0=$(date +%s)
  HF_XET_HIGH_PERFORMANCE=1 "$ROOT/venv/bin/hf" download "$repo" --revision "$rev" --local-dir "$dir" --max-workers 32
  say "$name: downloaded in $(( $(date +%s) - t0 )) s, $(du -sh "$dir" | cut -f1)"
  "$PY" - "$repo" "$rev" "$dir" <<'EOF'
import hashlib, os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
repo, rev, d = sys.argv[1:]
info = HfApi().model_info(repo, revision=rev, files_metadata=True)
if info.sha != rev:
    raise SystemExit(f"{repo}: revision resolves to {info.sha}, pinned {rev}")
def check(s):
    p = os.path.join(d, s.rfilename)
    if not os.path.exists(p):
        return f"missing {s.rfilename}"
    if os.path.getsize(p) != s.size:
        return f"size {s.rfilename}: {os.path.getsize(p)} != {s.size}"
    if s.lfs is not None:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while b := f.read(1 << 24):
                h.update(b)
        if h.hexdigest() != s.lfs.sha256:
            return f"sha256 {s.rfilename}"
    return None
with ThreadPoolExecutor(16) as ex:
    bad = [r for r in ex.map(check, info.siblings) if r]
n_lfs = sum(s.lfs is not None for s in info.siblings)
print(f"{len(info.siblings)} files ({n_lfs} LFS sha256-checked), {len(bad)} bad")
if bad:
    raise SystemExit("\n".join(bad[:20]))
EOF
  echo "$repo@$rev $(date -u +%FT%TZ)" > "$dir/.verified"
  say "$name: verified"
}

fetch v41 deepseek-ai/DeepSeek-V4.1-Flash 2cba9e42aa026125f3ed06c6d98c1db82f7ca027 500
[ "${VEXP:-0}" = 1 ] && fetch vexp deepseek-ai/DeepSeek-V4-Flash-Vision-Exp 6821d6ad3681a4b137b066b76094fa82ebd0a380 170
(cd "$ROOT/bundle" && grep '  calib/' SHA256SUMS | sha256sum --quiet -c) \
  || { say "FAILED: the bundle's calibration rows do not match SHA256SUMS"; exit 1; }
say "calibration rows: $(ls "$ROOT/bundle/calib") (sha256 ok)"
