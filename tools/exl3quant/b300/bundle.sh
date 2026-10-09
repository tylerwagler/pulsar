#!/usr/bin/env bash
# AT HOME (sparky or the VM): the one tarball the rental needs besides the Hub downloads.  Nothing private has to be
# reachable from the rented host: the engine repo, the L279 builder and the notes stay here.
#
#   bash tools/exl3quant/b300/bundle.sh [OUT.tar]            (from an engine checkout of work/v41-b300)
#
# Contents (bundle/ in the tar; unpack into $V41_ROOT on the host):
#   engine/tools/exl3quant/     this tree's quantizer + b300/ scripts (git archive HEAD: committed state only)
#   l279-container/             L279's tools/container (exl3_rates.admit: the authority the K map is graded by)
#   calib/                      calib-v41-code-v1.safetensors (2.2 MB; calib.py needs the reap corpus + local
#                               source trees, so it is built here, not rebuilt there)
#   refkit/                     the reference-capture kit from notes/reference-capture (ref_capture_v41.py, the
#                               story/code token bins, rig_capture_logits.py + rowdump_lp.py for the vLLM path)
#   MANIFEST.json, SHA256SUMS   what each piece is (shas, revisions); bootstrap.sh checks SHA256SUMS
set -euo pipefail
ENGINE=$(git rev-parse --show-toplevel)
OUT=${1:-$PWD/v41-b300-bundle.tar}
L279_REF=${L279_REF:-origin/work/l279}
NOTES=${NOTES:-$HOME/Projects/AI/pulsar/notes}
CALIB=${CALIB:-/mnt/models/v41-exl3-ours-est/calib-v41-code-v1.safetensors}
STAGE=$(mktemp -d "${TMPDIR_BUNDLE:-$HOME}/v41-bundle.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
B=$STAGE/bundle
mkdir -p "$B/engine" "$B/l279-container" "$B/calib" "$B/refkit"

[ -z "$(git -C "$ENGINE" status --porcelain -- tools/exl3quant)" ] || { echo "tools/exl3quant has uncommitted changes"; exit 1; }
git -C "$ENGINE" archive HEAD tools/exl3quant | tar -x -C "$B/engine"
git -C "$ENGINE" fetch -q origin work/l279 2>/dev/null || true
git -C "$ENGINE" archive "$L279_REF" tools/container | tar -x -C "$STAGE" && cp -a "$STAGE/tools/container/." "$B/l279-container/"
grep -q "def admit" "$B/l279-container/exl3_rates.py" || { echo "$L279_REF: tools/container has no exl3_rates.admit"; exit 1; }
cp "$CALIB" "$B/calib/"
K=$NOTES/reference-capture
cp "$K/v41/ref_capture_v41.py" "$K/rig_capture_logits.py" "$K/rowdump_lp.py" "$K/story.tokens.bin" "$K/code.tokens.bin" \
   "$B/refkit/"
cat > "$B/MANIFEST.json" <<EOF
{
 "made": "$(date -u +%FT%TZ)", "host": "$(hostname)",
 "engine": "$(git -C "$ENGINE" rev-parse HEAD)", "engine_branch": "$(git -C "$ENGINE" rev-parse --abbrev-ref HEAD)",
 "l279_container": "$(git -C "$ENGINE" rev-parse "$L279_REF")",
 "notes_reference_capture": "$(git -C "$NOTES" log -1 --format=%H -- reference-capture 2>/dev/null || echo unknown)",
 "exllamav3": "16a49792a3c93d8432d72e6c4bce800841566577",
 "v41_source": "deepseek-ai/DeepSeek-V4.1-Flash@2cba9e42aa026125f3ed06c6d98c1db82f7ca027",
 "vexp_source": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@6821d6ad3681a4b137b066b76094fa82ebd0a380",
 "calib": "$(basename "$CALIB")", "calib_sha256": "$(sha256sum "$CALIB" | cut -d' ' -f1)"
}
EOF
(cd "$B" && find . -type f ! -name SHA256SUMS -printf '%P\n' | sort | xargs sha256sum > SHA256SUMS)
tar -C "$STAGE" -cf "$OUT" bundle
echo "$OUT: $(du -h "$OUT" | cut -f1), $(wc -l < "$B/SHA256SUMS") files"
cat "$B/MANIFEST.json"
