#!/bin/sh
set -e

# ds4 (DwarfStar) release GGUFs live in this repo. Each file is a single,
# self-contained artifact: REAP-25-pruned DeepSeek-V4-Flash experts, MXFP8
# attention/shared/head, and the DSpark drafter merged in-file (auto-enabled
# on load). The repo is public; no token is required for the download.
REPO="twaggs88/DeepSeek-V4-Flash-REAP25-DSpark-ds4-GGUF"
# v3 is the IQ2_XXS_SOA (type-42) build as of v0.4.0 -- same values as the
# packed artifact, byte-identical logits, ~+2% prefill.  It requires a pulsar
# at or after v0.4.0; older engines reject type 42 at load, so the previous
# packed layout stays available as `v3-packed`.
V3_FILE="ds4flash-v3-soa.gguf"
V3_PACKED_FILE="ds4flash-v3.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${PULSAR_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
DeepSeek V4 Flash GGUF downloader (ds4 / DwarfStar)

Usage:
  ./download_model.sh v3 [--token TOKEN]
  ./download_model.sh v3-packed [--token TOKEN]

Targets:

  v3     Measured-allocation release build, about 92 GB on disk. Routed
         experts on an IQ2_XXS floor with byte-lossless MXFP4 promoted on the
         quality-sensitive layers (per-layer, per-role); MXFP8 attention,
         shared experts, and LM head; DSpark drafter merged in-file.
         Targets a single NVIDIA GB10 (~121 GB usable) with room for a 1M
         token context. Requires a pulsar engine built with CUDA_ARCH=sm_120f.
         Since v0.4.0 the 2-bit experts ship in the type-42 IQ2_XXS_SOA
         layout: identical values and byte-identical logits, ~+2% prefill.
         NEEDS PULSAR v0.4.0 OR NEWER -- older engines reject type 42.

  v3-packed
         The same build in the original type-16 IQ2_XXS layout. Use this if
         you are running a pulsar older than v0.4.0.

Options:
  --token TOKEN  Hugging Face token (optional; the repo is public). Otherwise
                 HF_TOKEN or the local HF token cache is used if present.

Environment:
  PULSAR_GGUF_DIR   Directory used for downloaded GGUF files.
                 Default: ./gguf

After download the script updates:
  ./ds4flash.gguf -> <download directory>/<selected model>

Then start the server (the shipped binary reads ./ds4flash.gguf by default):
  ./pulsar-server -m ds4flash.gguf --ctx 100000
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

MODEL=$1
shift

case "$MODEL" in
    v3) MODEL_FILE=$V3_FILE ;;
    v3-packed) MODEL_FILE=$V3_PACKED_FILE ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

find_hf_command() {
    if command -v hf >/dev/null 2>&1; then
        printf '%s\n' hf
        return 0
    fi
    return 1
}

# Prefer the official Hugging Face CLI when present: it is Xet-aware, so large
# GGUFs download chunk-deduplicated and resumable. Fall back to curl otherwise.
download_one() {
    file=$1
    out="$OUT_DIR/$file"
    part="$out.part"
    url="https://huggingface.co/$REPO/resolve/main/$file"

    mkdir -p "$OUT_DIR"

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    if [ -e "$part" ]; then
        echo "Found curl partial download: $part" >&2
        echo "Remove it before retrying, or the resume may be inconsistent." >&2
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$REPO"
    echo "If the download stops, run the same command again to resume it."

    HF_CMD=$(find_hf_command || true)
    if [ -n "$HF_CMD" ]; then
        if [ -n "$TOKEN" ]; then
            "$HF_CMD" download "$REPO" "$file" --repo-type model --local-dir "$OUT_DIR" --token "$TOKEN"
        else
            "$HF_CMD" download "$REPO" "$file" --repo-type model --local-dir "$OUT_DIR"
        fi
    else
        if [ -n "$TOKEN" ]; then
            curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
        else
            curl -fL --progress-meter -C - -o "$part" "$url"
        fi
        mv "$part" "$out"
    fi

    if [ ! -s "$out" ]; then
        echo "Download finished but expected file is missing: $out" >&2
        exit 1
    fi
}

download_one "$MODEL_FILE"

cd "$ROOT"
ln -sfn "$OUT_DIR/$MODEL_FILE" ds4flash.gguf
echo "Linked ./ds4flash.gguf -> $OUT_DIR/$MODEL_FILE"

echo
echo "Done."
