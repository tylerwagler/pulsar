#!/bin/sh
set -e

# The ds4 (DwarfStar) release GGUF lives in this repo.  One artifact, one
# target: the current release only.  It is a single self-contained file with
# the DSpark drafter merged in-file (auto-enabled on load).  The repo is
# public; no token is required for the download.
REPO="twaggs88/DeepSeek-V4-Flash-REAP25-DSpark-ds4-GGUF"
# Every release in this line is REAP-25 trimmed with a per-layer keep
# policy: layers 3-42 keep 192 of 256 routed experts, the first three
# layers keep the full 256 (the keep counts ship in the GGUF header,
# reap.layer.keep_count).  Earlier docs claimed the v4 line was unpruned;
# that was an error -- the shipped file's header says otherwise.
#
# v5 (current): the srcfmt line.  DeepSeek-V4-Flash-0731 weights carrying
# the checkpoint's native numerics end to end (BF16 source tensors, MXFP8
# E4M3 byte-lossless re-encode, MXFP4 byte-lossless experts on the
# quality-sensitive layers, 2-bit IQ2_XXS_MMQ aligned pre-store on the
# rest), 0731 DSpark drafter merged in-file, NVFP4 runtime KV.  Fidelity
# gated against unquantized B300 reference logits.
# v4 (previous): the pre-srcfmt build, same expert allocation shape.
V5_FILE="ds4flash-v5.gguf"
V4_FILE="ds4flash-v4.gguf"

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
  ./download_model.sh v5 [--token TOKEN]

Targets:

  v5     Current release build (srcfmt line), about 93 GB on disk:
         DeepSeek-V4-Flash-0731 weights carrying the checkpoint's native
         numerics end to end (BF16 source tensors, byte-lossless MXFP8/
         MXFP4 re-encodes), the 0731 DSpark drafter merged in-file, and
         the measured precision allocation -- 2-bit routed experts in the
         type-43 IQ2_XXS_MMQ aligned-SoA pre-store (tensor-core layout
         baked at quantize time: no boot-time repack), 16 quality-
         sensitive expert layers at CUTLASS MXFP4, MXFP8 attention/
         shared/head, REAP-25 per-layer expert trim.  Fidelity gated
         against unquantized B300 reference logits.  Targets a single
         NVIDIA GB10 (~121 GB usable) with room for very deep context.
         Requires a pulsar engine built with CUDA_ARCH=sm_120f.

  v4     Previous release (pre-srcfmt), about 92 GB on disk.

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
    v5) MODEL_FILE=$V5_FILE ;;
    v4) MODEL_FILE=$V4_FILE ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo "Use: ./download_model.sh v5   (or v4 for the previous release)" >&2
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
