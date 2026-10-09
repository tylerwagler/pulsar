#!/usr/bin/env bash
# Fresh Linux x86_64 host with N x B300 (or any Blackwell / Hopper), CUDA 13.x toolkit, NVIDIA driver >= 580:
# the quantizer's venv (python 3.14, torch 2.14.1+cu130, exllamav3 16a49792 with its extension built for this
# host's GPUs) and, with REFS=1, the reference-capture venv (DeepSeek's inference/ stack: python 3.12, torch
# 2.10.0+cu130, tilelang 0.1.8) -- the versions the existing captures and the sparky estimate ran.
#
#   V41_ROOT=/nvme/v41 [REFS=1] bash bootstrap.sh          (idempotent; re-run after a failure)
#
# Layout under $V41_ROOT: bundle/ (unpacked v41-b300-bundle.tar), venv/, venv-ref/, exllamav3/, src/, run/, refs/.
set -euo pipefail
ROOT=${V41_ROOT:?set V41_ROOT to a directory on the local NVMe}
B=$ROOT/bundle
EXL3_SHA=16a49792a3c93d8432d72e6c4bce800841566577
TORCH=2.14.1
TORCH_REF=2.10.0
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/bootstrap.log") 2>&1
say() { echo "[bootstrap $(date -u +%FT%TZ)] $*"; }
die() { say "FAILED: $*"; exit 1; }

[ "$(uname -m)" = x86_64 ] || die "this script is for x86_64 hosts (GB10/aarch64: see README, -mno-outline-atomics)"
[ -f "$B/MANIFEST.json" ] || die "$B/MANIFEST.json missing: unpack v41-b300-bundle.tar into $ROOT first"
(cd "$B" && sha256sum --quiet -c SHA256SUMS) || die "bundle checksums"

# ---- GPUs, arch list ----------------------------------------------------------------------------------------
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader || die "nvidia-smi"
caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd';')
# B300 is sm_103 (compute_cap 10.3), B200 sm_100.  The extension is built for exactly the archs present: torch's
# cu130 wheel carries sm_100 SASS (runs on sm_103: same major) but exllamav3_ext is compiled here.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$caps}
say "GPUs: $(nvidia-smi -L | wc -l), compute caps $caps -> TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
[ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc under CUDA_HOME=$CUDA_HOME (install the CUDA 13.x toolkit, or set CUDA_HOME)"
nvv=$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
case "$nvv" in 13.*) ;; *) die "nvcc $nvv: CUDA 13.x needed (sm_103 and torch cu130)";; esac
say "nvcc $nvv at $CUDA_HOME"

# ---- uv + the quantizer venv ----------------------------------------------------------------------------------
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
fi
[ -x "$ROOT/venv/bin/python" ] || uv venv --python 3.14 "$ROOT/venv"
PY=$ROOT/venv/bin/python
uv pip install --python "$PY" "torch==$TORCH" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$PY" -c "$B/engine/tools/exl3quant/b300/constraints.txt" \
  numpy safetensors transformers tokenizers huggingface_hub hf-xet ninja setuptools \
  rich typing_extensions pillow pyyaml marisa-trie pydantic llguidance

# ---- exllamav3 at the pinned commit, its extension built for this host -------------------------------------------
if [ ! -d "$ROOT/exllamav3/.git" ]; then
  git clone --filter=blob:none https://github.com/turboderp-org/exllamav3 "$ROOT/exllamav3"
fi
git -C "$ROOT/exllamav3" checkout -q "$EXL3_SHA"
[ "$(git -C "$ROOT/exllamav3" rev-parse HEAD)" = "$EXL3_SHA" ] || die "exllamav3 not at $EXL3_SHA"
# (on GB10 / aarch64 the build needs the -mno-outline-atomics patch -- notes/research/exl3-arm64-2026-09-24.patch;
# it does not apply to x86_64)
if ! "$PY" -c "import exllamav3_ext" 2>/dev/null; then
  say "building exllamav3_ext for $TORCH_CUDA_ARCH_LIST (MAX_JOBS=${MAX_JOBS:-32})"
  (cd "$ROOT/exllamav3" && MAX_JOBS=${MAX_JOBS:-32} uv pip install --python "$PY" --no-build-isolation \
     -c "$B/engine/tools/exl3quant/b300/constraints.txt" -e .)
fi
"$PY" - <<'EOF' || die "quantizer venv self-check"
import torch, exllamav3, exllamav3_ext, safetensors, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "arch list", torch.cuda.get_arch_list())
print("exllamav3", exllamav3.__version__ if hasattr(exllamav3, "__version__") else "?", "ext", exllamav3_ext.__file__)
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.matmul.allow_tf32 = True          # v41_stream.torch_setup's state
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3_batch
# one small EXL3 quantization (K3, mul1, out scales) on every GPU: the extension's kernels run on this arch, and the
# same input gives the same bytes on every GPU (what quant_worker.py's fan-out relies on)
g = torch.Generator().manual_seed(0)
w0 = torch.randn(1024, 512, generator=g, dtype=torch.float32) * 0.02
x0 = torch.randn(8192, 1024, generator=g, dtype=torch.float32)
first, bad = None, False
for i in range(torch.cuda.device_count()):
    dev = torch.device(f"cuda:{i}")
    x = x0.to(dev)
    hd = {"H": x.T @ x, "first_key": "selfcheck", "count": x.size(0), "finalized": False,
          "num_total": x.size(0) * x.size(1), "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}
    qa = {"seed": 1, "K": 3, "devices": [i], "device_ratios": None, "apply_out_scales": True, "mul1": True}
    err, out = quantize_exl3_batch([w0.clone()], [hd], [qa])[0]
    out = {k: v.cpu() for k, v in out.items()}
    same = first is None or all(torch.equal(out[k].reshape(-1).view(torch.uint8), first[k].reshape(-1).view(torch.uint8))
                                for k in first)
    first = first or out
    print(f"  cuda:{i} {torch.cuda.get_device_name(i)} cc{torch.cuda.get_device_capability(i)}: proxy {err:.5f}, "
          f"bytes {'== cuda:0' if same else 'DIFFER from cuda:0'}")
    bad |= not (0 <= err < 0.05) or not same
raise SystemExit(1 if bad else 0)
EOF
"$PY" "$B/engine/tools/exl3quant/test_refkernels.py" >/dev/null || die "test_refkernels (the CPU kernel port)"
say "quantizer venv ready: $PY"

# ---- the reference-capture venv (DeepSeek's inference/: tilelang kernels) ---------------------------------------
if [ "${REFS:-0}" = 1 ]; then
  [ -x "$ROOT/venv-ref/bin/python" ] || uv venv --python 3.12 "$ROOT/venv-ref"
  RPY=$ROOT/venv-ref/bin/python
  uv pip install --python "$RPY" "torch==$TORCH_REF" --index-url https://download.pytorch.org/whl/cu130
  # the snapshot's inference/requirements.txt, with the two pins the 2026-09-11 rentals found: tilelang 0.1.8's
  # declared tvm-ffi range resolves to a release that breaks its import -- pin the one current when 0.1.8 shipped
  uv pip install --python "$RPY" "apache-tvm-ffi==0.1.8.post2" "tilelang==0.1.8" "transformers==5.19.0" \
    "tokenizers==0.23.2" "safetensors==0.8.0" "numpy==2.5.3" "sympy==1.14.0" "Pillow==12.3.0" "tqdm==4.70.1" \
    "huggingface_hub==1.33.0" "hf-xet==1.7.0"
  "$RPY" -c "import torch, tilelang; print('ref venv: torch', torch.__version__, 'tilelang', tilelang.__version__)" \
    || die "ref venv"
  say "reference venv ready: $RPY (the fp8 GEMM go/no-go probe runs in refs/run_refs.sh, on the weights' GPUs)"
fi
say "done"
