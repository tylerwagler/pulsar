# R1: the BF16 reference for the Nemotron 3.5 Super VL format comparison (L296)

**What this is.**  The engine format decision for Nemotron 3.5 Super VL (NVIDIA's NVFP4 checkpoint vs our EXL3
packs) waits on one thing: how far each format's next-token distribution is from BF16.  BF16 is 249 GB and fits
neither Spark, so the reference is captured on a 4-GPU box with the card's own vLLM build (TP4), and graded back
on Sparky.  Nothing here writes outside the kit directory (`work/` by default) except Docker's image store.

## Needs

- **4 GPUs with >= 79 GB each** (H100 / H200 / A100-80G / B200 ...), idle: the check refuses any GPU holding
  > 2 GB.  Override `TP` / `MIN_GPU_MIB` only knowingly.  The GPU model is recorded; the grade quotes it.
- **A runtime**, either:
  - **Docker** with the NVIDIA runtime (default when `docker info` works): pulls the card's image
    `vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2` by digest (~25 GB); or
  - **a venv** (`RUNTIME=venv`; the usual case inside a rental container): `r1/run_reference.sh venv` installs the
    same build, vLLM `0.32.0rc1.dev190+g3f1bcfd82`, from vLLM's per-commit wheel index with `uv` (Python 3.12).
  `check` refuses any other vLLM version.
- **The BF16 checkpoint** `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, local (249 GB, ~1 h on a fast link):
  `hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 --local-dir /data/nemotron35-bf16`.
  `check` compares config, index, tokenizer, template and generation-config hashes, the shard count and the total
  bytes with the copy the suite was built from (`r1/bf16-identity.sha256`, `bf16-shard-count`, `bf16-shard-bytes`).
- ~2 GB free for results; `python3` (stdlib only) and `curl` on the host; internet for the image or the wheels.

## Steps (each resumable: rerun the same command)

```bash
sha256sum -c fmtcmp-r1-kit-YYYYMMDD.tar.sha256
tar -xf fmtcmp-r1-kit-YYYYMMDD.tar && cd fmtcmp-r1-kit
export MODEL=/data/nemotron35-bf16
# export RUNTIME=venv && r1/run_reference.sh venv     # only without Docker (~5 min)

r1/run_reference.sh check      # ~1 min. Fails fast on: kit checksums, vLLM build, GPUs, checkpoint identity.
r1/run_reference.sh capture    # server start 5-15 min, then ~1 min smoke, then the capture (~20-40 min).
r1/run_reference.sh bundle     # -> fmtcmp-r1-results-<host>-<date>.tar + .sha256
```

**capture** serves with `tools/fmtcmp/vllm_capture.flags` (the same file the NVFP4 side used): no speculation, no
prefix caching, `--max-logprobs 64`, Mamba state dtype at the checkpoint default; TP4, port 8300.

1. **Smoke, one minute after the server is up.**  One short document (`r1/SMOKE_DOC`) captured twice and graded
   against the NVFP4 capture of the same ids made on Sparky (`selfcheck/nvfp4/`).  A wrong checkpoint, tokenizer
   or flag reads KL >> 1 and top-1 < 60%; the gate is KL/token < 0.5 and top-1 > 80%.  The rerun's KL against the
   first pass is printed too: 0 / 100% means the capture is deterministic, anything else is the noise floor
   (report it; not a failure).  On FAIL nothing else runs.
2. **The capture**: 50 documents, 108,013 positions (48 chat rows <= 2,048 tokens, `code` 3,956, `story` 31,314),
   top-64 log-probs per position into `work/caps/bf16/` (~150-200 MB).  Resumable per document.


## Bring back

`fmtcmp-r1-results-*.tar` + `.sha256`: `caps/bf16/*.jsonl`, the smoke grade, logs (vLLM version, GPUs, flags,
server log), with `MANIFEST.sha256` over every file.  The scope is check -> capture -> bundle (no agentic eval).  Put it on the NAS under
`/mnt/models/fmtcmp/r1/` and tell the L296 thread.

## If something fails

- **Checksums:** the kit was damaged in transit; re-copy it.
- **Checkpoint identity:** a different revision was downloaded; do not capture against it, send `work/logs/run.log`.
- **vLLM version (venv):** the wheel index may have moved; send the log; do not substitute another version.
- **Server does not start:** `work/logs/server-capture.log`.  Out of memory: `GPU_UTIL=0.80`.  Never edit
  `vllm_capture.flags`: the NVFP4 side used the same file.
- **Smoke fails:** stop; send `work/logs/`.  The full capture would not be comparable.
