# R1 hand-off: the BF16 reference for the Nemotron 3.5 Super VL format comparison (L296)

**What this is.**  The engine format decision for Nemotron 3.5 Super VL (NVIDIA's NVFP4 checkpoint vs our EXL3 at K4
and K5) waits on one number: how far each format's next-token distribution is from BF16.  BF16 is 249 GB and fits
neither Spark, so the reference is captured here, on 4x H100, with the card's own vLLM build.  Grading happens back
on Sparky.  Nothing in this kit writes outside its own directory (`work/` by default) except Docker's image store.

## Needs

- 4x H100 80GB, idle (the check refuses if any GPU holds > 2 GB).
- Docker with the NVIDIA runtime, usable by your user; internet for one image pull (~25 GB) and, for the agentic
  step only, PyPI (tool-eval-bench installs from its lock file into a throwaway `uv` container).
- The BF16 checkpoint, `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, downloaded locally (249 GB):
  `hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 --local-dir /data/nemotron35-bf16`.
  The check compares its config, index, tokenizer and template hashes, shard count and total bytes against the copy
  the suite was built from (`h100/bf16-identity.sha256`, `bf16-shard-count`, `bf16-shard-bytes`).
- ~2 GB free for results; python3 (stdlib only) on the host.

## Steps (each resumable: rerun the same command)

```bash
tar -xf fmtcmp-h100-kit-YYYYMMDD.tar && cd fmtcmp-h100-kit
sha256sum -c ../fmtcmp-h100-kit-YYYYMMDD.tar.sha256     # from the directory holding the tar
export MODEL=/data/nemotron35-bf16

h100/run_reference.sh check      # ~1 min. Kit checksums, image + vLLM build, 4 idle H100s, checkpoint identity.
h100/run_reference.sh capture    # ~30-60 min total, see below.
h100/run_reference.sh agentic    # optional but wanted: BF16's own agentic score (~1-2 h).
h100/run_reference.sh bundle     # writes fmtcmp-r1-results-<host>-<date>.tar + .sha256
```

**capture** starts `vllm serve` (TP4, port 8300, never 8000) with the flags in `tools/fmtcmp/vllm_capture.flags`:
no speculation, no prefix caching, `--max-logprobs 64`, the Mamba state dtype left at the checkpoint default.  Then:

1. **Smoke, ~1 minute after the server is up.**  It captures one short document (`h100/SMOKE_DOC`) twice and grades
   it against the NVFP4 capture of the same ids made on Sparky (`selfcheck/nvfp4/`).  A wrong checkpoint, tokenizer
   or flag reads KL >> 1 and top-1 < 60%; the gate is KL/token < 0.5 and top-1 > 80%.  It also prints the
   rerun's KL against the first pass (0 and 100% means the capture is deterministic; anything else is the floor
   every other number is read against -- not a failure, just report it).  On FAIL nothing else runs.
2. **The full capture**: 50 documents, 108,013 positions (48 chat rows <= 2048 tokens, `code` 3,956, `story`
   31,314), top-64 log-probs per position, into `work/caps/bf16/` (~150-200 MB).  Resumable per document.

**agentic** starts the BF16 card's 4x H100 server (`h100/vllm_agentic_h100.flags`: MTP 3, the card's parsers) and
runs tool-eval-bench at the pinned commit `b60e2f0` with `h100/AGENTIC_ARGS` -- the same arguments the NVFP4 arm
used on Sparky.  Results land in `work/agentic/` (report markdown, sqlite, JSON).

## Bring back

`fmtcmp-r1-results-*.tar` and its `.sha256`.  It holds `caps/bf16/*.jsonl`, the smoke grade, the server logs
(vLLM version, flags, `nvidia-smi`), and `agentic/` if that ran, with `MANIFEST.sha256` over every file.  Copy it to
Sparky `/mnt/models/fmtcmp/r1/` (or anywhere on the NAS) and tell the L296 thread.

## If something fails

- `check` fails on checksums: the kit was damaged in transit; re-copy it.
- `check` fails on the checkpoint: a different revision was downloaded.  Do not capture against it; report the
  hashes in `work/logs/run.log`.
- The server does not start: `work/logs/server-capture.log` has vLLM's error.  Out of memory: retry with
  `GPU_UTIL=0.80`.  Do not change any flag in `vllm_capture.flags` -- the NVFP4 side used the same file.
- The smoke fails: stop and send `work/logs/` back; the full capture would not be comparable.
