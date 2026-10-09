# tools/exl3quant/b300 -- the V4.1 EXL3 quantization (and the owed reference captures) on one rented multi-B300 host

One rental does three things: our uniform-K3 EXL3 quantization of DeepSeek-V4.1-Flash's routed experts (40 trunk
layers x 384 + the DSpark drafter's 3 x 128), its per-block held-out grade, and the reference captures we owe (V4.1
source-precision logits for the gate's story/code prompts, the L248/L283 image prompts, the dsgrade short depths
64/128/256; Vision-Exp short depths and images with `VEXP=1`).  Everything is resumable; nothing on the host needs
access to our repos or NAS.

## Why it parallelises: the calibration stream is the SOURCE model's

`v41_stream.py` calibrates every layer on the **unquantized** model's activations: layer L+1's Hessians come from
the reference output of layer L (its `Stream` carries the FP4/FP8 source forward; no quantized layer ever runs in
the forward -- see the module docstring and `run_layer`).  So no block's quantization depends on another's, and the
job splits in two:

1. **one streamed forward** (`v41_stream.py --hessians-only`, one GPU): every layer's gate/up and per-expert down
   Hessians + the held-out rows' captures, then the drafter's;
2. **quantization in units** (`quant_worker.py`, one process per GPU): a unit is one `quantize_exl3_batch` call --
   14 gate/up projections over the block's shared Hessian, or 16 down projections (`v41_stream.Unit`,
   `block_units`; 79 units per trunk block, 27 per drafter stage, 3,241 in all).  Workers claim units (`O_EXCL`
   claim files) as soon as a block's Hessians land, so quantization chases the forward.  A unit is done iff
   `units/<block>-kK/<unit>.safetensors` and its `.sha256` exist; `assemble` merges a block's units into the same
   `exl3-kK/model-<block>.safetensors` + `proxy-<block>.json` that the single-process path writes (+ a `.sha256`),
   and prunes the units.

A unit's bytes depend only on the unit (its tensors, its Hessians, the seed = layer id, and the torch state
`v41_stream.torch_setup` sets): exllamav3 re-seeds the RNG per tensor, and the shared gate/up Hessian's
finalization (damping, sign flips, Hadamard, LDL) is a pure function of the Hessian and the seed, so a worker
finalizing its own copy equals quantize_block finalizing it once and reusing it.  `test_units.py` proves it on
real data (layer 20 of the sparky estimate, 2026-10-09): gu01 (reuses gu00's finalized Hessian inside
quantize_block) and dn11 (expert 180's Cholesky damping retry), run by two concurrent worker processes, are
byte-identical to the in-process order -- 56/56 and 64/64 tensors, proxy errors equal -- and to the shard the
original single-process run wrote on 2026-10-08.  `bootstrap.sh` additionally checks on the rented host that one
EXL3 quantization gives the same bytes on every GPU.

## Files

| file | what |
|---|---|
| `bundle.sh` | AT HOME: the tarball (quantizer + these scripts, L279's `tools/container`, the calibration rows, the reference-capture kit, MANIFEST.json + SHA256SUMS) |
| `bootstrap.sh` | the venvs: python 3.14 + torch 2.14.1+cu130 + exllamav3 16a49792 (extension built for the host's GPUs: `TORCH_CUDA_ARCH_LIST` = the compute caps present, `10.3` on B300); `REFS=1` adds python 3.12 + torch 2.10.0+cu130 + tilelang 0.1.8 for DeepSeek's inference/ |
| `fetch.sh` | `hf download deepseek-ai/DeepSeek-V4.1-Flash --revision 2cba9e42...` to `src/v41`, every file size- and sha256-checked against the Hub; `VEXP=1` adds Vision-Exp @ 6821d6ad |
| `run.sh` | forward + workers + assemble + grade + summary, resumable |
| `pack.sh` | the return tree (hard links) + MANIFEST.sha256; `push` rsyncs and verifies |
| `refs/run_refs.sh` | the reference captures on two GPUs (MP=2) -- see `refs/README.md` |
| `constraints.txt` | the sparky quantizer venv's pins (pip freeze of `~/pulsar-v41q/.venv`, 2026-10-09) |

The calibration rows ship in the bundle (`calib-v41-code-v1.safetensors`, 2.2 MB, sha256 6bc84c8e...): `calib.py`
needs `/mnt/models/reap-corpus` and our local source trees, which are not on the host -- it is not rebuilt there.

## Host requirements

| | minimum | recommended |
|---|---|---|
| GPUs | 4 x B300 (2 for the references at MP=2, 2 quantizing) | **8 x B300** (any Blackwell / Hopper works: the arch list follows the host) |
| GPU memory | forward ~25 GiB, a worker ~5 GiB; the V4.1 reference at MP=2 ~225 GiB per GPU | 288 GB B300s |
| NVMe | 1.6 TB (V4.1 476 GB + its MP=2 conversion 445 GB + Hessians 175 GB + units/shards 145 GB + stream 20 GB + refs) | **2.5-3 TB** with `VEXP=1` (+157 GB + 157 GB conversion) |
| RAM | 512 GB (DeepSeek's convert.py holds every MP shard's state dict until it saves: ~445 GB for V4.1) | 1-2 TB (what 8 x B300 hosts carry) |
| network in | >= 10 Gbps (476 GB from the Hub; Xet throttles per IP, so a slow transfer is upstream shaping) | 25+ Gbps |
| network out | ~155 GB back (shards + holdout + logs + refs), +175 GB with the Hessians | |
| software | Linux x86_64, NVIDIA driver >= 580, CUDA 13.x toolkit (`nvcc`; sm_103 needs 12.9+, torch cu130 needs 13), git, curl | |

## Expected wall time at N = 8 (ESTIMATES, extrapolated from one GB10 -- sparky, 2026-10-08/09)

Sparky measured: forward 70-90 s/layer, Engram prefetch 141-153 s (layers 1, 14; NFS-bound), Hessians 216 s/layer
over 262k tokens, K3 quantization 1556 s/layer = 55 gu units x ~19.5 s + 24 dn units x ~21 s (unit times measured
by test_units.py).  All units: 40 x 1556 + 3 x ~540 s = **~17.7 GB10-GPU-hours**.  B300 vs GB10 assumed: 3-5x on
the quantizer's trellis kernels (148 vs 48 SMs, compute-bound), 10-20x on the Hessian / forward GEMMs (2.2 PF dense
bf16, 8 TB/s), Engram tables read from local NVMe at 3-7 GB/s.

| stage | estimate | notes |
|---|---|---|
| bootstrap | 15-25 min | extension build ~5-10 min with MAX_JOBS=32; runs while fetch downloads |
| fetch V4.1 | 15-60 min + ~10 min sha256 | 476 GB; Xet per-IP throttling sets it |
| forward + Hessians | 25-45 min | 40 x (~10-15 s forward + 11-22 s Hessians) + Engram + drafter; on GPU 0 |
| quantization | 45-90 min | 3.5-5.9 B300-GPU-hours over 5 GPUs while the refs hold 2 (and GPU 0 the forward), then 8 |
| assemble + grade + summary | 10-20 min | grade: 46 s per 32 experts on GB10 (dry run) = ~9 min per full block there, ~1-3 min on a B300; 43 blocks over 8 GPUs |
| references (parallel to the above) | 1.5-2.5 h | convert 20-40 min, V4.1 text/anchors/dspark/greedy 30-60 min, images 10 min; VEXP=1 +45 min |
| pack + egress | 20-60 min | 155 GB; +175 GB with Hessians |
| **total** | **~3-4.5 h** of rental | the references are the critical path with VEXP=1; drop them (or run them first, on all 8) to cut ~1 h |

## Cost knobs

- **Hessians back or not** (175 GB; **Tyler: decide**).  They let us re-quantize any block at another K without
  the forward (`v41_stream.py --from-hessians`); without them a K change needs another forward (~1 h on a B300,
  ~4 h on sparky -- the forward does run on sparky).  Default: NOT packed (`HESSIANS=1 pack.sh` packs them).
- `VEXP=1` (Vision-Exp short depths + images): +157 GB download, +45 min.  Skip it if the vexp captures can wait.
- Fewer GPUs: N = 4 roughly doubles the quantization stage (~1.5-3 h); the refs need 2 regardless.
- `KEEP_CONVERTED=1` keeps the MP=2 conversions (only useful for a second capture round in the same rental).
- `LATE_GPUS` / `REF_GPUS`: the same two GPUs; run.sh adds them to the quantization when the refs end.

## Checklist

At home:
1. Tyler pushes `work/v41-b300` (the bundle records the engine sha; nothing on the host needs the repo).
2. `bash tools/exl3quant/b300/bundle.sh v41-b300-bundle.tar` (on the VM: needs `origin/work/l279`, `notes/`,
   `/mnt/models/v41-exl3-ours-est/calib-v41-code-v1.safetensors`).
3. Rent: 8 x B300, >= 2.5 TB NVMe, >= 1 TB RAM, CUDA 13 image, >= 10 Gbps.  Note the instance id; destroy only
   after the pull is verified.

On the host (`tmux`; `V41_ROOT` on the NVMe):
4. `mkdir -p $V41_ROOT && tar -xf v41-b300-bundle.tar -C $V41_ROOT`
5. `V41_ROOT=... REFS=1 bash $V41_ROOT/bundle/engine/tools/exl3quant/b300/bootstrap.sh` -- checks the bundle,
   builds both venvs, quantizes a test matrix on every GPU (bytes equal across GPUs).
6. `V41_ROOT=... [VEXP=1] bash .../b300/fetch.sh` (HF_TOKEN in the environment raises rate limits).
7. In two panes, together:
   - `V41_ROOT=... LATE_GPUS="6 7" bash .../b300/run.sh`
   - `V41_ROOT=... REF_GPUS=6,7 [VEXP=1] bash .../b300/refs/run_refs.sh`
   Watch: `$V41_ROOT/run/logs/status.json` (units done), `run/log.jsonl` (forward), `refs/logs/run_refs.log`.
   If `probe-v41` says NO-GO, the V4.1 references cannot be taken with DeepSeek's stack on this host (the B200
   failure mode): run.sh still finishes; decide then (see refs/README.md).
8. After a crash / preemption: re-run the same command(s); they resume.
9. `V41_ROOT=... [HESSIANS=1] bash .../b300/pack.sh` -> `$V41_ROOT/pack` + MANIFEST.sha256.
10. From home: `rsync -a --partial -e "ssh -p PORT" root@HOST:$V41_ROOT/pack/ /mnt/models/v41-exl3-ours/` then
    `cd /mnt/models/v41-exl3-ours && sha256sum -c --quiet MANIFEST.sha256 && echo VERIFIED`; only then destroy.

After: `summary.md` is the per-block grade table; `kmap.py assemble` + L279's `build.py emit --exl3` make the
container (tools/exl3quant/README.md); the references install as refs/README.md says.

## Dry run (sparky, 2026-10-09)

`LAYERS=2 QUANT_LAYERS=0-1 EXPERTS=32 DRAFTER=0 WORKER_GPUS="0 0"` with the smoke calibration rows (9 x 512): two
blocks, 14 units, two workers sharing the one GB10 (+ the forward's GPU joining after the forward), killed once
in the forward and once mid-quantization, resumed, assembled, graded, summarized and packed (MANIFEST verified).
`EXPERTS=N` is dry-run only: the shards say "N of 384" in their metadata and grade_layer grades those experts.
