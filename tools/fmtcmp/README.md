# fmtcmp — Nemotron 3.5 Super VL: NVIDIA's NVFP4 vs our EXL3 K4 / K5, graded against BF16 (L296)

The engine format decision for Nemotron waits on this.  NVIDIA's card shows the NVFP4 checkpoint (Model Optimizer +
quantization-aware distillation, W4A4 on Blackwell) holding most rows but dropping on agentic ones (Terminal-Bench
2.1 66.7 -> 60.8, DeepSWE 24.65 -> 18.2, WebVoyager 77.7 -> 66.3, MMLU-ProX 76.9 -> 73.8).  This harness measures
both formats the same way: teacher-forced next-token distributions against BF16, plus a small agentic eval.

## Arms

| arm | weights | runs on | logprob capture | agentic |
|---|---|---|---|---|
| **BF16** (reference) | `nemotron35-super-vl-bf16`, 249 GB | Tyler's 4x H100, vLLM TP4 (the card's build) | `h100/run_reference.sh capture` | `h100/run_reference.sh agentic` (card recipe, MTP 3) |
| **NVFP4** | `nemotron35-super-vl-nvfp4`, 70 GB | Sparky, vLLM, the card's Spark recipe | `sparky_vllm_job.sh capture` | `sparky_vllm_job.sh agentic` (LiLiCorr, card default) |
| **EXL3 K4 / K5** | quant session's packs (below) | Sparky, exllamav3's own forward | `sparky_exl3_job.sh capture` | `sparky_exl3_job.sh agentic` (`exl3_openai_shim.py`) |

vLLM is the pinned card image `vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2`
(`@sha256:cb9bc64f...`, vLLM 0.32.0rc1.dev190+g3f1bcfd82, multi-arch) on both hosts.

## NVFP4 activation modes (Tyler 2026-10-15: how much of NVFP4's loss is activation quantization, and W4A4's speed)

What the checkpoint actually is (`hf_quant_config.json`: `MIXED_PRECISION`, `kv_cache_quant_algo: FP8`):
routed experts (20,480 up + 20,480 down) and the shared experts are **NVFP4**; Mamba `in_proj` / `out_proj`,
attention q/k/v/o and the **LM head are FP8 (W8A8 per-tensor)**; latent fc1/fc2, router, conv1d and embeddings are
BF16; **the KV cache is FP8**.  vLLM's kernel choice (vLLM source, `kernels/linear/__init__.py`,
`fused_moe/oracle/nvfp4.py`; the old `VLLM_NVFP4_GEMM_BACKEND`-style env vars are gone):

| mode | out dir | extra serve args (`EXTRA_ARGS`) | experts | dense FP8 layers | KV |
|---|---|---|---|---|---|
| **W4A4** (card default) | `caps/nvfp4` | none: MoE `FLASHINFER_CUTLASS`, shared experts `FlashInferCutlassNvFp4LinearKernel` | FP4 x FP4 | W8A8 (`FlashInferFP8ScaledMM`) | FP8 |
| **W4A16** | `caps/nvfp4-w4a16` | `--moe-backend marlin --kernel-config {"linear_backend_per_quant":{"nvfp4_w4a4":"marlin"}}` | FP4 weights, bf16 acts (Marlin) | W8A8 | FP8 |
| **all-A16** | `caps/nvfp4-a16` | the above with `"fp8_w8a8":"marlin"` too, and `--kv-cache-dtype bfloat16` | Marlin | W8A16 (FP8 Marlin) | bf16 |
| W4A8 | -- | **not reachable**: `VLLM_MARLIN_INPUT_DTYPE=fp8` raises for NVFP4 weights (`marlin_utils_fp4.py`), and b12x's `w4a8_nvfp4` MoE mode cannot be selected from a ModelOpt checkpoint | | | |

W4A4 -> W4A16 isolates the experts' FP4 activation rounding; W4A16 -> all-A16 the dense layers' FP8 activations
and the FP8 KV; all-A16 vs BF16 is the weight rounding alone.  Each capture job also runs `speed.py` first (prefill
tok/s at 4k / 32k prompts, decode tok/s at 1 and 4 streams, no speculation) in the same exclusive gpuq job, and
records `kernels.txt` (vLLM's kernel-selection log lines) so each mode proves which kernels it ran.

## The prompt set (`build_suite.py` -> `/mnt/models/fmtcmp/suite-v1`, frozen, sha256 per file)

50 documents, 108,013 scored positions, all as token ids (every arm forwards the same ids; no arm tokenizes):
- **chat-00..47** — L251's full-model KL instrument: the 48 held-out rows of `calib-qwen38-v1.jsonl`, mapped back
  to their reap-corpus source conversations and re-rendered through **Nemotron's** chat template (tools, tool calls,
  tool results, reasoning), cut at 2048 tokens as L251 cut them.  7 classes: agentic glaive / hermes-fc /
  hermes-reasoning / swe, code-opencode, longprose-wikitext, prose-ultrachat.
- **code** (3,956) — the decode-reference "code" fixture (raw text, as its manifest renders it).
- **story** (31,314) — the decode-reference "story" fixture (system + user, generation prompt, thinking off): the
  long-context sample; grade.py reports it by depth.

## Capture format and the metric

Every capture is `ds4-fidelity-jsonl-v1` (tools/fidelity): per position the actual token's log-prob and the top-64.
vLLM gives top-K only (`prompt_logprobs`, `--max-logprobs 64`; full vocab over HTTP would be ~14 GB per arm), so
every arm, EXL3 included, is reduced to the same top-64 (+ the actual token) and graded by one estimator:
`fidelity_compare.trunc_kl` (union of the two top-K supports + one tail bucket) — a lower bound on full-vocab KL,
like for like across arms.  `grade.py` reports per group (chat48, each class, code, story, all): KL(BF16 || X) per
token with a document-bootstrap 95% CI, median, top-1 agreement with BF16, NLL of the actual tokens and dNLL vs BF16,
story by depth, and paired KL ratios between arms.  Without a reference it reports absolute NLL only.

Capture flags (`vllm_capture.flags`, shared by both vLLM hosts): no speculation, no prefix caching, max-num-seqs 1,
chunked prefill 8192, Mamba state dtype at the checkpoint default.  Sparky adds `--load-format fastsafetensors`
(the default mmap loader thrashes GB10's unified memory: 260 s per shard).

## Agentic eval

tool-eval-bench at the pinned commit `b60e2f0` (v2.8.0 + 1; our tool-eval anchor's instrument, L292), the standard
69 scenarios, args in `h100/AGENTIC_ARGS` (seed 42, the card's sampling T 1.0 / top-p 0.95), against each arm's
OpenAI-compatible endpoint.  One seed is a screen, not a grade: the anchor's seed-to-seed spread is 5 points, so a
verdict needs 4 seeds per arm.

## EXL3 arms: what the quant session produces

- **Source:** `/mnt/models/nemotron35-super-vl-bf16` — the snapshot the suite was built from
  (`h100/bf16-identity.sha256` pins config, index, tokenizer, template).
- **Converter:** our exllamav3 fork with plan 121's Q1 Nemotron-Omni adapter (Omni wrapper config,
  `layers_block_type` lists, `language_model.` prefix, vision skipped), at a recorded sha.  `CUDA_LAUNCH_BLOCKING=1`
  (L294's GB10 hang) and `calibration_all_experts`.
- **Two uniform packs** (L251: uniform beats mixed until a full-model KL says otherwise):
  - `/mnt/models/nemotron35-exl3/k4` — every quantized Linear at 4 bpw: routed experts, shared expert, Mamba-2
    in/out projections, attention q/k/v/o, latent fc1/fc2 at exllamav3's latent rate (`latent_hq_bits` default);
    head 6 bpw; embedding, router (f32 logits) and norms unquantized.  Plan 121: ~60 GiB.
  - `/mnt/models/nemotron35-exl3/k5` — the same at 5 bpw (head 6).  ~75 GiB.
  - MTP: include it if the adapter converts it; the grade does not use it.
- **Calibration** must not include the graded text: not the 48 held-out rows (`l251_row` in the suite manifest), nor
  the story / code fixtures.  exllamav3's default calibration set is fine.
- **Each pack directory** also holds `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
  `generation_config.json` copied from the source, and a `CONVERT.md` with the fork sha, the exact command, the
  calibration set and the wall time.
- **Check before handing over:** `capture_exl3.py --selfcheck --only chat-11` passes on the pack (it loads the pack,
  runs a cache-less forward and the chunked cached forward over the same 1,024 tokens, and requires them to agree).

How the harness runs them (on Sparky, copied to `/srv/models/nemotron35-exl3/k{4,5}` first for load speed):
```bash
gpuq submit --class short --timeout 20m --mem 90G --label fmtcmp --wait -- \
  ~/fmtcmp/tools/sparky_exl3_job.sh capture /srv/models/nemotron35-exl3/k4 ~/fmtcmp/caps/exl3-k4   # repeat until done
gpuq submit --class short --timeout 20m --mem 90G --label fmtcmp --wait -- \
  ~/fmtcmp/tools/sparky_exl3_job.sh agentic /srv/models/nemotron35-exl3/k4 ~/fmtcmp/teb-k4 -- $(cat h100/AGENTIC_ARGS)
python3 grade.py --suite /mnt/models/fmtcmp/suite-v1 --ref /mnt/models/fmtcmp/r1/caps/bf16 \
  nvfp4=~/fmtcmp/caps/nvfp4 k4=~/fmtcmp/caps/exl3-k4 k5=~/fmtcmp/caps/exl3-k5
```

## Files

| file | what |
|---|---|
| `build_suite.py` | builds the frozen suite (Sparky, CPU) |
| `vllm_capture.flags` | the capture server flags, one authority for both vLLM hosts |
| `vllm_agentic_spark.flags` | NVIDIA's Spark serving recipe (card Quick Start) |
| `sparky_vllm_job.sh` | one gpuq job: pinned vLLM container + capture or agentic client, inside an 18-min budget |
| `capture_exl3.py` | EXL3 capture through exllamav3 (chunked, recurrent state carried), `--selfcheck` |
| `exl3_openai_shim.py` | OpenAI chat-completions over exllamav3 for the EXL3 agentic arm |
| `sparky_exl3_job.sh` | one gpuq job for an EXL3 arm |
| `grade.py` | the grade (KL / NLL / top-1, CIs, ratios) |
| `h100/` | the R1 hand-off kit: `RUNBOOK.md`, `run_reference.sh`, `pack_kit.sh`, the H100 agentic flags |
