# Engine review — 2026-09-11 (static: decode / prefill / quality)

Target: `origin/dev` @ `d6d3feb5` (2026-09-10, "server: --capture-requests PATH …").
The working tree was fast-forwarded from `4d5dfddd` (502 commits behind) to this
tip before the review. `cutlass` is no longer a submodule (removed in
`5a681dd8`); `cutlass.pin` pins v4.7.0 `dcf215af`, and the local clone was
re-aligned to it.

This is a **static read**: no GPU, no model, no new measurements. Everything
below is a hypothesis until the measurement queue at the end prices it. Line
numbers are `origin/dev` content. Branch context: `v41-flash` is `origin/dev`
plus 17 L218 commits (V4.1 shape/CSA2/drafter); "v41: yes" means a working
forward-port template exists there, not that dev has the fix.

Settled ground and dead ends are listed first so nothing here is read as a
re-litigation of measured NO-GOs.

---

## Implementation status — branch `l219-review-fixes` (2026-09-11)

Landed, gate-verified on the GB10, and measured (see below). The list is the
commit map; the verification and locked-clock deltas are the authority on what
actually shipped.

Host-only verification actually run on the branch (in addition to the full
build of all five binaries, clean):
`./pulsar_test --server` PASS (its min_p out-of-range case exercises A3's new
clamp warning), `./pulsar_test --sampler` PASS (169 shape x config combos,
2496 fixed-seed trials), `./pulsar_test --sampler-prefilter` PASS,
`./pulsar_test --spec-math --lib-utf8 --lib-think --ctxmem` PASS,
`./pulsar-eval --self-test-extractors` PASS, `./pulsar_agent_test` PASS,
`make seam-check` PASS (83 host files CUDA-API clean).

GB10 verification (2026-09-11, sparky, sm_120f, CUDA 13.3, model
`v5mx4-0731-srcfmt-v1-reapfix-lt`):

- `make cuda-regression` PASS; `make gates` 8/8 **ALL PASS** on the final tip
  (`15bc79a`, after the C10a revert) including the runner's 27 sub-gates;
  `cuda-prefill-gate` byte-identical at all 7 depths;
  `cuda-minp-prefilter-gate` and `cuda-spec-sampling-gate` PASS on the B2
  sampler fast arm.
- **Two bugs the GPU gates caught, fixed on the branch:** C7's parallel
  normaliser loop had a compile-time bound that let `--use_fast_math`
  reassociate the add chain (every prefill logit moved) -- fixed in `09228ff9`;
  and A2's digest exposed glibc `fmemopen` null-terminating over the last byte
  of an exactly-full buffer, which had silently corrupted the last payload byte
  since before this branch -- fixed in `1cd0f145`.
- The decode byte baseline is re-anchored to `cf30211f` (B1): blob-to-blob KL
  1.26e-9..1.23e-7, top-1 identical at all three depths, top-5 4-5/5
  (`01c1b9ca`, provenance md next to the blob).
- **Locked-clock deltas vs `d6d3feb`** (pulsar-bench, teacher_forced_corpus,
  two stable rounds): prefill **+5.2% @4096** and **+5.4% @8192**; decode
  **+1.8% @2048** and **+2.0% @8192**.
- **Plain decode census on the current tree** (nsys, 8k and 28k decode
  windows): dense A8 GEMVs 39.6%, bf16 GEMVs 15.2% (the 1 GB BF16 output head
  alone is 4.15 ms/token at 94% of roofline), grouped A8 12.3%, IQ2 GEMV
  10.7%, MXFP4 experts 10.0%, attention 3.3%, indexer scorer <0.5% even at
  28k.  Plain decode is at the memory roofline; the actionable items were the
  top-k dispatch and the serving pool, both landed above (C10d: CUB top-k
  +0.8% @8k; pool cap 8 -> 16: +10.8% 12-way aggregate, no stalls at 16-way).

- A1 E8M0 `0xFF` bind-time refusal (`a43ec0b6`)
- A2 payload digest, format v10 (`117541f7`; fmemopen terminator fix `1cd0f145`);
  the one-byte corruption refusal is now asserted in
  `tests/session_payload_gate.cpp` (a flip of the last data byte, offset len-9 --
  the trailing 8 are the digest -- -> load refused, v10 digest); wired into the
  battery (`cuda-session-payload-gate` target) by the tail review below
- A3 sampler range clamps + warning (`bc9d70bb`)
- A4 AGENTS.md truth (`d6b4e93c`)
- A5 quantizer pre-flight shapes (`b9745c7e`)
- B1 IQ2 down decode GEMV, re-anchored and measured (`cf30211f`, anchor
  `01c1b9ca`)
- B2 full-nucleus sampler skips the below-floor expf (`13501b89`); the review's
  NaN-temperature hole is closed twice: `parse_sampling_key` refuses non-finite
  knobs (`api_parse.cpp`) and the full-nucleus fast arm requires
  `isfinite(temperature)` so the general arm's mass guard refuses it
- B3 greedy spec argmax readback (`4fa5e0b6`); the review's stale-compact bug
  (argmax branch did not reset `spec_compact_rows`) is fixed in `imatrix.cpp`
  and gated by the new `sampled/greedy alternating` shape in
  `tests/dspark_batch_gate.cpp` (verified FAIL on the bug, PASS after)
- B6 persistent plain/mixed lane logits (`e5b2979f`)
- B7 bulk `pulsar_tokens_copy` (`f59043f7`)
- B8 async drafter seed copies (`eb9b5a8b`)
- C2 MMQ gate/up fold emits the mid E4M3 itself, both-43 and mixed case B
  (`8a6986ea`); case A (type-40 gate/up) emits too, via the scatter kernel and
  the GEMV epilogue (`8dd124f8`)
- C3 `hc_expand` destination dedupe (bit-exact) (`ef576858`)
- C7 block-parallel softmax (bit-exact; runtime-bound fix `09228ff9`); the
  phase-2 tile invariants the review asked for are now `static_assert`s
- C8 16-byte `cp.async` staging (bit-exact) (`1dc537ed`)
- C9 refuted — already covered at block granularity (`8c67aeb0`)
- C10d indexer top-k uses CUB for 1025..4095 too (measured +0.8% @8k decode)
  (`3af5220b`)
- B9 (pool-cap half): bank-pool auto-cap raised 8 -> 16 after the requested
  re-measure (+10.8% 12-way aggregate, no TTFT stalls; 16-way 56.5 t/s)
  (`a8faa620`)
- C10a (indexed prefill gact/rope fusion) — **MEASURED NO-GO 2026-09-11** and
  reverted (`bedc2744`): the in-kernel epilogue costs attention +31.4 ms while
  removing 44.5 ms of fallback, a -0.3% wash at 4096 tokens that worsens with
  depth.  Kernel census and reasoning in the revert message.

Still open, in value order: B10 (sampled redraft), C4 (indexer f16 scores), C5
(`low` fusion), C6 (`mxf4nvf4`), C10b/c and the `attn_pack_store` retile.  B9's
row-budget half, B4, C1, D1 and D2 need dedicated campaigns.  B5 is a measured
NO-GO (below).

### Tail review — the five commits after the external review (2026-09-11)

`docs/engine-review-2026-09-11-review.md` scoped `dev..5bdadf04` (33 files,
+1597/−285).  Five commits landed after it and were never reviewed: `6801cdf6`
(B3), `bcaaa541` (B2), `2a32fdbf` (C7), `35a545c0` (A2), `2e51f069` (docs).  A
static read of that range found all four fixes correct, and closed three gaps
on `a9ea5210`:

- **A2 had no instrument in the battery.**  `cuda-session-payload-gate` was
  referenced only by its own Makefile definition -- absent from `GATE_TARGETS`
  and from `tests/gates_runner.cpp` -- so the new one-byte-corruption case
  could never fire on a landing.  It is in `GATE_TARGETS` now; folding it into
  the runner (one fewer model load) is the follow-up noted at the list.
- **The engine still accepted `+-Inf` temperature** (and a non-finite
  `top_p`/`min_p`), turning it into a uniform or unfiltered draw instead of a
  refusal, because those make a *finite* mass and so pass the mass guard.  All
  three are refused at the entry of `pulsar_sample_dist_build` now; the fast
  arm's `isfinite(temperature)` special case is gone with them.
- **The B3 gate's audit re-derived the readback arm from the temperatures
  alone** -- a second copy of the min-p contract, which also depends on
  `top_k`/`top_p`/`min_p`.  It now captures the engine's own armed flags before
  `arm_capture(0)` retires them, checks the live rows against that, and runs
  the audit before the per-bank round ends (the state the walk consumes)
  instead of after.  The pre-fix bug still fails it.

Also corrected: the A2 case flips the last DATA byte (offset `len-9`), not a
mid-payload field; and the `AF16_HEADS` `static_assert` message now names the
M-tile invariant it guards rather than the half-warp property that follows from
`AF16_ROWS`.

Host-only verification run at `a9ea5210` (authoring box, no GPU): all four
changed TUs compile with zero warnings (`g++ -Wall -Wextra`; `nvcc
-arch=sm_120f` for `attn_f16.cu`); `./pulsar_test --sampler`, `--server` and
`--sampler-prefilter` PASS; `make seam-check` PASS (83 host files).

GB10 verification at `d847440` (sparky; the bank tree fast-forwarded from
`2e51f069` by git bundle -- GitHub is unreachable from the authoring box -- and
the two gate binaries rebuilt, 0 warning/error lines):

- `cuda-session-payload-gate` **PASS** -- v10, 22,453,536 B payload, comp fnv
  and all 129,280 logits identical across the round trip.  The corruption case
  fired: `corruption at byte 22453531 refused (rc=1): KV checkpoint digest
  mismatch (corrupt payload)`.  First run of that assertion as a battery gate.
- `cuda-dspark-batch-gate` **PASS** at depth 0 (8 ticks), depth 1 (6) and depth
  4 (6); all three shapes `IDENTICAL`, including `sampled/greedy alternating`,
  the readback-arm switch the B3 audit exists for (24 / 18 / 18 bank-ticks).
- **The rewritten audit is mutation-validated.**  With the argmax branch's
  `spec_compact_rows = 0` removed, the alternating shape fails with
  `STALE COMPACT ROWS: the engine armed the argmax readback for 3 rows, but
  spec_compact_rows=3 is still live (argmax_rows=3)` on all four greedy ticks
  (gate FAIL, rc=1); it passes again after the restore.  The audit reaches the
  path it claims to reach.

Not run: the full `make gates` battery at this tip (the branch still owes one);
and no served A/B for the sampler entry guard -- a non-finite knob is
unreachable through the server parser, so every served path is unchanged by
construction.

### B5 measured NO-GO — the lane split costs a weight sweep (2026-09-11)

The per-member lane plan was built, gate-verified, and REFUTED by measurement.
It replaces the group verdict with two subsets (spec-eligible and plain) and
runs one quantum per subset in the same worker iteration.  That fixes the
demotion -- and loses badly:

| mix (192 tok each, 3 reps, greedy) | pre-B5, one shared sweep | B5 split, two sweeps |
| --- | --- | --- |
| 1 spec + 1 logprobs | 29.02 tok/s aggregate | 22.77 (**-22%**) |
| 3 spec + 1 logprobs | 42.40 tok/s aggregate | 32.27 (**-24%**) |

Decode is memory-bound: one M=4 sweep streams the weights once for every
client, while the split streams them twice.  Even the SPECULATING clients lose
(8.1 vs 11.7 tok/s at 3+1) -- the spec gain does not come close to a second
sweep.  **The demotion was buying co-batching, not broken.**  Reverted on
`l219-review-fixes`; the experiment is kept on branch `b5-lane-split` (`4e12f472`)
with the probe in `pulsar-notes/probes/b5-mix-probe.py` (1+1: 22.77 vs 29.02;
3+1: 32.27 vs 42.40).

The gate is real and discriminating, which is what makes the NO-GO trustworthy:
`tests/spec_batched_gate.py` case 5 drives a CHAT `/logprobs` partner against a
spec decoder and fails on the pre-B5 engine
(`A+logprobs={'spec-batched': 2, 'idle': 5, 'batched': 38}`, no drafts counted)
while passing on the split.  Getting it to be discriminating cost two
corrections: lane 2's metric name is `batched`, and `/v1/completions` does NOT
honor `logprobs`, so only the chat surface can create a non-spec decoder.

**If B5 is pursued, the design to try is the review's actual hint** -- base-only
members riding the SPEC quantum in ONE sweep (`worker_spec_batched_quantum`
tolerates them), not a second quantum.  That keeps the sweep count at one and
is the only version that can beat the measured baseline; it is a deeper change
(the spec rounds, the B3 readback arm, and full rows for the logprobs member
all have to agree).

**Filed separately (2026-09-11):** `/v1/completions` accepts `logprobs: true`,
returns HTTP 200 with no logprobs payload, and keeps speculation enabled (solo
probe: `draft_delta=61` with or without the flag).  `/v1/chat/completions` is
correct (`draft_delta=0`, payload present).  A client on the legacy surface
cannot tell the option was dropped -- fail-open, rules 1 and 9.

---

## Priority board

| ID | Item | Class | Expected value | Effort | Basis |
|----|------|-------|----------------|--------|-------|
| B1 | IQ2 **down** still runs the MMA tile at decode | decode | ~1.5–3% single-stream, more at verify width | medium | L210 measured gate/up GEMV ~85% roofline vs tile ~30–35%; down is ~1/3 of expert bytes |
| B2 | Plain/mixed-lane sampler: full-vocab `expf` + ~5 MB scratch per token | decode | ~2–6% at 2+ banks, GPU idle behind host | low/medium | L149 comment measures the pass at ~630 µs/position; scratch is 40 B × vocab |
| C2 | MMQ/mixed-40 SwiGLU `mid` written f32 then re-encoded | prefill | ~1.2–1.5% of a 4096 chunk | medium | byte count; L165 fixed only the small-batch lane |
| C6 | Indexer scorer: `mxf4nvf4` + packed nibbles (`idx_spread4` removal) | prefill+decode | scorer 2–4×, whole indexer ~2% of prefill | med-high | `92018615` measured 30.67 G inst/s at k=64; spread ~7 ALU/MMA suspected |
| C3 | `hc_expand` re-reads each residual carrier 4× | prefill | ~0.5–2% | medium | 172 launches / 3.5% of prefill; model says most of the traffic is redundant reads |
| C7 | Phase-2 softmax on 32 of 512 threads, serial | prefill+decode | attention is latency-bound; bit-exact restructure | low-med | `attn_f16.cu:711-740` |
| A1 | E8M0 `0xFF` unguarded and inconsistent (Inf vs NaN) | correctness | silent-corruption class | low | direct read of decoders + binder |
| B3 | Greedy spec rounds: full-row D2H + host argmax per row | decode | ~1–3% spec lane | low-med | device argmax already exists |
| B4 | Draft-depth controller stale after L214 | decode | several % where deep chains convert | low (measure) | `SPEC_DEPTH_MAX=5` from a sweep when a row cost 18.4 ms; now 7.17 |
| B5 | One `/logprobs` request latches its whole group to the plain lane | decode | removes a permanent demotion | medium | `batch_active` is one-way |
| A2 | Disk KV payload restored on trust (no integrity check) | correctness | silent bad attention from bit-rot | low-med | store is atomic; load has no digest |
| D1 | KV window rows share the pool's NVFP4 budget | quality | quality on the most-attended rows | high | v41 stores window E4M3×E8M0/32 + pool E2M1×E4M3/16 |
| C4 | Indexer scores materialized f32 `[n_tok×n_comp]` | prefill | ~1–3% at 128k | low (f16) | linear in depth |
| C1 | Expert recipe tier imbalance (9 MXFP4 + 7 mixed layers) | prefill/quality | old estimate up to ~20%, soft map ~4% | high | needs a fresh nsys census first |
| P10* | See C10: indexed-attn epilogue, compressor emit, decode-indexer blocks, top-k dispatch | prefill | ~0.2–0.4% each | medium | |

*(P-IDs from the review sessions are renumbered into B/C/D here; C10 bundles the
smaller structural items.)*

---

## Settled ground — do not reopen without new information

- **Whole-sweep CUDA graph "tape" and per-layer FFN graphs.** Head-only capture
  ships (+1–2% solo); FFN segments measured **−13%** on GB10 (43 small graph
  launches/round); whole tape ceiling +0.6% (`pulsar_gpu.h:229-236`).
- **Native packed-KV reads in prefill attention.** MEASURED NO-GO 2026-08-02:
  −1.6% @8k, −1.9% @16k, −2.7% @131k vs the f32 shadow; DSA top-k keeps the
  working set L2-hot. The shadow itself was later deleted; do not reconstruct it.
- **cuBLASLt algo autotune.** `REDUCTION_SCHEME_NONE` leaves 2–3 candidates;
  measured net-negative from per-shape tuning. The heuristic is right.
- **Prefill chunks below 4096.** 4096→1169 tok/s vs 2048→1055 (`469592de`).
- **`f32_to_f16`/norm/rope micro-optimisation.** Elementwise tier is at ~90–95%
  of 273 GB/s; only whole-pass removal helps.
- **Attention tile tuning knobs.** `AF16_MT=2`, `AF16_MINBLK=1` (spills at 2),
  `AF16_SPLITS=4` (16 splits host-bound), `IDX_MINBLK` 4/5 spills,
  `IDX_NB=8` no gain. The 512-thread/1-block shape is a measured local optimum.
- **Attention operand formats.** fp16 chosen; bf16 strictly dominated (same
  rate, 3–8× error); fp8 changes top-1 on 2–3% of head-token pairs.
- **E4M3 KV row revival.** Retired by L111; NVFP4 is the measured default.
- **Q8_K in MoE.** No Q8_K on any GPU path; the AGENTS.md Deferred Work line is
  stale, not pending work.
- **Sampler tie/stability rules and `top_k` clamps.** Gate-pinned decisions.
- **MXFP8 re-encoding is value-lossless** (source scales are already F8_E8M0).
- **Split-K decode `AF16_SPLITS` and the M-neutral row predicate** are
  measurement-backed; the `spec_sampling_gate` "deliberately NOT fixing" header
  is stale (L160 superseded it).
- **Indexer top-k comparator order** (`topk_pack_key`, zero canonicalisation
  `901e9a08`) is load-bearing; any restructuring must reproduce it exactly.

---

## A. Correctness and robustness

### A1. E8M0 scale byte `0xFF` is unguarded, and the arms disagree on its meaning
- Evidence: `gguf-tools/quantize/dsq_codecs.c:7-8` (`e8m0_to_f32`), used by
  FP8/FP4 dequant at `:83,:121`; runtime decoders `src/cuda/pulsar_cuda_matmul.cu:274,301,488,1824,1904,1919`
  and `src/cuda/pulsar_mxfp4_cutlass.cu:674-675` do `__int_as_float(byte<<23)`
  (255 → +Inf). `cublasLt` `VEC32_UE8M0` and CUTLASS block-scale MMAs treat
  0xFF as NaN per OCP. `src/engine/weights.cpp:948-960` validates type IDs only;
  nothing scans MX scale-factor planes.
- Today: writers clamp to ≤254, so a corrupt/foreign GGUF passes load and then
  multiplies by +Inf in the custom GEMVs (Inf·0 → NaN) while the tensor-core
  arms see NaN — two different failures from one byte.
- Change: validate MX SF planes at bind time (skip the f16 IQ2 `d` plane),
  naming the tensor; die or canonicalise, but pick one. One cold pass over
  ~1/32 of the weight bytes.
- Impact: silent-corruption class, no speed trade. Modelless test over a crafted
  33 B MXFP8 and 17 B MXFP4 block. `v41-flash`: not fixed.

### A2. Disk KV payloads have no integrity check on load
- Evidence: `src/engine/session_payload.cpp:602+` validates magic/version/shape/
  strides/saved_tokens, but KV rows, compressor state and `n_comp[il]` /
  `n_index_comp[il]` restore on trust (bounded by caps only).
  `src/lib/pulsar_kvstore.h:53-56,69`: `sha[41]` hashes the text key, not the
  payload. The store is atomic (`pulsar_kvstore.cpp:899-969`), so torn writes
  are mostly excluded; media bit-rot or an offline edit is not.
- Today: a flipped byte inside a multi-GB payload silently loads; an `n_comp`
  edited low quietly drops compressed recall.
- Change: payload digest in the header, verified on load; mismatch = cache
  miss / re-prefill. Bump `PULSAR_SESSION_PAYLOAD_VERSION` (currently 9). Cheap
  structural add-on: recompute the writer's `n_comp` from
  `(saved_tokens, ratio, prefill_frontier)` where the rule is exact.
- Impact: no output change on healthy payloads. `v41-flash`: not fixed.

### A3. `top_p`/`top_k` are silently reinterpreted
- Evidence: `src/server/api_parse.cpp:18-33` stores `top_p` verbatim, parses
  `top_k` with no range check (only `min_p` is clamped); `src/engine/tokenizer.cpp:1220-1222`
  rewrites `top_p<=0||>1 → 1.0`, `top_k>1024 → 1024`, `top_k<=0 → 0`.
- Today: `top_p=0` (a common near-greedy idiom) becomes a full nucleus; no log.
- Change: validate/clamp in `parse_sampling_key` with one log line, or document
  the clamps as contract. Unit-test the parse; no ppl effect. `v41-flash`: not
  changed.

### A4. Docs and process disagree with the code
- `AGENTS.md:92-94` claims `make test` runs `pulsar-eval --self-test-extractors`
  and `pulsar_agent_test`; `Makefile:1444` runs only `pulsar_test`. (QA doc
  `docs/QA_BEFORE_RELEASES.md` is correct; AGENTS.md is not.)
- `AGENTS.md:123` still names routed-expert types `IQ2_XXS`/`Q2_K`/`MXFP4` and
  `IQ2_XXS_SOA`; `weights.cpp` accepts only 40/43 (plus K-major variants).
  `AGENTS.md:194` still lists "Move MoE decode off Q8_K" as deferred work.
- `docs/engine-perf-map.md`/AGENTS.md carry the retired f32-shadow sentence.
- Stale comments: `src/cuda/pulsar_cuda_internal.h:86` and
  `src/engine/session_payload.cpp:293` say payload v7 (it is 9);
  `tests/spec_sampling_gate.cpp:55` calls a divergence "deliberately NOT
  fixing" that L160/L161 closed.
- `cuda-reference-gate` prints `SKIP` and succeeds when `PULSAR_REF_DIR` is
  unset (`Makefile:867-869`) and is not in `GATE_TARGETS`; `make gates` can
  pass comparing the engine only to itself. Consider a release gate that
  refuses without the reference fixture.
- Impact: reviewer/release correctness, trivial effort. No commit fixes these
  at dev tip.

### A5. Port v41's quantizer pre-flight shape validation
- v41 `1be641a2` adds `validate_plan_shapes` (headers-only scan before
  generation, `gguf-tools/quantize/dsq_generate.c:189`). Dev checks per tensor,
  so a wrong template dies only when that tensor is reached — possibly after
  hours of writing. Low effort, build-robustness only. `v41-flash`: fixed.

---

## B. Decode speed

### B1. IQ2 **down** still runs the prefill-shaped MMA tile at decode (largest decode lever)
- Evidence: gate/up takes the L210 k-major GEMV when
  `pulsar_gpu_matmul_batch_decode_rows() > 0`
  (`src/cuda/mmq/ds4_mmq_d2r.cu:1293`), but the single-tensor launch used for
  down has **no decode branch**: it builds a worklist and launches
  `gateup_iq2_d2r_pair_kernel<kNTileNarrow>` unconditionally
  (`ds4_mmq_d2r.cu:1420-1455`); down is reached via
  `pulsar_cuda_moe.cu:1096` → `ds4_mmq_iq2_xxs_moe_soa` → `ds4_mmq.cu:500`.
  The narrow tile was measured at 30.2–35.4% memory throughput (`e5b44652`);
  L210 priced the tile at ~60% of DRAM roofline vs the GEMV's ~85% on the same
  bytes. `kDecodeGemvMaxK` is 4096 at dev (5120 on v41); down's K is 2048.
- Change: a decode arm for the down tensor (single-tensor GEMV reading the
  producer's E4M3 `mid` like the pair GEMV reads `xq8`), dispatched on the same
  decode-row predicate. Different arithmetic (exact f32 weights, split-K order)
  means the decode byte gate re-anchors, exactly as L210 did for gate/up.
- Impact: down is ~1/3 of routed-expert bytes; extrapolating L210's +3.3%
  single-stream for 2/3 of the bytes gives **~1.5–3% single-stream**, more at
  verify widths. Needs A/B. `v41-flash`: not fixed.

### B2. Plain/mixed lanes pay the full-vocab sampler cost per token
- Evidence: `server_sched.cpp:1991` and `:2735` call `pulsar_sample_logits(...)`
  with no scratch; `pulsar_sample_dist_build`'s min-p path runs one `expf` per
  finite vocab entry (pass 2, `tokenizer.cpp:1222-1300`) after a full max pass;
  `sample_top_p_min_p` allocates a call-local scratch (`tokenizer.cpp:1567-1575`)
  — 40 B × 129,280 ≈ **5.2 MB malloc/free** per sampled token. The L149 comment
  measures the removed full-sum variant at **~630 µs per draft position**. The
  spec lane already avoids both via `pulsar_gpu_minp_prefilter_rows` +
  `pulsar_sample_dist_build_prefiltered` (`session_spec.cpp:803,814`).
- Change: (a) persistent per-slot scratch in the scheduler (small); (b) route
  the production shape (top_k≤0, min_p≥prefilter floor) through the existing
  device prefilter. Greedy and the carry path are unaffected.
- Impact: ~0.6 ms/bank/step of host time with the GPU idle, plus allocator/page
  churn; **~2–6% at 2+ banks**. The prefilter contract's ~3 ulp boundary class
  is already gate-pinned. `v41-flash`: not fixed.

### B3. Greedy spec rounds read back whole rows and host-argmax them
- Evidence: the compact candidate path is armed only for `temperature > 0`
  (`session_spec.cpp:1093`), so greedy takes the full readback in
  `imatrix.cpp:1163` (`head_runs × 517 KB` D2H, up to 8.3 MB) and then
  `sample_argmax` scans each row on the host (`session_spec.cpp:1861`, `:1425`).
  The classic lane computes the same argmaxes on device with
  `pulsar_gpu_argmax_tensor` and reads 4-byte ints (`imatrix.cpp:920-950`).
- Change: for greedy rounds run the device argmax over the head rows (0..n_batch-1
  yields bonus/carry too) and read ints; keep full rows only for sampled rounds.
- Impact: removes ~0.5–2 ms/round plus multi-MB D2H; **1–3% spec lane** (needs
  A/B). Low-medium effort; greedy semantics already defined by `row_tops`.
  `v41-flash`: not fixed.

### B4. Draft-depth controller and scheduler cost model are stale after L214
- Evidence: `pulsar_engine_internal.h:2976` (`SPEC_DEPTH_MIN=2, MAX=5`);
  calibration comment `session_spec.cpp:31-55` ("depth 6 lost on BOTH regimes")
  dates from a sweep when a verify row cost 18.37 ms; L214 re-fit the quench
  model to 45.0 + 7.17·n_batch (`2275ad3a`, `2a6ef45e`). The scheduler's
  `marginal_ms` is still 6.0 (`server_sched.cpp:2174`).
- Change: re-run the L107 depth sweep at [2,8] with production sampling on the
  current kernels; re-set `MAX`, the climb/veto thresholds and `marginal_ms`
  from one census.
- Impact: structured/code regimes that wanted depth 5 should move up; corrected
  overflow admission at c4+. Speed-only (verification is exact at any depth).
  Low effort; needs the rig. `v41-flash`: not fixed.

### B5. One non-spec decoder permanently demotes its whole group
- Evidence: lane select requires every decoder spec-enabled and `n_batched == 0`
  (`server_sched.cpp:1515-1521`); entering the plain quantum sets `batch_active`
  one-way for the request's life (`server_sched.cpp:1922`,
  `pulsar_server_internal.h:2316`); `dspark_spec_enabled = !req.logprobs`
  (`server_jobs.cpp:1207`).
- Today: a single overlapping `/logprobs` request forces spec-capable sessions
  onto the plain lane forever. Spec is worth ~+16% at N=1 and more in aggregate.
- Change: advance non-spec decoders in a separate plain step inside the same
  quantum instead of poisoning the group; `worker_spec_batched_quantum` already
  tolerates base-only members.
- Impact: large for mixed logprobs/spec traffic; zero for all-spec. Needs
  production-mix measurement. `v41-flash`: `server_sched.cpp` identical.

### B6. Per-quantum logits malloc in plain and mixed lanes
- Evidence: `server_sched.cpp:1925` allocates `n × vocab` f32 (up to 4.1 MB)
  and frees at `:2002`; the mixed lane at `:2630-2635` / `:2744`. The spec lane
  was converted to a persistent buffer for exactly this reason (L123,
  `:2113-2118`: "re-faulted 16.5 MB of demand-zero pages every quantum").
- Change: reuse a persistent server buffer sized to max live rows, both lanes.
- Impact: ~0.5–2 ms/quantum at N≥2. Low effort. `v41-flash`: not fixed.

### B7. Bank carry churn copies the full history and logits repeatedly per round
- Evidence: saves at `server_sched.cpp:2295,2387,2432` and restores at
  `:2308,2419`; each `bank_state_save/restore` copies the whole token
  checkpoint plus 517 KB logits (and pending qrows), `session_banks.cpp:549-560,
  595-608`; `pulsar_tokens_copy` is an O(len) push loop (`tokenizer.cpp:97-100`).
  The checkpoint only changes at `round_end`.
- Change: (a) bulk reserve+memcpy in `pulsar_tokens_copy` (few lines); (b)
  append-only/incremental carry copy and skip the logits copy when the shadow
  is unchanged.
- Impact: host time on the inter-round path; est. 1–4% at 50–130k (needs A/B).
  Low effort for (a), medium for (b). `v41-flash`: same calls at
  `session_banks.cpp:510/556`.

### B8. Drafter seed path: synchronous copies and per-row seeding
- Evidence: synchronous `pulsar_gpu_tensor_copy` at `gpu_decode.cpp:420`
  (`kv_rot ← kv_norm`) and `session_spec.cpp:498` (3 copies/committed position);
  the async twin exists at `pulsar_cuda_runtime.cu:1253` for exactly this
  ("destinations only consumed by a later kernel on the same stream"). Seeding
  loops `m <= commit` per row (`session_spec.cpp:1459,1488`) through
  `project_main_x` + `seed_draft_kv`, which loops 3 layers × rows with a copy,
  `rope_tail` and `store_raw_kv` per row (`gpu_decode.cpp:409-465`).
- Change: async copies where stream order suffices; batch the committed rows
  through one multi-row concat/GEMV/norm and one multi-row rope+store per layer
  using the existing per-row position meta.
- Impact: a full-accept K=3 round issues ~60–100 tiny launches + 12 host round
  trips; est. 0.5–2 ms/round. Medium effort; ring/position semantics must be
  preserved. `v41-flash`: not fixed.

### B9. Spec batch width capped at the 16-row budget; pool auto-cap still 8
- Evidence: `pulsar_gpu.h:117` (`PULSAR_SPEC_ROW_BUDGET = PULSAR_GPU_MNEUTRAL_ROWS_MAX`
  = 16) and `:128-135`; the allocator sheds depth when demand > 16
  (`server_sched.cpp:2040-2065`). `pulsar_server_internal.h:995-1003` documents
  the 8-bank auto-cap as a pre-L210 measurement that "awaits a re-measure".
- Change: a campaign to extend the M-independent decode instantiations and the
  row-neutrality gates to 24/32 (high effort), and separately re-run the 8 vs
  12/16 bank aggregate + multi-turn TTFT experiment (low effort).
- Impact: c4+ aggregate depth; could restore the ~25% the old cliff cost or
  confirm it still exists. Needs the rig. `v41-flash`: headers identical.

### B10. Sampled redraft serialises a host round trip per draft position
- Evidence: `session_spec.cpp:2016-2075`: per position a synchronous
  `tensor_write(prev_s)`, markov launch, prefilter launch + **synchronous**
  `tensor_read(sel)`, then host dist build/draw. Greedy banks run the whole
  chain in one device launch (`pulsar_gpu_dspark_markov_chain_banks_model`).
- Change: device-side draw with a deterministic per-bank RNG (preserving the
  exact q/residual store), or at minimum batch/async the per-position
  write/prefilter reads.
- Impact: depth-4 sampled drafting pays 4 dependent round trips per bank group;
  est. 0.3–1 ms per redraft group. Medium-high; T>0 only. `v41-flash`: not fixed.

---

## C. Prefill speed

### C1. Expert recipe tier imbalance remains the largest prefill item
- Evidence: `pulsar_mxfp4_cutlass.cu:43-63,235-243,308-319`;
  `pulsar_cuda_moe.cu:524-542,1369-1465`; `7550d1da` ("the 9 MXFP4 layers cost
  ~984 ms against ~26 ms/layer for the ~30 IQ2 D2R layers … the single largest
  remaining item, and it is a quality decision rather than an engine one").
  Every MXFP4 layer pays an E4M3→CUTLASS gather, three grouped block-scaled
  GEMMs with experts padded to a 128-row SF atom, plus a separate f32 swiglu;
  mixed 40/43 layers pay both paths plus padded bridges.
- Change: re-solve the PrismaQuant map with a prefill-runtime term, or demote
  the 9 MXFP4 layers (and the MXFP4 halves of the 7 mixed) to IQ2_XXS_MMQ.
  Engine-side, the 128-row padding and tile are measured optima — do not tune.
- Impact: old decomposition implies up to ~20% of a 4k prefill; map's soft
  estimate ~4% (attribution blurred by `cutlass3x` serving dense GEMMs too).
  **Re-run a fresh nsys census before any re-solve.** High effort (quantizer +
  ppl/KL gates); quality decision. `v41-flash`: not fixed (same two tiers).

### C2. MMQ/mixed-40 SwiGLU `mid` is written f32 then re-encoded
- Evidence: fold kernels write `float *mid_out` (`pulsar_cuda_moe.cu:927-966`),
  launch at `:1053`, then `pulsar_gpu_mxfp8_act_cache_encode_f32(mid, …)` at
  `:1261`; mixed-40 twins at `:795,:847`; encoder `matmul.cu:1350-1362`. L165
  fixed only the small-batch GEMV lane.
- Today: per type-43 layer the fold writes ~49 KB f32/token, the encoder reads
  it back and writes ~12 KB E4M3, and the down MMQ consumes the cache.
- Change: emit E4M3+E8M0 from the fold epilogue into the activation slot, as
  `expert_gemv_gu_swiglu_kernel` already does (`pulsar_mxfp4_cutlass.cu:989-993,
  1030-1034`); skip the f32 store when no dump consumes it.
- Impact: ~40–50 ms per 4096 chunk (~1.2–1.5%) plus one launch/layer;
  byte-identical by construction. Medium effort. `v41-flash`: unchanged.

### C3. `hc_expand` re-reads each residual carrier 4×
- Evidence: `pulsar_cuda_hc_router.cu:108-171` — one thread per
  `(t,dst,d)` element re-loads all 4 `comb`+`residual` values per output
  element; 172 launches / 0.96 ms / 3.5% of prefill in the map.
- Change: one thread per `(t,d)` computing all 4 destinations in registers,
  loading residuals/`block_out` once and storing 4 outputs; the per-destination
  accumulation order is unchanged, so it is bit-exact.
- Impact: removes 3/4 of the residual/block reads (worst >1 GB L2 traffic per
  expand at 4k); est. 0.5–2%. Medium; needs A/B. `v41-flash`: same kernel.

### C4. Indexer scores are materialised f32 and re-read by chunked top-k
- Evidence: `indexer_mxfp4.cu:435-444` writes `scores[tok][comp]` f32;
  `indexer.cu:704-874` bitonic chunk + tree merge reads f32; 512-row spans
  (`gpu_decode.cpp:90-94`). At 128k, per ratio-4 layer per 4096 chunk:
  8 spans × 512 × 32768 × 4 B ≈ 0.54 GB written + read → ~22 GB/chunk over 21
  layers.
- Change: emit f16 scores (selection-only; fp16 retained ~100% of exact top-512
  in the design fidelity table) or fuse a streaming top-k into the scorer
  epilogue (bigger).
- Impact: ~1–3% of a 128k chunk, ≈0 at 4k. Low/medium. A selection change must
  be gated against the top-k oracle. `v41-flash`: score dtype unchanged.

### C5. Attn-out `low` is written f32 then re-quantized in a separate pass
- Evidence: `matmul.cu:2787-2803` (`mxfp8_quant_act_kernel` over
  `[n_tok×8192]`), call at `:2906`, split suffix at `:2857`; `low` allocated f32
  at `gpu_diag.cpp:1852`.
- Change: cuBLASLt cannot emit E4M3, so the honest version is a CUTLASS
  block-scaled "a" GEMM with an E4M3 epilogue (weights already ship
  MXFP8_LT); a cheaper partial step is f16 `low` (halves the encode read), but
  that is a fidelity change needing the reference gate.
- Impact: full fusion ≈0.8% of prefill (26 ms/chunk); f16 narrowing ≈half.
  Medium-high / low. `v41-flash`: unchanged.

### C6. Indexer scorer is on `mxf8f6f4` + nibble spread; `mxf4nvf4` was never built
- Evidence: `indexer_mxfp4.cu:104-132` (`m16n8k32 e4m3×e2m1`), `:147-150`
  (`idx_spread4`), `:176,:403-405` (spreads per MMA); `tests/idx_mma_issue_bench.cu`
  in `92018615` measured both instructions at the same issue rate, with
  `mxf4nvf4` at **251.2 TMAC/s** vs `mxf8f6f4`'s 124.8; the scorer manages
  ~6.5 TMAC/s. The commit explicitly **re-opens** the fork that rejected E2M1 Q,
  because that rejection's premise ("the kernel is not math-bound") was measured
  false and Q has been E2M1 since L090.4 (`375e8cfe`, `f9fbafe0`).
- Change: emit `mma ... mxf4nvf4 2X m16n8k64 ... e2m1.e2m1` (timed and correct
  in the probe) and copy packed nibbles verbatim; no transfer of the old
  `docs/indexer-mxfp4-scorer.md` "FORK DECISION" — update that section.
- Impact: removes the ~7 ALU/MMA tax and doubles MACs/instruction; realistic
  scorer multiple 2–4×. Whole indexer complex ~2% of prefill, larger share of
  decode. Medium-high; PTX guard + SF lane layout for 2X. `v41-flash`: not
  fixed (`73537a81` only retiles 32 heads). **Selection fidelity must be graded
  first (D2).**

### C7. Phase-2 softmax runs on 32 of 512 threads
- Evidence: `pulsar_cuda_attn_f16.cu:711-740`: `tid < AF16_HPB` (32) does a
  serial max and a serial `exp`/half-round/sum loop; 480 threads wait at the
  barrier; `sS` is materialised only to be re-read by that warp.
- Change: parallelise exp across the block (one `(head,row)` per thread;
  segmented `__shfl_xor` max seeded from `sM`), then have lane `h` sum the
  already-rounded halves in ascending `r` — preserving the exact normaliser and
  bit-exactness; `sS` disappears.
- Impact: removes a serial exp chain from a latency-bound kernel (attention is
  the largest class); needs A/B. Low-medium. `v41-flash`: not fixed.

### C8. KV tile staging uses 8-byte `cp.async` on a 16-divisible row
- Evidence: `pulsar_cuda_attn_f16.cu:606-612` (`AF16_ROWB/8u`, size 8);
  `AF16_ROWB` is 384 (`PULSAR_ATTN_PACK_ROWBYTES(512)`), the 8-byte chunking is
  a leftover from the retired 584 B E4M3 row.
- Change: 16-byte chunks (`AF16_ROWB/16`). Pure byte move; smem base, pitch and
  all `comp_base*384` offsets are 16-aligned.
- Impact: halves staging LSU instructions (48 → 24/row); cheap A/B, likely
  0–1%. `v41-flash`: still 8-byte.

### C9. Prefill indexer computes the full causal rectangle
- **STATUS 2026-09-11: NOT A FINDING.** The block-level early-out
  (`indexer_mxfp4.cu:238-246`, `tile_c >= (pos0 + tok_base + ngroup)/ratio`)
  already writes `-INF` and returns before any MMA, so the per-block grid
  already carves the causal rectangle: hidden column tiles cost only the
  masking stores. A per-token bound would only trim the diagonal-crossing
  blocks, and a block's 8 tokens span just ~2 compressed columns at ratio 4,
  so the waste is a couple of columns per token, not the ~half the review
  estimated. Do not spend here unless a profile shows otherwise.
- Evidence: `indexer_mxfp4.cu:238-246` (early-out), `:442` (per-column mask).
  `v41-flash`: uses the candidate-mask scheme.

### C10. Smaller structural items (each ~0.2–0.4%)
- **Indexed prefill arm misses the in-kernel E4M3 epilogue and folded inverse
  rope**: `pulsar_cuda_attention.cu:410-582` launcher has no gact/fold_rope
  parameters; `gpu_prefill.cpp:2153-2177` calls it, then `:2226-2257` runs a
  separate rope_tail and `:2262-2267` a `gact_emit_heads` read-back quantize.
  All 21 ratio-4 layers at ctx>2048 pay a full heads read+write (~400 MB per
  4096 chunk) plus 21 launches. Thread the existing `gact_data` view and
  `fold_rope` through (precedents: `attn_f16_combine_kernel`). `v41-flash`:
  same fallback.
- **Compressor emit is 5–6 launches per compressed row at decode**:
  `norm_kv.cu:1634-1666` + `gpu_decode.cpp:144-150`; the L138 census names the
  small-kernel tail ~22% of launches. One single-row emit kernel (pool softmax →
  RMS → rope → pack) or batching same-bank rows. `v41-flash`: rewritten by
  `fde1ff4a`.
- **Decode indexer scorer runs 4–16 blocks for a whole layer**:
  `indexer_mxfp4.cu:480-481` (`IDX_NTILE=128`, `IDX_TOKGROUP=8`); at one token,
  ctx 8k → 16 blocks of 256 threads on 48 SMs. Split-K or a smaller N-tile for
  `n_tokens <= IDX_TOKGROUP`; determinism must be pinned. `v41-flash`: no.
- **Top-k dispatch cliffs**: `indexer.cu:720-761` uses bitonic for
  1025..4096 and CUB only at 4096/8192; `:870-873` falls back to a single-thread
  insertion kernel for `top_k != 512`; `attention.cu:530-541` sorts per row per
  layer. Use CUB for all `n_comp <= 8192` (already padded/tested), refuse
  non-512 loudly. Preserve `topk_pack_key`. `v41-flash`: superseded by CSA2.

---

## D. Quality

### D1. Raw-window KV precision: the most-attended rows share the pool's FP4 budget
- Evidence: `pulsar_gpu.h:1006-1043` + `pulsar_cuda_internal.h:68-85`: one
  384 B NVFP4 row for ring, pool, drafter and chunk. v41 `0ffef78d` stores two
  reference-matching rows: WINDOW E4M3×E8M0/32 (528 B) and MAIN E2M1×E4M3/16
  (288 B), selected per row in `v41 pulsar_cuda_kvrows.cu:31-79`.
- Change: adopt the two-format staging (window E4M3) — the 128-row raw window
  is what every query attends and the only thing a ratio-0 layer sees; +144 B
  per window row is memory-trivial.
- Impact: quality on the highest-attention region, matching the reference cache
  layout. Needs ppl/KL gates and a payload/bank version bump. High effort, v41
  is a working template.

### D2. Indexer selection fidelity is still ungraded end-to-end
- Evidence: `docs/indexer-mxfp4-scorer.md:225,293-294`: the end-to-end logits
  KL "suite-v1 run … is still outstanding". The kernel-level table shows E4M3 Q
  retains ~98.7% of the exact top-512 at the boundary. `make gates` includes no
  indexer-selection gate, and `cuda-reference-gate` skips by default.
- Change: run the owed long-context KL with selection engaged
  (`PULSAR_REF_DIR`, `--check-reference` + `--kl-baseline`); only if drift
  appears evaluate per-128 E8M0 or a small f32 rescore of the top ~1024.
- Impact: this is the gate that makes C6/C4 safe to land. Measurement first.
  `v41-flash`: reworks index sharing but does not grade the V4 artifact.

### D3. `emit_low`/other producer fidelity items
Nothing else in this class is free: `batch_indexer_q` f32 staging
(`gpu_diag.cpp:1844`, `gpu_prefill.cpp:1491-1515`) is required by the packer
(`norm_kv.cu:1305`) and narrowing it would round before the FP4 crush; the
`moe_sum` 6× re-read (`pulsar_cuda_moe.cu:185-196,293-308`) is the price of the
fixed ascending-expert accumulation order. Both are recorded as informational,
not work.

---

## Measurement queue (in order)

Each item is an A/B with clocks locked (`nvidia-smi -lgc 2600`, `-rgc` after);
refactors must be byte-identical (prefill/decode baseline blobs), numerics
changes need ppl + `pulsar-eval` q1..q4 + KL where attention/selection moves.

1. Fresh nsys kernel census at 4k/8k prefill and 2k/8k decode (the perf map
   predates L200–L214); attribute the current top-10.
2. Instrument the plain-lane sampler (B2) on served traffic: count, per-token
   cost, scratch churn.
3. B1 down GEMV: standalone kernel bench at decode widths, then locked-clock
   single-stream A/B + verify-width A/B; re-anchor the decode byte baseline.
4. C2 fold-epilogue emit: byte gate first (should be identical), then prefill
   A/B.
5. C7 + C8 attention: unit gate for bit-exactness, then prefill/decode A/B.
6. C6 `mxf4nvf4`: in-kernel scorer A/B and selection-set comparison against the
   existing top-k oracle; gate on D2.
7. C3 hc_expand, C4 f16 scores, C5 f16 `low`: standalone then
   in-engine.
8. B4 depth re-sweep [2,8] and the B9 8-vs-12/16 bank re-measure on the current
   kernels.
9. C1: nsys census before and after a Prisma re-solve candidate; ppl/KL gates.
10. D1: two-format KV row A/B (KL + ppl + drafter acceptance + memory).

## Addendum — directly verified during this review

- The down-path asymmetry in B1 is real at `ds4_mmq_d2r.cu:1293` vs
  `:1420-1455` (checked by direct read).
- B2's cost is real at dev tip: `pulsar_sample_dist_build` still runs one
  `expf` per finite entry and the plain lane passes no scratch
  (`tokenizer.cpp:1222-1300`, `:1567-1575`).
- C6's re-open is explicit in `92018615`'s message ("This re-opens the fork
  settled in b2c39ad"), and the old `docs/indexer-mxfp4-scorer.md` "FORK
  DECISION" still reads as settled — update the doc.
- C7 (serial phase 2) and C8 (8-byte `cp.async`) were checked in the dev-tip
  source.
