# Review of branch `l219-review-fixes` — 2026-09-12

Scope: `dev..l219-review-fixes` (35 commits, 23 code, 33 files, +1597/−285; merge-base
`d6d3feb5`), read against `docs/engine-review-2026-09-11.md` (the plan and its
implementation-status block).  Static read of the tree plus `git show` per commit; no GPU.
The one confirmed bug was re-verified by direct read of the tree, not only from the
reviewer's report.

## Verdict

One correctness bug (B3, `4fa5e0b6`).  Two low findings (B2 `13501b89`, C7 `6a152fbd`).
Two status-block claims the tree does not support (B3 "gate-verified", A2's corruption test).
Everything else in the range is clean or note-level.

## 1. CONFIRMED — B3: a stale `spec_compact_rows` hijacks the greedy verify (`4fa5e0b6`)

`src/engine/imatrix.cpp:1168-1179`.  The new argmax branch of
`gpu_graph_decode_multiseq_batch` does not reset `g->spec_compact_rows`; the compact branch
resets `spec_argmax_rows` and the full-read branch resets both:

```cpp
if (head_all_rows && g->spec_argmax_armed) {
    ok = gpu_graph_spec_argmax_read(g, 0u, head_runs);      // spec_compact_rows left as-is
} else if (head_all_rows && g->spec_compact_armed) {
    g->spec_argmax_rows = 0;
    ok = gpu_graph_spec_compact_read(g, 0u, head_runs);
} else {
    g->spec_compact_rows = 0; g->spec_argmax_rows = 0; ...
```

`spec_round_end_block` (`src/engine/session_spec.cpp:1849`) tests the compact arm FIRST:
`if (g->spec_compact_rows >= row0 + r->n_batch && g->spec_compact_host)`.  The only writers
of `spec_compact_rows` are `gpu_graph_spec_compact_read` (`imatrix.cpp:1207/1221`), the
fused lane (`session_spec.cpp:1633`, not used by the server) and `gpu_diag.cpp:2079`;
nothing in the batched lane clears it between steps.

Failure: step N, every round in the sparse min-p contract → compact arm,
`spec_compact_rows = R`.  Step N+1, every round greedy (the same client's next request at
temperature 0, or the sampling client leaves and a greedy one remains) → `spec_argmax_armed`,
the argmax readback runs, but `spec_compact_rows` is still `R >= row0 + n_batch`, so every
bank's round end takes the compact arm: `row_tops[i]` are the PREVIOUS step's argmaxes and the
walk gets the previous step's candidate block.  Drafts are accepted and committed against the
wrong targets with no error; the fresh `spec_argmax_host` is never read.

Fix: `g->spec_compact_rows = 0;` in the argmax branch, mirroring the compact branch.  The
gate that would have caught it does not exist: no gate alternates a min-p step with a greedy
step in the batched lane.  Add that case before re-marking B3 landed.

## 2. PLAUSIBLE (low) — B2: the fast nucleus arm defeats the non-finite-mass refusal (`13501b89`)

`src/engine/tokenizer.cpp:1389`: `if (!sum_inert && (sum <= 0.0f || !isfinite(sum)))`.  The
guard's own comment says only a NaN temperature reaches it.  On the fast arm with a NaN
temperature: `floor_logit` is NaN so nothing is skipped, `p = expf(NaN)` is NaN, only
`max_id` survives, `sum = INFINITY`, the guard is bypassed, and `sample_nucleus_emit` emits
`probs[0] = NaN/NaN`.  Before this commit the same input refused.  Reachable only if
`json_number`/CLI parsing lets a NaN temperature through (not traced).

The membership analysis is sound for finite temperatures: an entry skipped by
`v < floor_logit` has `p < min_p·e^-1e-3`, ~1e-3 relative below `prefilter = min_p·(1-4e-6)`,
far outside float rounding, and any candidate with `p < min_p` is cut by `min_e` in the emit
anyway.  The sampled set and probabilities are unchanged.

## 3. Low — C7: shuffle-tree invariants asserted in a comment (`6a152fbd`)

`src/cuda/pulsar_cuda_attn_f16.cu:727-738`.  The phase-2 block assumes
`AF16_HPB * AF16_ROWS == AF16_THREADS` (one pass, `i = tid` unguarded) and `AF16_ROWS == 16`
(xor tree `8,4,2,1` inside a half-warp).  Both hold today (32·16 = 512) but only a comment
says so; the surrounding `for (i = tid; i < HPB*ROWS; i += THREADS)` loops tolerate a
mismatch, this block would index `sS[h]` out of bounds if `AF16_MT` changed.  Rule 4: make it
a `static_assert`.  Numerics are correct: `sS[h][r]` is `-INFINITY` for `r >= nr` and masked
rows (line 711), so including them in the max is exact; `09228ff9` correctly restored the
runtime `nr` bound on the `l` sum.

## 4. The status block vs the tree

- **B3 "landed, gate-verified"** — landed, and the gates that ran pass, but the commit says
  "Not measured" and no gate covers the min-p→greedy alternation.  Status should read
  "landed, unverified for alternating traffic" until the gate case exists.
- **A2 "a one-byte corruption case"** (commit `117541f7` and the status block) —
  `tests/session_payload_gate.cpp` contains no digest/corruption case.  Add it or strike the
  sentence.
- Everything else in the status block matches the commit map, including the C10a no-go and
  its clean revert (`git diff d1e47ad8^ bedc2744` is empty).

## 5. Notes (not defects)

- `117541f7` payload digest: coverage is complete (every byte through `payload_io`, header
  through tensors; digest appended raw and excluded from itself; `payload_bytes()` adds 8;
  chunk-independent fold).  A mismatch is detected only after the bytes are in the live bank,
  but both callers (`src/lib/pulsar_kvstore.cpp:1127`, `src/agent/kvstore_session.cpp:212`)
  call `pulsar_session_invalidate`, so corrupt rows are never served.
- `1cd0f145` fmemopen: correct; `load_snapshot` still opens with `snap->len`, and that reader
  is the only other `fmemopen`.
- `f59043f7` tokens_copy: message says "growth policy is unchanged" but `token_vec_push` seeds
  capacity at 64 (`tokenizer.cpp:70`) and the copy seeds at 16.  Harmless; message inaccurate.
- `a43ec0b6` E8M0 scan: geometry verified per consumer (38 `[E8M0][32×E4M3]`; 41 data then
  swizzled scale; 40 `cutlass_mxfp4_expert_layout`; 46 scales `[E][V/32]` then payload).  The
  scan runs twice on the merged artifact — `weights_bind` and `dspark_weights_bind`
  (`session.cpp:410`) both pass `&e->model` — ~2× (1/32 of the file) touched at cold load.
- `bc9d70bb` sampling clamps: the parser duplicates the engine's clamp constants (`1024`,
  `(0,1]`) from `pulsar_sample_dist_build` — a kept-in-sync copy in rule-4 terms.  The
  `static bool warned` is unsynchronised across request threads (worst case two warnings).
- `3af5220b` top-k CUB: `smem > 0 ? CUB : bitonic` is a two-implementation dispatch
  (pre-existing at 4096/8192, now 1025..4095 too); outputs are identical by the shared
  `topk_pack_key` total order, so not a numeric fallback, but a silent alternate arm.  The
  `if (n_comp > 1024u)` inside the `<= 4096` branch (`pulsar_cuda_indexer.cu:742`) is always
  true since `<= 2048` was handled above.
- `cf30211f` down GEMV: `PAIR=false` arm verified — `qu/du` computed but never
  dereferenced, gate accumulation order untouched, same decode-rows predicate and
  `K <= 4096` refusal as the pair launch; re-anchored in `01c1b9ca`.
- `8a6986ea` / `8dd124f8` fold emits: warp/lane-group alignment to 32-element blocks holds in
  all three kernels (`mid_dim % 32` guarded at the slot; scalar warp = 32 consecutive
  elements at a 32-aligned offset; v4 8-lane groups = 32 columns; scatter warps step by
  `blockDim` = 256).  Same helpers as `mxfp8_quant_act_kernel`; the slot's scale-slab memset
  still covers padding.  `pulsar_gpu_mxfp8_act_cache_encode_f32` retains callers in `tests/`.
- `1dc537ed`, `ef576858`, `eb9b5a8b`, `e5b2979f`, `b9745c7e`, `a8faa620`: clean.  cp.async
  16-byte chunks are legal (row pitch 384, `__align__(16)` dynamic smem); hc_expand's `NHC`
  stride is only used on the `n_hc == 4` dispatch; the async seed copies sit on
  `cudaStreamPerThread` with `--default-stream per-thread` and both sources are only rewritten
  by later same-stream kernels; `lane_logits` is `(POOL_CAP+1)×vocab` with `n_dec <= POOL_CAP`;
  `db_tensor` exits on a missing HF tensor; `POOL_AUTO_MAX = 16 = PULSAR_MSEQ_MAX`.

## 6. Clean commits

`a43ec0b6`, `bc9d70bb`, `b9745c7e`, `cf30211f`, `1dc537ed`, `ef576858`, `eb9b5a8b`,
`f59043f7`, `e5b2979f`, `6a152fbd`+`09228ff9` (with §3), `117541f7`, `8a6986ea`, `8dd124f8`,
`1f8c86b0`, `1cd0f145`, `01c1b9ca`, `13501b89` (with §2), `3af5220b`, `a8faa620`,
`d1e47ad8`/`bedc2744` (net zero).  Needs a fix: `4fa5e0b6`.

## 7. What the misses have in common

B3's "gate-verified" label, A2's phantom test, and (on the Vision-Exp track, see
`pulsar-notes/rows/L216.md`) the type-43 expert tensors built for a loader that has accepted
only type 44 since 2026-09-07: in each case a check that proves the work RAN was recorded as
a check that proves the work is RIGHT.  The mechanical fix, per item, is to name the gate
that exercises the failure mode before writing "verified", and to grep for the test before
citing it.

## 8. Next, in order

1. B3: the one-line reset + a batched-lane gate case that runs a min-p step followed by a
   greedy step.
2. A2: add the one-byte corruption case to `tests/session_payload_gate.cpp`, or strike the
   claim from the commit map and the status block.
3. C7: `static_assert` the two tile invariants.
4. B2: decide whether a NaN temperature is reachable at the parser; if it is, refuse it there.
