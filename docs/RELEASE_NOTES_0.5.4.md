# Pulsar 0.5.4 release notes

0.5.4 is 0.5.3 with the rest of the L380 performance pass and L381: short prompt chunks cost about half as much,
the dense decode kernel is 9-10 % faster at verify widths, the checkpoint a finished turn takes no longer blocks
or wastes work, and the server's readers see a decoder in the plain batch.  No model output changes: every new
kernel is byte-identical to the one it replaces, gate-checked.  No stored-KV format change.  The full list is
`git log v0.5.3..v0.5.4`.

## Faster

- **Short prompt chunks read the EXL3 dense weights in place** (L381).  A prefill chunk used to cost the same at 5
  rows as at 4096: every dense EXL3 linear decoded its whole weight to fp16 and ran the fixed cuBLASLt GEMM.  A
  prompt launch of up to 256 rows now decodes the trellis in registers and accumulates in the vendor GEMM's own
  order (a plain ascending chain of m16n8k16 MMAs, established on 26 shapes), so its bytes are the GEMM's.  Dense
  cost a chunk on the GB10: Qwen3.8-Flash-Next 66 -> 17 ms at 5 rows, 72 -> 34 at 256; Nemotron 3.5 Super 147 -> 24
  and 163 -> 68.  Served: Qwen, a tail of 30 tokens or fewer, prefill 146 -> 98 ms and first token 177 -> 127 ms;
  Nemotron short turns, prefill 491 -> 267 ms and first token 529 -> 305 ms.  Larger chunks and every other
  architecture run the code they ran.  **The first launch of each shape in a process runs both paths and compares
  every byte**; on a mismatch the server says so once, turns the arm off for the process and returns the GEMM's
  result -- a cuBLASLt change cannot split prompt bytes between small and large chunks.
- **The EXL3 dense decode kernel takes an 8-warp CTA, two an SM, on large grids** (L380).  On sm_12x, at up to 8
  rows, a launch of 6 or more waves runs `exl3_dense_kernel_8w` (two accumulator sets a warp, the 16-warp kernel's
  reduction order): 0 of 924 gate outputs differ from the 16-warp CTA's bytes.  Dense ms a round: Qwen 13.06 ->
  11.87 at 1 row, 13.32 -> 11.93 at 4, 12.65 -> 12.06 at 8; Nemotron 20.2 -> 19.2 at 1 row.  Decode floor on Qwen:
  every workload 1.014-1.034 of the baseline.
- **The terminal checkpoint yields, waits, and is skipped when it is not used** (L380).  The re-prefill of a
  finished turn's reply starts no sooner than 300 ms after the turn ended and yields at a chunk end to a queued
  request; before, a request arriving inside it waited for all of it (445 ms in one measured turn).  A conversation
  whose next request leaves the previous reply behind (a client that re-renders its history) stops arming it for
  replies of that kind: on Nemotron with such a client, 26 of 41 checkpoints skipped, each 480-706 ms of GPU work
  whose rows were discarded.  A side effect: a bank keeps the resume points near its prompt end, so a divergent
  next turn is served from memory where it used to load from disk.

## Fixed

- **A decoder in the plain batch reads its position and its history** (L380).  While a slot decoded in the plain
  batch on the installed bank, the server read its position as 0 and its history as empty.  `/metrics` slot
  positions now advance; the wait for a re-sent request after a disconnect, the eviction ranking, the duplicate
  census, the prefix-wait decision and the warm-match protection see the decoder (its history at batch entry plus
  the batch's commits).
- **DeepSeek: an idle installed bank kept no history while other banks stepped in the plain batch** (L380): its
  tool-result continuation found no owner and it could be handed out as empty.  The batch step now drops the
  installed bank's view only when it moved that bank, as the other families do.  Traced by reading; the release
  tier does not serve DeepSeek.

## New in the log

- `kv req:` gains `reply=1|0|-`: this request continued from the previous reply, left it behind, or had none.
- `terminal checkpoint: ... yielded to a queued request, the rest when idle` and `terminal checkpoint: bank B
  skipped: ...`.
- `EXL3 dense prompt rows K=... N=...: the in-place kernel wrote the fixed GEMM's bytes ...` once a shape, and the
  starred `... are NOT the fixed GEMM's on this device ...` line if a check ever fails.

## API

- `pulsar_bank_view`, `pulsar_session_bank_tokens_as`, `pulsar_session_bank_resume_at_as`; the resume-rule entries
  take a view.  `pulsar_session_bank_prefix_match` and `pulsar_session_bank_common_prefix` are removed (their two
  callers read through `server::slot_history`); `pulsar_tokens_common_prefix` is public.

## Tests and rules

- `exl3_dense_gate`: section 9 (both CTA forms forced on 1200 launches, every byte) and section 10 (the in-place
  prompt arm against the GEMM arm at every rate, shape class and M to 257; the faulted check); `--plan` walks
  section 10's fixtures on the host.  `lt-order-gate` (kernels tier): the vendor GEMM is still the chain the
  in-place kernel follows -- run `tests/lt_order_probe` after a cuBLASLt move or on a new architecture.
- `resume-deepest` takes its disk turns from a restart on the same store, not from the terminal checkpoint's timing.
- A decode-floor baseline for Nemotron 3.5 Super.
- Rules: a release run that is red because a test is wrong is not run again; fix the test, run that gate, record
  both shas (`docs/ENGINEERING-RULES.md` §10).

## Known limits

- The in-place prompt arm and the 8-warp CTA are sm_12x only: nothing was measured on sm_75.  The 256-row threshold
  is the last measured width that ships; in the bench the in-place kernel is ahead to 4096 rows.
- `session-contract-gate` on the DeepSeek fixture fails its 128-token cut in the dev tier's reused engine, at
  v0.5.3 as here (not in the release tier).
- Not run for this release: the DeepSeek three-stream scenario for the idle-bank fix, and the served scenarios for
  the history readers (a disconnect and re-send beside a plain batch; eviction with a dominated bank).
- A terminal checkpoint that does run over a reply crossing a 128-token grid point can still drop the last grid
  slot near the previous prompt end.
