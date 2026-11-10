# Pulsar 0.5.3 release notes

0.5.3 is 0.5.2 with three server changes, each from evidence read off production: the server keeps a record of what
bank routing and the disk KV store do (L378), a request no longer evicts the bank that matched it and scratch banks
are evicted first (L379), and a short turn costs one prefill chunk fewer and no longer decodes on the plain lane by
accident (L380).  No kernel, numeric or output-byte change; no stored-KV format change.  The full list is
`git log v0.5.2..v0.5.3`.

## Faster

- **A turn that resumes its own conversation makes no cut at its message endpoint** (L380).  A prefill chunk costs
  about 90 ms on the GB10 whatever its row count (Qwen3.8-Flash-Next at ~37k context: 2 rows 89 ms, 20 rows 130 ms),
  and every request cut one at the prompt's message endpoint to leave a resume record for a client whose re-rendered
  history parts at the assistant header.  A request that resumed at or past its bank's previous prompt end, from the
  bank's own state, now waives the cut, unless a resume-grid point lies between the endpoint and the prompt's end
  (there the cut stays: the grid checkpoint below would not stand in for the record).  Same bytes; one chunk and one
  record fewer.  Measured on the plain server, arms alternated twice: prefill of a tail of 30 tokens or fewer
  240 -> 145 ms, first token 270 -> 176 ms; tails of 100-650 tokens 60-100 ms less.  Callers of the engine API, the
  CLI and the terminal checkpoint cut as before.
- **Lane prices: a decoder's depth is the same on either lane** (L380).  The lane pick read a decoder's depth from
  its bank's history, which the plain batch invalidates while it runs: a lone decoder read depth 0 on the plain lane
  and its real depth on the spec lane, so every lane switch moved the price key and dropped both lanes' measured
  prices, and a probe of the plain lane ran on at a key its prices were not for.  Short requests after a server start
  decoded on the plain lane (Qwen: ~33 tok/s against ~64).  The depth is now the decoder's on either lane, a probe
  ends when its decoder count or depth band really moves, and the lane line says what moved.

## Changed

- **A request does not evict the bank it matched; scratch banks go first** (L379).  One victim pick for the fresh path
  and the bind loop.  When the bank an eviction would free is the request's own best match, the request runs in place
  and nothing is evicted.  A scratch bank (last taken by a read-only request, or below the disk tier's minimum) is
  evicted, and spilled, after dominated and dead banks and before any conversation.

## New in the log

- **`kv req:`** -- one line a generation request at its end: conversation and shared-preamble keys (token hashes),
  the client's session and agent hints, the class, the bank, the route and resume pick with why the best bank had no
  resume point, tokens reused / prefilled, the eviction it caused, queue wait, restore, prefill and first-token times
  (L378), and `epcut=1|0|-`: the endpoint cut made, waived, or no endpoint ahead (L380).
- **`kv ledger:`** -- hourly and at shutdown: routes, picks, evictions by who-for-whom, disk bytes written / read /
  removed, restores, spills, preemptions, 503s by cause.  `/metrics` reads the same variables.  (L378)
- **`systemone req:`** -- one line a `/v1/systemone` request.  (L378)
- **Lane lines name a moved price key**: "depth band 3 -> 0", "decoders 1 -> 3"; a lane that holds through a move
  prints `decode lane N holds (...)`; `priced` reads `priced: clears the other lane` when it holds on the clearance.

`docs/SERVER.md`, "KV evidence", documents the fields.

## Upgrade notes

- Nothing to clear or convert.  Log readers that match the lane line's reason text exactly should allow the new
  prefix and the `holds` line.
- A side request that branches exactly at another bank's message endpoint adopts that bank's grid checkpoint (up to
  128 tokens lower) once the bank has waived its cut.

## Tests

- `test_l380_lane_probe_key`, `test_l380_lane_decoder_depth`, the waiver cases in `test_ckpt_records`; L378 / L379's
  line formats, token hash, conversation key, the store's read bit and CT 130's replayed eviction shapes.
- `make release` on the GB10 at the release's tree: pass.

## Known limits

- A user turn that follows an in-place side call on the same bank still makes the endpoint cut.
- Other readers of a plain-batch bank's position (`/metrics` slot positions, the live-continuation bookkeeping) see
  the same invalidated history the lane pick did; they are unchanged in this release.
