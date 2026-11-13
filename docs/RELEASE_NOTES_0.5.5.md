# pulsar 0.5.5 -- release notes

0.5.4 plus the L386 / L388 fixes home has served since 2026-11-11 (dev `8ab1e6ed`), the L295 family-module refactor,
L382's DeepSeek release lane, L335's post-rc1 fixes and L334's strong types throughout the engine.  **No model output
changes**: the refactors and the types are value-identical by construction, gate-checked.  **One stored-KV change:**
the disk KV identity now carries the arithmetic that computed the bytes (L386), so existing disk KV stores are wiped
once on first start.

## Fixed

- **Tool arguments are typed through a schema's unions and local references** (`anyOf` / `oneOf` / `$ref`, string
  enums / consts; L386): a STRING parameter declared that way (pydantic's `Optional[str]`, a `$ref` into `$defs`) was
  read as "not a string", so an argument whose text was `"123"`, `"true"`, `"null"` or `"[1]"` reached the client as a
  number, bool, null or array -- in the call's arguments and its streamed fragments.
- **DeepSeek's MXFP8 prefill arm takes one kernel per shape**, chosen at the canonical row count (L382): cuBLASLt picked
  a kernel per row count, so a 128-row resume could differ from the cold prefill.  Pinned at 4096 rows;
  `tests/mxfp8_lt_pin_bench.cu`.
- **The disk KV identity includes the compute capability, cuBLASLt and the CUDA runtime** (L386), so a store written by
  another toolchain is never read as this one's.
- **A context overflow on `/v1/messages` says Anthropic's "prompt is too long: N tokens > M maximum"** (L252 P5).
- **The server's `/v1/models` context field** went through printf varargs as a typed object (undefined behaviour); it
  is now its integer.  Every printf-style logger is format-checked, so a typed value through varargs is a compile
  error.
- **L335's post-rc1 findings:** the NULL-engine helpers refuse instead of crashing; the instance lock, the think
  helpers, `expect_source_set` and the attention-layout install refuse instead of exiting; a short `compress_ratios`
  is one counted refusal; Qwen's QSA scratch and DSpark's out-of-vocabulary confidence; the device compute capability
  per device.
- **Four HC launchers took their live row count from the output's allocation** (the shape of f3a617af / d0b2fdb5):
  they take it from the caller now and refuse an output too small.  Every caller passes exactly what it passed before.

## Changed (no behaviour change)

- **L295 F8: each model family is a module** -- its chat frontend, output parser, server tables, tokenizer hooks and
  engine state live in `src/families/<name>/` and register themselves; Qwen moved into `src/families/qwen/`.  A new
  family touches two or three places outside its module.
- **L334: strong types throughout the engine.**
  - Positions and counts (`TokPos` / `TokCount`), batched-step rows (`StepRow`), headed rows (`HeadRow`), compressed
    rows (`CompRow`) and the factor between them (`TokPerRow`).
  - Token ids (`TokId`, ids only compare) and bank ids (`BankId`).
  - GPU tensor bytes (`TensorBytes`), the session-cost ledger in `KvBytes`, rendered text bytes (`TextBytes`).
  - Each crossing between spaces is a named function.
  - The six hazards the survey found are closed as typed or asserted crossings: a verify's head row as a step row; the
    store's "0 = none"; "no limit"; the frontier's length; row distances as token counts; a payload's pool row.
  - The slot-equals-bank rule fae579f0 relied on is one checked lookup.
  - The `(bank, il)` / `(il, bank)` argument orders that compiled when swapped no longer do.
- **L382: DeepSeek V4 has a lane in the release tier** -- session contract, chunk neutrality and multiseq on the
  fixture.

## Known, not changed (reported in L334, for a later release)

- Pool-mode admission does not consult the session-cost ledger; `est_cost_bytes` holds two meanings (classic vs
  pool); a per-bank marginal is compared with MemAvailable as demand-pagable bytes; the per-bank charge divides by
  the bank count including the judge bank.
- A disk chain's live-side depth is ranked and priced as prompt tokens (four sites, each now a marked crossing); the
  prefill progress line mixes a request-side total with a live position.

## Tests

`make release` (`make gates`) at `3f623a66`, the sources this tag ships: **RELEASE PASS, 112 of 112**, on Sparky
(GB10) via gpuq #717, 67 minutes; build 0 warnings.  The VM check tier at every step of the series (23 of 23); the
warnings gate (153 test programs + the mutants, 0 warnings) and the doxygen gate clean.
