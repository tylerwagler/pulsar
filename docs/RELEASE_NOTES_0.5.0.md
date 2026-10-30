# Pulsar 0.5.0 release notes

0.5.0-rc1 covers everything on `dev` since `v0.4.2` (2026-08-03): about 1,450
commits. Ledger rows (Lnnn) are named where a commit message gives one. The full
list is `git log v0.4.2..`.

0.4.2 served one model, DeepSeek V4 Flash, from GGUF. 0.5.0 is a
multi-family server that reads pulsar safetensors containers. Its production
model is Qwen3.8-Flash-Next.

## New

**Model families** (plug-ins under `src/engine/family_*.cpp` and
`src/families/`; L251, L272, L295):

- **Qwen3.8-Flash-Next** (L251, L266, L284, L288, L299): EXL3 weights from
  turboderp's packs. It drafts with the DFlash2 block drafter (its default) or the
  MTP module. Its PLE n-gram table is a row file streamed from disk, with read-ahead
  (L329b2). Image input (L268), and 1M context with `--rope-yarn 4` (L280).
- **DeepSeek V4.1 Flash** (L218, L242, L269), next to V4 Flash 0731 and
  Vision-Exp. It has the Engram conditional-memory table on disk and EXL3 routed
  experts (L245, L287).
- **Nemotron 3.5 Super** (L296, L298): EXL3 K4, with the LiLiCorr block drafter
  (NVFP4 W4A16) or the MTP module.
- **Inkling** (L294, L295): single session.
- **Nemotron 3.5 Embed-8B, Rerank-8B and Parse** (L321): `/v1/embeddings`,
  `/v1/rerank` and `/v1/parse`. Several of them run in one process (`-m`
  repeated), and they run on Turing (`sm_75`) GPUs.

**Models as containers** (L247, L279): `tools/container/build.py` builds the
safetensors container directly from Hugging Face checkpoints or EXL3 packs, from
one recipe schema (`pulsar.recipe.v1`). The engine converts nothing at load.

**Serving:**

- **Tensor-parallel serving** on a GB10 pair over RDMA (L102, L238–L243, L250,
  L260, L264, L269) for DeepSeek and Qwen. It covers expert and attention splits,
  the row lane and a mirrored disk KV cache. `/health` reports every rank.
- **Structured output** (L303): `response_format`, `text.format` and
  `output_config.format` (json_object, json_schema, regex, grammar,
  structural_tag) through vendored xgrammar. Forced and strict tool calls go
  through the same grammar, and speculation stays on (L309).
- **Image input** (L216, L261, L268, L273, L281) on the Chat, Responses and
  Messages surfaces, for DeepSeek Vision-Exp and Qwen. Images are inline base64
  PNG or JPEG. They may appear anywhere in a conversation, and disk KV chains
  carry them.
- OpenAI `logprobs` / `top_logprobs` on Chat Completions.
- `GET /metrics/stream`, an SSE change feed.
- `--capture-requests`.
- Claude Code conformance on `/v1/messages` (L252).
- Codex tool search and deferred tools rendered where they are referenced
  (L325, L326).

**Multi-session scheduling** (`src/server/server_sched.cpp`):

- One batched step for all decoders. The classic single-session lane is gone
  (L118).
- Queued prompts ride the speculative step (L260), with value-per-time round
  selection (L312).
- **Preemption and exact readmission** (L314).
- Resume from the deepest valid state, from the bank's own state, another bank's
  checkpoint or disk (L313, L318).
- Restore only when reading beats recomputing (L310).
- **Shared-prefix wait** (L329a).
- The bank auto-size cap went from 8 to 16 (L219).

**Disk KV cache v2** (L264, L323, L330): content-addressed segment chains on a
fixed grid, so copies and branches deduplicate. Restores run in slices between
rounds. A duplicate-prefix census is exported (L324).

**Speculative decoding:**

- Adaptive draft depth (L112) and priced chains (L288).
- A measured verify-cost curve with a throughput-ratio pick (L301), and
  per-drafter calibration on `/metrics`.
- Every draw, argmax, logprob and accept walk runs on the device (L302),
  bit-identical to the host sampler it replaced.
- New flags `--drafter`, `--spec-depth` and `--spec-tau`.

**Memory safety** (L319, L328, L332):

- One derived **memory floor** (an OS margin, the kernel's zone watermarks and
  one worker step's demand), checked before every bank, prefill chunk and persist.
- KV admission counts what it has promised.
- A **stall watchdog**.

## Behaviour changes

- **`cached_tokens` is correct at token seams** (L334). When a client re-spells
  text the model generated, the reported cached count is now the request's own
  tokens that the reused state serves. Before, it was a position on the live side.
  Strong unit types (`KvRows`, `KvBytes`, `LiveToks`, `PromptToks`; L333, L334)
  now keep those currencies apart in the code.
- **Memory-floor refusals.** A request that cannot get memory above the floor
  is answered with a retryable **503**, after the server has tried spilling idle
  banks and preempting. It is no longer run until the box stalls.
  - The 503 carries `Retry-After` and `X-Should-Retry` (L253).
  - Anthropic requests get Anthropic's error envelope.
  - The refusals are counted in `pulsar:admission_refusals_total{reason="mem_floor"}`
    and `pulsar:prefill_mem_refusals_total`.
- **`/health` can answer 503 `stalled`.** When the stall watchdog fires
  (`PULSAR_STALL_WATCHDOG_S`, default 120 s), `/health` reports
  `"status":"stalled"` and `"stalled": seconds`, and the log gets one full dump.
  `/health` also gained `stalled`, `slots.capacity` and a `tp` block.
- **A pool with no measured headroom refuses to start** (L319). Lower `--ctx` or
  `PULSAR_MSEQ_BANKS`.
- **A context past the model's trained positions is refused** (L284). The
  default `--ctx` is still 1M, so a Qwen container needs `--rope-yarn 4` or a
  `--ctx` of 262144 or less.
- `GET /v1/models/{id}` answers only for the served id; other ids are 404.
- **Thinking effort is per family.** No context size demotes an effort any more.
  DeepSeek's default is the encoder's (V4.1 high, 0731 low; L239).
- Graph capture refuses instead of falling back to eager (L306).
- A served tool call at T > 0 is decided greedy or sampled per position, as in
  plain decode (L296).
- Every KV buffer uses one NVFP4 row format (L111, L292). On-disk KV from 0.4.x
  is not read. The old `KVC` files are ignored and rewritten. The segment payload
  is v6.

## New flags and environment

- Flags:
  - `--drafter`, `--spec-depth`, `--spec-tau` and `--rope-yarn`.
  - `--capture-requests`.
  - Tensor parallelism: `--tp-role`, `--tp-peer`, `--tp-port`, `--tp-rank`,
    `--tp-nranks` and `--tp-peers`.
  - `-m` may be repeated for encoder and Parse models.
- Environment:
  - `PULSAR_STALL_WATCHDOG_S`, `PULSAR_SERVER_PREFIX_WAIT_MS` (default 60000)
    and `PULSAR_PLE_READ_AHEAD`.
  - `PULSAR_SPEC_DEPTH` and `PULSAR_SPEC_TAU`.
  - `PULSAR_TP_TIMEOUT_SEC` and `PULSAR_TP_RDMA_DEV`, `PULSAR_TP_RDMA_DEV2`,
    `PULSAR_TP_RDMA_GID_INDEX`.
  - Test hooks: `PULSAR_SERVER_SCHED`, `PULSAR_SERVER_KV_RESTORE_SLICE_MS` and
    `PULSAR_GRAMMAR_FAIL_AFTER`.

  See [SERVER.md](SERVER.md#environment-variables).

## Removed

- **GGUF**: the reader, the ggml type vocabulary and the GGUF tooling are gone.
  The tooling is archived at the tag `archive/gguf-tooling-2026-09-24`.
- **DeepSeek V4 Pro**: no longer a loadable shape.
- **Server flags**:
  - Engine selection: `--backend`, `--cuda`, `-t/--threads`, `--warm-weights`,
    `--quality`.
  - Drafter: `--dspark`, `--dspark-draft`, `--dspark-confidence` (replaced by
    `--drafter`, `--spec-depth` and `--spec-tau`).
  - Prefill and generation: `--prefill-chunk` (use `PULSAR_CUDA_PREFILL_CHUNK`),
    `-n/--tokens`.
  - Serving and networking: `--cors`, `--chdir`, `--served-model-id`,
    `--served-model-name`.
  - Tool replay: `--tool-memory-max-ids`, `--disable-exact-dsml-tool-replay`.
  - Disk KV policy: the `--kv-cache-*` policy flags (`-min-tokens`,
    `-cold-max-tokens`, `-continued-interval-tokens`,
    `-boundary-trim-tokens`, `-boundary-align-tokens`, `-reject-different-quant`).
- **Environment variables**: 133 `PULSAR_*` names went down to 44. The deleted
  ones are opt-outs and profiles with no caller, including:
  - `PULSAR_ATTN_PACK`, `PULSAR_IDX_FP4`, `PULSAR_RAW_F16`
  - `PULSAR_CUDA_DISABLE_*_FUSION`, `PULSAR_CUDA_NO_*`, `PULSAR_CUDA_WEIGHT_*`
  - `PULSAR_DSPARK_DISABLE`, `PULSAR_WARM_FORK`, `PULSAR_SERVER_SPILL_DIR`
  - `PULSAR_MIXED_CHUNK`, `PULSAR_MIXED_DEEP_GUARD_ROWS`, `PULSAR_THREADS`

  A deleted name does nothing if set, except `PULSAR_KV4` and `PULSAR_ATTN_MX`,
  which refuse startup.
- **The Anthropic `web_search` server tool** (L197). Server-tool entries are
  dropped at parse time.
- The CPU forward path, the classic decode lane, the f32 attention kernels and
  the vendored llama.cpp MMQ kernels that nothing reached (L066, L131, L166).
- CUTLASS is no longer a submodule. It is pinned in `cutlass.pin`, moved to
  v4.8.0 (L244).

## Known limits

- **Test coverage is uneven.** Only Qwen3.8-Flash-Next has good coverage
  (production and most live legs). DeepSeek Vision-Exp has broad battery gates,
  but no production use in this release. V4.1, Nemotron and Inkling are thinner.
  The encoder and Parse families are gated by hand. See the README's model table.
- **Nemotron 3.5 Super image input is not in this release.** Its vision front is
  not wired, and an image is refused by name. DeepSeek V4.1 and Inkling take no
  images either.
- **VMM (CUDA virtual memory management) for KV was explored and paused.** KV
  stays on managed, demand-paged memory.
- Qwen tensor parallelism has not been graded on a real pair in the battery. It
  is a declared gap in `tests/contracts.tsv`, as are Qwen whole-session payloads
  and mixed-prefill neutrality.
- Batched and speculative decode are not bit-reproducible across load. A greedy
  near tie can flip with what else is co-scheduled. Under sampling the
  distribution is preserved.
- DeepSeek V4.1 containers report the served id `deepseek-v4-pro`, a leftover
  of the old Flash/Pro naming.
- Without `--kv-disk-space-mb` the disk KV cache has no size limit, although
  `--help` still says 65536.
- The default context (1M) is refused by a container whose trained context is
  smaller. Pass `--ctx` or `--rope-yarn`.
