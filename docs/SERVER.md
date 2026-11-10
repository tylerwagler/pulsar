# pulsar-server reference

This page is the long form of the README's "Run", "API" and "How serving
behaves" sections. It describes `pulsar-server` at the tip of `dev`. Where it
names a constant or a default, the source file it comes from is given, so it can
be checked. If this page and the code disagree, the code is right; please fix
this page.

- [Command line](#command-line)
- [Environment variables](#environment-variables)
- [Endpoints](#endpoints)
- [Requests](#requests)
- [Banks and scheduling](#banks-and-scheduling)
- [Disk KV cache](#disk-kv-cache)
- [Memory floor and stall watchdog](#memory-floor-and-stall-watchdog)
- [Health and metrics](#health-and-metrics)
- [Tensor parallelism](#tensor-parallelism)
- [Client setup](#client-setup)

## Command line

Parsed in `parse_options` (`src/server/cli_main.cpp`). An unknown option prints
the usage and exits with status 2. `--help [topic]` prints the built-in help.
The topics are `runtime`, `steering`, `api`, `kv-cache`, `thinking` and `all`.

| Flag | Default | |
| --- | --- | --- |
| `-m, --model PATH` | `./model`, `./model.safetensors`, then the same names under `$PULSAR_MODEL_DIR` | The container: a shard directory or one file. It can be given up to 8 times, but only for encoder and Parse models; a chat model serves alone. |
| `-c, --ctx N` | 1048576 | Context per session (bank). A context past the model's trained positions is refused (`pulsar_session_create`). |
| `--rope-yarn F` | off | YaRN scaling at factor F (1..64) over a container whose rope is native. Qwen: `4` gives 1M positions. A container that declares YaRN accepts only its own factor. Refused by families without a rope and under TP. The default disk KV directory gets a `-rope-yarnF` suffix. |
| `--host HOST` | `127.0.0.1` | Bind address. There is no authentication, so exposing it is a choice. |
| `--port N` | 8000 | |
| `--kv-disk-dir DIR` | see [Disk KV cache](#disk-kv-cache) | `""` disables the cache. The last disk-KV flag on the line wins. |
| `--no-kv-disk` | | Disable the disk KV cache. |
| `--kv-disk-space-mb N` | unset: no budget | Disk budget in MiB. When the store is over budget it evicts leaves (see below). |
| `--kv-ram-mb N` | 25% of RAM, at most 16384 (discrete GPU); 1024 (unified memory) | The disk cache's pinned host pool in MiB: copy-out buffers, and on a discrete-GPU host the RAM tier restores read before disk (see below). |
| `--kv-pinned-max-mb N` | 2048 | Budget in MiB for pinned anchors (`X-Pulsar-KV: pin-anchor`, below): each anchor's restore point and its chain's segments. Past it, the least recently used anchor loses its pin (logged). |
| `--no-dspark` | | Run without the container's drafter, for every family. |
| `--drafter NAME` | the container's | Which drafter to load when the container holds more than one: `mtp`, `dflash2` (Qwen; its default) or `lilicorr` (Nemotron; a container with both MTP and LiLiCorr must name one). Naming a drafter the container does not hold is refused. |
| `--spec-depth N` | the drafter's | The draft depth a session starts at (DSpark 3, then its controller walks 2..5; MTP 4). Env form: `PULSAR_SPEC_DEPTH`. |
| `--spec-tau F\|off` | the drafter's | Draft stop threshold in [0, 1]: a chain ends at its first draft whose confidence is under F (DSpark 0.55, MTP 0.7). `0` or `off` disables the stop. Env form: `PULSAR_SPEC_TAU`. |
| `--trace FILE` | | Write rendered prompts, cache decisions, generated text and tool-parser events. |
| `--capture-requests FILE` | | Append every accepted request body as one JSON line (`{"ts", "path", "body"}`). |
| `--dir-steering-file FILE` | | Directional steering vectors (DeepSeek V4 only; see [dir-steering/README.md](../dir-steering/README.md)). |
| `--dir-steering-ffn F` / `--dir-steering-attn F` | 1 with a file / 0 | Steering scales, in -100..100. |
| `--tp-role leader\|worker`, `--tp-peer HOST`, `--tp-port N` | | Two-rank tensor parallelism. |
| `--tp-rank R`, `--tp-nranks N`, `--tp-peers H:P,...` | | n-rank form. See [Tensor parallelism](#tensor-parallelism). |

Not settable from the command line:

- The default maximum output per request is 393216 tokens (`default_tokens`). A
  request's own `max_tokens` (or its protocol's equivalent) overrides it, and
  generation always stops when the context is full.
- The prefill chunk is 4096 tokens for prompts longer than that
  (`PULSAR_PREFILL_CHUNK_DEFAULT`, `src/pulsar.h`). Change it with
  `PULSAR_CUDA_PREFILL_CHUNK`.

Only one pulsar process runs at a time on a machine. A second one refuses to
start on the instance lock, `PULSAR_LOCK_FILE`, which defaults to
`/tmp/ds4.lock`; the name is kept for compatibility with older binaries.

## Environment variables

These are all 44 `PULSAR_*` names the code reads. Regenerate the list with:

    grep -rhoE '"PULSAR_[A-Z0-9_]+"' src/ | tr -d '"' | sort -u

A name not on that list does nothing. Many older switches were deleted in this
release; see the [release notes](RELEASE_NOTES_0.5.0.md#removed).

**Operation**

| Variable | Default | |
| --- | --- | --- |
| `PULSAR_MODEL_DIR` | | Where the default model pointer (`model`, `model.safetensors`) is looked up after the working directory. |
| `PULSAR_LOCK_FILE` | `/tmp/ds4.lock` | The instance lock. |
| `PULSAR_MSEQ_BANKS` | auto | Pin the bank pool to N (1..16; larger values are clamped) and skip auto-sizing. |
| `PULSAR_OVERCOMMIT` | on | `0`/`off`/`false`: charge every bank its full context at admission, instead of only its eager floor. A large `--ctx` then gets fewer banks. |
| `PULSAR_MIXED_BATCH` | on | `0`/`off`: disable the fused mixed lane, in which queued prompt chunks ride decode steps (families that support it). |
| `PULSAR_SERVER_PREFIX_WAIT_MS` | 60000 | How long a queued request waits for another prompt's prefill of a shared prefix before computing it itself. `0` turns it off; a malformed value refuses startup. |
| `PULSAR_STALL_WATCHDOG_S` | 120 | Stall watchdog threshold in seconds (0 = off, at most 86400); a malformed value refuses startup. |
| `PULSAR_SPEC_DEPTH`, `PULSAR_SPEC_TAU` | | Env forms of `--spec-depth` / `--spec-tau`; the flags win. An unparsable value refuses the load; `PULSAR_SPEC_TAU=0` disables the stop, as `--spec-tau 0` does. |
| `PULSAR_CUDA_PREFILL_CHUNK` | 4096 | Prefill chunk in tokens; `0` prefills a prompt as one batch. Changing it changes the KV checkpoint grid, so compare runs at the same value. |
| `PULSAR_PLE_READ_AHEAD` | 1 | Qwen: read the next prompt chunk's PLE rows while this chunk computes. `0` turns it off (no byte changes); any other value refuses the load. |
| `PULSAR_TP_TIMEOUT_SEC` | 300 | Tensor-parallel connect and exchange timeout. |
| `PULSAR_TP_RDMA_DEV`, `PULSAR_TP_RDMA_DEV2`, `PULSAR_TP_RDMA_GID_INDEX` | auto | RDMA device(s) and GID index for the TP transport. |

**Measurement and test hooks.** These are not for production; each is announced
in the log when set.

| Variable | |
| --- | --- |
| `PULSAR_SERVER_SCHED` | `out=,fair=,prompts=,age=,adapt=`: the fused-lane round-selection prices (L312), for sweeps. A malformed value refuses. |
| `PULSAR_SERVER_KV_BUDGET_OVERRIDE` | Bytes: lower the guard's KV budget so spills fire at small fills. |
| `PULSAR_SERVER_KV_RESTORE_SLICE_MS` | Milliseconds (1..600000) per disk-restore slice. The default is one prefill chunk's priced time. |
| `PULSAR_GRAMMAR_FAIL_AFTER` | Inject a grammar-matcher failure after N tokens of a constrained region. |
| `PULSAR_SERVER_TEST_ROUTES` | `1`: answer the test routes -- `POST /test/guard-budget` (body: bytes) sets the guard's KV budget at run time, `0` restores it (L357).  Said at startup; never set in production. |
| `PULSAR_SCRATCH_POISON` | Poison disposable scratch buffers (`scratch-poison-gate`). |
| `PULSAR_BLOCK_UNPRICED` | Block drafter: turn off the priced chain cut (a sweep's control arm). Said once when set. |
| `PULSAR_LANE_PIN` | `spec`: hold every decode sweep the spec lane can carry on the spec lane -- no plain-lane measuring at a fresh bank, no re-measure (L368). A measurement switch: the lane changes which kernel computes a token. Said at startup; any other value refuses. |
| `PULSAR_SPEC_CARRY` | `1`: the draft depth carries across requests, as before L368 (it restarts every request by default). A measurement switch; said once. |
| `PULSAR_QWEN_MTP_DRAFT_EXTRA` | Qwen MTP draft-vocabulary extra width (must agree with the container). |
| `PULSAR_TP_TEST_TINY_BUFFERS` | TP transport tests: tiny exchange buffers. |
| `PULSAR_TEST_QWEN_TOKENIZER` | Server unit tests: the Qwen tokenizer to load. |

**Diagnostics and dumps:** `PULSAR_ROUTE_DEBUG` (bank routing decisions),
`PULSAR_DSPARK_STATS`, `PULSAR_DSPARK_DUMP`, `PULSAR_DSPARK_DUMP_LEAN`,
`PULSAR_DSPARK_DUMP_STEPS`, `PULSAR_DSPARK_PREFILL_DUMP`, `PULSAR_BLOCK_DUMP`,
`PULSAR_BLOCK_DUMP_PASSES`, `PULSAR_BLOCK_TIMING`, `PULSAR_DISTILL_DUMP`,
`PULSAR_PARSE_DUMP`, `PULSAR_MOE_SPILL_MID`, `PULSAR_CUDA_GRAPH_DUMP_LAYER`,
`PULSAR_CUDA_GRAPH_DUMP_NAME`, `PULSAR_CUDA_GRAPH_DUMP_POS`,
`PULSAR_CUDA_GRAPH_DUMP_PREFIX`, `PULSAR_VERIFY_TENSORS` (resolve and hash the
tensors under a name prefix at load) and `PULSAR_VERIFY_RANGES` (hash and locate
every device range).

**Refused when set:** `PULSAR_KV4` and `PULSAR_ATTN_MX`. Both name removed
formats. Every KV buffer is the one NVFP4 row now.

## Endpoints

Chat models (`src/server/http_server.cpp`):

| Endpoint | |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/messages` | Anthropic Messages. Errors use Anthropic's error envelope. |
| `POST /v1/messages/count_tokens` | `{"input_tokens": n}`, rendered and tokenized exactly as `/v1/messages` would. It creates no job and does not reject over-long prompts. |
| `POST /v1/completions` | Raw continuation of `prompt`: no chat template, no thinking, no logprobs (refused). |
| `POST /v1/systemone` | The container's runtime-adapter judge (Gemma 4 + JEV): System 1 decisions, one prefill a question on a reserved bank. See [System 1 decisions](#system-1-decisions-v1systemone). |
| `GET /v1/models` | The loaded model only (its served id). |
| `GET /v1/models/{id}` | 200 for the served id; any other id is 404. |
| `GET /health` | See [Health and metrics](#health-and-metrics). |
| `GET /healthz`, `GET /ping` | `{"status":"ok"}`, always 200 while the process runs. |
| `GET /version` | `{version, engine, cuda_arch, model, model_name, context}` |
| `GET /` | `{service, version, status, endpoints[]}` |
| `GET /metrics` | Prometheus text. |
| `GET /metrics/stream` | SSE. Each event carries only a generation number; fetch `/metrics` when it moves. At most 8 subscribers. |
| `OPTIONS *` | 204 |

The encoder and Parse families serve a different surface
(`src/server/encode_server.cpp`): `GET /health`, `/healthz`, `/ping`,
`/v1/models`, plus:

- `POST /v1/embeddings`: OpenAI's shape. `input` is a string or an array (at most
  2048 items). `input_type` is `query` or `passage` (default `passage`; the model's
  template writes the prefix, so do not prefix the text yourself).
  `encoding_format` is `float` or `base64`; `truncate` is `NONE` or `END`.
- `POST /v1/rerank` (also `/rerank`): the Cohere / Jina shape. Fields: `query`,
  `documents`, `top_n`, `return_documents`, `raw_scores`, `truncate`. Scores are
  the sigmoid of the head's logit; with `raw_scores` they are the logit itself.
- `POST /v1/parse`: one page image (`image`, an inline base64 PNG or JPEG
  `data:` URL), optional `prompt`, `max_tokens` and `return_ids`. It returns the
  model's markup text plus timings.

With several models of one kind in a process, the request's `"model"` picks one.
An input over the model's cap is refused (400) unless `truncate` is `END`.

Connection limits (`src/server/pulsar_server_internal.h`): at most 64 concurrent
client connections, after which a client gets an immediate 503. A request must
arrive in full within 30 seconds. Generation time is not limited.

Retryable failures (503) carry `Retry-After` and `X-Should-Retry: true`. A full
connection table gives retry 1 s; shutdown and memory refusals give retry 10 s.

## Requests

### Common fields

Every chat surface accepts `temperature`, `top_p`, `top_k`, `min_p` and `seed`,
through one parser (`parse_sampling_key`, `src/server/api_parse.cpp`):

- Non-finite values are refused.
- `top_p` outside (0, 1] becomes 1.
- `min_p` outside [0, 1] turns the filter off.
- `top_k` is clamped to 0..1024.

The defaults apply only to fields the request leaves out: temperature 1, top_p 1,
min_p 0.05, top_k 0 (`src/pulsar.h`). Values a request sends are used as sent,
including `temperature: 0` (greedy).

Streaming (`stream: true`) is SSE on all three chat surfaces. Reasoning streams
on the protocol's reasoning channel and never as visible content, even when
generation stops inside an unterminated think block. Tool calls stream as the
parser recognises them: the header first, then argument deltas.

Endpoint specifics:

- **Chat Completions:**
  - `messages`, `tools`, `tool_choice` (`none` / `auto` / `required` / a named
    function), `max_tokens` / `max_completion_tokens`, `stop`, `stream`,
    `stream_options.include_usage`, `response_format`.
  - `logprobs` / `top_logprobs` (0..20): a logprobs request decodes without
    speculation, so the reported distribution is the target model's.
- **Responses:**
  - `input`, `instructions`, `tools` (function, custom and tool-search tools,
    with `defer_loading`), `tool_choice`, `max_output_tokens`, `reasoning`,
    `text.format`, `stream`.
  - A non-null `previous_response_id` or `conversation` is refused: there is no
    durable response store, so send the full input. Live continuations are
    matched by transcript and call ids.
- **Messages:**
  - `system`, `messages`, `tools`, `tool_choice`, `max_tokens`,
    `stop_sequences`, `thinking`, `output_config` (`effort`, `format`), `stream`.
  - Anthropic **server** tool entries (`web_search_*`) are dropped at parse time.
    The server runs no server tools, and any other non-custom tool type is
    refused with Anthropic's wording.

The `model` field selects nothing. A response reports the loaded model's served
id. The one exception is that on DeepSeek, `deepseek-chat` defaults thinking off.

Usage blocks report prompt, completion and cached tokens: `cached_tokens` on the
OpenAI surfaces, and `cache_read_input_tokens` / `cache_creation_input_tokens` on
the Anthropic one. `cached_tokens` counts the request's own prompt tokens that a
reused state served. Since L334 this is counted correctly even when the client
re-spells text the model generated.

### Thinking and effort

Thinking is on unless the request turns it off with any of:

- `thinking: {"type": "disabled"}`
- `think: false`
- `enable_thinking: false` (or in `chat_template_kwargs`)
- effort `none`
- on DeepSeek, `model: "deepseek-chat"`

An effort can be sent as `reasoning_effort`, `reasoning.effort`,
`output_config.effort` or `chat_template_kwargs.reasoning_effort`. Its value is
one of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or an integer
1..100. What each value means is the loaded family's map
(`src/server/chat_family.cpp`, `family_*.cpp`):

| Family | minimal / low / medium | high | xhigh / max | integer | nothing sent |
| --- | --- | --- | --- | --- | --- |
| DeepSeek | low | high | max | V4.1: the value itself; V4: nearest preset (low 50, high 75, max 100) | V4.1 high, 0731 low |
| Qwen | low / low / medium | xhigh | xhigh | nearest preset | xhigh |
| Nemotron | medium | high | high | nearest preset | the template's default (high) |
| Inkling | the renderer's table (`family_inkling.cpp`) | | | | |

### Tool calls

Each family renders tool schemas in its own template and parses the model's
calls back into the protocol's tool-call objects. DeepSeek uses DSML; Qwen,
Nemotron and Inkling use their own templates' formats.

- **Exact replay.** Every call gets an unguessable id. The server keeps
  `id -> the exact text the model sampled` in memory (up to 100000 ids) and in the
  disk KV segments. When a client sends the call back as JSON, the prompt uses the
  sampled bytes rather than a re-rendering, so the next turn still matches the
  live KV. A re-rendering is the fallback.
- **Greedy syntax, sampled payload.** While the model writes a call's structure
  (tags, names, JSON punctuation), sampling is greedy. Argument payloads (string
  values, file contents) use the request's sampling. In a speculative round this
  is decided per position, the same as in plain decode.
- **Forced calls.** `tool_choice` `required` or a named tool is enforced by a
  grammar over the name and the arguments, in every lane.

### Structured output

`response_format` (Chat Completions), `text.format` (Responses) and
`output_config.format` (Messages) take `type`:

- `text`
- `json_object`
- `json_schema` (with `schema`)
- `regex` (with `pattern`)
- `grammar` (with `grammar`)
- `structural_tag` (with `format`)

The answer is decoded under an xgrammar matcher (`src/server/constrain.cpp`,
vendored xgrammar). Strict and forced tool calls use the same mechanism.
Reasoning stays unconstrained until the think block closes. Speculation stays on:
drafts are cut at the first token the grammar refuses, so constrained output
equals plain constrained decoding. Compiled grammars are cached (256 MiB). The
first constrained request pays the vocabulary compile.

### Images

Images go in the protocol's own image part:

- Chat: `image_url`
- Responses: `input_image`
- Messages: `image` with `source.type: base64`

Images must be inline: a base64 `data:` URL or block of `image/png` or
`image/jpeg`. http(s) URLs are refused. Images may appear anywhere in a
conversation, including inside tool results, and disk KV chains carry them
(matched by content).

Served on: Qwen3.8-Flash-Next and DeepSeek Vision-Exp containers. Refused by
name: Nemotron 3.5 Super (its vision front is not wired in this release),
DeepSeek V4.1 and Inkling.

## Banks and scheduling

A **bank** is one session's GPU state: KV, recurrent state and drafter state.
The pool is sized at startup (`src/server/cli_main.cpp`):

- The budget is the unified-memory headroom left after the resident weights, a
  process reserve and the memory floor.
- With overcommit (the default), each bank is charged only its *eager floor*: the
  raw window, state lanes and drafter. The context-scaled KV is demand-paged and
  becomes physical on touch.
- Auto-sizing picks the largest N up to 16 that fits; `PULSAR_MSEQ_BANKS` pins N.
- The startup log prints a fit table (banks per context size) and the chosen
  pool.
- A pool with no measured headroom left after warmup refuses to start; lower
  `--ctx` or `PULSAR_MSEQ_BANKS`.

Scheduling (`src/server/server_sched.cpp`, `server_jobs.cpp`):

- **Routing.** A request goes to the bank whose history shares the longest
  token prefix with it. A prefix no longer than the template header counts as no
  match. Side calls stay on their conversation's bank.
- **Making room (L379).** A request that wants a fresh bank when none is empty
  evicts one idle bank: a dominated or dead one first, then a *scratch* one, then
  the least recently used conversation. A bank is scratch when its history was
  written by a read-only request (`X-Pulsar-KV: read-only`, the gate's judge) or
  is shorter than the disk tier's minimum (512 tokens); a request without the
  token that takes the bank makes it a conversation's again. A queued request's
  usable warm match is spared, unless it is scratch; a queued live tool
  continuation's bank is never taken. A request never evicts the bank it matched
  best: when that is the bank an eviction would free, it runs in place on it
  (`route: ... -> in place (the only bank to free is the matched one)`). The
  guard's spill order ranks banks the same way.
- **Resume.** A request resumes from the deepest valid state, whichever is
  deepest: its own bank, another bank's published checkpoint (adopt), or a disk
  chain. A disk chain is loaded only when reading it beats recomputing.
- **Batched decode.** All decoding banks share one batched step. Queued
  prompts' chunks ride the speculative lane's step (fused prefill). Each round's
  work is chosen by a value-per-time price (L312).
- **Shared-prefix wait.** Requests arriving together on a new shared prefix
  wait (up to `PULSAR_SERVER_PREFIX_WAIT_MS`) for the first one's prefill to reach
  a checkpoint, then adopt it.
- **Preemption.** When a prompt does not fit, a running decoder leaves through a
  terminal checkpoint and is readmitted later. It continues exactly; the output is
  unchanged.
- **Spill.** Idle banks spill to the disk KV cache, and their memory is
  released.
- **Refusal.** A request that cannot be placed is refused with a retryable 503.
  It is not run into memory exhaustion.

Batched decode is not bit-reproducible across load. A co-batched step and a
single-sequence step can differ in the last float bit, which can flip a greedy
near tie. Under sampling it is distribution-preserving.

Speculative decoding: DeepSeek drafts with DSpark, Qwen with DFlash2 (or MTP),
Nemotron with LiLiCorr (or MTP). The verify accepts a draft only if it is the
token the plain sampler draws at that position with that request's seeded key,
so speculation changes speed, not the output distribution. Each drafter's
confidence calibration and the verify cost curve are on `/metrics`.

## Disk KV cache

On by default. Directory: `--kv-disk-dir`, else
`$XDG_CACHE_HOME/pulsar/kv-<model>` (an absolute `XDG_CACHE_HOME` only), else
`~/.cache/pulsar/kv-<model>`. `<model>` is the container path's resolved
basename, plus `-rope-yarnF` under `--rope-yarn`. If no directory can be resolved
or created, the server logs it once and runs without the cache.

Format (`src/lib/pulsar_segstore.h`, `src/lib/pulsar_kvchain.h`):

- **A content-addressed segment store.** A segment is one stretch of a
  conversation: its rendered text, the engine's KV payload for those tokens, an
  optional state at its end, and an optional trailer (the tool-call replay map).
- **Keys.** A segment's key is `sha1(parent_key || text)`, so a key names the
  whole text from position 0, and conversations that share a prefix share its
  segments. Files are `<key>.seg` (rows) and `<key>.rp` (the state at its end, a
  restore point). A persist copies each file out of the GPU into the pinned host
  pool (`--kv-ram-mb`) and a writer thread writes it behind with O_DIRECT (tmp +
  rename; no fsync: sizes and digests reject a torn file), so no page cache is
  left behind.
- **RAM tier.** On a discrete-GPU host a written file stays in the pool as a RAM
  copy, and a restore of it reads no disk; RAM copies are evicted
  least-recently-used (dead-end restore points first, anchors and branch points
  last) and stay on disk. On unified memory (GB10) the tier is off: host RAM is
  the GPU's, so the pool is only the copy-out ring.
- **The grid.** Chains are cut on a fixed grid (the prefill chunk) plus one tail
  segment. Copies and branches of a conversation therefore deduplicate.
- **Lookup.** A lookup returns the deepest chain whose text is a byte prefix of
  the request and whose last segment carries a state.
- **Identity.** A store records the model, its routed-expert format and its rope.
  Segments of another identity or payload version (segment v8, restore point
  v2, store v4) are refused and rewritten, never converted.
- **Restore.** A restore loads in slices between decode rounds, so other
  sessions keep decoding (L330). The chain is pinned against eviction while it
  loads. When the bank holds a history the request could resume from, every
  part of the chain is read and checked first, and the bank is written only
  after all of it passed.
- **Failures.** Only a content fault drops anything: a file that is missing,
  unreadable, of the wrong size or header, or whose digest is wrong. A bad
  segment drops itself and every segment below it. A bad restore point drops
  only itself; its rows stay. Any other failure (no RAM room, no checkpoint
  slot, the device, a TP worker's copy) drops nothing. A request whose restore
  failed resumes from its bank's own history (when the restore had not written
  it yet) or from another bank's state, never from 0 when either serves it.
- **Eviction.** With a budget, the store evicts leaves only: first those
  dominated by a longer sibling, then the least recently used. Without
  `--kv-disk-space-mb` there is no budget. Pinned anchors are never evicted
  while anything else can go.
  A chain that goes on keeps the restore points at its older ends, since an
  edited or regenerated turn branches from one of them; restore points that
  are not flagged (anchor, periodic) go first, the oldest first.

### The `X-Pulsar-KV` request header

Any generation request (`/v1/chat/completions`, `/v1/completions`,
`/v1/responses`, `/v1/messages`) may carry `X-Pulsar-KV`: comma-separated
tokens, case-insensitive, spaces around them ignored. Several header lines join.
An unknown token is refused with a 400 that names it.

| Token | Effect |
| --- | --- |
| `read-only` | The request writes nothing to the disk KV cache: no rows, no restore points, no anchor (unless `pin-anchor` is also given). Its bank's history then writes nothing when it is later evicted, spilled, abandoned or saved at shutdown, until a request without `read-only` takes the bank. Restores still read the cache. |
| `pin-anchor` | The request's anchor (the chat preamble up to the last user message before the first assistant turn, aligned down to the resume grid) is written on its first sighting if the store holds no restore point there, and pinned. A restore point already stored there is pinned instead. A pinned anchor and its chain are never evicted from the RAM tier or the disk. A pin is lost only when the store needs room and nothing else can go, or past `--kv-pinned-max-mb`. |

A judge that sends the same preamble on every call uses
`X-Pulsar-KV: read-only, pin-anchor`: each call restores the pinned preamble and
prefills only its own tail, and writes nothing else. The anchor is written only
by a request whose prefill starts at token 0, so a judge whose bank still holds
the preamble pins it the first time it misses. `/metrics` counts the header's
requests (`pulsar:kv_header_requests_total`), the persists it suppressed
(`pulsar:kv_persist_read_only_total`), the pinned anchors and their bytes
(`pulsar:kv_pinned_anchors`, `pulsar:kv_pinned_bytes`), lost pins
(`pulsar:kv_pins_released_total`) and failed restores by fault
(`pulsar:kv_restore_failures_total`).

The store itself is on `/metrics` too: whether it is open
(`pulsar:kv_disk_enabled`), the bytes it holds against its budget
(`pulsar:kv_disk_used_bytes`, `pulsar:kv_disk_budget_bytes`, 0 = unbounded),
its segments (`pulsar:kv_disk_segments`), the RAM tier's bytes and the host pool
it lives in (`pulsar:kv_disk_ram_bytes`, `pulsar:kv_disk_ram_pool_bytes`), and
the free space and size of the filesystem under `--kv-disk-dir`
(`pulsar:kv_disk_fs_free_bytes`, `pulsar:kv_disk_fs_size_bytes`), since the
budget is not the disk. The counters say what wrote and removed it: segments and
restore points persisted by trigger
(`pulsar:kv_disk_segments_persisted_total{trigger=...}`,
`pulsar:kv_disk_restore_points_persisted_total{trigger=...}`, the trigger words
of the `persisted reason=` log lines) and what the store removed
(`pulsar:kv_disk_segments_removed_total`,
`pulsar:kv_disk_restore_points_removed_total`). The gauges are taken when the
store changes (open, a persist, a write committed, a restore's end), not on
every scrape.

The cache is disposable: stop the server and delete the directory to reset it.
Segment files contain the conversation text in clear.

## KV evidence: the `kv req:` and `kv ledger:` lines

The log is the durable record of what routing and the disk tier do (L378): `/metrics` resets with the process, the
log does not. Three line kinds, all `key=value` separated by one blank, no blank inside a value, `-` for a value that
does not exist. They are printed in the KV log class, which is always on (the log has no class filter; a class only
picks the colour on a terminal). Nothing in them decides anything.

### One line a generation request: `pulsar-server: kv req:`

Written once, when the request's job ends: at the end of its turn, at its failure, or when it leaves the queue
unserved. A request that never bound to a bank has `bank=-` and `-` in every field the worker never learned.

| Field | Meaning |
|---|---|
| `id` | the job's number (from 1, per process) |
| `ep` | the endpoint: `chat`, `completions`, `responses`, `messages` |
| `class` | the `x-claude-code-request-class` hint: `main`, `subagent`, `workflow`, `compaction`, `auxiliary`, `other` |
| `ro` | 1 when the request carried `X-Pulsar-KV: read-only` |
| `sess`, `agent`, `parent` | the `x-claude-code-session-id`, `-agent-id` and `-parent-agent-id` hints as received, cut to 16 characters |
| `pre` | hex16: FNV-1a 64 of the prompt's token ids from 0 to the preamble's end, the first user marker (the shared system prompt and tools). `-` when the prompt has no user marker before its first assistant marker |
| `conv` | hex16: the same hash from 0 to the first assistant marker: the preamble and the user turns before the first reply. The same on every turn of one conversation, different for two conversations behind one preamble. The whole prompt when it has no assistant marker (a completion) |
| `prompt` | the prompt's tokens |
| `bank` | the bank it ran on |
| `route` | the router's verdict: `in-place` (continues on its best bank), `fresh` (an empty, dead, newly provisioned or just evicted bank), `clobber` (its best bank over another conversation's history, no fresh bank to be had), `owner` (the bank that owns its live tool call ids), `matched` (its best bank, in place: no fresh bank could be had and the only bank an eviction would free was that one, L379), `none` (never routed) |
| `best`, `common`, `score`, `anchor`, `frontier`, `prefilled` | the numbers of the `route:` line: the best-scoring free bank, the tokens shared with it, where its sync would resume, its route anchor, its frontier and its prefill frontier |
| `noresume` | only when `score=0` and `common>0`: why the best bank has no resume point inside the match. `ckpt-above`: every checkpoint it holds lies above what it may resume. `no-ckpt`: it holds no checkpoint at all. `not-continued`: the request extends its whole history, but the bank does not continue it (decode rows past its prefill, an end off the grid, stale rows) and no checkpoint lies below. `images`: a checkpoint within the match was refused, the bank's images are not the request's. `no-match` cannot print (it has `common=0`) |
| `pick` | the state the sync started from: `own` (the bank's history), `adopt` (another bank's state, copied), `disk` (a disk chain), `live` (a live Responses / Anthropic tool continuation), `spilled` (the bank's own history, read back from its spill chain), `none` (token 0) |
| `from` | the bank an adopt copied from; `disk` for a disk chain |
| `at` | the position the sync started computing at |
| `reused` | the request's tokens served from state (the cache read) |
| `prefilled_now` | tokens this request prefilled |
| `below` | tokens between `at` and the deepest match any free bank had when the request was routed: what a resume at that match would have saved |
| `evict` | the bank this request's bind evicted (the last one, when it evicted several) |
| `evict_tokens`, `evict_conv`, `evict_ro` | that bank's history: its tokens, its `conv` key (the same rule over its tokens), and 1 when it was a read-only request's |
| `snapshot` | the eviction line's word with its blank removed: `disk`, `none`, `none(<rule>)` |
| `qwait_ms` | acceptance to bind |
| `restore_ms` | disk chain reads the request waited for |
| `prefill_ms` | prefill start to decode start (after a retried attempt: to the last attempt's decode start) |
| `ttft_ms` | acceptance to the first generated token |
| `out` | tokens generated, every attempt |
| `finish` | the turn's finish label (`stop`, `length`, `tool_calls`, `abandoned`, `error`, ...), or the server's word for an end that is no turn: `failed`, `busy`, `gone`, `conflict` (409), `stream-failed`, `refused-memory` (a 503 mid-prefill), and for a request that never bound `gone-queued`, `refused-context` (400), `refused-bank` (500), `refused-kv` (503) |

```text
pulsar-server: kv req: id=4711 ep=messages class=subagent ro=1 sess=0123456789abcdef agent=agent_7 parent=- pre=00000000000000ab conv=fd1f0f4381eb0395 prompt=60812 bank=1 route=fresh best=0 common=60460 score=0 anchor=31104 frontier=60987 prefilled=60900 noresume=ckpt-above pick=disk from=disk at=31104 reused=31104 prefilled_now=29708 below=29356 evict=1 evict_tokens=69233 evict_conv=84c39a079fc08121 evict_ro=0 snapshot=none(read-only) qwait_ms=3 restore_ms=456 prefill_ms=37300 ttft_ms=38012 out=16 finish=stop
```

### The cumulative ledger: `pulsar-server: kv ledger:`

Written every hour while the server runs (by the stall watchdog's thread, so not with `PULSAR_STALL_WATCHDOG_S=0`) and
once at shutdown, after the shutdown persists. Every value counts from the process start and is a variable `/metrics`
publishes: the line and `/metrics` are two prints of the same counters.

| Field | Meaning | `/metrics` |
|---|---|---|
| `uptime_s` | seconds since the listener came up | `/health` |
| `requests` | `kv req:` lines written | `pulsar:kv_requests_total` |
| `route_in-place`, `route_fresh`, `route_clobber`, `route_owner`, `route_matched`, `route_none` | ... by `route` | `pulsar:kv_routes_total{route}` |
| `pick_own`, `pick_adopt`, `pick_disk`, `pick_live`, `pick_spilled`, `pick_none` | ... by `pick` | `pulsar:kv_request_picks_total{pick}` |
| `evictions` | banks evicted | `pulsar:slot_evictions_total` |
| `evict_<victim>_by_<requester>` | of them, by the evicted history (`conv`: a conversation's, `ro`: a read-only request's) and the request it was evicted for | `pulsar:kv_evictions_by_total{victim,requester}` |
| `evict_own_best` | evictions whose victim was the requester's own best bank (0 since L379: such a request runs in place, `route=matched`) | `pulsar:kv_evictions_own_best_total` |
| `disk` | 1 when the disk tier is on (the disk fields are 0 when it is off) | |
| `seg_written_bytes`, `rp_written_bytes`, `segs_written`, `rps_written` | row segments and restore points committed: file bytes and files | `pulsar:kv_disk_io_bytes_total{op="written"}`, `pulsar:kv_disk_io_files_total{op="written"}` |
| `seg_read_bytes`, `rp_read_bytes`, `segs_read`, `rps_read` | files mapped for a load, from the RAM tier or the disk, each map | `...{op="read"}` |
| `removed_bytes` | file bytes removed, any reason | `pulsar:kv_disk_io_bytes_total{op="removed"}` |
| `removed_budget_bytes` | ... of them, evicted for room (the budget, a full disk) | `...{op="removed_budget"}` |
| `removed_unread_bytes` | ... of `removed_bytes`, files no load had mapped since the process started. The store keeps the bit in memory only, so a file left by an earlier run counts as unread until this run reads it | `...{op="removed_unread"}` |
| `restores_begun`, `restores_done`, `restores_failed` | disk chain restores; the rest are running or were cancelled | `pulsar:kv_disk_restores_total{outcome}` |
| `restored_tokens` | tokens the loaded chains added past the banks' own resumes | `pulsar:kv_disk_restored_tokens_total` |
| `dup_banks`, `dup_pages_resident`, `dup_pages_shared`, `dup_bytes`, `dup_bytes_peak`, `dup_deep_pairs`, `dup_deep_longest` | the duplicate census as last taken: resident banks with a whole page, pages held, distinct pages two banks share, the bytes held twice and their peak, bank pairs sharing more than the preamble and their longest shared prefix | `pulsar:kv_dup_*` |
| `spills`, `preemptions`, `readmissions` | guard spills, decoders preempted and readmitted | `pulsar:bank_spills_total`, `pulsar:decoder_preemptions_total`, `pulsar:decoder_readmissions_total` |
| `refused_<cause>` | 503 answers: `admission` (the prompt's KV does not fit), `image_busy`, `parse_busy`, `connections`, `shutdown`, `systemone` | `pulsar:refused_503_total{cause}` |
| `refused_prefill` | prefills stopped by the memory floor (503) | `pulsar:prefill_mem_refusals_total` |

### One line a `/v1/systemone` request: `pulsar-server: systemone req:`

Written when the handler returns, whatever the answer: `questions` (0 when it was refused before the queue), `tokens`
(the questions' rendered prompts, summed: the state once per question), `cached` (of those, held by the judge bank
when each question began), `prefilled` (tokens the worker prefilled for it), `ms` (the handler's wall time) and
`status` (the HTTP status).

## Memory floor and stall watchdog

The GB10 shares one memory pool between CPU and GPU, and the kernel's own
reclaim threshold is where GPU page faults start to fail. Pulsar keeps one
**memory floor** (`server_mem_floor_init`, `src/server/cli_main.cpp`). It is
derived at startup and logged, as the sum of:

- a 4 GiB OS margin;
- every zone's `high` watermark from `/proc/zoneinfo`;
- the largest host demand one worker step makes between checks.

Every `MemAvailable` check uses it: bank provisioning, the guard's live budget,
each prefill chunk, and each disk-KV persist.

When a prefill chunk would cross the floor, the server tries in order:

1. spill an idle bank;
2. preempt a decoder;
3. refuse the request with a retryable 503. The bank keeps what it already
   prefilled.

A persist that does not fit is skipped and counted. The decode guard keeps
touched KV under a ceiling measured after warmup.

The **stall watchdog** (`src/server/watchdog.cpp`) is a thread that checks once a
second. It fires when work is queued or running and the worker has not marked
progress for `PULSAR_STALL_WATCHDOG_S` seconds (default 120). It then:

- writes one dump to the log: slots, queue, KV and memory figures, the last
  section entered, every thread's state, and the worker's own backtrace;
- makes `/health` answer 503 `"status":"stalled"` with `"stalled": seconds`
  until progress resumes.

It never touches the engine.

## System 1 decisions (`/v1/systemone`)

A container that carries a runtime adapter and its decision head (Gemma 4's JEV judge: `build.py --adapter`) serves
`POST /v1/systemone` (L338 P2.4, `src/server/systemone.cpp`). Each question is one prefill of the adapter's bare-v1
template on a bank reserved for it at startup (`--judge-banks`, default 1 with an adapter: the pool's last bank, outside
the chat pool, so chat traffic never touches its K / V; it needs a pool of three). The worker serves the judge's queue
between decode quanta, a 1024-token chunk a turn.

```json
{"state": {"user_request": "...", "working_directory": "...", "action": {"tool": "Bash", "input": {"command": "..."}}},
 "questions": [{"kind": "noul", "question": "Does this tool call match one of the block rules below? ..."},
               {"kind": "score", "question": "How harmful is running the tool call ..."},
               {"kind": "choice", "question": "...", "options": ["...", "..."]}]}
```

- `state`: a string used verbatim, or an object serialized as the adapter's training data was, `json.dumps(state,
  indent=1)` (ASCII-escaped).
- `kind`: one the head answers (`judge_config.json` `slots.ranges`). `options` default to `false`/`true` (noul) and
  `0`..`5` (score); a choice question's are rendered `A) ...`.
- Answer: `{"object":"systemone.decisions","decisions":[{"kind","options","probs","choice", "p_block" (noul) |
  "score" (score: the expectation)}],"usage":{"prompt_tokens","cached_tokens"},"ms"}` -- `probs` = softmax over the
  kind's slots of `softcap * tanh((W h + b) / softcap) / T_kind` (`calibration_gold.json`), h the last row's final-norm
  hidden.
- 404 without a judge; 400 for a malformed request; 503 when its prefill fails.

## Health and metrics

`GET /health` returns:

```json
{"status": "ok", "version": "...", "model": "...", "uptime_s": 0, "stalled": 0,
 "slots": {"total": 0, "running": 0, "waiting": 0, "capacity": 0},
 "kv_cache_usage": 0.0,
 "tp": {"nranks": 1, "transport": null, "failed": false, "ranks": [...]}}
```

- It answers 200 when serving.
- It answers 503 when `status` is `draining` (shutdown requested) or `stalled`
  (watchdog).
- `slots.total` counts banks provisioned so far; `capacity` is the pool size.
- `tp.ranks` lists every rank's host, address, build and RDMA device. A
  non-tensor-parallel server reports itself as a one-rank group.

`GET /metrics` is Prometheus text. It uses vLLM's names where one exists, so
vLLM-oriented scrapers work:

- `vllm:num_requests_running`, `vllm:num_requests_waiting`,
  `vllm:kv_cache_usage_perc`
- `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`
- `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total`
- the latency histograms: `vllm:time_to_first_token_seconds`,
  `vllm:time_per_output_token_seconds`, `vllm:e2e_request_latency_seconds`,
  `vllm:request_prompt_tokens`, `vllm:request_generation_tokens`
- speculation: `vllm:spec_decode_num_drafts_total`,
  `vllm:spec_decode_num_draft_tokens_total`,
  `vllm:spec_decode_num_accepted_tokens_total` and the per-position counters

Everything else is in the `pulsar:` namespace. The main groups:

- **Per slot:** `pulsar:slot_phase` (idle, prefill_cold, prefill_main, decode,
  finish, restore), `pulsar:slot_position_tokens`, `pulsar:slot_context_tokens`,
  prefill progress, draft depth.
- **Admission and memory:**
  - `pulsar:admission_refusals_total{reason}`, where `reason` is one of
    `pool_full`, `ledger_full`, `mem_floor`, `create_fail`, `kv_touched`
  - `pulsar:prefill_mem_refusals_total`, `pulsar:mem_floor_bytes`
  - `pulsar:kv_eager_floor_bytes`, `pulsar:kv_touched_bytes`,
    `pulsar:kv_touched_ceiling_bytes`, `pulsar:kv_ledger_*`
  - `pulsar:queue_blocked_reason`, `pulsar:worker_stalled_seconds`
- **KV evidence (L378):** `pulsar:kv_requests_total`, `pulsar:kv_routes_total`,
  `pulsar:kv_request_picks_total`, `pulsar:kv_evictions_by_total`,
  `pulsar:kv_evictions_own_best_total`, `pulsar:kv_disk_io_bytes_total`,
  `pulsar:kv_disk_io_files_total`, `pulsar:kv_disk_restores_total`,
  `pulsar:refused_503_total` -- the counters of the hourly `kv ledger:` line.
- **Scheduling:** `pulsar:decoder_preemptions_total`,
  `pulsar:decoder_readmissions_total`, `pulsar:resume_decisions_total`,
  `pulsar:prefix_waits_total`, `pulsar:prefix_wait_gain_tokens_total`,
  `pulsar:bank_spills_total`, `pulsar:bank_restores_total`,
  `pulsar:slot_evictions_total`.
- **Disk KV:** `pulsar:kv_disk_*`, `pulsar:kv_restore_slices_total`,
  `pulsar:kv_restore_stall_max_seconds`, `pulsar:kv_persist_skipped_total`,
  `pulsar:kv_restore_failures_total`, the `X-Pulsar-KV` counters
  (`pulsar:kv_header_requests_total`, `pulsar:kv_persist_read_only_total`,
  `pulsar:kv_pinned_anchors`, `pulsar:kv_pinned_bytes`,
  `pulsar:kv_pins_released_total`), and the duplicate-prefix census
  `pulsar:kv_dup_*`.
- **Speculation:** `pulsar:spec_draft_depth`, `pulsar:spec_verify_*` (the
  measured verify cost curve), `pulsar:spec_calib_*` (per-drafter confidence
  calibration), and the round phase timers.
- **Qwen PLE read-ahead:** `pulsar:ple_read_ahead_*`.

`/metrics` and `/health` read a snapshot the worker publishes once per round.
They never call into the engine.

## Tensor parallelism

DeepSeek V4 / V4.1 and Qwen3.8-Flash-Next can be split across GPUs in separate
processes, normally two GB10s linked by RDMA. Nemotron, Inkling and the
encoder/Parse families refuse it. Every rank runs `pulsar-server` with the same
container:

```sh
# rank 0 (serves HTTP):
./pulsar-server -m /path/to/container --tp-role leader --tp-peer 0.0.0.0 --tp-port 9000
# rank 1 (serves no HTTP; follows the leader):
./pulsar-server -m /path/to/container --tp-role worker --tp-peer <leader-address> --tp-port 9000
```

The n-rank form gives every rank `--tp-rank R --tp-nranks N --tp-peers
host:port,...` (the same ordered list everywhere). `--rope-yarn` is refused under
TP. Each rank keeps its own copy of the disk KV segments (`<kv-dir>/tp-seg`). A
peer loss fails the group, and `/health` reports it in `tp.failed`. The design
records are [tensor-parallel-port.md](tensor-parallel-port.md) and
[tensor-parallel-split.md](tensor-parallel-split.md).

## Client setup

Set the client's context limit no higher than the server's `--ctx`. The examples
use `qwen3.8-flash-next`; substitute the served id of the model you loaded
(`GET /v1/models`). The server ignores the API key, but some clients need one.

**opencode** (`~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "pulsar": {
      "name": "pulsar (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8000/v1", "apiKey": "local" },
      "models": {
        "qwen3.8-flash-next": {
          "name": "Qwen3.8-Flash-Next (pulsar)",
          "limit": { "context": 262144, "output": 65536 }
        }
      }
    }
  }
}
```

**Codex CLI** (`~/.codex/config.toml`), on the Responses API:

```toml
[model_providers.pulsar]
name = "Pulsar"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

Then run `codex --model qwen3.8-flash-next -c model_provider=pulsar`.

**Claude Code**, on the Anthropic endpoint:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=local
export ANTHROPIC_MODEL=qwen3.8-flash-next
export ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8-flash-next
export ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-flash-next
export ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8-flash-next
export CLAUDE_CODE_SUBAGENT_MODEL=qwen3.8-flash-next
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000
exec claude "$@"
```

Claude Code's first request is large (often around 25k tokens). Keep the disk KV
cache on, so later turns and restarted sessions reuse that prefix.
`/v1/messages/count_tokens` is served, so Claude Code's context accounting
matches the server's rendering.

**Pi** and other OpenAI-compatible clients: point them at
`http://127.0.0.1:8000/v1` with the served id. Effort names reach the model
through `reasoning_effort` (see [Thinking and effort](#thinking-and-effort)).
