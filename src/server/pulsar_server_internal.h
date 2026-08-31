/* pulsar_server_internal.h — internal shared declarations for the server sources.
 * Produced by the multi-TU split of pulsar_server.c; edit freely (the
 * generator is not part of the build). */
#ifndef PULSAR_SERVER_INTERNAL_H
#define PULSAR_SERVER_INTERNAL_H

#include "pulsar.h"
/* L117: the spec-batched lane sizes its row arrays and admission budget from
 * the engine's slab authority (PULSAR_SPEC_LOGITS_ROWS) instead of a mirrored
 * literal -- one constant, no comment-enforced sync. The header is the
 * self-contained backend seam (stdlib includes only). */
#include "pulsar_gpu.h"
#include "pulsar_help.h"
#include "pulsar_kvstore.h"
#include "rax.h"

/* OpenAI/Anthropic compatible local server.
 *
 * HTTP is intentionally simple: each client connection is handled by a small
 * blocking thread that parses one request, then queues a job to the single
 * GPU worker.  The worker owns the pulsar_session and therefore owns all live KV
 * cache state.  That keeps session reuse, disk checkpointing, and future
 * batching decisions in one place instead of spreading graph mutations across
 * client threads. */

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>


/* Build-time version string (Makefile passes -DPULSAR_VERSION_STR from
 * `git describe`); fall back to "unknown" for non-Makefile/ad-hoc builds. */
#ifndef PULSAR_VERSION_STR
#define PULSAR_VERSION_STR "unknown"
#endif

/* ---- shared macros ---- */



#define PULSAR_SERVER_IO_TIMEOUT_SEC 10
#define PULSAR_SERVER_SEND_STALL_TIMEOUT_MS 2000
/* Trusted-LAN posture, but a single stuck or hostile peer must not exhaust
 * threads (one thread per connection) or hold a socket open forever while
 * trickling bytes (slowloris). The cap bounds concurrent client threads;
 * the read deadline bounds the total time a request may take to ARRIVE
 * (generation time is not counted - streaming responses run as long as the
 * model needs). */
#define PULSAR_SERVER_MAX_CLIENTS 64
#define PULSAR_SERVER_REQUEST_READ_DEADLINE_SEC 30

/* Multi-session serving increment 2: the worker steps each job as a resumable
 * state machine in bounded quanta instead of running it to completion. A
 * decode quantum yields back to the worker loop once it has emitted at least
 * this many tokens — the bound is checked between sampling iterations, so one
 * speculative burst (up to 17 accepted tokens) can overshoot it (a prefill
 * quantum is one engine chunk, bounded by the engine's chunked-prefill
 * machinery). With one slot the quanta run back-to-back and behavior is
 * identical to run-to-completion; increment 3 interleaves slots at these
 * boundaries. */
#define PULSAR_SERVER_DECODE_QUANTUM_TOKENS 16

/* OpenAI caps chat-completions top_logprobs at 20; a request above it is a 400,
 * so this also bounds the per-token capture buffer (see logprob_ledger). */
#define PULSAR_SERVER_MAX_TOP_LOGPROBS 20

/* Ceiling for bytes queued in a slot_writer for a client that stops reading.
 * The stall timeout is the real slow-client guard; this cap only bounds worst
 * case memory if a client keeps draining just enough to defeat it. */
#define PULSAR_SERVER_WRITER_MAX_PENDING_BYTES (64u * 1024u * 1024u)


/* The request parser only understands the API fields we use and skips the
 * rest.  Skipping is recursive because JSON values nest, so keep an explicit
 * ceiling: without it, a useless ignored field like {"x":[[[...]]]} can spend
 * the whole C stack before the request is rejected. */
#define JSON_MAX_NESTING 256


#define PULSAR_DSML "｜DSML｜"
#define PULSAR_DSML_SHORT "DSML｜"
#define PULSAR_TOOL_CALLS_START "<" PULSAR_DSML "tool_calls>"
#define PULSAR_TOOL_CALLS_END "</" PULSAR_DSML "tool_calls>"
#define PULSAR_INVOKE_START "<" PULSAR_DSML "invoke"
#define PULSAR_INVOKE_END "</" PULSAR_DSML "invoke>"
#define PULSAR_PARAM_START "<" PULSAR_DSML "parameter"
#define PULSAR_PARAM_END "</" PULSAR_DSML "parameter>"
#define PULSAR_TOOL_CALLS_START_SHORT "<" PULSAR_DSML_SHORT "tool_calls>"
#define PULSAR_TOOL_CALLS_END_SHORT "</" PULSAR_DSML_SHORT "tool_calls>"
#define PULSAR_INVOKE_START_SHORT "<" PULSAR_DSML_SHORT "invoke"
#define PULSAR_INVOKE_END_SHORT "</" PULSAR_DSML_SHORT "invoke>"
#define PULSAR_PARAM_START_SHORT "<" PULSAR_DSML_SHORT "parameter"
#define PULSAR_PARAM_END_SHORT "</" PULSAR_DSML_SHORT "parameter>"


/* =========================================================================
 * Tool Call Text Memory.
 * =========================================================================
 *
 * The model speaks DSML, while OpenAI and Anthropic clients round-trip tool
 * calls as JSON.  Re-rendering that JSON is not always the same byte sequence:
 * clients may preserve, sort, or rebuild object keys differently.  Tool call
 * ids are the bridge between both worlds.  For every generated tool call we
 * remember the exact DSML block sampled by the model under a random id.  When
 * the client later sends the same id back in conversation history, we replay
 * the sampled DSML verbatim and keep the KV cache aligned with the live model
 * state.
 */

#define PULSAR_TOOL_MEMORY_DEFAULT_MAX_IDS 100000
#define PULSAR_TOOL_MEMORY_MAX_BYTES (512u * 1024u * 1024u)


/* =========================================================================
 * KV Cache.
 * =========================================================================
 *
 * The server has one live GPU session.  We persist reusable DS4 session
 * snapshots when a cold prompt reaches a useful prefix, when a long continued
 * conversation has grown far enough, and when a request evicts the live session.
 * The cache key is the SHA1 of the rendered byte prefix.  The payload still
 * stores exact token IDs and graph state; the filename only selects a checkpoint
 * whose decoded transcript bytes are a prefix of the next rendered request.
 *
 * Files are loaded with plain read/write I/O into the existing graph tensors;
 * mmap is deliberately avoided here so cache restore cannot add more VM
 * mappings to a process that already maps a very large GGUF.
 *
 * Stores are created only when the live graph is already at the checkpoint we
 * want to persist.  For long cold prompts this means prefill reaches the stable
 * boundary first, writes that prefix, and then continues with the suffix.  We
 * never roll the session backward just to build a disk cache entry: that would
 * turn cache population into a second hidden prefill.
 *
 * File layout:
 *
 *   "KVC" version
 *   quant bits, save reason, token count, hit count, context size
 *   creation time, last-used time, payload byte count
 *   rendered text byte count + rendered text for human inspection
 *   DS4 engine payload written by pulsar_session_save_payload()
 *   optional tool-id map section
 *
 * The filename is SHA1(cache text bytes), not SHA1(token ids).  For ordinary
 * checkpoints the cache text is the rendered token prefix.  For live hidden
 * state it can instead be the client-visible transcript: the payload still
 * contains sampled reasoning KV, but the lookup key must be what the client can
 * replay after a process restart or session switch.
 *
 * The optional tool-id map is not part of model state, but it is needed to
 * render future client JSON back to the exact DSML sampled by the model.  We
 * persist only mappings whose DSML block appears in the saved cache text.
 */

#define KV_CACHE_FIXED_HEADER PULSAR_KVSTORE_FIXED_HEADER
#define KV_CACHE_HIT_HALF_LIFE_SECONDS PULSAR_KVSTORE_HIT_HALF_LIFE_SECONDS
#define KV_EXT_TOOL_MAP PULSAR_KVSTORE_EXT_TOOL_MAP
#define KV_EXT_RESPONSES_VISIBLE PULSAR_KVSTORE_EXT_RESPONSES_VISIBLE
#define KV_EXT_THINKING_VISIBLE PULSAR_KVSTORE_EXT_THINKING_VISIBLE
#define KV_TOOL_MAP_MAGIC0 'K'
#define KV_TOOL_MAP_MAGIC1 'T'
#define KV_TOOL_MAP_MAGIC2 'M'
#define KV_TOOL_MAP_VERSION 1u
#define KV_TOOL_MAP_HEADER 8u


/* =========================================================================
 * Trace Diagnostics.
 * =========================================================================
 *
 * The human transcript is not enough to debug prompt-cache misses.  The model
 * may generate text that is semantically accepted as a tool call, while the
 * next OpenAI request re-renders a slightly different canonical DSML block.
 * That creates a token mismatch even if the conversation "looks" continuous.
 *
 * When --trace is enabled we therefore record the exact cache decision and a
 * small token window around the first mismatch between the live KV checkpoint
 * and the incoming prompt.  Normal server logs stay compact; trace files get
 * enough data to diagnose tokenizer-boundary and canonicalization problems.
 */

#define TRACE_CACHE_BEFORE 8
#define TRACE_CACHE_AFTER  8
#define TRACE_CACHE_WINDOW (TRACE_CACHE_BEFORE + 1 + TRACE_CACHE_AFTER)

/* ---- shared types ---- */

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf;

typedef enum {
    REQ_CHAT,
    REQ_COMPLETION,
} req_kind;

typedef enum {
    API_OPENAI,
    API_ANTHROPIC,
    API_RESPONSES,
} api_style;

typedef struct server server;

typedef struct {
    char *id;
    char *name;
    char *arguments;
} tool_call;

typedef struct {
    tool_call *v;
    int len;
    int cap;
    char *raw_dsml;
} tool_calls;

typedef struct {
    int mem;
    int disk;
    int canonical;
    int missing_ids;
} tool_replay_stats;

typedef struct {
    char *name;
    char *wire_name;
    char *tool_namespace;
    /** Distinguish the Responses hosted tool from a normal function that
     * happens to be named "tool_search". */
    bool responses_tool_search;
    /** Anthropic web_search server tool: the SERVER executes calls to this
     * name mid-request (web_search.cpp); they never stop the turn. */
    bool server_web_search;
    char **prop;
    int len;
    int cap;
} tool_schema_order;

typedef struct {
    tool_schema_order *v;
    int len;
    int cap;
} tool_schema_orders;

typedef struct {
    char *role;
    char *content;
    char *reasoning;
    char *tool_call_id;
    char **tool_call_ids;
    int tool_call_ids_len;
    int tool_call_ids_cap;
    tool_calls calls;
    /** True when this system-role entry carries the request's top-level
     * system/instructions FIELD (the parser appends it to the array). Field
     * content always renders in the system region; inline role:system
     * MESSAGES follow the leading-run rule instead (L113) — position in the
     * array cannot distinguish the two, so this flag is the one authority. */
    bool system_field;
} chat_msg;

typedef struct {
    chat_msg *v;
    int len;
    int cap;
} chat_msgs;

typedef struct {
    char **v;
    int len;
    int cap;
    size_t max_len;
} stop_list;

/* Per-response timing metrics, surfaced in the additive "timings" object that
 * sits next to "usage" on the OpenAI chat/completions response (and the final
 * include_usage SSE chunk). Filled once by gen_step_finish from counters the
 * worker already keeps (g->t0/decode_t0/first_token_t, prompt/completion
 * counts, per-session DSpark deltas) — there is NO hot-path work here, and
 * these fields never influence sampling/rng/logits. All rates are derived in
 * the JSON emitter (guarded divisions), so a zero denominator omits a rate
 * rather than emitting NaN/inf. */
typedef struct {
    bool valid;
    double ttft_s;  ///< wall time from request start to first emitted token
    double prefill_s;  ///< wall time spent in prefill (request start -> decode start)
    double decode_s;  ///< wall time spent decoding (decode start -> finish)
    int prompt_n;  ///< total prompt tokens (prefill target)
    int cached_n;  ///< prompt tokens served from a cache (<= prompt_n)
    int decode_n;  ///< completion tokens emitted
    bool spec_active;  ///< DSpark speculative decode ran this request
    uint64_t spec_accepted;  ///< accepted draft tokens (this request)
    uint64_t spec_draft;  ///< proposed/verified draft tokens (this request)
    uint64_t spec_drafts;  ///< draft rounds (this request)
    uint64_t spec_gen;  ///< tokens emitted by the spec loop (this request)
} req_timings;

/* Fixed-bucket Prometheus histogram. The bucket bounds are shared per metric
 * family rather than stored per instance, so an instance is just counters —
 * cheap to keep under mu and to zero with the rest of the server struct.
 * Buckets are non-cumulative here; send_metrics accumulates them on the way
 * out, which is the form Prometheus wants. */
#define PULSAR_HIST_BUCKETS 14
typedef struct {
    uint64_t bucket[PULSAR_HIST_BUCKETS];
    uint64_t count;  ///< also serves as the +Inf bucket
    double   sum;
} pulsar_hist;

/* Defined in http_server.cpp, next to the emitter that prints the le= labels
 * so the bounds and their advertised values can never drift apart. */
extern const double pulsar_hist_seconds_bounds[PULSAR_HIST_BUCKETS];
extern const double pulsar_hist_tokens_bounds[PULSAR_HIST_BUCKETS];

/* Record one observation. Values above the last bound fall only into +Inf,
 * which the emitter derives from count, so no bucket is touched for them. */
static inline void pulsar_hist_observe(pulsar_hist *h, const double *bounds, double v) {
    if (!(v >= 0.0)) return;  ///< drops NaN as well as negatives
    int i = 0;
    while (i < PULSAR_HIST_BUCKETS - 1 && v > bounds[i]) i++;
    if (v <= bounds[i]) h->bucket[i]++;
    h->count++;
    h->sum += v;
}

/* One generated token's TARGET-model distribution, captured at the instant the
 * token was drawn and kept until the token's bytes reach the client.
 *
 * `end_off` is the offset into gen_state.text just past this token's piece.
 * The protocol projections release BYTES, not tokens (they hold tails for stop
 * strings, partial UTF-8 and half-written DSML tags), so the byte watermark is
 * what decides which SSE chunk an entry rides on; without it a chunk could
 * carry logprobs for a token whose text it has not sent yet. */
typedef struct {
    int    token;
    char  *piece;  ///< owned raw token bytes (may be partial UTF-8)
    size_t piece_len;
    float  logprob;
} logprob_token;

typedef struct {
    logprob_token tok;  ///< the token that was actually emitted
    size_t end_off;
    int    n_top;
    logprob_token *top;  ///< owned; n_top alternatives, descending
} logprob_entry;

/* Per-request OpenAI logprobs state.  Inert (and allocation-free) unless the
 * client asked for logprobs: every capture site is behind `enabled`, so a
 * request that did not ask pays one predictable not-taken branch per token.
 *
 * COVERAGE: an entry is recorded for EVERY generated token, in generation
 * order, over the whole raw completion — reasoning bytes and DSML tool-call
 * bytes included, not just the visible content substring.  The streamed
 * entries therefore concatenate to exactly the non-streaming array, which is
 * the invariant a client can check.
 *
 * `pending_*` holds the distribution of the token that has been SAMPLED but not
 * yet emitted.  Every decode lane draws its next token from a row it has in
 * hand (the session's own logits, or one row of a batched step's output) and
 * only later feeds it to gen_emit_token, so capture happens at the draw and is
 * consumed at the emit.  Exactly one token is ever in flight per slot. */
typedef struct {
    bool enabled;
    int  top_k;  ///< client's top_logprobs, 0..PULSAR_SERVER_MAX_TOP_LOGPROBS
    logprob_entry *v;
    int  len;
    int  cap;
    int  streamed;  ///< entries already written to an SSE chunk
    bool pending_valid;
    float pending_logprob;
    int  pending_n_top;
    pulsar_token_score pending_top[PULSAR_SERVER_MAX_TOP_LOGPROBS];
} logprob_ledger;

typedef struct {
    req_kind kind;
    api_style api;
    pulsar_tokens prompt;
    char *model;
    bool model_from_request;
    stop_list stops;
    char *raw_body;
    char *prompt_text;
    tool_schema_orders tool_orders;
    /** >0 iff the request advertised the Anthropic web_search server tool (and
     * the server has a backend configured): the remaining-use budget. */
    int web_search_max_uses;
    int max_tokens;
    int top_k;
    float temperature;
    float top_p;
    float min_p;
    /** Presence flags: true iff the CLIENT sent the parameter in the request
     * body. request_init() fills the value fields with engine defaults at
     * parse time, so the values alone cannot distinguish "explicitly 1.0"
     * from "absent" — downstream policy (e.g. think-mode defaults in
     * generate.cpp) must consult these and default only what is absent.
     * Zeroed by request_init's memset; set only in api_parse.cpp. */
    bool has_temperature;
    bool has_top_k;
    bool has_top_p;
    bool has_min_p;
    /** OpenAI logprobs (chat completions only).  `logprobs` alone reports the
     * chosen token's logprob; `top_logprobs` additionally asks for that many
     * alternatives per position and is rejected without logprobs:true, per the
     * OpenAI contract. */
    bool logprobs;
    int  top_logprobs;
    uint64_t seed;
    bool stream;
    bool stream_include_usage;
    int cache_read_tokens;
    int cache_write_tokens;
    req_timings timings;
    pulsar_think_mode think_mode;
    bool has_tools;
    /** tool_choice="required" (OpenAI) / {"type":"any"|"tool"} (Anthropic):
     * force a tool call. The prompt is prefilled into an open DSML tool_calls
     * block (thinking skipped) and generate_job seeds the output with the
     * SAME opener so the model must complete an invoke. forced_tool_name is
     * set for Anthropic {"type":"tool","name":X}: the opener then includes
     * the named invoke so the model can only fill in the parameters. */
    bool force_tool_call;
    char *forced_tool_name;
    bool prompt_preserves_reasoning;
    /** For /v1/responses: emit reasoning_summary_* events / fields only when the
     * client opted in via reasoning.summary. Other APIs leave this false; the
     * field is ignored on those code paths. */
    bool reasoning_summary_emit;
    /** Responses continuation contract:
     *
     * A live Responses tool loop is not a normal "new prompt with a long
     * prefix" request.  The protocol gives tool outputs a call_id that binds
     * them to a prior assistant tool call.  If that call_id is still known in
     * memory, the live KV is the authoritative prefix, including any hidden
     * thinking that the client did not replay.  These fields carry the parsed
     * evidence needed by generate_job() to append only the new suffix.
     *
     * A tool-output-only request has no stateless prefix to match.  If the live
     * call_id binding is gone by the time the worker executes it, DS4 must ask
     * for a full replay rather than cold-prefilling a prompt that starts with a
     * naked tool result.  Similarly, if live state is gone, a reasoning-mode
     * tool replay must contain the prior reasoning item (or an equivalent
     * opaque reasoning state from a future implementation). */
    bool responses_requires_live_tool_state;
    bool responses_requires_live_reasoning;
    stop_list responses_live_call_ids;
    char *responses_live_suffix_text;
    bool anthropic_requires_live_tool_state;
    stop_list anthropic_live_call_ids;
    char *anthropic_live_suffix_text;
    tool_replay_stats tool_replay;
} request;

typedef struct {
    char *key;
    char *value;
    bool is_string;
    bool used;
} json_arg;

typedef struct {
    json_arg *v;
    int len;
    int cap;
} json_args;

typedef enum {
    OPENAI_STREAM_THINKING,
    OPENAI_STREAM_TEXT,
    OPENAI_STREAM_TOOL,
    OPENAI_STREAM_SUPPRESS,
} openai_stream_mode;

typedef enum {
    DSML_TOOL_BETWEEN_INVOKES,
    DSML_TOOL_BETWEEN_PARAMS,
    DSML_TOOL_PARAM_VALUE,
    DSML_TOOL_DONE,
    DSML_TOOL_ERROR,
} dsml_tool_stream_state;

/* Shared states for protocol-specific DSML stream projections.  The model
 * still samples DSML; these states only translate already-sampled bytes into
 * OpenAI / Anthropic wire events while final parsing remains authoritative. */
typedef struct {
    dsml_tool_stream_state state;
    const char *tool_calls_end;
    const char *invoke_start;
    const char *invoke_end;
    const char *param_start;
    const char *param_end;
    size_t parse_pos;
    int index;
    bool active;
    bool emitted_any;
    bool args_open;
    bool first_param;
    bool param_is_string;
    char **ids;
    int ids_cap;
} openai_tool_stream;

typedef struct {
    openai_stream_mode mode;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    /** Thinking+tools: hold tentative answer text after the first </think>
     * until a tool marker, stream end, or a SECOND close proves whether it
     * is answer text or another reasoning pass (upstream ds4 fe2d3b0). */
    bool guard_second_reasoning;
    bool sent_reasoning;
    bool sent_content;
    /** Borrowed (never owned): the request's logprob ledger, so the delta
     * emitters can attach the entries whose bytes the delta releases.  NULL
     * whenever the client did not ask for logprobs — openai_stream_start
     * zeroes it and only the job binds it. */
    logprob_ledger *lp;
    openai_tool_stream tool;
} openai_stream;

typedef enum {
    DSML_DECODE_OUTSIDE,
    DSML_DECODE_STRUCTURAL,
    DSML_DECODE_STRING_BODY,
    DSML_DECODE_JSON_STRUCTURAL,
    DSML_DECODE_JSON_STRING,
} dsml_decode_state;

typedef enum {
    DSML_TRACK_SEARCH,
    DSML_TRACK_STRUCTURAL,
    DSML_TRACK_STRING_BODY,
    DSML_TRACK_JSON_PARAM,
    DSML_TRACK_DONE,
} dsml_track_mode;

typedef struct {
    const char *tool_calls_start;
    const char *tool_calls_end;
    const char *invoke_start;
    const char *invoke_end;
    const char *param_start;
    const char *param_end;
} dsml_syntax;

typedef struct {
    dsml_track_mode mode;
    dsml_decode_state decode;
    const dsml_syntax *syn;
    size_t pos;
    bool json_in_string;
    bool json_escaped;
} dsml_decode_tracker;

typedef enum {
    RESP_STREAM_THINKING,
    RESP_STREAM_TEXT,
    RESP_STREAM_SUPPRESS,
} responses_stream_mode;

typedef struct {
    responses_stream_mode mode;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    /** See openai_stream: second-reasoning-pass hold (upstream ds4 fe2d3b0;
     * upstream left Responses out, but our leak is identical). */
    bool guard_second_reasoning;
    bool reasoning_item_opened;
    bool reasoning_item_closed;
    bool reasoning_summary_started;
    bool reasoning_closed_naturally;
    bool message_item_opened;
    bool message_text_part_open;
    bool message_item_closed;
    bool reasoning_emitted_any;
    bool message_emitted_any;
    buf reasoning_text;
    buf message_text;
    char response_id[40];
    char reasoning_id[40];
    char message_id[40];
    int reasoning_index;  ///< output_index of the reasoning item (0 if present)
    int message_index;  ///< output_index of the assistant message item
    int next_output_index;  ///< monotonic counter for upcoming output items
    int sequence;  ///< monotonic per-event sequence_number Codex consumes
} responses_stream;

/* Item identity per tool call must be stable across added/done/completed. */
typedef struct {
    char fc_id[40];
    char call_id[64];
    bool is_custom;
    int output_index;
} responses_tool_item;

typedef enum {
    ANTH_STREAM_THINKING,
    ANTH_STREAM_TEXT,
    ANTH_STREAM_TOOL,
    ANTH_STREAM_SUPPRESS,
} anthropic_stream_mode;

typedef enum {
    ANTH_BLOCK_NONE,
    ANTH_BLOCK_THINKING,
    ANTH_BLOCK_TEXT,
    ANTH_BLOCK_TOOL,
} anthropic_block_type;

typedef struct {
    dsml_tool_stream_state state;
    const dsml_syntax *syn;
    size_t parse_pos;
    int index;
    bool active;
    bool emitted_any;
    bool args_open;
    bool first_param;
    bool param_is_string;
    char **ids;
    int ids_cap;
} anthropic_tool_stream;

/* Anthropic streaming uses the same sampled DSML bytes that will later be
 * parsed and remembered for exact continuation.  This state is only a wire
 * projection: it turns an in-progress DSML block into content_block/tool_use
 * SSE events, and never rewrites the model-visible transcript or cache key. */
typedef struct {
    anthropic_stream_mode mode;
    anthropic_block_type open_block;
    /** Borrowed from the request for the stream's lifetime: lets the tool-block
     * opener type server-executed tools as server_tool_use on the wire. */
    const tool_schema_orders *orders;
    int next_index;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    /** See openai_stream: second-reasoning-pass hold (upstream ds4 fe2d3b0).
     * has_tools is copied from the request at start so the round reset can
     * re-arm without a request pointer. */
    bool guard_second_reasoning;
    bool has_tools;
    bool sent_thinking;
    bool sent_text;
    anthropic_tool_stream tool;
} anthropic_stream;

typedef struct job job;

/* ---- deferred slot writer (multi-session increment 2) ----
 *
 * The GPU worker must never sleep in poll() waiting for a slow client while it
 * could be advancing the model. While a job is bound to a slot, the worker
 * installs a slot_writer for the job's (non-blocking) socket: send_all() on
 * that fd becomes best-effort non-blocking — bytes that do not fit in the
 * socket buffer are queued in order and flushed at every quantum boundary,
 * then drained fully before the job is signalled done. Client threads never
 * install a writer, so their send_all() keeps the bounded-blocking behavior.
 * Failure semantics match the blocking path: a hard socket error fails
 * immediately, and a peer that accepts no bytes for
 * PULSAR_SERVER_SEND_STALL_TIMEOUT_MS (or overflows the pending cap) fails the
 * stream, which the generation loop reports exactly as before. */
typedef struct {
    int fd;
    buf pending;  ///< accepted-but-unsent bytes, in wire order
    size_t off;  ///< consumed prefix of pending
    long long stall_deadline_ms;  ///< 0 = disarmed (nothing pending)
    bool failed;
    /** Wall clock of the last client bytes ACCEPTED by this writer (queued or
     * sent), 0 until the first. Every streamed byte funnels through
     * slot_writer_send, so this is the one honest "when did the client last
     * hear from us" clock — the heartbeat below reads it. */
    long long last_write_ms;
} slot_writer;

/* Decode-phase stream heartbeat (ledger L006, upstream Entrpi 406bf93).
 * A long prefill (156 s cold re-prefill at depth is normal for us — see the
 * KV-replay-divergence finding) sends the client NOTHING, so proxies and
 * client-side idle timeouts can kill a request that is progressing fine.
 * Every quantum boundary the worker asks whether a streaming slot has been
 * silent for PULSAR_SERVER_HEARTBEAT_MS and, if so, emits a keepalive that is
 * a NO-OP at the protocol level for the surface in question. */
#define PULSAR_SERVER_HEARTBEAT_MS 5000

void slot_writer_init(slot_writer *w, int fd);
void slot_writer_install(slot_writer *w);  ///< thread-local; NULL uninstalls
bool slot_writer_flush(slot_writer *w);  ///< non-blocking best effort
/* True when this writer has an armed clock and has been silent >= interval. */
bool slot_writer_idle_for(const slot_writer *w, long long now_ms, long long interval_ms);
bool slot_writer_drain(slot_writer *w);  ///< blocking, stall-timeout bounded
void slot_writer_free(slot_writer *w);

typedef pulsar_kvstore_entry kv_entry;

typedef pulsar_kvstore_options kv_cache_options;

typedef pulsar_kvstore kv_disk_cache;

typedef enum {
    TOOL_MEMORY_RAM = 0,
    TOOL_MEMORY_DISK = 1,
} tool_memory_source;

typedef struct tool_memory_entry tool_memory_entry;

typedef struct {
    char *dsml;
    size_t len;
    size_t bytes;
    int refs;
    uint64_t seen;
    tool_memory_entry *entries;
} tool_memory_block;

struct tool_memory_entry {
    char *id;
    tool_memory_block *block;
    size_t bytes;
    uint64_t stamp;
    tool_memory_source source;
    tool_memory_entry *prev;
    tool_memory_entry *next;
    tool_memory_entry *block_next;
};

typedef struct {
    rax *by_id;
    rax *by_block;
    tool_memory_entry *head;
    tool_memory_entry *tail;
    int entries;
    int max_entries;
    size_t bytes;
    size_t max_bytes;
    uint64_t clock;
    uint64_t scan_clock;
} tool_memory;

typedef struct {
    bool valid;
    /** Token frontier of a live assistant tool-call turn. Continuing from this
     * point preserves hidden thinking and sampled DSML bytes that are not
     * necessarily present in the client-visible replay. */
    int live_tokens;
    /** Optional rendered conversation text that the client is expected to replay.
     * Responses uses this because visible replay can omit hidden reasoning.
     * Anthropic currently uses only the call-id side of the state. */
    char *visible_text;
    size_t visible_len;
    /** Tool-call ids generated at the same live frontier. A following tool
     * result for these ids is a direct protocol continuation and should not
     * trigger prompt-prefix matching or checkpoint canonicalization. */
    stop_list call_ids;
} live_tool_state;

typedef struct {
    bool valid;
    /** Token frontier of the live sampled session.  The visible text below is
     * what clients will replay, but the payload at this frontier may also
     * contain hidden thinking tokens that are intentionally absent from that
     * visible replay. */
    int live_tokens;
    char *visible_text;
    size_t visible_len;
} visible_live_state;

/* ---- Session pool (multi-session serving, increment 1) ----
 *
 * PROCESS-GLOBAL CUDA STATE AUDIT (Tier 1 §1.5 — must-do, read before growing
 * the pool or adding GPU threads).
 *
 * All GPU work in this server runs on the SINGLE worker thread (worker_main in
 * generate.cpp). Client threads only parse HTTP and block on a per-job condvar
 * until the worker finishes (http_server.cpp handle path); they never touch a
 * pulsar_session_* or pulsar_gpu_* entry point. Verified by grepping every
 * pulsar_session_/pulsar_gpu_ call site under src/server: all of them are reached
 * only from the generation state machine (gen_* in generate.cpp) driven by
 * worker_main (or from cli_main startup/shutdown, before the worker starts and
 * after it joins) — with two deliberate exceptions, both plain loads of data
 * immutable after startup (no CUDA behind them): http_server.cpp reads
 * pulsar_session_ctx(server.sess) on client threads, and client paths read the
 * model id via server_model_id_from_engine (pulsar_engine_model_id, a static
 * shape constant). Nothing else: /metrics in particular makes NO engine
 * calls — the worker publishes per-slot KV positions and the spec-decode
 * counters into plain server fields under mu (m_slot_pos/m_slot_ctx/m_spec,
 * server_publish_metrics_snapshot in generate.cpp, refreshed at bind time and
 * once per quantum) and send_metrics reads only those snapshots. No CUDA
 * call is made off the worker thread. This is a correctness invariant, not
 * an accident.
 *
 * It matters because the CUDA layer keeps process-global, NON-thread-safe state
 * that all sessions share:
 *   - g_cublas / g_cublaslt handles (pulsar_cuda_runtime.cu, pulsar_cuda_matmul.cu),
 *     bound to cudaStreamPerThread;
 *   - the MXFP8 weight map g_fp8_mx_by_offset (std::unordered_map) plus its
 *     direct-mapped front cache fc_off/fc_ptr (pulsar_cuda_matmul.cu);
 *   - the function-local static lt_shape_cache (pulsar_cuda_matmul.cu);
 *   - the determinism setting CUBLASLT_REDUCTION_SCHEME_NONE (pulsar_cuda_matmul.cu).
 *
 * Because these are shared and unlocked, Tier 1 keeps ONE GPU worker thread and
 * multiplexes sessions by time-slicing on that thread. Tier 2 must NOT naively
 * spawn a second GPU thread: cudaStreamPerThread would give it a distinct
 * stream but it would still race on g_cublas/g_cublaslt and the weight/shape
 * caches. Any future concurrency stays on the single GPU lane (batched kernels),
 * not multiple GPU threads, until these globals are made per-context.
 *
 * Increment 1 was pure structural plumbing (pool of capacity 1). Increment 2
 * made the generation path re-entrant: each job runs as a per-slot resumable
 * state machine (gen_state in generate.cpp) that the worker steps in bounded
 * quanta — one prefill chunk, or up to PULSAR_SERVER_DECODE_QUANTUM_TOKENS decode
 * tokens, per step. All GPU work still happens on the single worker thread.
 * Increment 3 adds the scheduler: the worker binds queued jobs to free slots
 * (FIFO, warmest-prefix slot choice, lazily provisioning extra slots under the
 * KV admission budget) and round-robins one quantum at a time over the bound
 * slots. Increment 4 adds LRU eviction: when the queue head cannot be placed
 * cleanly (no fitting free slot, or only a warm slot it would clobber) and
 * provisioning was refused by a constraint eviction can relieve (full pool /
 * full admission ledger — deliberately NOT the MemAvailable floor, which
 * freed CUDA memory does not promptly move), the worker evicts the
 * least-recently-serviced IDLE slot — snapshot to the disk kv cache, free
 * the session, release its ACTUAL bytes from the ledger — and provisions in
 * its place (see the increment-4 block in generate.cpp; slot 0 is pinned).
 * Batched decode is a later increment (Tier 2). */

/* Pool capacity (increment 3). Slot 0 is provisioned at startup with the
 * configured --ctx-size; the rest are provisioned lazily, only when a job
 * arrives while every provisioned slot is busy (or would clobber another
 * conversation's warm KV) AND the packed-KV admission budget still has room.
 * A single client therefore always runs on slot 0, byte-identical to the
 * increment-2 single-session server. */
/* Raised 5 -> 8 (2026-08-10): banks are warm-state slots, not decode
 * streams, and 5 was below the fast-lane boundary for no reason.  8 is the
 * measured sweet spot: the batched custom-nt matmul lane and the split-KV
 * decode gate cap their fast paths at 8 rows, and N=12 aggregate decode
 * holds 29.2 tok/s at 8 banks vs 21.9 at 12 (the >8-row steps fall to the
 * slow lanes).  The engine allows up to PULSAR_MSEQ_MAX=16 via an operator
 * PULSAR_MSEQ_BANKS pin for TTFT-focused deploys that accept that cliff.
 * KNOWN LIMIT, measured: with active conversations == banks, pool-full
 * eviction is LRU and cyclic traffic evicts exactly the next returning
 * conversation's bank (domino, everyone cold); warm reuse needs headroom
 * (convs < banks) until the victim policy is smarter than LRU. */
#define PULSAR_SESSION_POOL_CAP 16
/* Auto-sizing cap: the batched custom-nt matmul lane and the split-KV decode
 * gate cap their fast paths at 8 rows, and N=12 aggregate decode holds 29.2
 * tok/s at 8 banks vs 21.9 at 12 (>8-row steps fall to the slow lanes) — so
 * the DEFAULT config never auto-sizes past 8.  POOL_CAP above is the hard
 * array bound = PULSAR_MSEQ_MAX, so an operator PULSAR_MSEQ_BANKS pin up to
 * 16 is safe (it was an out-of-bounds walk when the pin exceeded the array,
 * a latent bug up to and including the 5-slot era). */
#define PULSAR_SESSION_POOL_AUTO_MAX 8

/* Default context for lazily provisioned secondary slots (plan Tier 1 §1.4:
 * keep the default per-session context far below the lone-session maximum;
 * compressed-KV cost scales with ctx, so concurrency is bounded by the sum of
 * context sizes). A request that needs more than this gets a slot sized to
 * its need (capped at slot 0's ctx), admission permitting. */
#define PULSAR_SERVER_EXTRA_SLOT_CTX_TOKENS 65536

/* The rendered-prompt BOS marker. One definition for every render site AND
 * the startup trivial-match-threshold derivation (cli_main.cpp), so the
 * derived threshold can never silently drift from what rendered prompts
 * actually begin with. (Unit tests keep independent string literals on
 * purpose — they pin the wire bytes, not this macro.) */
#define PULSAR_SERVER_RENDER_BOS "<｜begin▁of▁sentence｜>"

/* Slot-routing trivial-match allowance (task #30, 2026-07-16). The router's
 * choose-vs-provision gate treats a candidate slot's common token prefix as
 * TRIVIAL — "just the shared rendered-template header, not a warm
 * continuation" — below a threshold of
 *     tokens(BOS + think-max preamble) + this allowance,
 * measured once per model at startup (cli_main.cpp). The derived part is the
 * largest template-injected text two UNRELATED conversations can share; the
 * allowance covers incidental natural-language prologue overlap between
 * distinct conversations (measured 3–8 tokens beyond the header across real
 * conversation pairs in the task-#24 bounce repro — 64 gives ~8x margin)
 * while staying an order of magnitude below any warm state worth a slot:
 * the 512-token disk-snapshot floor (KV_CACHE_DEFAULT_MIN_TOKENS) and the
 * multi-thousand-token preambles the session pool exists for. Used only by
 * the routing decision (server_slot_match_is_trivial); prefill reuse of a
 * chosen slot still honors arbitrarily short common prefixes. */
#define PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS 64

/* Admission-control budget (Tier 1 §1.4). GB10 unified memory is ~121 GiB
 * usable; weights are queried at runtime (pulsar_engine_weights_resident_bytes).
 * The overhead reserve is the fixed process footprint measured on the GB10,
 * independent of session count. Re-measured 2026-07-15 over three clean
 * server restarts (production v5mx gguf, ctx=98304, prefill_chunk=2048,
 * drop_caches before each load; MemAvailable deltas net of weights_resident
 * 85.04 GiB and slot 0's 2.68 GiB ledgered cost):
 *   - startup component (CUDA context, GPU page tables, pinned staging
 *     buffers): 8.54 / 9.87 / 9.41 GiB;
 *   - lazy first-request component (cuBLASLt workspaces at ~32 MiB/GEMM,
 *     FP8 workspaces, MXFP4 expert staging, GEMV activation buffers):
 *     8.71 / 8.71 / 8.72 GiB — stable to 10 MiB across restarts;
 *   - total steady state: 17.25 / 18.59 / 18.13 GiB, mean 17.99.
 * No further erosion after the first generation (MemAvailable flat within
 * ±0.01 GiB over subsequent generations and minutes of idle), so 18 GiB IS
 * the steady-state reserve; MemAvailable-based checks made after warm-up
 * must not re-reserve any part of it (see PULSAR_SERVER_MEM_FLOOR_BYTES).
 *
 * F1 addendum (task #32, 2026-07-17): the lazy first-request component had
 * grown to ~9.2 GiB (instrumented 3-client greedy burst: MemAvailable fell
 * 13.8 GiB in ~1.5 s, of which 4.59 GiB was a ledgered slot-1 create and
 * the rest this working set + measured total overhead ~18.7-19.0 GiB), and
 * it materialized mid-burst, AFTER every admission check had already read a
 * stale-high MemAvailable.  Two changes de-fang it:
 *   - cli_main.cpp runs a warmup generation at startup, so the working set
 *     materializes before the listener opens and before any admission math;
 *   - the admission budget is then re-derived from MEASURED post-warmup
 *     MemAvailable, min()'d with the static formula below, so these
 *     constants are an upper bound rather than the load-bearing estimate.
 * Re-measure the constants when the measured/static gap logged at startup
 * ("session admission: measured budget") exceeds ~2 GiB. */
#define PULSAR_SERVER_USABLE_BYTES          (121ull * 1024ull * 1024ull * 1024ull)
#define PULSAR_SERVER_PROCESS_OVERHEAD_BYTES (18ull * 1024ull * 1024ull * 1024ull)

/* Free-memory floor (2026-07-13 lockup postmortem; re-sized 2026-07-15):
 * kernel/OS breathing room ONLY — the last-resort backstop for when other
 * accounting is wrong. It deliberately does NOT cover any process overhead:
 * that is PULSAR_SERVER_PROCESS_OVERHEAD_BYTES' job. The original 6 GiB was
 * sized before the 18 GiB overhead constant existed and double-counted
 * caution on a warmed box: the ~8.7 GiB lazy first-request allocations
 * erode MemAvailable inside the reserve the ledger already subtracted, so
 * the live floor check vetoed sessions the ledger legally admitted
 * (measured 2026-07-14: third 2.5 GiB session refused at 8.39 GiB avail vs
 * the 8.50 the 6 GiB floor demanded, with ~5.9 GiB genuinely free at full
 * commit). 4 GiB backstop sizing (2026-07-15): the incident kernel died
 * near 0 and watchdogs fire at ~5; the measured warmed-box steady state
 * with three live sessions is 5.96 GiB avail, so 4 GiB re-admits the
 * incident shape (needs 2.5 + 4 = 6.5 <= 8.39) while still refusing a
 * further session at that 5.96 steady state (would leave ~3.4). Measured
 * per-session fault-in overshoot beyond the ledgered estimate is ~0.13 GiB
 * (2.63 actual vs 2.50 committed), so the realized post-admission floor
 * stays >= ~3.8 GiB.
 * Two guards use it:
 *   - server_kv_budget_bytes subtracts it from the admission budget, so the
 *     ledger can never legally commit the machine to zero free;
 *   - server_mem_floor_admits: provision_slot (and the eviction precheck)
 *     additionally refuse to create a session unless
 *     MemAvailable >= estimated cost + this floor.
 * The MemAvailable read is a coarse belt-and-suspenders guard: driver 610's
 * UVM accounting lags MemAvailable (and under UVM pressure MemTotal itself
 * SHRINKS — the incident box reported MemTotal 866 MiB — so percentage-based
 * monitors like earlyoom are useless here). Sessions also fault in AFTER
 * the check (measured: two sessions provisioned within 1 s both passed at
 * 11.23 GiB avail before either faulted), so the floor bounds intent, not
 * the instantaneous worst case. One /proc/meminfo read per provisioning
 * attempt, never on a hot path. */
#define PULSAR_SERVER_MEM_FLOOR_BYTES        (4ull * 1024ull * 1024ull * 1024ull)

typedef enum {
    SLOT_IDLE = 0,     /* no live job; the slot's KV may be warm and reusable */
    SLOT_PREFILLING,   /* ingesting a prompt (chunked prefill) */
    SLOT_DECODING,     /* generating tokens */
    SLOT_EVICTED,      /* session freed, ledger released; KV state spilled to
                          the disk kv cache (snapshot-on-evict) so a returning
                          client restores via the normal disk-text path instead
                          of a cold prefill. The slot entry (provisioned ==
                          false) is reusable by the next provisioning. */
} slot_state;

/* Resumable per-job generation state (defined in generate.cpp). Owns everything
 * that used to be a local of the run-to-completion generate_job: prompt/cache
 * resolution results, prefill progress, stream writers for all four API
 * surfaces, decode-loop trackers, and the deferred socket writer. */
typedef struct gen_state gen_state;

/* A pool slot is a pure bank descriptor over the server's ONE session
 * (server.sess). Slot 0 is provisioned at startup and pinned; slots 1..cap-1
 * are provisioned lazily and evicted LRU-first by the scheduler (worker
 * thread only) — an evicted slot is a reusable hole (provisioned == false,
 * state SLOT_EVICTED) below the n_slots high-water mark. */
typedef struct {
    bool         provisioned;  ///< false until admitted; cleared on eviction (the slot is a reusable hole). Every reader that used to skip sess == NULL skips this.
    uint32_t     bank;  ///< Tier-2: this slot's bank id in the shared pool (slot i -> bank i). 0 in classic (non-pooled) mode.
    int          committed_pos;  ///< Tier-2: this bank's committed KV frontier length (== pulsar_session_pos when this bank is the live one). Kept current at every op boundary so routing/metrics can read a non-live bank's position without a bank swap.
    struct job  *active_job;  ///< request bound to this slot, or NULL
    gen_state   *gen;  ///< resumable state for active_job
    slot_state   state;
    int          ctx_size;  ///< context this slot was admitted for
    uint64_t     est_cost_bytes;  ///< ledger-committed session cost (ACTUAL resident bytes once the session exists; the true-cost estimate only gates admission before the create)
    uint64_t     tokens_emitted;  ///< decode bookkeeping for the scheduler
    /** L114 counter watermark: session position up to which computed prefill
     * rows have been ticked into w_prefill_chunk_tokens. Advanced by BOTH the
     * classic progress callback and the mixed lane's fused sub-chunk commit
     * (same position coordinate), so their overlap can never double-count.
     * Reset to the resume/cached position at each prefill start. Monotone
     * per prefill: a rebuild that recomputes below the mark deliberately does
     * not re-tick (position-progress semantics, not GPU-work accounting). */
    int          prefill_counted;
    uint64_t     last_serviced_us;  ///< last quantum wall-clock (scheduler)
    /** Per-conversation continued-store frontier (see kv_cache_tracker_bind):
     * the shared pulsar_kvstore keeps one continued_last_store_tokens field, but
     * the schedule it tracks belongs to this slot's conversation. */
    int          continued_last_store_tokens;
    /** Tier-2 task #55 increment 2b — proactive-eviction guard. `spilled` means this
     * bank's comp/index PHYSICAL was cudaFree'd (raw KV bit-identical on disk at
     * <spill_dir>/spill-bank-<bank>.kv) while its conversation stays bound here; it
     * is restored (alloc_physical + kv_load) before this slot next decodes. Distinct
     * from SLOT_EVICTED (which frees the bank for a DIFFERENT conversation). */
    bool         spilled;
    /** Protocol live bindings for THIS slot's sampled KV frontier (guarded by
     * server.tool_mu — client threads read them at parse time). They bind
     * tool-call ids / visible transcripts to the session they were sampled on,
     * so a continuation can never match another slot's frontier. */
    live_tool_state responses_live;
    live_tool_state anthropic_live;
    visible_live_state thinking_live;
} session_slot;

/* Forward declarations / relocated types referenced by struct server's member
 * methods (C++ port) whose full definitions appear later in this header or (for
 * provision_refusal) in server_sched.cpp. Pointer params only need a forward
 * declaration; provision_refusal is an unscoped enum and must be complete here. */
struct job;
struct server_prefill_progress;
struct thinking_state;
struct trace_cache_diag;
typedef enum {
    PROVISION_OK = 0,
    PROVISION_REFUSED_POOL_FULL,   /* no free slot entry — eviction helps */
    PROVISION_REFUSED_ADMISSION,   /* ledger full — eviction helps */
    PROVISION_REFUSED_MEM_FLOOR,   /* machine physically tight (incl. an
                                      unreadable /proc/meminfo, fail closed) —
                                      eviction does NOT promptly help */
    PROVISION_REFUSED_CREATE_FAIL, /* allocation failed — eviction unsafe to
                                      chain on (same physical pressure) */
    PROVISION_REFUSAL_COUNT,       /* sentinel: array bound for the /metrics
                                      per-reason counters, not a reason */
} provision_refusal;

struct server {
    pulsar_engine *engine;
    /** The ONE session (created at startup, freed once at shutdown): classic
     * mode == 1 bank-less slot (slot 0), pool mode == pool_banks banks over
     * it. Slots are pure bank descriptors; all engine work goes through this
     * pointer. */
    pulsar_session *sess;
    /** Session pool. slots[0..n_slots) are provisioned; the worker thread is
     * the only mutator of slot fields and n_slots (n_slots additionally
     * published under mu for readers on client threads). */
    session_slot slots[PULSAR_SESSION_POOL_CAP];
    int          n_slots;  ///< provisioned slots (worker-owned; published under mu)
    /** Tier-2 bank-pool state (worker thread only). `pool_banks` > 0 means the
     * shared-pool flip is active: all live slots share server.sess and each
     * owns one bank; `live_bank` is the bank whose device views + host carry are
     * currently installed on that session (server_bank_switch lazily saves the
     * old and restores the new). L118: the three-way scheduler's spec_max_live
     * knob and the classic decode lane are deleted — decode runs through the
     * batched quanta at every n >= 1. */
    int          pool_banks;
    int          live_bank;
    /** plan-34 phase-2 inc 5: fused mixed-batch lane (PULSAR_MIXED_BATCH, default OFF,
     * read once at startup). When ON, the worker folds ONE prefilling slot's next
     * chunk (a K-row prefill run) into the decode quantum's first mixed step
     * (pulsar_session_decode_mixed) instead of advancing it as a separate classic
     * sweep — true continuous batching (P=1). OFF => today's exact decode-quantum +
     * separate one-prefill-chunk time-slice (byte-identical). Only meaningful in
     * pool mode (pool_banks>0). */
    bool         mixed_batch_enabled;
    /** Deep-concurrent guard for the fused lane: when the aggregate committed
     * depth (sum of committed_pos) of the active decode set exceeds this many
     * rows, worker_find_fuse_prefill refuses to fuse — the decode step is
     * already bandwidth-saturated and folding prefill in displaces decode
     * (the measured -48% tg regime). 0 disables the guard. */
    int          mixed_deep_guard_rows;
    /** Prefill rows folded into EACH decode step of a fused quantum (PULSAR_MIXED_CHUNK,
     * read once; default 32). Spreading the prefill uniformly across the quantum's
     * steps (vs one big chunk on one step) is what trades the time-slice's per-
     * interval decode STALL for a small uniform per-token cost — the p99 lever. */
    int          mixed_chunk_tokens;
    /** plan-33 inc B: warm full-prefix FORK routing (PULSAR_WARM_FORK, default on;
     * read once at startup). When a request's prompt token-extends an idle warm
     * bank's committed history, the router forks that trunk into a FREE bank and
     * continues there, leaving the trunk intact for siblings. */
    bool         warm_fork_enabled;
    /** plan-33 inc D: minimum shared-prefix TOKEN count for a PARTIAL fork-cut to
     * be worth it (below this, reusing so few tokens loses to a plain cold
     * prefill; a full-prefix match still forks regardless). PULSAR_WARM_PARTIAL_MIN,
     * read once at startup; floored to the ratio-4 align (partial cuts below R
     * reuse nothing). */
    int          warm_partial_min;
    /** The pool's shared per-bank context (boot --ctx).  Its own field
     * because slot 0's ctx_size used to double as this reference — and a
     * uniform eviction of slot 0 zeroed it, silently poisoning every later
     * provision with ctx 0 (banks the router then skipped forever). */
    int          pool_ctx_size;
    /** PULSAR_EVAL_PIN=1: history-independent serving for reproducible evals.
     * Kills every cross-request reuse channel at its choke point — thinking-
     * bind routing, warm forks, and in-place prefix continuation (common
     * prefix reported as 0, so every request cold-prefills from position 0).
     * Same request -> same output regardless of what the server did before.
     * The 2026-08-09 TEB investigation measured +-10 final points of history
     * dependence on identical requests at temperature 0; this flag is how an
     * eval pins behavior WITHOUT changing production defaults. */
    bool         eval_pin;
    uint64_t     bank_marginal_bytes;  ///< Tier-2: per-bank ledger charge in pooled mode (even split of the admitted pool cost; conservative, demand-paged reality is smaller). 0 in classic mode.
    uint64_t     kv_budget_bytes;  ///< admission ceiling computed at startup
    uint64_t     kv_committed_bytes;  ///< sum of est_cost_bytes over live slots (under mu)
    /** Tier-2 task #55 increment 2b — proactive-eviction guard. `guard_enabled`
     * gates the whole mechanism (on iff overcommit sized N>1 banks and a spill dir
     * exists). `guard_touched_budget` is the resident-KV ceiling the guard keeps
     * touched_kv under = kv_budget − eager_reserved (banks may grow to 1M but total
     * physical is bounded); `guard_eager_bytes` the eager floor already resident.
     * `guard_evictions` counts spills for metrics. `spill_dir` is a LOCAL fast-disk
     * scratch (NOT the NAS; NOT tmpfs — either would defeat physical reclaim). */
    bool         guard_enabled;
    uint64_t     guard_touched_budget;
    uint64_t     guard_eager_bytes;
    uint64_t     guard_evictions;
    char         spill_dir[512];
    /** Trivial-match threshold for the choose-vs-provision routing decision:
     * template-header tokens measured at startup +
     * PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS (cli_main.cpp; immutable after
     * startup, worker thread reads only). */
    int slot_trivial_common_tokens;
    int default_tokens;
    kv_disk_cache kv;
    tool_memory tool_mem;
    const char *web_search_url;  ///< see server_config.web_search_url
    pthread_mutex_t tool_mu;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t clients_cv;
    job *head;
    job *tail;
    bool stopping;
    time_t started;  ///< wall-clock when the listener came up (uptime for /health)
    int clients;
    /** /metrics scheduler + prefill gauges (all under mu). n_queued = jobs
     * enqueued not yet bound to a slot; n_generating = jobs bound to slots
     * (0..n_slots, time-sliced by the single worker). m_* are cumulative
     * prefill counters feeding the Prometheus prompt-throughput and
     * prefix-cache-hit metrics. */
    int n_queued;
    int n_generating;
    /* Tokens actually emitted, counted at the shared emit path (gen_emit_token)
     * rather than inside the DSpark fused verify loop. The engine's
     * spec_gen_tokens only advances on the spec lane, so at spec_max_live decode
     * banks or fewer it tracks generation and above that it stops dead — which
     * made vllm:generation_tokens_total report zero throughput on a fully busy
     * server. The worker thread is the only writer; publish_metrics_snapshot
     * copies it under mu. */
    uint64_t w_gen_tokens;  ///< worker-owned, no lock
    uint64_t m_gen_tokens;  ///< published copy, read by send_metrics
    /** L114: chunk-granular prefill counter (computed rows only, accumulated in
     * server_progress_cb per prefill_chunk event). vllm:prompt_tokens_total
     * stays request-granular (finish-time, cache-exact); this one advances
     * every ~4s chunk so a scraper can build prefill rate CURVES — flat spots
     * = scheduler/store overhead, lower slope = per-chunk slowdown (the L114
     * attribution instrument). */
    uint64_t w_prefill_chunk_tokens;
    uint64_t m_prefill_chunk_tokens;
    /** L117: live EMA of ms per emitted token in the spec-batched lane
     * (worker-owned). Denominator of the overflow argmax's cost threshold:
     * a marginal draft row is admitted while survival >= marginal_ms / this.
     * 0 until the first quantum; the argmax uses a 45 ms prior until then. */
    float    spec_ms_per_tok_ema;
    /** L123: the batched lane's shared ALL_ROWS logits landing buffer
     * (PULSAR_SPEC_LOGITS_ROWS x vocab floats, ~16.5 MB), allocated once on
     * first quantum. It was a per-quantum malloc/free — 16.5 MB of
     * demand-zero pages faulted back in on every touch cycle, a measured
     * ~1-2 ms/round of host tax. Worker-owned like the EMA above. */
    float   *spec_lane_logits;
    /** Which decode lane the scheduler is on: 0 idle, 1 spec, 2 batched. The
     * spec-decode counters cannot advance on the batched lane (it never enters
     * the fused loop), so a scraper needs this to tell "acceptance really is
     * this" from "no speculative decoding ran at all". */
    int w_decode_lane;
    int m_decode_lane;
    uint64_t m_prompt_tokens;  ///< cumulative prompt tokens prefilled
    uint64_t m_prefix_queries;  ///< cumulative prompt tokens seen (hit-rate denom)
    uint64_t m_prefix_hits;  ///< cumulative prompt tokens served from prefix cache
    /* Worker-published /metrics snapshots (under mu). The CUDA-state audit
     * above forbids engine calls on client threads, so the worker exports
     * per-slot KV positions/contexts and the engine spec-decode counters here
     * (server_publish_metrics_snapshot, at bind time and once per quantum);
     * send_metrics reads only these. */
    int m_slot_pos[PULSAR_SESSION_POOL_CAP];  ///< pulsar_session_pos per provisioned slot
    int m_slot_ctx[PULSAR_SESSION_POOL_CAP];  ///< ctx_size per provisioned slot
    /** Per-slot generation phase and prefill progress. Without these a scraper
     * cannot tell a slot that is prefilling from one that is decoding: both
     * only show m_slot_pos advancing, and a prefill chunk and a decode quantum
     * are indistinguishable once sampled at scrape cadence.
     * Stored as gen_phase + 1, so 0 (a zeroed server, or a slot with no bound
     * job) reads as idle rather than as GEN_PREFILL_COLD. */
    int m_slot_phase[PULSAR_SESSION_POOL_CAP];
    int m_slot_depth[PULSAR_SESSION_POOL_CAP];  ///< L112: adaptive draft depth per slot (0 = n/a)
    int m_slot_prefill_done[PULSAR_SESSION_POOL_CAP];  ///< tokens synced so far
    int m_slot_prefill_total[PULSAR_SESSION_POOL_CAP];  ///< prefill target, 0 if not prefilling
    pulsar_spec_metrics m_spec;  ///< engine spec-decode counters
    /* Request-latency histograms. req_timings is already computed for every
     * request (generate_job_end) and was previously only serialized into the
     * response body; observe_request_timings folds it in here so /metrics can
     * report TTFT and per-token latency. Worker thread writes, under mu. */
    pulsar_hist m_h_ttft;  ///< seconds to first emitted token
    pulsar_hist m_h_tpot;  ///< seconds per output token (decode_s/decode_n)
    pulsar_hist m_h_e2e;  ///< seconds, request start -> finish
    pulsar_hist m_h_prompt_tok;  ///< prompt tokens per request
    pulsar_hist m_h_gen_tok;  ///< completion tokens per request
    uint64_t m_requests_finished;
    /* Slot-pool lifecycle. Eviction forces the next turn of that conversation
     * to replay from a checkpoint, which is the dominant tail-latency source
     * on a busy pool, and none of it was previously observable. */
    uint64_t m_evictions;  ///< slots evicted to make room
    uint64_t m_spills;  ///< banks spilled to disk by the guard
    uint64_t m_restores;  ///< spilled banks brought back
    uint64_t m_restore_failures;  ///< spilled bank could not be restored
    /** Why the queue is stuck. Counted once per job (job::refusal_counted) so a
     * head that cannot bind for many quanta registers once, not once a tick. */
    uint64_t m_refusals[PROVISION_REFUSAL_COUNT];
    int m_queue_block_reason;  ///< provision_refusal + 1 of a stuck head; 0 = not blocked
    uint64_t seq;
    FILE *trace;
    pthread_mutex_t trace_mu;
    uint64_t trace_seq;

    /** @name Server methods (C++ port)
     *  1:1 mirror of the server_ / worker_ verb family; bodies keep the
     *  `auto *s = this` alias, numerics/logic verbatim.
     *  @{
     */
    bool send_model(int fd, const char *id);
    bool send_models(int fd);
    /** Liveness probe (/healthz, /ping): is the process alive at all? Always 200
     * while the process runs — deliberately independent of readiness/drain state,
     * so a k8s liveness probe never restarts a server that is merely draining.
     * Lock-free, engine-free (safe on a client thread).
     */
    bool send_liveness(int fd);
    /** Readiness + status (/health): is the server ready to accept work, and what
     * is it doing right now? 200 {"status":"ok",...} when serving; 503
     * {"status":"draining",...} once shutdown has been requested so a load
     * balancer stops routing to it. Reads only the worker-published snapshot under
     * mu (same discipline as /metrics — no engine calls on the client thread).
     */
    bool send_health(int fd);
    /** Version + build identity (/version), vLLM/OpenAI convention. Version is the
     * git-describe string baked in at build time (see Makefile).
     */
    bool send_version(int fd);
    /** Root banner so a bare GET / (browsers, uptime probes) gets a 200 with the
     * version and a pointer to the real endpoints instead of a 404.
     */
    bool send_root(int fd);
    /** Prometheus /metrics — DSpark speculative-decode counters in vLLM naming, so
     * tool-eval-bench --spec-live (and any vLLM-oriented scraper) reads acceptance
     * rate, acceptance length, and the per-position waterfall unchanged. All
     * counters are cumulative since engine open; gauges are point-in-time.
     * Series in the pulsar: namespace are additions with no vLLM equivalent; a
     * vLLM-oriented scraper ignores them.
     */
    bool send_metrics(int fd);
    void client_done();
    bool kv_tool_map_measure_locked(const char *text, uint32_t *count_out, uint64_t *bytes_out);
    bool kv_tool_map_serialized_size(const char *text, uint64_t *bytes_out);
    bool kv_tool_map_write(FILE *fp, const char *text, uint64_t *written_bytes);
    int kv_tool_map_load_from_pos(FILE *fp, const stop_list *wanted);
    void kv_cache_restore_tool_memory_for_messages(const chat_msgs *msgs);
    pulsar_kvstore_trailer_hooks kv_cache_tool_map_hooks(const stop_list *wanted);
    bool kv_cache_store_live_prefix_text(session_slot *sl, const pulsar_tokens *tokens, int store_len, const char *reason, const char *cache_text_override, uint8_t cache_text_ext, const char *cache_text_key);
    bool kv_cache_store_live_prefix(session_slot *sl, const pulsar_tokens *tokens, int store_len, const char *reason);
    bool kv_cache_store_current(session_slot *sl, const char *reason);
    void kv_cache_discard_failed_disk_entry(session_slot *sl, const char *path);
    void kv_cache_tracker_bind(session_slot *sl);
    void kv_cache_tracker_flush(session_slot *sl);
    void kv_cache_maybe_store_continued(session_slot *sl);
    int kv_cache_try_load_text(session_slot *sl, const char *prompt_text, pulsar_tokens *effective_prompt, char **loaded_path_out, uint8_t *loaded_ext_flags_out, bool responses_protocol);
    int kv_cache_try_load(session_slot *sl, const request *req, pulsar_tokens *effective_prompt, char **loaded_path_out, uint8_t *loaded_ext_flags_out);
    int live_text_prefix_prompt(session_slot *sl, const request *req, pulsar_tokens *effective_prompt);
    /** Tool-output-only Responses continuation.
     * Some clients send just the new tool outputs after a tool call.  There is no
     * long visible prefix to match in that shape; the call_id itself is the
     * protocol binding to the previous live assistant output.  Use it only when the
     * remembered live frontier and call-id set match exactly.
     */
    int responses_live_continuation_prompt(session_slot *sl, const request *req, int live_pos, pulsar_tokens *effective_prompt, int *matched_ids);
    /** Tool-result Anthropic continuation.
     * /v1/messages has no server-side response object like the OpenAI Responses
     * API, but its tool_use_id is still a precise continuation handle inside a live
     * local agent loop.  When the IDs and live token frontier match, continue from
     * the sampled DSML state and append only the user tool_result suffix.
     */
    int anthropic_live_continuation_prompt(session_slot *sl, const request *req, int live_pos, pulsar_tokens *effective_prompt, int *matched_ids);
    /** Visible-replay Responses continuation.
     * Other clients send the full visible transcript on every turn even though the
     * API semantics still make the request a continuation.  For Responses, exact
     * token-prefix matching is the wrong first question: hidden reasoning may be
     * live in KV but absent from the replay by design.  Instead, verify that the
     * request's rendered text begins with the visible transcript remembered at the
     * live frontier.  If it does, continue from the live token prefix and tokenize
     * only the bytes after that visible boundary.
     * If this check fails, DS4 has no special Responses state to trust.  The caller
     * then uses normal token/text/disk matching, which is the correct fallback for
     * cold starts, edits, restarts, or cross-client replays.
     */
    int responses_live_visible_prefix_prompt(session_slot *sl, const request *req, int live_pos, pulsar_tokens *effective_prompt);
    /** Tool-less thinking continuation.
     * Chat/completions and Anthropic do not have a previous_response_id object that
     * binds a later request to the last sampled turn.  Still, after a normal
     * tool-less thinking answer, the next prompt renderer intentionally omits that
     * hidden reasoning.  The live KV state is richer than the visible transcript.
     * Remembering the visible transcript as a key lets us keep the sampled hidden
     * KV when the next request clearly extends that same visible history.  This is
     * the same byte-prefix idea used by the disk cache: the client-visible text
     * selects the checkpoint, while the payload stays the exact sampled token
     * frontier.  If the visible key does not match, callers fall back to ordinary
     * token/text/disk matching.
     */
    int thinking_live_visible_prefix_prompt(session_slot *sl, const request *req, int live_pos, pulsar_tokens *effective_prompt);
    /** Routing probe (choose_slot_for_job): does this slot's live thinking
     * binding mark it as the warm continuation of req's visible transcript?
     * Same guards as thinking_live_visible_prefix_prompt above, but byte-prefix
     * check only — no tokenization, no effective-prompt build (gen_begin redoes
     * the full resolution on the chosen slot). The router needs this because
     * the token common prefix UNDERSTATES relatedness for thinking chats: the
     * client replays visible content while the slot's sampled frontier holds
     * the hidden reasoning too, so a short token match can still be the same
     * conversation. Returns the matched visible-key length (>0) so the caller
     * can prefer the most recent frontier if several slots hold bindings for
     * prefixes of one conversation, or 0 for no match. Never dereferences
     * s->sess (the caller passes the slot's live position).
     */
    size_t thinking_live_binds_prompt(session_slot *sl, const request *req, int live_pos);
    /** Validate Responses tool outputs before rendering.
     * A tool output with a call_id is meaningful only if either:
     * 1. DS4 still has the matching live assistant call in memory, or
     * 2. the same request replays the prior assistant call item.
     * Case 1 is the fast, protocol-native continuation path: keep the live KV and
     * append only the tool result.  Case 2 is stateless replay after restart or
     * branching.  In thinking mode, case 2 is less faithful if the replay omits
     * reasoning state for the assistant call.  Official Responses clients can
     * carry that state with reasoning items / encrypted reasoning content; when
     * they do not, the request is still renderable as visible history.  Mark that
     * condition so generate_job() can prefer live / visible checkpoints and emit a
     * warning if it must fall back to visible replay instead of aborting the
     * session.
     */
    bool responses_validate_tool_outputs(const chat_msgs *msgs, pulsar_think_mode think_mode, bool *requires_live_tool_state, bool *requires_live_reasoning, char *err, size_t errlen);
    /** Validate Anthropic tool results before rendering.
     * A tool_result.tool_use_id is valid if it is either still bound to the live
     * Anthropic assistant tool-call frontier or the same request replays the prior
     * assistant tool_use block.  The first case is the fast path: keep the sampled
     * KV and append only the tool-result suffix.  The second case is a normal
     * stateless replay, where exact DSML tool memory can restore the sampled tool
     * bytes before prefix matching.  A tool-result-only request with an unknown
     * live id has no safe prefix to reconstruct, so report a clear client error.
     */
    bool anthropic_validate_tool_results(const chat_msgs *msgs, bool *requires_live_tool_state, char *err, size_t errlen);
    bool append_rendered_suffix_to_live_session(session_slot *sl, const char *suffix, int *tokens_appended, char *err, size_t errlen);
    bool continue_after_invalid_dsml(session_slot *sl, const request *r, const thinking_state *thinking, const char *detail, int *tokens_appended, char *err, size_t errlen);
    /** Execute one server-side web_search round: run the query against the
     * configured SearXNG backend, surface the result to the client as
     * server_tool_use/web_search_tool_result content, splice the result text into
     * the live session as an ordinary tool_result turn, and re-enter decode within
     * the same request.  Returns false when this call is not server-executed (or
     * the splice failed) — the caller then finishes the turn as a normal
     * client-visible tool_use.
     */
    bool gen_web_search_round(session_slot *sl, const tool_calls *calls, const char *pre_content, const char *pre_reasoning);
    void send_prefill_failure_response(const job *j, const server_prefill_progress *progress, const char *ctx, const char *flags, const char *err);
    void remember_thinking_checkpoint(session_slot *sl, const job *j, const char *ctx, uint64_t trace_id, const char *content);
    /** Tool-call finish WITH thinking on: the model emitted <think>reasoning</think>
     * before the DSML tool call, so the reasoning tokens sit in the live KV.  We
     * remember the exact bytes the NEXT request will render for this turn as a visible
     * key, keeping the live tokens (reasoning included) as the sampled frontier.  The
     * next request byte-matches the key and continues from live KV — no rewrite, no
     * rebuild — and, critically, an evicted-then-reloaded checkpoint is keyed by that
     * same visible transcript on disk (kv_cache_store_current).
     * render_chat_prompt_text ALWAYS re-renders the reasoning inside <think>…</think>
     * for a tool-context turn (prompt_render.cpp: `tool_context || i > last_user_idx`),
     * because agentic clients (opencode et al.) replay reasoning_content verbatim so
     * the model keeps its chain of thought across tool rounds.  So the key MUST carry
     * the reasoning too — an earlier version dropped it (<think></think>), which byte-
     * diverges from every reasoning-preserving replay at the first reasoning byte and
     * made the live alias AND the disk key miss, forcing a full cold re-prefill of the
     * whole conversation on eviction (opencode's ~4-minute-per-message symptom).  The
     * toolless thinking path (remember_thinking_checkpoint) still strips: it only fires
     * for non-tool-context requests (should_remember_thinking_checkpoint bails when
     * prompt_preserves_reasoning), i.e. clients that DO drop reasoning on replay.
     */
    void remember_tool_thinking_checkpoint(session_slot *sl, const job *j, const char *ctx, uint64_t trace_id, const char *content, const char *reasoning, const tool_calls *calls);
    /** After a successful tool-call finish, make the live checkpoint match what the
     * next request will render.  Usually that is just the exact DSML remembered by
     * tool id.  If a client sends a tool call without an id we know, the fallback
     * renderer still builds valid DSML from JSON, and this function either rewrites
     * the short suffix in place or reloads an older disk checkpoint before replay.
     */
    void canonicalize_tool_checkpoint(session_slot *sl, const job *j, const char *ctx, uint64_t trace_id, const char *content, const char *reasoning, const tool_calls *calls);
    /** Shared failure epilogue for both prefill phases (the old duplicated blocks
     * after each pulsar_session_sync failure). Token vectors and the disk path are
     * freed centrally by gen_state_free.
     */
    void gen_prefill_fail(session_slot *sl);
    /** Resolve the prompt against every cache layer and decide the prefill plan.
     * Clients resend full prompts as text.  The worker first tries the old exact
     * token-prefix hit, then a rendered-text prefix hit for the live checkpoint,
     * then disk text-prefix restart snapshots, then a cold prefill.  On text-prefix
     * hits we build a fresh effective prompt from the checkpoint's exact token
     * history plus a newly tokenized string suffix; the canonical full-prompt
     * tokens are not sliced because BPE may merge across the byte boundary.  Cold
     * prompt caching is handled before generation: if the stable checkpoint is
     * shorter than the full prompt, we prefill to that boundary, store it, and
     * immediately continue to the real prompt.  The live graph therefore always
     * moves forward.
     */
    void gen_begin(session_slot *sl);
    /** One prefill quantum: (re-)issue the sync toward the phase's target; the
     * cancel callback stops it after one completed chunk and the checkpoint
     * carries the progress to the next quantum.
     */
    void gen_step_prefill(session_slot *sl);
    /** Runs once, in the same quantum that completed the main prefill: clear stale
     * live bindings, persist checkpoints, emit response identity, and start the
     * protocol stream projections that persist across all decode quanta.
     */
    void gen_stream_begin(session_slot *sl);
    /** (Re)initialize a decode attempt: the body of the old decode_again label.
     * Runs both for a fresh request and after a tool-error recovery appended a
     * model-visible correction to the live session.
     */
    void gen_decode_init(session_slot *sl);
    /** Emit one already-decoded token into the response stream: append it to the
     * accumulated text, feed the thinking/DSML trackers, run stop-string and
     * tool-marker detection, and drive every active protocol stream projection
     * (plain SSE / OpenAI / Anthropic / Responses). Returns true when the decode
     * loop must STOP after this token (EOS, a stop string, a completed tool_calls
     * block, or a client write error), with g->finish (and g->err on error) set.
     * Factored out of the classic decode loop's per-token inner loop (Tier-2 Step 5)
     * so every driver shares ONE emit path: the classic spec/plain decode loop
     * below AND the batched multi-session lanes (which sample each live bank's
     * row on the host, then call this to stream that bank's slot). It touches
     * ONLY host state hung off sl->gen + j->req + the client fd — no engine/CUDA
     * call except pulsar_token_text. That host-only property is what makes the
     * L116 tool admission to the batched lanes sound: all tool-marker tracking,
     * thinking-recovery, and stop handling here runs identically in every lane.
     * Behavior for the single-session path is byte-identical to the
     * pre-factoring inner loop.
     */
    bool gen_emit_token(session_slot *sl, int token);
    /** Post-decode epilogue: tool repair/recovery, final parse, protocol live
     * state, checkpoints, the final response, and logging. Recovery paths loop
     * back to GEN_DECODE_INIT (the old goto decode_again).
     */
    void gen_step_finish(session_slot *sl);
    void gen_state_free(session_slot *sl);
    /** Bind a dequeued job to the slot and resolve its prompt (the first quantum). */
    void generate_job_begin(session_slot *sl, job *j);
    /** Advance the job by one quantum. */
    void generate_job_step(session_slot *sl);
    /** Unbind: drain deferred client bytes, free the resumable state. */
    void generate_job_end(session_slot *sl);
    void thinking_live_clear(session_slot *sl);
    void thinking_live_remember(session_slot *sl, const char *visible_text);
    void responses_live_remember(session_slot *sl, const char *visible_text, const tool_calls *calls);
    void anthropic_live_remember(session_slot *sl, const tool_calls *calls);
    void responses_live_clear(session_slot *sl);
    void anthropic_live_clear(session_slot *sl);
    bool responses_live_has_call_id(const char *id);
    bool anthropic_live_has_call_id(const char *id);
    bool responses_live_matches_request(const session_slot *sl, const stop_list *ids, int live_tokens);
    bool anthropic_live_matches_request(const session_slot *sl, const stop_list *ids, int live_tokens);
    /** Scheduler routing (worker thread): find the slot whose live binding holds
     * ALL of the request's continuation ids at that slot's current frontier, so
     * the job can be bound to the session that owns its conversation.
     */
    session_slot * live_slot_for_ids(const stop_list *ids, bool anthropic);
    session_slot * responses_live_slot_for_ids(const stop_list *ids);
    session_slot * anthropic_live_slot_for_ids(const stop_list *ids);
    bool tool_memory_has_id(const char *id);
    void tool_memory_remember(const tool_calls *calls);
    void tool_memory_put_source(const char *id, const char *dsml, tool_memory_source source);
    void tool_memory_put(const char *id, const char *dsml);
    void tool_memory_attach_to_messages(chat_msgs *msgs, tool_replay_stats *stats);
    void assign_tool_call_ids(tool_calls *calls, api_style api);
    void trace_piece(uint64_t id, const char *piece, size_t len);
    void trace_event(uint64_t id, const char *fmt, ...);
    void trace_write_cache_diag(const trace_cache_diag *d, const tool_replay_stats *tool_replay, int cached, const char *cache_source, int disk_cached, const char *disk_path);
    uint64_t trace_begin(const job *j, int cached, int effective_prompt_tokens, const trace_cache_diag *cache_diag, const char *cache_source, int disk_cached, const char *disk_path);
    void trace_finish(uint64_t id, const request *r, const char *final_finish, int completion, bool saw_tool_start, bool saw_tool_end, const char *parsed_content, const char *parsed_reasoning, const tool_calls *parsed_calls, double elapsed);
    bool enqueue(job *j);
    void close_resources();
    bool bank_switch(int bank);
    /** Evict exactly one NON-trunk victim so a warm fork gets a free bank. Trunk is
     * always protected (a sibling still matches it); LRU-superseded victims go
     * first, else plain LRU (worker_evict_one's picker). Reuses the proven eviction
     * body (snapshot + ledger release + bank reset). Worker thread only; returns
     * true when a bank was freed.
     */
    bool fork_make_room(const session_slot *trunk, bool superseded_only = false);
    bool bank_restore_spilled(int bank);
    /** plan-33 inc D victim policy: an idle bank is LRU-SUPERSEDED when its whole
     * committed history is a token-prefix of ANOTHER live bank's history (a sibling
     * that already extends past it) — its KV is redundant, so evicting it loses the
     * least. Returns such a slot's index (the least-recently-served among them), or
     * -1. Pure host reads (pulsar_session_bank_tokens / _common_prefix are the same
     * host-carry reads routing already uses on idle banks; no CUDA, no install).
     */
    int pick_superseded_idle(const bool *protect);
    /** Spill one idle bank: install it, snapshot its KV to disk, save its host carry,
     * repoint AWAY (free_physical refuses the cur bank), then cudaFree its physical.
     */
    bool spill_bank(session_slot *victim);
    /** LRU-idle smallest-frontier victim: NOT bank 0 (pinned), NOT in the live decode
     * set, no active job, not already spilled. -1 if none.
     */
    int guard_pick_victim(session_slot **dec, int n);
    void guard_maybe_evict(session_slot **dec, int n);
    /** Publish the /metrics snapshots — per-slot KV position/context and the
     * engine spec-decode counters — into plain server fields under mu. Client
     * threads must never call into the engine (CUDA-state audit,
     * pulsar_server_internal.h), so the worker exports these at startup (cli_main,
     * before the worker thread runs), after binds, and once per quantum;
     * send_metrics reads only the snapshots. Host-int copies, no GPU work.
     */
    void publish_metrics_snapshot();
    /** Fold one finished request's timings into the /metrics histograms. Called by
     * the worker from generate_job_end, where req_timings has just been computed;
     * this only reads that struct, so it adds no hot-path work.
     */
    void observe_request_timings(const req_timings *t, double e2e_s);
    /** Record why the head job could not be bound. Counted once per job, not once
     * per bind attempt: the worker retries the head every quantum, so counting
     * each attempt would turn one stuck request into thousands of "refusals". The
     * companion gauge reports the reason the queue is blocked right now.
     */
    void note_provision_refusal(job *j, provision_refusal refusal);
    /** Bump a /metrics counter under mu. Worker-thread callers only; the lock is
     * what keeps the value coherent for send_metrics reading on a client thread.
     * Never called with mu already held — the mutex is not recursive.
     */
    void count_metric(uint64_t *counter);
    /** Parse-time id lookups run on client threads before the request is bound to
     * a slot, so they scan every provisioned slot's live binding. n_slots is
     * published under mu (its owning lock) — take mu for the snapshot rather than
     * asserting cross-lock visibility. A momentarily stale snapshot would only
     * miss a slot provisioned this instant, whose bindings are still empty.
     */
    int n_slots_snapshot();
    /** Mark slots some QUEUED live-tool-state continuation still needs: that KV
     * frontier exists only on its owner slot, so evicting it would turn the
     * queued job into a 409 the moment it binds. The queue is snapshotted under
     * mu; the job pointers stay valid afterwards because only this worker pops
     * jobs and each client thread blocks on its job condvar until then. The
     * owner lookups (tool_mu + session pos) run after mu is released — the two
     * locks are never nested.
     */
    void worker_protect_queued_owner_slots(bool protect[PULSAR_SESSION_POOL_CAP]);
    /** Soft eviction protection: OR into protect[] every bank that is some QUEUED
     * job's best USABLE warm match. Without this the fresh-path domino recurs:
     * job A's eviction lands on job B's warm trunk, and B — often the very next
     * bind — cold-replays its whole history. "Usable" is the operative word: a
     * match whose partial cut the raw ring has scrolled past (a compacted client)
     * is dead warmth and stays evictable — protecting it would evict live warmth
     * in its stead. Best-effort by contract: callers retry without this overlay
     * when it leaves no victim, so binding always progresses. Worker thread only
     * (slot_common_prefix reads engine host carries).
     */
    void worker_protect_queued_warm_matches(bool protect[PULSAR_SESSION_POOL_CAP]);
    /** Pointless-eviction guard #2: evicting is only worth its cost if releasing
     * idle sessions can actually admit the provisioning the head job needs.
     * If even reclaiming EVERY unprotected idle slot leaves admission refusing,
     * skip eviction entirely — the head is genuinely waiting for a busy slot to
     * free, and evicting warm idle sessions would only churn snapshots. (Host
     * arithmetic only: pulsar_engine_session_cost_bytes is the same sizing code the
     * allocator uses, no CUDA work; runs only on failed bind attempts. Guard #1
     * is the provisioning-refusal reason check in worker_try_bind.)
     */
    bool worker_eviction_could_help(const job *j, const bool *protect);
    /** Evict one idle slot (LRU victim): snapshot to the disk kv cache when
     * possible, free the session, release the ledger, and leave the slot entry
     * (provisioned == false) for provision_slot to reuse. Failure honesty: a failed or
     * unavailable snapshot only costs the returning client a re-prefill — the
     * eviction itself proceeds, and the response always belongs to the right
     * conversation because the freed KV can never be read again. Returns false
     * when nothing is evictable. Worker thread only.
     */
    bool worker_evict_one(bool protect[PULSAR_SESSION_POOL_CAP]);
    /** Bind the head job to a slot if routing allows it. Strict FIFO: when the
     * head must wait (its owner slot is busy, or no fitting slot is free), later
     * jobs wait behind it — simple and starvation-free. Returns true when the
     * head was consumed: bound to a slot, or failed explicitly (a continuation
     * that cannot fit its owner slot — see choose_slot_for_job). When the head
     * cannot be placed cleanly (nothing fits, or only a warm slot it would
     * clobber), it is not waiting on a busy owner, and the provisioning refusal
     * is one eviction can relieve (full pool / full ledger — never the
     * MemAvailable floor, see the increment-4 block above), idle slots are
     * evicted LRU-first until the head binds without clobbering or eviction
     * stops helping — then the clobber fallback binds it exactly like the
     * increment-3 scheduler did.
     */
    bool worker_try_bind();
    /** Detach a finished job from its slot and wake its client thread. */
    void worker_finish_slot(session_slot *sl);
    void worker_batched_decode_quantum(session_slot **dec, int n);
    /** plan-34 inc 6: the SPEC batched quantum. Same skeleton as
     * worker_batched_decode_quantum, but each sweep runs one speculative ROUND
     * per bank instead of one token: per bank under its restored state we draw
     * the base token (carry or fresh), begin the round (guards, frontier
     * snapshot, checkpoint push), and contribute its rows; ONE decode_mixed
     * ALL_ROWS forward covers every bank's rows with the drafter capture +
     * Stage-B saves armed; then per bank round_end walks its slice, rolls state,
     * redrafts, and we emit the accepted tokens through the same slot machinery
     * the plain lane uses. Tokens per weight-stream compound: batching x
     * acceptance ([[L076]]).
     * Emission mirrors gen_decode's L073 discipline: a mid-emit stop (tool-call
     * end, stop string) rewinds the ghost tail so the bank's history never
     * carries tokens the client did not see (pulsar_session_rewind also clears
     * the pendings/carry, which is exactly right -- they were conditioned on the
     * ghosts).
     */
    void worker_spec_batched_quantum(session_slot **dec, int n);
    /** plan-34 phase-2 inc 5 — find ONE prefilling slot to FOLD into the fused mixed
     * quantum (P=1). Admissible = main-prefill (not cold), already past its FIRST chunk
     * (bank pos>0, so the driver's pos-0 reject is satisfied — the first chunk stays
     * classic), and with more than one fold-chunk of prompt still left (the FINAL tail
     * <= chunk stays classic; it carries the prefill->decode completion bookkeeping).
     * Its bank is necessarily DISTINCT from every decode bank (different phase). NULL
     * when the flag is off, not in pool mode, or nothing qualifies.
     */
    session_slot * worker_find_fuse_prefill();
    /** plan-34 phase-2 inc 5 — FUSED mixed-batch quantum. One decode quantum whose EVERY
     * step folds a small (s->mixed_chunk_tokens) prefill run for `pf` into the SAME
     * pulsar_session_decode_mixed sweep as the decode banks (true continuous batching,
     * P=1). Decode banks advance exactly as worker_batched_decode_quantum — the inc-4
     * neutrality gate proves a co-scheduled prefill does not perturb them — so their
     * per-request output is unchanged in shape. The prefill advances up to
     * QUANTUM*chunk tokens, SPREAD uniformly across the steps so no single decode step
     * eats a whole chunk (the p99 lever vs the time-slice's per-interval decode stall).
     * pf's FIRST chunk (pos 0) and FINAL tail (<= chunk) stay CLASSIC: the tail's
     * pulsar_session_sync carries the prefill->decode completion (kv-cache store,
     * gen_stream_begin), so this never reimplements that handoff. Reconciliation of
     * pf's bank is the exact recipe the decode lane uses (bank_state_restore +
     * note_committed_tokens).
     */
    void worker_mixed_batch_quantum(session_slot **dec, int n, session_slot *pf);
    /** Provision a bank in the shared pool (Tier-2). No GPU allocation happens here:
     * the whole n-bank pool was allocated and admitted ONCE at startup, so this is
     * pure host bookkeeping — find a free bank slot, install it, reset it to an
     * empty conversation (so gen_begin sees pos 0 / no stale prefix), and charge
     * the even-split per-bank marginal to the ledger. The only runtime pressure is
     * the demand-paged comp/index pages a bank touches as it fills; the belt-and-
     * suspenders MemAvailable floor still guards each provision. Returns NULL with
     * *refusal set on a full pool or a tight box (never on a create/admission
     * failure — there is no runtime create).
     */
    session_slot * provision_bank(provision_refusal *refusal);
    session_slot * provision_slot(int ctx, provision_refusal *refusal);
    /** Route the job to a slot. Preferences, in order:
     * 1. A live-tool-state continuation binds to the slot that owns its call
     * ids (waiting for it if busy — running it elsewhere could only 409).
     * A continuation whose prompt cannot fit its owner slot's context can
     * never run: it must not run elsewhere (the live tool state exists only
     * on the owner), and leaving it queued would wedge the FIFO forever
     * behind an unbindable head — so it is failed explicitly through
     * *reject_ctx with the same context_length_exceeded client error the
     * front door sends (http_server.cpp / request_exceeds_context; the front
     * door checks against slot 0's ctx and cannot see the owner's smaller
     * one).
     * 2. A free fitting slot whose live thinking binding byte-matches the
     * request's visible transcript is that conversation's warm continuation
     * and wins outright (thinking_live_binds_prompt): for thinking chats
     * the client replays visible content while the slot's frontier holds
     * the hidden reasoning, so the token common prefix understates
     * relatedness and must not out-vote the binding.
     * 3. Among free slots with enough context, the longest common token prefix
     * wins, keeping a client's follow-ups on their warm KV.
     * 4. A job whose best token match is TRIVIAL — header-deep only, against a
     * slot holding meaningful warm state past the match
     * (server_slot_match_is_trivial) — prefers a fresh lazily provisioned
     * slot over clobbering that conversation (budget permitting); with the
     * pool exhausted it falls back to the warmest free slot exactly like
     * the single-session server did. (Through v0.2.0 this gate required
     * common == 0, which rendered chat traffic can never produce — every
     * rendered prompt shares the template header, measured 4–9 common
     * tokens across distinct conversations — so sequential conversations
     * always clobbered slot 0 and the pool never provisioned; task #24
     * bounce repro, fixed in task #30.)
     * Returns NULL when the job must wait for a slot to free — except when
     * *reject_ctx is set nonzero (the owner slot's ctx_size), which means the
     * job can never run and must be failed, not left queued. *waiting_owner is
     * set when the NULL means "the continuation's owner slot is busy": eviction
     * cannot help that job, only the owner finishing can. *clobbers is set when
     * the returned slot would overwrite another conversation's warm KV — the
     * caller may prefer evict(LRU)+provision over that (increment 4). *refusal
     * reports why a fresh provisioning was refused (PROVISION_OK when none was
     * attempted or it succeeded) so the eviction path can act only on refusals
     * eviction relieves.
     */
    session_slot * choose_slot_for_job(job *j, int *reject_ctx, bool *waiting_owner, bool *clobbers, provision_refusal *refusal);
    /** const readers (take a const server *s in the C predecessor). */
    bool should_canonicalize_tool_checkpoint(const tool_calls *calls) const;
    int slot_common_prefix(const session_slot *sl, const pulsar_tokens *prompt) const;
    /** L115: the one prefix-reuse question, asked of a slot.  Callers that need
     * the request-side count (accounting, prefill bounds) read `prompt_cut`;
     * callers that need KV rows or a cut point read `live_cut`.  Across a seam
     * these differ, and keeping them in one struct is what stops a live-side
     * count from being used to index the request array.
     */
    void slot_prefix_match(const session_slot *sl, const pulsar_tokens *prompt,
                           pulsar_prefix_match *out) const;
    /** Context a request needs from a slot: prompt plus generation budget (plus a
     * small allowance for tool-error recovery injections), capped at the largest
     * (startup) slot so every request can always run on slot 0.
     */
    int job_needed_ctx(const job *j) const;
    int slot_frontier_pos(const session_slot *sl) const;
    /** Context a lazily provisioned slot would be created with for this job: the
     * secondary-slot default, raised to the job's need, capped at slot 0's ctx.
     * Shared by the provisioning path and the eviction could-it-help precheck so
     * they price the same session shape.
     */
    int provision_ctx_for_job(const job *j) const;
    /** @} */
};

/* Jobs are stack-owned by the client thread.  The worker signals completion
 * after the response has been written, so request data and the socket remain
 * valid without heap-allocating per-request job objects. */
struct job {
    int fd;
    request req;
    bool done;
    /** Set (under mu) by the client thread when its socket dies while the job
     * is still queued; the worker reaps the job pre-bind. The client thread
     * NEVER unlinks or frees — it stays parked on cv until the worker
     * signals done, preserving the worker-only pop/free invariant the
     * queued-job protect helpers depend on (see worker_try_bind). */
    bool cancelled;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    job *next;
    /** Which provisioning refusal this job has already been counted against, so
     * the /metrics counter records jobs blocked rather than bind retries (the
     * worker re-attempts the head every quantum). PROVISION_OK = not counted;
     * jobs are memset to zero at creation, so that is the natural initial
     * value. */
    provision_refusal refusal_counted;
};

typedef enum {
    KV_REASON_UNKNOWN   = PULSAR_KVSTORE_REASON_UNKNOWN,
    KV_REASON_COLD      = PULSAR_KVSTORE_REASON_COLD,
    KV_REASON_CONTINUED = PULSAR_KVSTORE_REASON_CONTINUED,
    KV_REASON_EVICT     = PULSAR_KVSTORE_REASON_EVICT,
    KV_REASON_SHUTDOWN  = PULSAR_KVSTORE_REASON_SHUTDOWN,
    KV_REASON_SYS_PREFIX = PULSAR_KVSTORE_REASON_SYS_PREFIX,
} kv_cache_reason;

typedef struct trace_cache_diag {
    bool valid;
    int old_pos;
    int prompt_len;
    int common;
    int start;
    int count;
    int live_id[TRACE_CACHE_WINDOW];
    int prompt_id[TRACE_CACHE_WINDOW];
} trace_cache_diag;

typedef struct server_prefill_progress {
    server *srv;
    session_slot *slot;  ///< slot whose session is prefilling (worker thread)
    req_kind kind;
    int prompt_tokens;
    int cached_tokens;
    char ctx[48];
    const char *phase;
    bool has_tools;
    bool responses_protocol;
    double t0;
    double last_t;
    int last_current;
    bool seen;
    /** SSE keepalive during long prefill: send HTTP/SSE headers ahead of
     * generation and emit a `:` comment line every few seconds so HTTP/TCP
     * idle timeouts on the client side don't close the connection while the
     * server is busy doing prefill. */
    int fd;
    bool stream;
    bool headers_sent;
    bool stream_failed;
    double last_keepalive;
} server_prefill_progress;

typedef struct thinking_state {
    bool inside;
    char tail[8];  ///< Long enough for "</think>".
    int tail_len;
    bool tail_ends_with(const char *s) const;  ///< was thinking_tail_ends_with
    void feed(const char *p, size_t len);  ///< was thinking_state_feed
} thinking_state;

/* Resumable per-job generation state machine (moved verbatim from
 * generate.cpp when the scheduler split into its own TU): server_jobs.cpp owns
 * the lifecycle, but the scheduler/worker (server_sched.cpp) steps jobs by
 * phase and drives the batched/fused decode lanes through the batch_*
 * fields, so the definition is shared here. */
typedef enum {
    GEN_PREFILL_COLD = 0, /* syncing the cold-store prefix, one chunk/quantum */
    GEN_PREFILL_MAIN,     /* syncing the effective prompt, one chunk/quantum */
    GEN_DECODE_INIT,      /* (re)initialize a decode attempt (old decode_again) */
    GEN_DECODE,           /* sampling loop, K tokens per quantum */
    GEN_FINISH,           /* parse, checkpoints, final response */
    GEN_DONE,
} gen_phase;

struct gen_state {
    job *j;
    gen_phase phase;

    /** prompt/cache resolution (owned by gen_begin, read by later phases) */
    char err[160];
    pulsar_tokens effective_prompt;
    const pulsar_tokens *prompt_for_sync;  ///< &j->req.prompt or &effective_prompt
    bool responses_protocol;
    bool responses_live_continuation;
    bool anthropic_live_continuation;
    bool thinking_live_continuation;
    char *disk_cache_path;
    int prompt_tokens;
    double t0;
    double first_token_t;  ///< wall time the first output token was produced (TTFT); 0 until set. Request-lifetime: survives decode_again so it reflects the genuinely first token emitted.
    uint64_t trace_id;
    char ctx_span[48];
    char req_flags[64];
    server_prefill_progress progress;  ///< stable address: callback userdata
    int cold_store_len;
    bool cold_store_is_anchor;  ///< cold_store_len is a chat_anchor_pos preamble cut (shared system prompt + tools, before the task message) -> stored as "sys-prefix" so eviction keeps the one file every new conversation can text-prefix restore from
    int suppressed_continued_last;
    pulsar_tokens cold_prefix;

    /* prefill quantum policy (see gen_prefill_cancel_cb) */
    uint32_t prefill_min_suffix;  ///< 0 = interrupting is never exact
    int prefill_chunks_done;  ///< chunks completed in the current sync call
    int prefill_last_current;
    int prefill_total;

    /** response identity + per-protocol stream projections; these live across
     * quanta AND across decode_again recovery attempts */
    char id[96];
    bool structured_stream;
    anthropic_stream anthropic_live;
    openai_stream openai_live;
    responses_stream responses_live;
    bool openai_live_chat;
    bool responses_live_chat;
    long responses_created_at;
    bool dsml_recovery_attempted;
    /** Server-executed web_search state (request lifetime).  completion_total
     * accumulates tokens across decode attempts so continued generations spend
     * one shared max_tokens budget; web_rounds_json holds prebuilt Anthropic
     * content blocks (thinking/text/server_tool_use/web_search_tool_result,
     * each ','-terminated) for completed rounds of non-streaming requests. */
    int completion_total;
    int web_search_uses;
    buf web_rounds_json;
    uint64_t rng;

    /** decode attempt state (reset by GEN_DECODE_INIT) */
    buf text;
    logprob_ledger logprobs;  ///< per-token target distributions; see the type
    size_t plain_stream_pos;
    size_t stop_scan_from;
    const char *finish;
    int completion;
    int max_tokens;
    bool saw_tool_start;
    bool saw_tool_end;
    bool saw_orphan_tool_end;
    size_t tool_scan_from;
    int next_tool_progress;
    int next_decode_log;
    double decode_t0;
    /* L119: request-scoped DSpark counters, accumulated by the spec-batched
     * lane per round (the ONLY decode lane that speculates post-L118). The
     * old design diffed the SHARED pool session's cumulative counters, which
     * per-round bank save/restore rolls and concurrent banks mix — the
     * response reported impossible values (tokens/step 27.4). Filled where
     * the truth lives: the lane knows each round's drafted rows and
     * committed tokens for exactly this slot. */
    uint64_t req_spec_draft;  ///< draft tokens proposed+verified, this request
    uint64_t req_spec_accepted;  ///< draft tokens accepted, this request
    uint64_t req_spec_rounds;  ///< spec rounds (incl. base-only quenched), this request
    uint64_t req_spec_gen;  ///< tokens emitted by spec rounds, this request
    double last_decode_log_t;
    int last_decode_log_completion;
    thinking_state thinking;
    bool thinking_gates_tool_markers;
    bool tool_scan_waiting_for_think_close;
    size_t think_recovery_scan_from;
    bool dspark_spec_enabled;
    dsml_decode_tracker dsml_tracker;

    /** Tier-2 batched-decode lane state (worker_batched_decode_quantum). A slot
     * becomes batch_active when it joins the shared multiseq lane; it stays
     * there until it finishes (no mid-conversation batched->classic switch, so
     * no stale-logits hazard). batch_feed_token is the next token to commit at
     * batch_feed_pos (the bank's KV frontier); batch_pending holds the tokens
     * committed via multiseq since the bank's last host-checkpoint save, to be
     * reconciled onto the checkpoint when the slot returns to a classic op
     * (finish/store). */
    bool batch_active;
    bool batch_feed_valid;
    int  batch_feed_token;
    int  batch_feed_pos;
    pulsar_tokens batch_pending;
    /** plan-34 inc 5: this prefill slot is not fusable (a fused step rejected its
     * run as not-position-true, e.g. a cache-warm resume); route it CLASSIC. Set
     * once by the fused quantum on giveup; the classic path handles it correctly. */
    bool no_fuse;

    /** deferred, non-blocking client writes (installed for send_all) */
    slot_writer writer;
};

/* Emit the surface-appropriate keepalive if this slot has gone quiet (see
 * PULSAR_SERVER_HEARTBEAT_MS). Returns true if one was written. Non-blocking
 * and idempotent-by-clock, so it is safe to call every worker pass. */
bool gen_stream_heartbeat(gen_state *g);

typedef struct {
    char method[8];
    char path[256];
    char *body;
    size_t body_len;
} http_request;

typedef struct {
    server *srv;
    int fd;
} client_arg;

typedef struct {
    pulsar_engine_options engine;
    const char *host;
    int port;
    int ctx_size;
    int default_tokens;
    const char *trace_path;
    const char *kv_disk_dir;
    bool kv_disk_disable;
    uint64_t kv_disk_space_mb;
    kv_cache_options kv_cache;
    /** Base URL of the SearXNG-compatible backend for the Anthropic web_search
     * server tool (e.g. http://searxng.defense.lan:8888). NULL disables the
     * feature: web_search tool entries are then dropped at parse so the model
     * never emits un-executable calls. */
    const char *web_search_url;
} server_config;

/* ---- shared globals ---- */

extern volatile sig_atomic_t g_stop_requested;
extern volatile sig_atomic_t g_listen_fd;
extern const dsml_syntax dsml_syntaxes[3];

/* ---- shared functions ---- */

void stop_signal_handler(int sig);
void die(const char *msg);
void *server_xmalloc(size_t n);
void *server_xrealloc(void *p, size_t n);
char *xstrdup(const char *s);
bool random_bytes(void *dst, size_t len);
void pulsar_die(const char *msg);  ///< engine util.cpp; aborts the process
char *xstrndup(const char *s, size_t n);
void buf_append(buf *b, const void *p, size_t n);
void buf_putc(buf *b, char c);
void buf_puts(buf *b, const char *s);
void buf_printf(buf *b, const char *fmt, ...);
char *buf_take(buf *b);
void buf_free(buf *b);
void json_ws(const char **p);
bool json_lit(const char **p, const char *lit);
bool json_string(const char **p, char **out);
bool json_string_n(const char **p, char **out, size_t *out_len);
size_t trim_truncated_dsml_close_tail(const char *raw, size_t start, size_t len);
bool json_number(const char **p, double *out);
bool json_int(const char **p, int *out);
bool json_bool(const char **p, bool *out);
bool json_skip_value(const char **p);
bool json_raw_value(const char **p, char **out);
char *json_minify_raw_value(const char *json);
bool json_content(const char **p, char **out);
void random_tool_id(char *dst, size_t dstlen, api_style api);
void tool_calls_free(tool_calls *calls);
void tool_calls_push(tool_calls *calls, tool_call tc);
void chat_msg_add_tool_call_id(chat_msg *m, const char *id);
void chat_msgs_free(chat_msgs *msgs);
void chat_msgs_push(chat_msgs *msgs, chat_msg msg);
void tool_schema_orders_free(tool_schema_orders *orders);
const tool_schema_order *tool_schema_orders_find(const tool_schema_orders *orders, const char *name);
void request_init(request *r, req_kind kind, int max_tokens);
void request_free(request *r);
pulsar_think_mode think_mode_from_enabled(bool enabled, pulsar_think_mode effort);
bool parse_reasoning_effort_name(const char *s, pulsar_think_mode *out);
bool parse_reasoning_effort_value(const char **p, pulsar_think_mode *out);
bool parse_thinking_control_value(const char **p, bool *thinking_enabled);
bool parse_output_config_effort(const char **p, pulsar_think_mode *effort);
bool model_alias_disables_thinking(const char *model);
bool model_alias_enables_thinking(const char *model);
const char *server_model_id_from_engine(pulsar_engine *engine);
/* Advertised model id ("id"/"root"/metrics): the built-in id derived from the
 * loaded GGUF shape. */
const char *server_served_model_id(const server *s);
/* Advertised display name ("name"): the built-in shape name. */
const char *server_served_model_name(const server *s);
void stop_list_clear(stop_list *stops);
void stop_list_push(stop_list *stops, char *s);
bool parse_stop(const char **p, stop_list *out);
bool stop_list_find_from(const stop_list *stops, const char *text,
                                size_t from, size_t *pos, size_t *len);
size_t stop_list_stream_safe_len(const stop_list *stops, size_t text_len);
size_t utf8_stream_safe_len(const char *s, size_t start,
                                   size_t limit, bool final);
bool parse_stream_options(const char **p, bool *include_usage);
void tool_schema_orders_add_json(tool_schema_orders *orders, const char *json);
bool parse_tools_value(const char **p, char **out, tool_schema_orders *orders,
                       bool web_search_enabled, int *web_search_max_uses);
bool parse_messages(const char **p, chat_msgs *msgs);
bool parse_anthropic_messages(const char **p, chat_msgs *msgs);
bool parse_anthropic_system(const char **p, char **out);
void append_tool_result_text(buf *b, const char *s);
bool append_dsml_arguments_from_json(buf *b, const char *json, const tool_schema_order *order);
void append_json_object_or_empty(buf *b, const char *json);
void append_dsml_tool_calls_text(buf *b, const tool_calls *calls);
bool chat_history_uses_tool_context(const chat_msgs *msgs,
                                           const char *tool_schemas);
bool chat_history_preserves_reasoning(const chat_msgs *msgs,
                                             const char *tool_schemas);
char *render_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas,
                                     const tool_schema_orders *tool_orders,
                                     pulsar_think_mode think_mode);
void responses_prepare_live_continuation(request *r,
                                                const chat_msgs *msgs);
void anthropic_prepare_live_continuation(request *r,
                                                const chat_msgs *msgs);
bool parse_chat_request(pulsar_engine *e, server *s, const char *body, int def_tokens,
                               int ctx_size, request *r, char *err, size_t errlen);
bool parse_anthropic_request(pulsar_engine *e, server *s, const char *body, int def_tokens,
                                    int ctx_size, request *r, char *err, size_t errlen);
bool parse_responses_input(const char **p, chat_msgs *msgs,
                                  buf *loaded_tool_schemas,
                                  tool_schema_orders *orders);
bool parse_responses_request(pulsar_engine *e, server *s, const char *body, int def_tokens,
                                    int ctx_size, request *r, char *err, size_t errlen);
bool parse_completion_request(pulsar_engine *e, const char *body, int def_tokens,
                                     int ctx_size, request *r, char *err, size_t errlen);
bool send_all(int fd, const void *p, size_t n);
void json_escape(buf *b, const char *s);
void json_escape_n(buf *b, const char *s, size_t n);
void json_escape_fragment_n(buf *b, const char *s, size_t n);
const char *find_any_tool_start(const char *s);
const char *find_any_tool_end(const char *s);
bool complete_tool_call_inside_thinking(const char *text, size_t len, size_t *scan_from);
void observe_tool_markers(const char *scan, bool *saw_start,
                                 bool *saw_end, bool *orphan_end);
size_t trim_tool_separator_ws(const char *raw, size_t start, size_t limit);
const char *find_last_substr(const char *s, const char *needle);
char *dsml_unescape_text(const char *s);
char *dsml_attr(const char *tag, const char *name);
bool parse_generated_message_ex(const char *text, bool require_thinking_closed,
                                       char **content_out, char **reasoning_out,
                                       tool_calls *calls);
bool try_repair_dsml(const char *s, size_t len, buf *out);
bool parse_generated_message_for_response(const char *text,
                                                 bool has_tools,
                                                 bool saw_tool_start,
                                                 bool require_thinking_closed,
                                                 const char **finish_io,
                                                 char *err,
                                                 size_t errlen,
                                                 char **content_out,
                                                 char **reasoning_out,
                                                 tool_calls *calls,
                                                 bool *recovered_out);
void append_json_object_string(buf *b, const char *json);
void append_tool_calls_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                   const tool_schema_orders *orders);
void append_tool_call_deltas_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                         const tool_schema_orders *orders);
bool http_response(int fd, int code, const char *type, const char *body);
bool http_error(int fd, int code, const char *msg);
bool http_error_anthropic(int fd, int code, const char *msg);
void request_forced_tool_seed(const request *r, buf *out);
void request_apply_forced_tool_prefill(request *r);
bool request_exceeds_context(const request *r, int ctx_size);
bool gen_client_disconnected(int fd);
bool http_error_context_length_exceeded(int fd,
                                               const request *r,
                                               int n_prompt_tokens,
                                               int ctx_size);
bool sse_headers(int fd);
bool sse_error_event(int fd, const request *r, const char *msg);
bool sse_chunk(int fd, const request *r, const char *id, const char *text, const char *finish);
int clamp_usage_tokens(int value, int max);
/* The one shared sampling-knob parser every protocol surface routes through
 * (temperature/top_p/min_p/top_k/seed). Non-static so the test suite can pin
 * its contract -- /responses silently dropping seed is the bug class. */
int parse_sampling_key(const char *key, const char **p, request *r);
void resolve_cache_split(int *cache_read, int *cache_write, int total);
void append_openai_usage_json(buf *b, const request *r,
                                     int prompt_tokens, int completion_tokens);
/* Emit the additive ",\"timings\":{...}" fragment (leading comma included) from
 * r->timings, or nothing when r->timings.valid is false. Rates are derived here
 * with guarded divisions; a zero denominator omits that rate. */
void append_openai_timings_json(buf *b, const request *r);
bool sse_done(int fd, const request *r, const char *id,
                     int prompt_tokens, int completion_tokens);
bool sse_chat_finish(int fd, const request *r, const char *id, const char *content,
                            const char *reasoning, const tool_calls *calls, const char *finish,
                            int prompt_tokens, int completion_tokens);
void openai_stream_start(const request *r, openai_stream *st);
void openai_stream_free(openai_stream *st);
bool raw_full_lit(const char *raw, size_t raw_len, size_t pos, const char *lit);
bool raw_partial_any(const char *raw, size_t raw_len, size_t pos,
                            const char *a, const char *b);
const char *find_lit_bounded(const char *s, size_t n, const char *lit);
dsml_decode_state dsml_decode_state_for_text(const char *raw, size_t raw_len);
bool dsml_decode_state_is_tool(dsml_decode_state state);
bool dsml_decode_state_uses_payload_sampling(dsml_decode_state state);
void dsml_decode_tracker_init(dsml_decode_tracker *dt);
void dsml_decode_tracker_update(dsml_decode_tracker *dt,
                                       const char *raw, size_t raw_len);
size_t tool_param_value_stream_safe_len(const char *raw, size_t start,
                                               size_t raw_len, const char *param_end,
                                               bool is_string);
bool openai_sse_stream_update(int fd, server *s, const request *r, const char *id,
                                     openai_stream *st,
                                     const char *raw, size_t raw_len,
                                     bool final);
bool openai_sse_finish_live(int fd, server *s, const request *r, const char *id,
                                   openai_stream *st, const char *raw,
                                   size_t raw_len, const tool_calls *calls,
                                   const char *finish, int prompt_tokens,
                                   int completion_tokens);
bool request_uses_openai_live_stream(const request *r);
bool request_uses_responses_live_stream(const request *r);
bool request_uses_structured_stream(const request *r);
void responses_stream_init(const request *r, responses_stream *st);
void responses_stream_free(responses_stream *st);
bool responses_sse_created(int fd, const request *r, responses_stream *st,
                                  long created_at);
void responses_append_function_call_item(buf *b, const tool_call *tc,
                                                const responses_tool_item *item,
                                                const char *item_status,
                                                bool with_args,
                                                const tool_schema_orders *orders);
bool responses_sse_completed(int fd, const request *r,
                                    responses_stream *st,
                                    const tool_calls *calls,
                                    const responses_tool_item *tool_items,
                                    const char *finish,
                                    int prompt_tokens, int completion_tokens,
                                    long created_at);
bool responses_sse_stream_update(int fd, const request *r,
                                        responses_stream *st,
                                        const char *raw, size_t raw_len,
                                        bool final);
bool responses_sse_finish_live(int fd, const request *r,
                                      responses_stream *st,
                                      const char *raw, size_t raw_len,
                                      const char *recovered_content,
                                      const tool_calls *calls,
                                      const char *finish,
                                      int prompt_tokens, int completion_tokens,
                                      long created_at);
bool responses_final_response(int fd,
                                     const request *r, const char *id,
                                     const char *text, const char *reasoning,
                                     const tool_calls *calls, const char *finish,
                                     int prompt_tokens, int completion_tokens);
bool final_response(int fd,
                           const request *r, const char *id, const char *text,
                           const char *reasoning, const tool_calls *calls, const char *finish,
                           int prompt_tokens, int completion_tokens,
                           const logprob_ledger *lp);
void append_anthropic_content(buf *b, const char *text, const char *reasoning,
                                     const tool_calls *calls, const char *id_prefix,
                                     const tool_schema_orders *orders,
                                     const char *prior_blocks_json);
bool anthropic_final_response(int fd,
                                     const request *r, const char *id, const char *text,
                                     const char *reasoning, const tool_calls *calls, const char *finish,
                                     int prompt_tokens, int completion_tokens,
                                     const char *prior_blocks_json);
bool anthropic_sse_start_live(int fd, const request *r, const char *id,
                                     int prompt_tokens, anthropic_stream *st);
void anthropic_stream_free(anthropic_stream *st);
size_t text_stream_safe_limit(const char *raw, size_t start,
                                     size_t raw_len, bool has_tools,
                                     bool final);
bool anthropic_sse_stream_update(int fd, server *s, const request *r, const char *id,
                                        anthropic_stream *st,
                                        const char *raw, size_t raw_len,
                                        bool final);
bool anthropic_sse_finish_live(int fd, server *s, const request *r, const char *id,
                                      anthropic_stream *st, const char *raw,
                                      size_t raw_len, const tool_calls *calls,
                                      const char *finish, int completion_tokens);
double server_now_sec(void);
void server_log(pulsar_log_type type, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
int tool_memory_max_entries(const tool_memory *m);
tool_memory_block *tool_memory_find_block_locked(tool_memory *m,
                                                        const char *dsml,
                                                        size_t len);
void tool_memory_free(tool_memory *m);
void live_tool_state_free(live_tool_state *st);
void visible_live_free(visible_live_state *st);
/* Live protocol bindings are per-slot (they describe one session's sampled
 * frontier); has_call_id scans every provisioned slot because request parsing
 * runs before the job is bound to a slot. */
/* Slots whose live binding contains all of the request's continuation ids
 * (worker thread; used to route a continuation to the session that owns it). */
void apply_openai_stream_tool_ids(tool_calls *calls,
                                         const openai_stream *st);
/* web_search.cpp — Anthropic web_search server tool (SearXNG backend). */
#define WEB_SEARCH_DEFAULT_MAX_USES 8
bool web_search_tool_entry(const char *raw_tool_json, int *max_uses);
const char *web_search_schema_line(void);
char *web_search_query_from_arguments(const char *arguments_json);
bool web_search_run(const char *base_url, const char *query,
                    buf *model_text, buf *client_content_json,
                    char *logerr, size_t errlen);
void web_search_run_exhausted(buf *model_text, buf *client_content_json);
char *web_search_rebuild_result_text(const char *content_json);

/* Streamed emission of one completed web_search round: closes any open block,
 * emits the web_search_tool_result block, and rewinds the text-relative stream
 * state so the continued decode attempt streams from a fresh g->text. */
bool anthropic_sse_web_search_result_live(int fd, anthropic_stream *st,
                                          const char *tool_use_id,
                                          const char *content_json);
void anthropic_sse_round_reset(anthropic_stream *st, bool thinking_enabled);

void apply_anthropic_stream_tool_ids(tool_calls *calls,
                                            const anthropic_stream *st);
kv_cache_options kv_cache_default_options(void);
void le_put32(uint8_t *p, uint32_t v);
void sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
bool id_list_contains(const stop_list *ids, const char *id);
void id_list_push_unique(stop_list *ids, const char *id);
void id_list_free(stop_list *ids);
void collect_tool_call_ids(const chat_msgs *msgs, stop_list *ids);
char *path_join(const char *dir, const char *name);
void kv_fill_header(uint8_t h[KV_CACHE_FIXED_HEADER], uint8_t quant_bits,
                           uint8_t reason, uint8_t ext_flags,
                           uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                           uint64_t created_at, uint64_t last_used,
                           uint64_t payload_bytes);
double kv_entry_eviction_score(const kv_entry *e, const pulsar_tokens *live,
                                      uint64_t now,
                                      const pulsar_kvstore_eviction_context *incoming);
void kv_cache_evict(kv_disk_cache *kc, const pulsar_tokens *live,
                           uint64_t extra_bytes,
                           const pulsar_kvstore_eviction_context *incoming);
bool kv_cache_open(kv_disk_cache *kc, const char *dir, uint64_t budget_mb,
                          bool reject_different_quant, kv_cache_options opt);
void kv_cache_close(kv_disk_cache *kc);
char *render_tokens_text(pulsar_engine *engine, const pulsar_tokens *tokens, size_t *out_len);
void tokens_copy_prefix(pulsar_tokens *dst, const pulsar_tokens *src, int n);
void build_prompt_from_exact_prefix_and_text_suffix(
        pulsar_engine *engine,
        const pulsar_tokens *exact_prefix,
        const char *suffix_text,
        pulsar_tokens *out);
int kv_cache_store_len(const kv_disk_cache *kc, int tokens);
int kv_cache_sys_prefix_cut(const kv_disk_cache *kc, int anchor);
int kv_cache_chat_anchor_pos(const kv_disk_cache *kc,
                                    const pulsar_tokens *prompt,
                                    int user_token_id,
                                    int assistant_token_id);
int kv_cache_continued_store_target(const kv_disk_cache *kc, int live_tokens);
bool kv_cache_file_size_fits(const kv_disk_cache *kc,
                                    uint64_t text_bytes,
                                    uint64_t payload_bytes,
                                    uint64_t tool_map_bytes,
                                    uint64_t *file_bytes_out,
                                    uint64_t *required_bytes_out);
/* Returns whether a checkpoint file was actually written — eviction uses this
 * for failure honesty (evict-without-snapshot falls back to client re-prefill;
 * older callers ignore the result as before). */
/* The continued-store frontier (lib field kc->continued_last_store_tokens) is
 * per-conversation state on a kvstore shared by every slot. Every
 * tracker-touching operation brackets itself with these on the single worker
 * thread: bind loads the acting slot's frontier into the shared struct, flush
 * writes it back (2026-07-14 review: without this, slot A's high-water mark
 * suppressed slot B's continued checkpoints, and a cold request on B reset
 * A's schedule). */
void kv_cache_note_store(kv_disk_cache *kc, int tokens);
int kv_cache_suppress_continued_store(kv_disk_cache *kc, int tokens);
void kv_cache_restore_suppressed_continued(kv_disk_cache *kc,
                                                  int old_tokens,
                                                  int suppressed_tokens);
int kv_cache_find_text_prefix(kv_disk_cache *kc, const char *prompt_text,
                                     int quant_bits, int ctx_size);
/* Routing probe: does this slot's live thinking binding mark it as the warm
 * continuation of req's visible transcript? Same guards as
 * thinking_live_visible_prefix_prompt but byte-prefix check only — no
 * tokenization, no effective-prompt build. Returns the matched visible-key
 * length (>0), or 0 for no match (defined in kv_cache.cpp; unit-tested in
 * cli_main.cpp). */
/* Trivial-match classifier for the router's choose-vs-provision decision
 * (defined in generate.cpp; unit-tested in cli_main.cpp). */
bool server_slot_match_is_trivial(int common, int slot_pos,
                                         int share_ceiling, int protect_floor);
/* Admission predicate (defined in cli_main.cpp; unit-tested there). */
bool server_kv_admits(uint64_t kv_budget_bytes,
                             uint64_t committed_bytes,
                             uint64_t incoming_bytes);
/* Live MemAvailable floor predicate: kernel-breathing-room backstop applied
 * at provisioning time on top of the ledger (defined in cli_main.cpp;
 * unit-tested there). avail == 0 (unreadable /proc/meminfo) fails closed. */
bool server_mem_floor_admits(uint64_t avail_bytes, uint64_t est_bytes);

/* MemAvailable from /proc/meminfo, in bytes (0 on parse failure — callers
 * fail closed).  Never called on a token/layer hot path (defined in
 * generate.cpp; also used by startup warmup/budget derivation in cli_main.cpp). */
uint64_t server_mem_available_bytes(void);
/* Log estimate-vs-actual for a freshly created session, warn loudly on >10%
 * drift (sizing code out of sync with the allocator), and return the value
 * the ledger must commit — the actual (defined in generate.cpp). */
uint64_t server_reconciled_session_cost(int slot_idx, int ctx,
                                               uint64_t est_bytes,
                                               uint64_t actual_bytes);
/* Eviction ledger release: subtract an evicted slot's committed bytes from
 * the admission ledger total, warning loudly (and clamping to 0, which fails
 * toward over-admission being caught by the MemAvailable floor rather than
 * leaking budget forever) if the pairing ever underflows (defined in
 * generate.cpp; unit-tested in cli_main.cpp). */
uint64_t server_ledger_release(uint64_t committed_total, uint64_t slot_cost);
/* Tier-2 bank-aware frontier position of `sl`, correct whether or not sl->bank
 * is the currently-installed bank of the shared pool session (a non-live bank
 * reads its saved host carry via pulsar_session_bank_pos; the live bank reads the
 * live checkpoint). In classic (non-pooled) mode == pulsar_session_pos(s->sess).
 * Worker-thread scheduling reads AND the client/worker tool-id lookups use this
 * instead of pulsar_session_pos so a non-live bank's frontier is never misread as
 * the pool's live cursor (defined in generate.cpp). */
/* Install `bank` on the shared pool session: lazily saves the outgoing bank's
 * carry, reloads a guard-spilled target from disk, and repoints the graph's
 * views.  Returns false WITHOUT installing on any failure — callers must fail
 * the request rather than run against a half-repointed view set. */
/* LRU eviction victim: least-recently-serviced idle provisioned slot,
 * tie-broken by smallest committed bytes; slot 0 pinned; protect[i] (may be
 * NULL) marks slots a queued live continuation still needs. Returns a slot
 * index or -1. Pure selection over host fields — never touches the session
 * (defined in generate.cpp; unit-tested in cli_main.cpp). */
int server_evict_pick_victim(const session_slot *slots, int n_slots,
                                    const bool *protect,
                                    bool allow_slot0 = false);
void trace_cache_capture(
        trace_cache_diag *d,
        const pulsar_tokens *live,
        const pulsar_tokens *prompt,
        int old_pos,
        int common);
const char *trace_cache_miss_reason(const trace_cache_diag *d);
void request_ctx_span(char *buf, size_t len, int cached, int prompt);
void log_flags(char *buf, size_t len, bool responses_protocol,
                      bool tools, bool thinking,
                      bool dsml_start, bool dsml_end);
void log_decode_progress(req_kind kind, int prompt_tokens, int completion,
                                bool responses_protocol,
                                bool tools, bool thinking,
                                bool dsml_start, bool dsml_end,
                                double decode_t0,
                                double *last_t, int *last_completion);
thinking_state thinking_state_from_prompt(const request *r);
char *build_web_search_result_suffix(const request *r,
                                     const thinking_state *thinking,
                                     const char *result_text);
char *build_invalid_dsml_tool_error_suffix(const request *r,
                                                  const thinking_state *thinking,
                                                  const char *detail);
bool should_remember_thinking_checkpoint(const request *r,
                                                const thinking_state *thinking,
                                                const char *finish);
char *build_tool_checkpoint_suffix(const request *r, const char *content,
                                          const char *reasoning, const tool_calls *calls);
char *build_responses_visible_assistant_suffix(const request *r,
                                                      const char *content,
                                                      const char *reasoning,
                                                      const tool_calls *calls);
char *build_toolless_thinking_visible_text(const request *r,
                                                  const char *content);
void *worker_main(void *arg);
/* Job-lifecycle entry points (server_jobs.cpp), driven by the scheduler/
 * worker (server_sched.cpp): bind/step/unbind plus the three per-token
 * helpers the batched and fused mixed-batch quanta share with the classic
 * decode loop. */
void gen_resolve_sampling_decode(const gen_state *g, float *temperature,
                                 int *top_k, float *top_p, float *min_p);
void gen_resolve_sampling(const request *req, float *temperature,
                          int *top_k, float *top_p, float *min_p);
/* OpenAI logprobs ledger (generate.cpp).  capture_* records the distribution a
 * token was DRAWN from; logprob_commit binds it to that token as it is emitted
 * and returns false if a capture was missing (the ledger then reports nothing
 * for the request rather than a distribution from the wrong position).
 * logprob_stream_ready reports how many entries a byte watermark releases. */
void logprob_ledger_reset(logprob_ledger *lg);
void logprob_ledger_free(logprob_ledger *lg);
void logprob_capture_session(logprob_ledger *lg, pulsar_session *sess, int token);
void logprob_capture_row(logprob_ledger *lg, const float *logits, int n_vocab, int token);
bool logprob_commit(logprob_ledger *lg, pulsar_engine *engine, int token,
                    const char *piece, size_t piece_len, size_t end_off);
int  logprob_stream_ready(const logprob_ledger *lg, size_t upto);
/* The OpenAI logprobs object for ledger entries [from, to), leading comma
 * included (append_openai_timings_json's convention); nothing when the client
 * did not ask for logprobs. */
void append_openai_logprobs_json(buf *b, const logprob_ledger *lg, int from, int to);
void append_model_json_values(buf *b, const char *id, const char *name,
                                     int ctx, int default_tokens);
void *client_main(void *arg);
int listen_on(const char *host, int port);
void configure_client_socket(int fd);
void set_client_socket_nonblocking(int fd);
void usage(FILE *fp, const char *topic);

/* ---- shared inline helpers ---- */


#endif /* PULSAR_SERVER_INTERNAL_H */
