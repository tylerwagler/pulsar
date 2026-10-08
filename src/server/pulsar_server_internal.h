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
#include "pulsar_kvtext.h"
#include "pulsar_dsml.h"
#include "pulsar_utf8.h"
/* The JSON scanner is shared with the engine's safetensors reader, so it lives
 * in src/lib and is declared there. */
#include "pulsar_json.h"
/* L251: the Qwen family's chat renderer, effort authority and output parser. */
#include "qwen_chat.h"

#include <new>
#include <string>
#include <unordered_map>

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

/* Long-lived /metrics/stream subscribers, capped separately from
 * PULSAR_SERVER_MAX_CLIENTS. A stream holds its connection (and its thread)
 * until the client goes away, so sharing the request budget would let a couple
 * of idle dashboards lock real requests out of the server. */
#define PULSAR_SERVER_MAX_STREAMS 8

/* Silence on the stream for this long means the connection is unverified, so
 * the pump sends an SSE comment. Comfortably inside the idle timeouts of the
 * proxies and NATs this is expected to run behind. */
#define PULSAR_METRICS_STREAM_KEEPALIVE_MS 15000

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
 * the whole C stack before the request is rejected.  The ceiling itself is
 * defined in pulsar_json.h, beside the scanner that enforces it. */
#ifndef JSON_MAX_NESTING
#define JSON_MAX_NESTING 256
#endif


/* The DSML tag literals and the syntax table live in src/lib/pulsar_dsml.h
 * (one authority for the server AND the agent, L184). */


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
 * L264: the disk KV cache is a content-addressed SEGMENT store
 * (src/lib/pulsar_segstore.h, src/server/kv_cache.cpp).  A segment is one grid
 * span of a conversation -- tokens [G_prev, G), their comp/index-K rows and the
 * grid checkpoint at G (pulsar_session_save_segment) -- plus the rendered text
 * of those tokens; its key is sha1(parent_key || text), so a chain's last key
 * names the whole text from position 0 and conversations sharing a prefix share
 * its segments.  A lookup returns the deepest chain whose text is a byte prefix
 * of the rendered request; loading it leaves the bank live at the chain's end.
 *
 * kv_cache_persist extends a bank's stored chain by the segments past its last
 * one (a write costs the new tokens, never the history), at prompt end, on
 * progress, at the system-prefix cut, on evict and on a Tier-2 spill -- always
 * from checkpoints the live bank already holds; the session is never rolled
 * back to build an entry.  The store is keyed by pulsar_segstore_identity
 * (model + routed-expert format), evicts least-recently-used LEAVES first, and
 * writes durably without leaving page cache (L261).
 *
 * The optional tool-id map rides as a segment TRAILER.  It is not model state,
 * but it is needed to render future client JSON back to the exact DSML the
 * model sampled; only mappings whose DSML block appears in the segment's text
 * are kept.
 */

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

/** The engine's public span type, under the server's name for it. */
typedef pulsar_text_span chat_text_span;

/** Growable byte buffer. The server's workhorse for accumulating response
 * text, JSON, and SSE payloads. */
typedef struct {
    char *ptr;   ///< bytes, owned
    size_t len;  ///< bytes used
    size_t cap;  ///< bytes allocated
    /** L223: the byte ranges of this buffer that came from CLIENT data.
     *
     * The renderer writes client text (message content, reasoning, tool
     * results, DSML argument values, the tool-schema blob) into the same string
     * as its own control markers, and the prompt is then tokenised as one
     * string.  pulsar_tokenize_rendered_chat matches a special-token spelling at
     * ANY position, so without this record a client typing `<|Assistant|>` or
     * `｜DSML｜` into a message gets a real control token -- the agent guards
     * against exactly that for user text, and the server did not.
     *
     * A range is recorded by buf_text_begin()/buf_text_end() around the writes
     * that copy client bytes, and the tokeniser skips special matching inside
     * one.  The ranges are in the FINAL string's coordinates: the only buffer
     * that moves bytes between buffers is the system region, which carries its
     * spans with buf_spans_carry(). */
    chat_text_span *spans;         ///< client-data ranges, ascending and disjoint, owned
    uint32_t n_spans;              ///< ranges recorded
    uint32_t cap_spans;            ///< ranges allocated
    uint32_t span_open;            ///< >0 while a range is being recorded (nesting depth)
    uint32_t span_lo;              ///< where the outermost open range started
} buf;

void buf_text_begin(buf *b);
void buf_text_end(buf *b);
/** Move `src`'s ranges into `dst`, shifted by `offset` (the number of bytes `src`
 * will occupy in front of them), and free `src`'s array.  Used when a region is
 * assembled in its own buffer and then concatenated. */
void buf_spans_carry(buf *dst, buf *src, size_t offset);
/** Append `text` (whose CLIENT-DATA ranges are `spans`, in `text`'s own
 * coordinates) and shift those ranges into `dst`.  Ranges past the appended
 * bytes are dropped, so the result stays in bounds and ascending. */
void buf_puts_spanned(buf *dst, const char *text, const pulsar_text_span *spans,
                      uint32_t n_spans);

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

/** One tool call, as the client sees it. Arguments stay as raw JSON TEXT
 * rather than a parsed structure -- reproducing the prefix later needs the
 * exact bytes, and a re-serialised parse is not byte-identical. */
typedef struct {
    char *id;         ///< call id the client echoes back with the result, owned
    char *name;       ///< tool name, owned
    char *arguments;  ///< arguments as raw JSON text, owned
} tool_call;

/** The tool calls parsed from one assistant turn, plus the bytes they came
 * from. */
typedef struct {
    tool_call *v;     ///< the calls, in emission order
    int len;          ///< calls present
    int cap;          ///< calls allocated
    /** The exact sampled DSML bytes these calls were parsed from, owned. Kept
     * because reproducing a prefix requires the bytes the model actually
     * produced, not a re-serialisation of the parse. */
    char *raw_dsml;
} tool_calls;

/** How a replayed conversation's tool calls were resolved.
 *
 * `missing_ids` is the one that matters operationally: an unresolved id means
 * the exact bytes are gone, the prefix cannot be reproduced, and the turn
 * cold-prefills. It is counted rather than swallowed so a slow turn can be
 * explained after the fact.
 */
typedef struct {
    int mem;          ///< calls resolved from in-memory tool memory
    int disk;         ///< calls resolved from a KV-file trailer
    int canonical;    ///< calls resolved by canonicalising the client's own text
    int missing_ids;  ///< calls that could not be resolved at all
} tool_replay_stats;

/** One declared tool: its names, and the order its properties were declared
 * in. See ::tool_schema_orders for why the order is kept. */
typedef struct {
    char *name;            ///< tool name as the model sees it, owned
    char *wire_name;       ///< name as the client sent it, owned; these can differ
    char *tool_namespace;  ///< namespace qualifying the tool, owned; NULL when none
    /** Distinguish the Responses hosted tool from a normal function that
     * happens to be named "tool_search". */
    bool responses_tool_search;
    char **prop;  ///< property names in DECLARED order, owned
    int len;      ///< properties present
    int cap;      ///< properties allocated
} tool_schema_order;

/** Declaration order for every tool in a request.
 *
 * Order is preserved because the rendered prompt must be byte-stable: JSON
 * objects are unordered, but a re-render that permutes properties produces
 * different bytes, and different bytes miss the prefix cache. */
typedef struct {
    tool_schema_order *v;  ///< one entry per declared tool
    int len;               ///< tools present
    int cap;               ///< tools allocated
} tool_schema_orders;

/** One inline image attached to a chat message: the ENCODED image FILE
 * (PNG or JPEG) exactly as the client's base64 data: URL carried it.  The
 * engine decodes; the server never converts pixels. */
typedef struct {
    uint8_t *bytes;  ///< the encoded image file, owned
    size_t   len;    ///< its length in bytes
} chat_image;

/** One message in a chat request, after parsing and before rendering. */
typedef struct {
    char *role;           ///< "system", "user", "assistant", or "tool", owned
    char *content;        ///< message text, owned
    /** Inline images in content order.  Each contributes one
     * PULSAR_IMAGE_PLACEHOLDER to `content` at the position its block occupied,
     * so the rendered prompt places the sentinel block where the client asked;
     * the two lists must stay the same length. */
    chat_image *images;   ///< owned, `images_len` entries
    int   images_len;     ///< images present
    int   images_cap;     ///< images allocated
    /** Byte offset in `content` of each image's placeholder, parallel to
     * `images` (same length, owned).  The renderer emits exactly these ranges
     * OUTSIDE the message's client-text span, which is how the tokeniser knows
     * to resolve them to the image token -- the surrounding client text keeps
     * its BPE-only span.  A placeholder occurrence the CLIENT typed is not in
     * this list and therefore stays ordinary text (the L223 invariant). */
    size_t *image_ph_off; ///< owned, `images_len` entries
    char *reasoning;      ///< the assistant's reasoning for this turn, owned; NULL when absent
    char *tool_call_id;   ///< for a tool-result message, the call it answers, owned
    char **tool_call_ids; ///< for a multi-result message, the calls it answers, owned
    int tool_call_ids_len;  ///< ids present
    int tool_call_ids_cap;  ///< ids allocated
    tool_calls calls;     ///< for an assistant message, the calls it made
    /** True when this system-role entry carries the request's top-level
     * system/instructions FIELD (the parser appends it to the array). Field
     * content always renders in the system region; inline role:system
     * MESSAGES follow the leading-run rule instead (L113) — position in the
     * array cannot distinguish the two, so this flag is the one authority. */
    bool system_field;
    /** L267: this entry continues the previous one's Anthropic message.  An Anthropic message is
     * read into parts in block order -- its text (role as sent) and each tool_result (role "tool",
     * tool_call_id, the result's own text and images) -- because a family template places tool
     * results itself; DeepSeek's folds them back into the one user turn they arrived in
     * (anthropic_fold_tool_results). */
    bool anthropic_continues;
} chat_msg;

/** A parsed conversation. */
typedef struct {
    chat_msg *v;  ///< the messages, in request order
    int len;      ///< messages present
    int cap;      ///< messages allocated
} chat_msgs;

/** A list of strings, used for stop sequences and for sets of tool-call ids.
 *
 * `max_len` is cached rather than recomputed: a stop scan must back up by the
 * longest entry to catch a sequence straddling a chunk boundary, and that
 * happens on every chunk. */
typedef struct {
    char **v;         ///< the strings, owned
    int len;          ///< entries present
    int cap;          ///< entries allocated
    size_t max_len;   ///< longest entry in bytes; the scan back-up distance
} stop_list;

/* Per-response timing metrics, surfaced in the additive "timings" object that
 * sits next to "usage" on the OpenAI chat/completions response (and the final
 * include_usage SSE chunk). Filled once by gen_step_finish from counters the
 * worker already keeps (g->t0/decode_t0/first_token_t, prompt/completion
 * counts, per-session DSpark deltas) — there is NO hot-path work here, and
 * these fields never influence sampling/rng/logits. All rates are derived in
 * the JSON emitter (guarded divisions), so a zero denominator omits a rate
 * rather than emitting NaN/inf. */
/** Per-response timings -- see the note above for the emission contract. */
typedef struct {
    bool valid;     ///< the timings below were populated; suppresses a half-filled object
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
/** Fixed-bucket Prometheus histogram instance -- counters only; see the note
 * above for why the bounds live elsewhere. */
typedef struct {
    uint64_t bucket[PULSAR_HIST_BUCKETS];  ///< non-cumulative counts; send_metrics accumulates on the way out
    uint64_t count;  ///< also serves as the +Inf bucket
    double   sum;    ///< sum of observed values, for the _sum series
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
/** One token and its logprob, carrying the token's RAW bytes.
 *
 * The bytes are kept rather than re-derived because a token's text is not
 * recoverable from the response stream: it can be partial UTF-8, or inside a
 * string the projection has rewritten. */
typedef struct {
    int    token;      ///< token id
    char  *piece;  ///< owned raw token bytes (may be partial UTF-8)
    size_t piece_len;  ///< length of `piece` in bytes
    float  logprob;    ///< the token's logprob
} logprob_token;

/** One committed logprob entry: the emitted token, where its bytes end in the
 * response, and the alternatives considered at that position. */
typedef struct {
    logprob_token tok;  ///< the token that was actually emitted
    size_t end_off;  ///< byte watermark: where this token's text ends in the response
    int    n_top;    ///< alternatives recorded in `top`
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
/** Per-token logprobs for one request, captured at the draw and consumed at
 * the emit.
 *
 * The split matters: the sampler has the distribution in hand and the emitter
 * does not, so the `pending_*` fields hold exactly one token's worth of data
 * between those two points. Exactly one token is ever in flight per slot,
 * which is why one pending slot suffices rather than a queue.
 */
typedef struct {
    bool enabled;     ///< the client asked for logprobs
    int  top_k;  ///< client's top_logprobs, 0..PULSAR_SERVER_MAX_TOP_LOGPROBS
    logprob_entry *v; ///< committed entries, one per emitted token
    int  len;         ///< entries present
    int  cap;         ///< entries allocated
    int  streamed;  ///< entries already written to an SSE chunk
    bool pending_valid;    ///< a drawn-but-not-yet-emitted token's data is held below
    float pending_logprob; ///< that token's logprob
    int  pending_n_top;    ///< alternatives captured for it
    pulsar_token_score pending_top[PULSAR_SERVER_MAX_TOP_LOGPROBS];  ///< those alternatives
} logprob_ledger;

/* ---- L272 P3: the model-family half of the server, ONE table per template family --------------------
 * (chat_family.cpp).  A request carries the loaded family's (request::family); every per-family branch
 * in the server reads one of these fields where it used to read a flag.  Step 1 of the server audit's
 * order: the table replaces the chat_v41 / chat_qwen flags; the output parser as a plugin and the
 * decode hook join it next (steps 2-3), the suffix hooks after (step 4). */
struct chat_conversation;
struct server;
struct request;
/** How a family's generated text is read. */
typedef enum {
    SERVER_PARSER_DSML = 0,   ///< DeepSeek's DSML machinery: the stream walk, the decode tracker, recovery, tool memory, the canonical rewrite, live tool state
    SERVER_PARSER_QWEN = 1,   ///< ONE qwen_output_parser per generation (gen_state::qwen); the turn ends at the family's stop token
} server_parser_kind;
typedef struct server_family_ops {
    const char *name;
    pulsar_chat_format format;
    /** DeepSeek's template variant -- V4.1 or V4 (0731): the DSML tag spelling, the effort line, how
     *  consecutive user-side messages merge, whether a mid-conversation system message is an in-place
     *  System token or a <system-reminder> note, which assistant turns replay reasoning.  Read by the
     *  DSML renderer and every KV-key suffix builder, so the live KV and the replay cannot disagree
     *  about which template produced the bytes (L218 s123).  Meaningless to another family. */
    bool v41;
    server_parser_kind parser;
    /** Thinking is ON (the shared rule, chat_family.cpp): the level the template has nearest the asked
     *  effort (a row of the one effort-name table, an integer, or nothing sent: the template's default),
     *  into r->think_mode and the family's own r->family_effort. */
    void (*effort)(pulsar_engine *e, const struct chat_effort_ask *a, struct request *r);
    /** Render the conversation into r->prompt_text / prompt_spans / prompt; refuse by name what the
     *  template cannot express (a forced tool call, a live tool continuation, ...). */
    bool (*render)(pulsar_engine *e, struct server *s, struct chat_conversation *c, struct request *r, char *err,
                   size_t errlen);
    /** The rendered prefix two UNRELATED prompts share by template construction, in tokens: the slot
     *  router's trivial-match header (server::slot_trivial_common_tokens). */
    int (*trivial_header_tokens)(pulsar_engine *e);
    /** How the family's generated text is read (steps 2-3). */
    const struct server_output_parser_ops *output;

    /* ---- step 4: the suffix hooks.  Each is OPTIONAL: a family without one does not have the feature
     *      built on it, and the server refuses or skips that feature by name (never a DeepSeek
     *      rendering on another family's KV).  Every hook renders exactly the bytes the family's full
     *      replay render produces for the same turn, so the live KV and the next prompt agree (L196). */
    /** The assistant turn as SAMPLED, appended after the prompt's generation prefix: `reasoning`
     *  (NULL = none replayed), the content, the calls, and the EOS only when the turn sampled one.
     *  The canonical tool-checkpoint rewrite and the Responses visible memory key on it. */
    char *(*assistant_turn_sampled)(const request *r, bool think, const char *reasoning, const char *content,
                                    const tool_calls *calls, chat_text_span **spans_out, uint32_t *n_spans_out);
    /** The tail after a tool-call turn: the turn's EOS, the new user-side messages from `start` (tool
     *  results, a mid-loop system message), the generation prefix -- the live tool continuation. */
    char *(*tool_result_tail)(const request *r, const chat_msgs *msgs, int start, chat_text_span **spans_out,
                              uint32_t *n_spans_out);
    /** A model-visible tool error (a malformed call) as ONE tool result appended mid-turn, with the
     *  system prompt reminded -- the parser's retry. */
    char *(*tool_error_suffix)(const request *r, const struct thinking_state *thinking, const char *detail,
                               chat_text_span **spans_out, uint32_t *n_spans_out);
    /** A forced tool_choice (required / a named function): the text the generated output is SEEDED with
     *  -- the family's spelling of "thinking skipped, a call opened" -- read by the parser before the
     *  model's first token. */
    void (*forced_call_seed)(const request *r, buf *out);
    /** The rendered prompt's rewrite for that forced call: on entry `*keep` is the prompt's length;
     *  the hook lowers it to drop a server-written opener the seed replaces and writes the bytes to
     *  append, so prompt + output = the family's render of the turn (request_apply_forced_tool_prefill
     *  does the bookkeeping).  With forced_call_seed, both or neither. */
    void (*forced_call_prefill)(const request *r, const char *prompt, size_t *keep, buf *append);
    /** L272: an UNNAMED forced call's seed ends where the function name's OPENER starts; the opener, the
     *  name and the closer (Qwen: "=" and ">"; DeepSeek: ` name="` and `">`, L284) are then sampled under a
     *  mask that keeps them a prefix of opener + a declared tool's name + closer, so "required" with several
     *  tools cannot name an undeclared one (Qwen sampled "ask_user").  The seed stops BEFORE the opener
     *  (token healing): Qwen's tokenizer merges "=" with a name's first piece ("=get"), so a prompt ending
     *  in a lone "=" is a state the model saw only before names it does not merge ("=", "convert"), and
     *  "required" chose convert_currency for every question.  The closer is the whole tag end for the same
     *  reason (DeepSeek's `">` is one token).  Every family has both. */
    const char *forced_name_open;
    const char *forced_name_close;
    /** Tool memory: the earliest complete tool-call block at or after `p` in a transcript's text
     *  (`*end` = one past it); NULL = none.  The block's bytes are the replay key. */
    const char *(*find_call_block)(const char *p, const char **end);
} server_family_ops;

/* ---- L272 P3 steps 2-3: the family's OUTPUT PARSER ----------------------------------------------
 * One reading of the generated text per family, behind this table: per emitted token it tracks the
 * decode-time state (the greedy region of a tool call, whether a closed call ends the turn) and
 * projects the released bytes into the request's protocol sink (chat_sink: text deltas, section ends,
 * the generic tool events); at the finish it gives the turn's final reading.  DeepSeek's is the
 * <think> / DSML machinery (parser_deepseek.cpp), Qwen's the incremental qwen_output_parser
 * (parser_qwen.cpp).  The server owns the stop-string scan, the plain (completion) stream, the
 * protocol finish and the response. */
struct gen_state;
struct session_slot;
/** The turn's final reading. */
typedef struct {
    char *content;          ///< owned; NULL = the raw text stands (completions)
    char *reasoning;        ///< owned, or NULL
    tool_calls calls;       ///< the turn's calls, ids assigned
    const char *finish;     ///< the finish label after the reading ("tool_calls", "length", "stop", "error", ...)
    bool parse_failed;      ///< a malformed call: the raw text was promoted to content
    /** The parser ran its model-visible retry (a tool error appended to the live session): the
     *  generation loops back to a fresh decode attempt instead of answering. */
    bool retry;
    /** Text the live stream did not send that the final reading returned to content (the Responses
     *  finish sends it as output_text); NULL = none. */
    const char *tail;
    size_t tail_len;
} server_turn;
typedef struct server_output_parser_ops {
    /** The parser's state for one decode attempt, after the protocol sink is set up (g->sink); NULL
     *  with `err` = the request fails. */
    void *(*create)(server *s, struct gen_state *g, char *err, size_t errlen);
    void (*destroy)(void *st);
    /** Optional.  A forced tool_choice prefill seeded g->text: read it as if the model had emitted it. */
    void (*seed)(void *st, struct gen_state *g);
    /** Per emitted token, after g->text grew: `upto` = the bytes the stop-string scan released, to be
     *  projected into the sink (final = false); at the finish, `final` flushes what the projection held.
     *  false = a client write failed. */
    bool (*feed)(void *st, server *s, struct gen_state *g, size_t upto, bool final);
    /** The decode sits on a tool call's STRUCTURE: greedy.  A call's payload -- a string-typed value, or a
     *  JSON string inside any other value -- is sampled (L284 P6: one rule, every family). */
    bool (*in_tool_call)(const void *st, const struct gen_state *g);
    /** The turn is complete before the stop token (DeepSeek: a closed DSML block). */
    bool (*turn_complete)(const void *st, const struct gen_state *g);
    /** Optional, diagnostics: a call has opened / closed in this attempt. */
    void (*tool_progress)(const void *st, bool *opened, bool *closed);
    /** The turn's final reading (repair, parse, ids, the family's own retry and memory).  false = a
     *  client write failed while finishing the stream. */
    bool (*finish)(void *st, server *s, struct session_slot *sl, struct gen_state *g, server_turn *out);
} server_output_parser_ops;
extern const server_output_parser_ops k_parser_deepseek;
extern const server_output_parser_ops k_parser_qwen;
/** The table for a chat format (every format has one), and for the loaded engine's. */
const server_family_ops *server_family_for_format(pulsar_chat_format fmt);
const server_family_ops *server_family_for_engine(const pulsar_engine *e);

/** One parsed request, normalised across the three supported APIs.
 *
 * `kind` and `api` are separate on purpose: the same logical operation (a chat
 * completion) arrives in three different wire shapes, and the response has to
 * go back in the shape it came from. Everything below is protocol-neutral;
 * `api` is what the emitters branch on.
 */
typedef struct request {
    req_kind kind;             ///< what the request asks for (completion, chat, embedding, ...)
    api_style api;             ///< which wire protocol it arrived on; the response must match
    pulsar_tokens prompt;      ///< the rendered prompt as tokens
    /** Images for pulsar_session_sync_mm(), in message/content order, with
     * `start_pos` already resolved by pulsar_expand_image_placeholders() at
     * parse time.  Empty (n_images == 0) for every text-only request, which
     * keeps pulsar_session_sync() as its exact path.  `bytes` are owned. */
    pulsar_image_ref *images;  ///< owned, `n_images` entries
    int n_images;              ///< images in the request
    char *model;               ///< model name to serve, owned
    bool model_from_request;   ///< the client named the model (vs the server default)
    stop_list stops;           ///< client-supplied stop sequences
    char *raw_body;            ///< the original request body, owned; kept for tracing and replay
    char *prompt_text;         ///< the rendered prompt as text, owned
    pulsar_text_span *prompt_spans;  ///< L223: prompt_text's client-data ranges, owned
    uint32_t prompt_n_spans;   ///< ranges in prompt_spans
    tool_schema_orders tool_orders;  ///< tool declaration order, preserved so re-renders stay byte-stable
    int max_tokens;      ///< generation cap
    int top_k;           ///< top-k sampling cutoff
    float temperature;   ///< sampling temperature
    float top_p;         ///< nucleus-sampling mass cutoff
    float min_p;         ///< floor on candidate probability, relative to the top token
    /** Presence flags: true iff the CLIENT sent the parameter in the request
     * body. request_init() fills the value fields with engine defaults at
     * parse time, so the values alone cannot distinguish "explicitly 1.0"
     * from "absent" — downstream policy (e.g. think-mode defaults in
     * generate.cpp) must consult these and default only what is absent.
     * Zeroed by request_init's memset; set only in api_parse.cpp. */
    bool has_temperature;  ///< the client sent `temperature`
    bool has_top_k;        ///< the client sent `top_k`
    bool has_top_p;        ///< the client sent `top_p`
    bool has_min_p;        ///< the client sent `min_p`
    /** OpenAI logprobs (chat completions only).  `logprobs` alone reports the
     * chosen token's logprob; `top_logprobs` additionally asks for that many
     * alternatives per position and is rejected without logprobs:true, per the
     * OpenAI contract. */
    bool logprobs;               ///< report the chosen token's logprob
    int  top_logprobs;           ///< also report this many alternatives per position
    uint64_t seed;               ///< sampling seed; 0 = unseeded
    bool stream;                 ///< deliver the response as SSE
    bool stream_include_usage;   ///< append a usage event to the stream
    int cache_read_tokens;       ///< prompt tokens served from cache (reported back as usage)
    int cache_write_tokens;      ///< prompt tokens written to cache (reported back as usage)
    req_timings timings;         ///< per-phase timings for this request
    pulsar_think_mode think_mode;///< whether the model may emit reasoning, and how it is surfaced
    bool has_tools;              ///< the request declared tools
    /** tool_choice="required" (OpenAI) / {"type":"any"|"tool"} (Anthropic):
     * force a tool call. The prompt is prefilled into an open DSML tool_calls
     * block (thinking skipped) and generate_job seeds the output with the
     * SAME opener so the model must complete an invoke. forced_tool_name is
     * set for Anthropic {"type":"tool","name":X}: the opener then includes
     * the named invoke so the model can only fill in the parameters. */
    bool force_tool_call;      ///< the model must emit a tool call this turn
    char *forced_tool_name;    ///< force this specific tool; NULL = any, owned
    /** The replayed prompt still contains the model's prior reasoning. When
     * false the reasoning was stripped by the client, and the server must not
     * assume the live KV and the replayed text agree about it. */
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
    bool responses_requires_live_tool_state;  ///< tool-output-only: needs the live call_id binding or a full replay
    bool responses_requires_live_reasoning;   ///< reasoning-mode replay: needs the prior reasoning item to be live
    stop_list responses_live_call_ids;        ///< call_ids this request's tool outputs refer to
    char *responses_live_suffix_text;         ///< the new suffix to append when the live prefix is authoritative, owned
    pulsar_text_span *responses_live_suffix_spans;  ///< its CLIENT-DATA ranges (L223), owned
    uint32_t responses_live_suffix_n_spans;   ///< ranges in responses_live_suffix_spans
    bool anthropic_requires_live_tool_state;  ///< Anthropic equivalent of responses_requires_live_tool_state
    stop_list anthropic_live_call_ids;        ///< Anthropic tool_use ids this request refers to
    char *anthropic_live_suffix_text;         ///< Anthropic new-suffix text, owned
    pulsar_text_span *anthropic_live_suffix_spans;  ///< its CLIENT-DATA ranges (L223), owned
    uint32_t anthropic_live_suffix_n_spans;   ///< ranges in anthropic_live_suffix_spans
    tool_replay_stats tool_replay;            ///< what the replay matched, for logging and metrics
    /** L272 P3: the loaded model's template family -- the server's per-family table (server_family_ops,
     *  chat_family.cpp).  Every per-family branch reads one of its fields; a parser that holds the engine
     *  sets it from the engine (server_family_for_engine), request_init defaults it to the compile-time
     *  profile's (DeepSeek V4.1) so hand-built requests keep that profile's bytes.  Never NULL. */
    const struct server_family_ops *family;
    /** The family's own resolved thinking control, when its template has one beyond think_mode (Qwen's
     *  qwen_effort); written by family->resolve, read by family->render. */
    int family_effort;
    /** L251: the request's tools array as the client sent it (JSON text), for
     * the Qwen output parser's argument typing; owned, NULL = no tools. */
    char *qwen_tools_json;
} request;

/** One key/value pair from a parsed JSON object. */
typedef struct {
    char *key;       ///< the key, owned
    char *value;     ///< the value as text, owned
    /** The value was written as a quoted string -- the only way to tell the
     * string "true" from the boolean true once both are held as text. */
    bool is_string;
    /** This occurrence has been consumed. Lookups take the next UNUSED match,
     * so a duplicate key resolves to its occurrences in order rather than
     * collapsing to one. */
    bool used;
} json_arg;

/** A parsed JSON object, kept as an ordered list rather than a map -- order is
 * preserved for byte-stable re-rendering, and duplicate keys are retained. */
typedef struct {
    json_arg *v;  ///< the pairs, in document order
    int len;      ///< pairs present
    int cap;      ///< pairs allocated
} json_args;


typedef enum {
    DSML_TOOL_BETWEEN_INVOKES,
    DSML_TOOL_BETWEEN_PARAMS,
    DSML_TOOL_PARAM_VALUE,
    DSML_TOOL_DONE,
    DSML_TOOL_ERROR,
} dsml_tool_stream_state;

/* Shared states for the DSML stream projection.  The model still samples
 * DSML; these states only translate already-sampled bytes into OpenAI /
 * Anthropic wire events while final parsing remains authoritative. */
/** Projects a streaming DSML tool call into a protocol's wire events.
 *
 * A projection, not a parser: it translates already-sampled bytes into wire
 * events so the client sees the call forming, while the FINAL parse of the
 * complete text stays authoritative. The two can disagree mid-stream -- an
 * incomplete marker, a malformed block -- and when they do, the final parse
 * wins.
 *
 * ONE state machine (dsml_tool_stream_update / _finalize, genmsg.cpp) drives
 * both OpenAI tool_call deltas and Anthropic tool_use blocks; what differs
 * per protocol -- the call header, the argument-fragment event, the
 * block/stop lifecycle -- is the protocol's ::sink_tool_ops.  Before
 * L184 the machine existed twice, verbatim, and `"stream": true` could
 * return a different tool-call reading than the final parse on one protocol
 * but not the other. */
typedef struct dsml_tool_stream dsml_tool_stream;
struct dsml_tool_stream {
    dsml_tool_stream_state state;  ///< where the projection is in the block
    const pulsar_dsml_syntax *syn; ///< the marker literals this block opened with; borrowed
    size_t parse_pos;              ///< how far into the generated text the projection has consumed
    int index;                     ///< invocation index: OpenAI's tool_call index / Anthropic's block counter
    bool active;                   ///< a tool block is being projected
    bool emitted_any;              ///< at least one event has gone out
    bool args_open;                ///< the arguments JSON object is open and needs closing
    bool first_param;              ///< next parameter needs no leading comma
    bool param_is_string;          ///< current value is a JSON string, so it needs quoting and escaping
    char **ids;                    ///< generated call ids, one per invocation, owned
    int ids_cap;                   ///< entries allocated in `ids`
};

typedef struct chat_sink chat_sink;
/** A protocol's emitters for a family's live tool-call events (L272 P3: generic -- the DSML projection
 * and the Qwen output parser both drive them).  Each returns false on a client write failure, which
 * aborts the update. */
typedef struct {
    /** Call `index` opens, named `name`, with the id the client will see: OpenAI sends the tool_call
     * start delta, Anthropic starts the tool_use block.  The argument object's text follows in `args`. */
    bool (*begin)(chat_sink *k, int index, const char *id, const char *name);
    /** A fragment of call `index`'s argument object JSON text (keys, values, braces). */
    bool (*args)(chat_sink *k, int index, const char *text, size_t len);
    /** Call `index` is complete (its "}" went out): Anthropic stops the content block, OpenAI has
     * nothing to send. */
    bool (*end)(chat_sink *k, int index);
} sink_tool_ops;

/** L267: where DeepSeek's raw generated text is, for its one stream projection (deepseek_stream.cpp):
 * every protocol's live response is fed by the same walk over \<think\>, the answer and DSML blocks. */
typedef enum {
    DS_WALK_THINKING,   ///< inside the reasoning block
    DS_WALK_TEXT,       ///< the answer
    DS_WALK_TOOL,       ///< a DSML block, decoded into tool-call events as it arrives
    DS_WALK_SUPPRESS,   ///< nothing more goes out live (a block the finish sends, or the end)
} deepseek_walk_mode;

typedef struct {
    deepseek_walk_mode mode;
    size_t emit_pos;             ///< bytes of generated text already turned into events
    bool checked_think_prefix;   ///< the leading `<think>` check has been done once
    /** Thinking+tools: hold tentative answer text after the first \</think\>
     * until a tool marker, stream end, or a SECOND close proves whether it
     * is answer text or another reasoning pass (upstream ds4 fe2d3b0). */
    bool guard_second_reasoning;
    dsml_tool_stream tool;       ///< the DSML block being decoded (DS_WALK_TOOL)
} deepseek_stream_walk;

/** L267: one protocol's live response, as a family's output parser drives it: text as it is released,
 * the end of a reasoning or answer section, and -- on protocols that stream calls as they decode --
 * DSML tool-call events.  The protocol owns the wire shape; the family owns where the text is.  Built by
 * openai_sink_init / anthropic_sink_init / responses_sink_init. */
struct chat_sink {
    int fd;
    server *s;
    const request *r;
    const char *id;
    void *st;              ///< the protocol's stream state
    /** Reasoning or answer text; `release_upto` is the byte offset in the generation the text ends at
     * (OpenAI's logprob entries ride with the delta that releases their bytes). */
    bool (*text)(chat_sink *k, bool reasoning, const char *text, size_t len, size_t release_upto);
    /** The current section ended; `think_closed`: at the model's own \</think\>. */
    bool (*end)(chat_sink *k, bool think_closed);
    /** Live tool-call events; NULL: calls go out with the finish. */
    const sink_tool_ops *tool_ops;
    /** Also start streaming a block that first appears in the final flush (else the finish sends it). */
    bool tools_on_final;
};

/** OpenAI chat-completions SSE projection for one response. */
typedef struct {
    bool active;          ///< the stream has started
    bool sent_reasoning;  ///< a reasoning delta has been emitted
    int tools_streamed;   ///< tool calls whose start delta has gone out
    /** Borrowed (never owned): the request's logprob ledger, so the delta
     * emitters can attach the entries whose bytes the delta releases.  NULL
     * whenever the client did not ask for logprobs — openai_stream_start
     * zeroes it and only the job binds it. */
    logprob_ledger *lp;
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

/** Tracks where generation is inside a DSML block, at DECODE time.
 *
 * Runs during sampling rather than after it, so it can tell the sampler when a
 * structured region is in progress. The two JSON flags exist because a
 * parameter value can be JSON, and a marker-looking byte sequence inside a
 * JSON string is text, not a marker.
 */
typedef struct {
    dsml_track_mode mode;      ///< what the tracker is watching for
    dsml_decode_state decode;  ///< where it currently is
    const pulsar_dsml_syntax *syn;  ///< marker literals this block opened with; borrowed
    size_t pos;                ///< how far into the generated text it has consumed
    bool json_in_string;       ///< inside a JSON string, where markers do not apply
    bool json_escaped;         ///< previous byte was a backslash, so this one is literal
} dsml_decode_tracker;

/** L272 P3: DeepSeek's output-parser state for one decode attempt (parser_deepseek.cpp). */
typedef struct {
    deepseek_stream_walk walk;        ///< the live projection into the sink (walk_on: a streamed chat)
    bool walk_on;
    dsml_decode_tracker tracker;      ///< decode-time DSML marker tracking: the greedy region of a call
    bool saw_tool_start;              ///< a tool-call opening marker has appeared
    bool saw_tool_end;                ///< a tool-call closing marker has appeared (the turn is complete)
    /** A closing tool marker arrived with no opening one. Logged once per
     * request rather than per token -- the flag exists to keep a model that
     * emits the marker repeatedly from flooding the log. */
    bool saw_orphan_tool_end;
    size_t tool_scan_from;            ///< where the tool-marker scan resumes (a marker can straddle a piece)
    int next_tool_progress;           ///< token count at which to emit the next tool-progress event
    /** Tool markers inside a reasoning block are NOT tool calls -- the model is
     * thinking about calling something. True when the request has thinking
     * enabled, so marker scanning must wait for the block to close. */
    bool thinking_gates_tool_markers;
    bool tool_scan_waiting_for_think_close;  ///< a marker was seen inside reasoning; scan resumes after the block
    size_t think_recovery_scan_from;  ///< where to resume scanning after a malformed reasoning block
} deepseek_parser;

/** /v1/responses SSE projection for one response.
 *
 * The Responses protocol is ITEM-structured rather than delta-structured: a
 * reasoning item and a message item are opened, filled, and closed, each with
 * a stable id and output_index. Most of the booleans here exist because those
 * open/close events must be emitted exactly once and in order, even when the
 * generation is interrupted, recovered, or resumed on a later quantum.
 */
typedef struct {
    bool active;                 ///< the stream has started
    bool reasoning_item_opened;   ///< the reasoning item's added event has gone out
    bool reasoning_item_closed;   ///< its done event has gone out
    bool reasoning_summary_started;  ///< a reasoning_summary part has been opened
    /** The reasoning block ended because the model closed it, not because the
     * stream was cut short. Distinguishes a complete reasoning item from one
     * truncated by a stop or an error, which must be closed differently. */
    bool reasoning_closed_naturally;
    bool message_item_opened;     ///< the assistant message item's added event has gone out
    bool message_text_part_open;  ///< a text part inside that item is open
    bool message_item_closed;     ///< the message item's done event has gone out
    bool reasoning_emitted_any;   ///< reasoning produced at least one delta
    bool message_emitted_any;     ///< the message produced at least one delta
    buf reasoning_text;           ///< accumulated reasoning, for the item's final value
    buf message_text;             ///< accumulated message text, for the item's final value
    char response_id[40];         ///< id of the response object
    char reasoning_id[40];        ///< id of the reasoning item
    char message_id[40];          ///< id of the assistant message item
    int reasoning_index;  ///< output_index of the reasoning item (0 if present)
    int message_index;  ///< output_index of the assistant message item
    int next_output_index;  ///< monotonic counter for upcoming output items
    int sequence;  ///< monotonic per-event sequence_number Codex consumes
} responses_stream;

/* Item identity per tool call must be stable across added/done/completed. */
/** Identity of one tool call in a Responses stream, held stable across its
 * added / done / completed events. */
typedef struct {
    char fc_id[40];     ///< function-call item id
    char call_id[64];   ///< call id the client echoes back with the tool output
    bool is_custom;     ///< a custom tool rather than a function tool
    int output_index;   ///< the item's position in the output array
} responses_tool_item;

/** Which content block is currently open on the Anthropic wire. */
typedef enum {
    ANTH_BLOCK_NONE,      ///< no block open
    ANTH_BLOCK_THINKING,  ///< a thinking block is open
    ANTH_BLOCK_TEXT,      ///< a text block is open
    ANTH_BLOCK_TOOL,      ///< a tool_use block is open
} anthropic_block_type;

/* Anthropic streaming uses the same sampled DSML bytes that will later be
 * parsed and remembered for exact continuation.  This state is only a wire
 * projection: it turns an in-progress DSML block into content_block/tool_use
 * SSE events, and never rewrites the model-visible transcript or cache key. */
/** Anthropic messages SSE projection for one response. */
typedef struct {
    anthropic_block_type open_block;    ///< which content block is open
    int next_index;              ///< content-block index for the next block opened
    bool active;                 ///< the stream has started
    bool sent_thinking;  ///< a thinking delta has been emitted
    bool sent_text;      ///< a text delta has been emitted
    int tools_streamed;  ///< tool_use blocks streamed and stopped
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
/** Deferred, non-blocking writer for one bound job's socket. See the note
 * above for the contract. */
typedef struct {
    int fd;       ///< the job's non-blocking client socket
    buf pending;  ///< accepted-but-unsent bytes, in wire order
    size_t off;  ///< consumed prefix of pending
    long long stall_deadline_ms;  ///< 0 = disarmed (nothing pending)
    bool failed;  ///< the stream has failed; further writes are refused
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
/** A prefill chunk is done for the slot whose gen_state is `ud`: progress, SSE keepalive,
 * the prefill log line and the L114 counter (server_jobs.cpp). */
void gen_prefill_progress_cb(void *ud, const char *event, int current, int total);
bool slot_writer_flush(slot_writer *w);  ///< non-blocking best effort
/* True when this writer has an armed clock and has been silent >= interval. */
bool slot_writer_idle_for(const slot_writer *w, long long now_ms, long long interval_ms);
bool slot_writer_drain(slot_writer *w);  ///< blocking, stall-timeout bounded
void slot_writer_free(slot_writer *w);

/** L264 S4: the disk KV cache's placement policy. */
typedef struct {
    int min_tokens;          ///< a conversation shorter than this is not persisted, nor restored
    int sys_prefix_align;    ///< the sys-prefix cut lands on a multiple of this (a resume-grid multiple)
    int sys_prefix_margin;   ///< and this far below the chat anchor, clear of preamble jitter
} kv_cache_options;

typedef struct pulsar_segstore pulsar_segstore;
/** L264 S4: the disk KV cache -- the segment store (lib/pulsar_segstore.h). */
typedef struct {
    pulsar_segstore *st;     ///< NULL while disabled
    char *dir;
    bool enabled;
    kv_cache_options opt;
} kv_disk_cache;

/** Where a remembered tool entry came from. */
typedef enum {
    TOOL_MEMORY_RAM = 0,   ///< recorded live, this process
    TOOL_MEMORY_DISK = 1,  ///< restored from a KV-file trailer
} tool_memory_source;

typedef struct tool_memory_entry tool_memory_entry;

/** One remembered DSML tool-call block, shared by every call inside it.
 *
 * Blocks are refcounted rather than copied per call: one assistant turn can
 * contain several invocations, and all of them need the same exact bytes to
 * reproduce the prefix. */
typedef struct {
    char *dsml;    ///< the block's exact sampled bytes, owned
    size_t len;    ///< length in bytes
    size_t bytes;  ///< accounted size, charged against tool_memory::max_bytes
    int refs;      ///< entries referring to this block; freed at zero
    /** Visit stamp for one scan pass. Compared against tool_memory::scan_clock
     * so a scan can skip blocks it has already counted without allocating a
     * visited set. */
    uint64_t seen;
    tool_memory_entry *entries;  ///< list of entries in this block, via block_next
} tool_memory_block;

/** One remembered tool call, keyed by the id the client will echo back. */
struct tool_memory_entry {
    char *id;                    ///< the call id, owned; the lookup key
    tool_memory_block *block;    ///< block holding this call's bytes
    size_t bytes;                ///< accounted size of this entry
    uint64_t stamp;              ///< tool_memory::clock value at last use; the LRU key
    tool_memory_source source;   ///< live or restored
    tool_memory_entry *prev;     ///< LRU list, towards most recent
    tool_memory_entry *next;     ///< LRU list, towards least recent
    tool_memory_entry *block_next;  ///< next entry sharing the same block
};

/** Bounded LRU of tool calls and their exact sampled bytes.
 *
 * Exists so a replayed conversation can be matched against what the model
 * actually produced. The client replays a tool RESULT and its call id; the
 * bytes of the call itself may never have been visible to it, so without this
 * the prefix cannot be reproduced and the turn cold-prefills.
 *
 * Bounded on BOTH count and bytes, and evicted LRU: this is a cache, and a
 * conversation whose calls have aged out simply replays instead. Guarded by
 * server::tool_mu, since client threads read it.
 */
/** Exact-match lookup for the two tool-memory indexes.
 *
 * These were a vendored radix tree (rax) using five of its twenty-six entry
 * points -- new, find, insert, remove, free.  Nothing ever iterated, seeked a
 * prefix or walked a range, which is the entire reason to prefer a radix tree
 * over a hash map, so three thousand lines were serving as a map.
 *
 * Keys are built with an explicit length rather than from a C string, because a
 * block key is raw dsml bytes; today those bytes come from strlen at the call
 * site and cannot contain a NUL, but the key type should not be what enforces
 * that.  Held by POINTER so tool_memory stays trivially memset-able -- its
 * free() zeroes the whole struct. */
typedef std::unordered_map<std::string, tool_memory_entry *> tool_memory_id_map;
typedef std::unordered_map<std::string, tool_memory_block *> tool_memory_block_map;

typedef struct {
    tool_memory_id_map    *by_id;     ///< id -> entry
    tool_memory_block_map *by_block;  ///< block bytes -> block, for deduplication
    tool_memory_entry *head;   ///< most recently used
    tool_memory_entry *tail;   ///< least recently used; the eviction victim
    int entries;               ///< entries live
    int max_entries;           ///< entry ceiling
    size_t bytes;              ///< bytes accounted
    size_t max_bytes;          ///< byte ceiling
    uint64_t clock;            ///< monotonic; stamps entries on use for the LRU order
    uint64_t scan_clock;       ///< monotonic; one value per scan pass, compared against block::seen
} tool_memory;

/** A slot's binding between tool-call ids and the frontier they were sampled
 * at -- the state that lets a tool result continue its own conversation rather
 * than cold-prefilling. */
typedef struct {
    bool valid;  ///< the binding below is populated
    /** Token frontier of a live assistant tool-call turn. Continuing from this
     * point preserves hidden thinking and sampled DSML bytes that are not
     * necessarily present in the client-visible replay. */
    int live_tokens;
    /** Optional rendered conversation text that the client is expected to replay.
     * Responses uses this because visible replay can omit hidden reasoning.
     * Anthropic currently uses only the call-id side of the state. */
    char *visible_text;
    size_t visible_len;   ///< length of visible_text in bytes
    /** Tool-call ids generated at the same live frontier. A following tool
     * result for these ids is a direct protocol continuation and should not
     * trigger prompt-prefix matching or checkpoint canonicalization. */
    stop_list call_ids;
} live_tool_state;

/** What a slot's frontier looks like from the CLIENT's side.
 *
 * The distinction this type exists to hold: the live payload and the text the
 * client can replay are not the same thing, because hidden reasoning sits in
 * one and not the other. Matching a replayed prompt against the live session
 * has to compare against `visible_text`, not against the frontier itself.
 */
typedef struct {
    bool valid;  ///< the state below has been populated
    /** Token frontier of the live sampled session.  The visible text below is
     * what clients will replay, but the payload at this frontier may also
     * contain hidden thinking tokens that are intentionally absent from that
     * visible replay. */
    int live_tokens;
    char *visible_text;   ///< the replayable text at that frontier, owned
    size_t visible_len;   ///< its length in bytes
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
 * measured sweet spot when it was set: N=12 aggregate decode held 29.2 tok/s
 * at 8 banks vs 21.9 at 12 (the dense-step lanes then capped their fast
 * paths at 8 rows; the M-neutral range is 16 since L150/L152 and the
 * attention kernel is row-count-neutral since L161/L166, so this number is
 * due a re-measure).  The engine allows up to PULSAR_MSEQ_MAX=16 via an operator
 * PULSAR_MSEQ_BANKS pin for TTFT-focused deploys that accept that cliff.
 * KNOWN LIMIT, measured: with active conversations == banks, pool-full
 * eviction is LRU and cyclic traffic evicts exactly the next returning
 * conversation's bank (domino, everyone cold); warm reuse needs headroom
 * (convs < banks) until the victim policy is smarter than LRU. */
#define PULSAR_SESSION_POOL_CAP 16
/* Auto-sizing cap: raised 8 -> 16 on 2026-09-11 after the re-measure the old
 * comment asked for (L219).  The 2026-08-10 measurement (N=12 aggregate held
 * 29.2 tok/s at 8 banks vs 21.9 at 12) predated the row-neutral lanes; the
 * dense, spec and attention lanes are 16-row neutral now, and the cliff is
 * gone.  Locked-clock one-shot chat, thinking off, 200 tokens, ctx 8192:
 *
 *   banks  N=8          N=12                  N=16
 *     8    54.2 t/s     50.7 (5x 25-31 s TTFT stalls)   --
 *    12    53.9         56.2 (no stalls)      54.6 (4x 34-38 s stalls)
 *    16    --           56.5                   56.5 (no stalls)
 *
 * 16 is the row-neutral maximum (PULSAR_MSEQ_MAX), so the cap moves there;
 * auto-sizing still fits N to the KV budget below the cap (huge ctx yields
 * fewer banks), and POOL_CAP above remains the hard array bound. */
#define PULSAR_SESSION_POOL_AUTO_MAX 16

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
/* The chat template's role markers and turn terminator (DeepSeek V4).  Written
 * only by the renderers in prompt_render.cpp; read by the few places that
 * parse a rendered prompt back (generate.cpp rendered_chat_system_region). */
#define PULSAR_RENDER_SYSTEM "<｜System｜>"   /* V4.1: leads a thinking conversation or one opening with system text; marks mid-conversation system messages */
#define PULSAR_RENDER_USER "<｜User｜>"
#define PULSAR_RENDER_ASSISTANT "<｜Assistant｜>"
#define PULSAR_RENDER_EOS "<｜end▁of▁sentence｜>"

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
/** One session-pool slot -- see the note above for the lifecycle.
 *
 * A slot is a pure BANK DESCRIPTOR, not a session: all engine work goes through
 * the single shared server.sess, and a slot names which bank of it holds this
 * conversation.
 */
typedef struct session_slot {
    bool         provisioned;  ///< false until admitted; cleared on eviction (the slot is a reusable hole). Every reader that used to skip sess == NULL skips this.
    uint32_t     bank;  ///< Tier-2: this slot's bank id in the shared pool (slot i -> bank i). 0 in classic (non-pooled) mode.
    int          committed_pos;  ///< Tier-2: this bank's committed KV frontier length (== pulsar_session_pos when this bank is the live one). Kept current at every op boundary so routing/metrics can read a non-live bank's position without a bank swap.
    struct job  *active_job;  ///< request bound to this slot, or NULL
    gen_state   *gen;  ///< resumable state for active_job
    slot_state   state;  ///< lifecycle state; SLOT_EVICTED marks a reusable hole
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
    /** Tier-2 task #55 increment 2b — proactive-eviction guard. `spilled` means this
     * bank's comp/index PHYSICAL was cudaFree'd, its history persisted as a
     * segment chain in the disk KV cache (L264), while its conversation stays
     * bound here; it is restored (alloc_physical + the chain's load) before this
     * slot next decodes. Distinct from SLOT_EVICTED (which frees the bank for a
     * DIFFERENT conversation). */
    bool         spilled;
    /** Protocol live bindings for THIS slot's sampled KV frontier (guarded by
     * server.tool_mu — client threads read them at parse time). They bind
     * tool-call ids / visible transcripts to the session they were sampled on,
     * so a continuation can never match another slot's frontier. */
    live_tool_state responses_live;      ///< Responses call-id binding at this slot's frontier
    live_tool_state anthropic_live;      ///< Anthropic tool_use binding at this slot's frontier
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
                                      eviction helps in pool mode only (the
                                      evicted bank's resident pages are
                                      reused); see server_refusal_evictable */
    PROVISION_REFUSED_CREATE_FAIL, /* allocation failed — eviction unsafe to
                                      chain on (same physical pressure) */
    PROVISION_REFUSAL_COUNT,       /* sentinel: array bound for the /metrics
                                      per-reason counters, not a reason */
} provision_refusal;

/** Rate limiter for a warning about a PER-REQUEST condition that the worker
 * re-evaluates every quantum (L190 C1): one line per period, carrying the
 * count of occurrences the period swallowed, so the log shows the condition
 * is still live without a line per retry -- and never a single line per
 * process for something that happens per request. */
typedef struct {
    double last_sec;      ///< wall clock of the last line printed; 0 = never printed
    unsigned suppressed;  ///< occurrences since that line
} warn_limiter;
/** True when the caller should log now; *skipped receives the occurrences
 * suppressed since the previous line (0 on the first). */
bool warn_limiter_due(warn_limiter *w, double now_sec, double period_sec, unsigned *skipped);
/** Period of the MemAvailable-floor provisioning refusal line. */
#define PULSAR_SERVER_MEM_FLOOR_WARN_SEC 10.0

/** L284 lane cost: the scheduler's priced choice between the plain and spec decode lanes, one rule for every
 *  family (server_sched.cpp lane_price_pick).  The spec lane's acceptance is measured here, per bank per decode
 *  round; both lanes' step costs are the engine's fits (pulsar_engine_lane_cost).  All-zero is the start state. */
typedef struct {
    float tau;            ///< EW tokens a bank commits per spec round, base included (0: never measured)
    float rho;            ///< EW rows a bank carries per spec round, base included
    int lane;             ///< the priced lane while spec could carry the decoders: 2 or 3 (0: none yet -> 3)
    int probe;            ///< the lane being measured against `lane` (0: none)
    uint32_t probe_n0;    ///< that lane's step count when the probe began
    uint32_t held_n0;     ///< `lane`'s step count when it was chosen or last weighed (hold and re-probe count from it)
    int meas_n;           ///< the decoder count the measured prices below are for
    double meas_tok[PULSAR_LANE_COUNT];        ///< EW tokens a step committed there, per pulsar_decode_lane
    double meas_ms[PULSAR_LANE_COUNT];         ///< EW wall of those steps, ms
    uint32_t meas_steps[PULSAR_LANE_COUNT];    ///< steps measured there
    double t_plain;       ///< the last prices, tokens/s (0: no number) -- announced with the lane
    double t_spec;
    bool meas_plain;      ///< ...and whether each was measured at the decoders' count (else predicted)
    bool meas_spec;
    const char *why;      ///< the last pick's reason, announced with the lane
} lane_price;

/** The whole server: engine, session pool, scheduler queue, caches, metrics.
 *
 * ONE worker thread does every piece of engine work; client threads only parse,
 * enqueue, and write. That split is a hard rule rather than a style choice --
 * CUDA state is not safe to touch from the client threads -- which is why the
 * metrics below are published snapshots instead of live reads, and why `mu`
 * guards the queue rather than the engine.
 */
struct server {
    pulsar_engine *engine;  ///< the loaded model; worker thread only
    /** The ONE session (created at startup, freed once at shutdown): classic
     * mode == 1 bank-less slot (slot 0), pool mode == pool_banks banks over
     * it. Slots are pure bank descriptors; all engine work goes through this
     * pointer. */
    pulsar_session *sess;
    /** L272: every token's bytes (pulsar_token_text over the logits width), built on the first constrained
     *  tool name and kept: the mask reads them per draw.  A pointer, so the struct stays memset-clean;
     *  freed with the session at shutdown.  Worker thread only. */
    std::vector<std::string> *token_bytes;
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
    int          pool_banks;  ///< banks in the shared pool; >0 means the pool flip is active
    int          live_bank;   ///< bank whose device views + host carry are currently installed
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
    uint64_t     bank_marginal_bytes;  ///< Tier-2: per-bank ledger charge in pooled mode (even split of the admitted pool cost; conservative, demand-paged reality is smaller). 0 in classic mode.
    uint64_t     kv_budget_bytes;  ///< admission ceiling computed at startup
    /** Context-scaled KV one bank holds at the configured context — the
     *  compressed rows plus their index. Demand-paged under overcommit, so it is
     *  reserved as VA and resident only on touch; published because dividing the
     *  budget by it is the only way to answer "how many tokens of KV does this
     *  box actually hold", which the configured slot count does not answer. */
    uint64_t     kv_bank_bytes;
    uint64_t     kv_committed_bytes;  ///< sum of est_cost_bytes over live slots (under mu)
    /** Tier-2 task #55 increment 2b — proactive-eviction guard. `guard_enabled`
     * gates the whole mechanism (on iff overcommit sized N>1 banks and the disk
     * KV cache is open: a spill persists to it, L264). `guard_touched_budget` is the resident-KV ceiling the guard keeps
     * touched_kv under = kv_budget − eager_reserved (banks may grow to 1M but total
     * physical is bounded); `guard_eager_bytes` the eager floor already resident.
     * `guard_evictions` counts spills for metrics. */
    bool         guard_enabled;          ///< the proactive-eviction guard is active
    uint64_t     guard_touched_budget;   ///< resident-KV ceiling the guard keeps touched_kv under
    uint64_t     guard_eager_bytes;      ///< eager floor already resident
    uint64_t     guard_evictions;        ///< spills performed, for metrics
    warn_limiter mem_floor_warn;         ///< the per-request MemAvailable-floor refusal line, rate-limited (L190 C1)
    /** Trivial-match threshold for the choose-vs-provision routing decision:
     * template-header tokens measured at startup +
     * PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS (cli_main.cpp; immutable after
     * startup, worker thread reads only). */
    int slot_trivial_common_tokens;
    /** L272 B5: the template's turn markers (pulsar_chat_turn_markers at startup; zeroed when the
     * tokenizer does not spell them, which leaves the sys-prefix cold store and the last-turn routing
     * anchor off).  Immutable after startup. */
    pulsar_turn_markers turn_markers;
    int default_tokens;      ///< generation cap when the request does not set one
    kv_disk_cache kv;        ///< on-disk prompt-prefix KV cache
    tool_memory tool_mem;    ///< tool call/result bodies kept for replay matching
    pthread_mutex_t tool_mu;     ///< guards tool_mem, which client threads also read
    pthread_mutex_t mu;          ///< guards the queue, client count, and every published metric
    pthread_cond_t cv;           ///< wakes the worker when a job is enqueued or state changes
    pthread_cond_t clients_cv;   ///< signals shutdown waiters as clients drain
    /** Broadcast on every metrics publish and once at shutdown, waking
     * /metrics/stream waiters. Separate from `cv` on purpose: `cv` wakes the
     * single worker thread, and waking it on every metrics publish would spin
     * it against an empty queue for no reason. */
    pthread_cond_t stream_cv;
    job *head;                   ///< queue head; the next job to bind
    job *tail;                   ///< queue tail; where enqueue appends
    bool stopping;               ///< shutdown in progress; stop accepting and drain
    time_t started;  ///< wall-clock when the listener came up (uptime for /health)
    int clients;                 ///< connected clients, for the shutdown drain
    int stream_clients;          ///< live /metrics/stream subscribers (under mu)
    /** Bumped by publish_metrics_snapshot on every publish. A stream client
     * waits for this to change rather than for anything in the payload: the
     * snapshot is a dozen fields, and comparing it to decide "did anything
     * move" is exactly the mistake pulsar-gui made and had to undo. */
    unsigned long long metrics_generation;
    /** /metrics scheduler + prefill gauges (all under mu). n_queued = jobs
     * enqueued not yet bound to a slot; n_generating = jobs bound to slots
     * (0..n_slots, time-sliced by the single worker). m_* are cumulative
     * prefill counters feeding the Prometheus prompt-throughput and
     * prefix-cache-hit metrics. */
    int n_queued;      ///< jobs enqueued but not yet bound to a slot
    int n_generating;  ///< jobs bound to slots, time-sliced by the single worker
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
    uint64_t w_prefill_chunk_tokens;  ///< worker-owned running count
    uint64_t m_prefill_chunk_tokens;  ///< published copy, read by send_metrics
    /** L117: live EMA of ms per emitted token in the spec-batched lane
     * (worker-owned). Denominator of the overflow argmax's cost threshold:
     * a marginal draft row is admitted while survival >= marginal_ms / this.
     * 0 until the first quantum; the argmax uses a 45 ms prior until then. */
    float    spec_ms_per_tok_ema;
    /** L136: how often the overflow argmax actually binds.  overflow_rounds
     * counts sweeps whose demand exceeded the shared row budget; thr_cut_rows
     * counts candidate rows the COST THRESHOLD priced out (budget still
     * unspent) — the only rows whose fate the marginal_ms constant decides.
     * Worker-owned/mirrored like the counters above. */
    uint64_t w_spec_overflow_rounds;
    uint64_t m_spec_overflow_rounds;
    uint64_t w_spec_thr_cut_rows;
    uint64_t m_spec_thr_cut_rows;
    /** L123: the batched lane's shared ALL_ROWS logits landing buffer
     * (PULSAR_SPEC_LOGITS_ROWS x vocab floats, ~16.5 MB), allocated once on
     * first quantum. It was a per-quantum malloc/free — 16.5 MB of
     * demand-zero pages faulted back in on every touch cycle, a measured
     * ~1-2 ms/round of host tax. Worker-owned like the EMA above. */
    float   *spec_lane_logits;
    /** L219: the plain and mixed batched lanes' logits landing buffer
     * ((PULSAR_SESSION_POOL_CAP + 1) x vocab floats).  Same rationale as
     * spec_lane_logits, one lane over: the per-quantum malloc/free re-faulted
     * ~4 MB of demand-zero pages on every quantum of a
     * --no-dspark/plain-serving workload.  Worker-owned; the two lanes run
     * sequentially in one worker. */
    float   *lane_logits;
    /** Which decode lane the scheduler is on: 0 idle, 1 spec (retired), 2
     * batched, 3 spec-batched, 4 family-spec (server_pick_decode_lane). The
     * spec-decode counters cannot advance on the batched lane (it never enters
     * the fused loop), so a scraper needs this to tell "acceptance really is
     * this" from "no speculative decoding ran at all". */
    int w_decode_lane;  ///< worker-owned current lane
    int m_decode_lane;  ///< published copy, read by send_metrics
    /** L284 lane cost: the priced choice between the plain (2) and spec (3) lanes while the spec lane could carry
     *  every decoder (server_sched.cpp lane_price_pick).  Worker-owned. */
    lane_price w_lane_price;
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
    uint64_t m_requests_finished;  ///< requests completed; the count these histograms summarise
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
    FILE *trace;               ///< trace sink, or NULL when tracing is off
    pthread_mutex_t trace_mu;  ///< serialises trace writes across threads
    uint64_t trace_seq;        ///< monotonic trace event counter, for ordering
    FILE *capture;             ///< --capture-requests sink (append), or NULL
    pthread_mutex_t capture_mu;///< serialises capture writes across client threads

    /** @name Server methods (C++ port)
     *  1:1 mirror of the server_ / worker_ verb family; bodies keep the
     *  `auto *s = this` alias, numerics/logic verbatim.
     *  @{
     */
    /** Write one model object to `fd` as the /v1/models item shape. */
    bool send_model(int fd, const char *id);
    /** Write the full /v1/models listing to `fd`. */
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
    /** The /health `tp` block's fixed part, built once by build_tp_health()
     *  before any client thread runs: one JSON object per rank, each missing
     *  its closing brace so send_health can append the live `connected`.
     *  Ranks, hosts, builds and devices do not change for the life of the
     *  transport.  `tp` is read on client threads only for
     *  pulsar_tp_failed(), an atomic load; clients drain before the engine
     *  (and with it the transport) closes. */
    char **tp_rank_json;
    int *tp_rank_ids;           ///< the rank each fragment describes
    int tp_rank_json_n;
    int tp_nranks;
    int tp_self_rank;
    const char *tp_transport;   ///< "rdma" | "tcp" | NULL when not TP
    struct pulsar_tp *tp;
    void build_tp_health();
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
    /** GET /metrics/stream: hold the connection open and emit one SSE frame
     * per metrics publish. Returns false if the client went away or the
     * stream cap was already reached. */
    bool send_metrics_stream(int fd);
    /** Drop the connected-client count and signal anyone waiting on the
     * shutdown drain. Called from client threads. */
    void client_done();
    /** @name Tool-map trailer (KV cache files)
     *  A cached prefix is only replayable if the tool calls and results it
     *  refers to are still known, so they ride along in a trailer on the KV
     *  file rather than living only in memory.
     *  @{
     */
    /** Measure the trailer for `text`: how many entries and how many bytes.
     * Caller must already hold `tool_mu` -- hence the name. */
    bool kv_tool_map_measure_locked(const char *text, uint32_t *count_out, uint64_t *bytes_out);
    /** Serialized trailer size for `text`, taking `tool_mu` itself. */
    bool kv_tool_map_serialized_size(const char *text, uint64_t *bytes_out);
    /** Append the trailer for `text` to `fp`.
     * @param fp             open file, positioned where the trailer goes
     * @param text           the prefix text whose tool entries are written
     * @param written_bytes  bytes emitted
     * @return true on success. */
    bool kv_tool_map_write(FILE *fp, const char *text, uint64_t *written_bytes);
    /** Read a trailer from the CURRENT position of `fp`, keeping the entries
     * named in `wanted`. @return entries loaded; 0 at clean EOF or on a
     * malformed trailer, which is treated as "no trailer" rather than an error
     * so an older file still loads. */
    int kv_tool_map_load_from_pos(FILE *fp, const stop_list *wanted);
    /** @} */
    /** Re-populate tool memory with the calls and results `msgs` refers to, so
     * a replayed conversation can be matched against the live prefix. */
    void kv_cache_restore_tool_memory_for_messages(const chat_msgs *msgs);
    /** L264 S4: persist the INSTALLED bank's history as segments -- the chain the
     * store already holds for this text, extended to the bank's deepest grid
     * checkpoint.  @return segments written (0: nothing new, or disabled). */
    int kv_cache_persist(session_slot *sl, const char *reason);
    /** Load the deepest stored chain whose text is a byte prefix of
     * `prompt_text` into the installed bank (live at the chain's end), and build
     * the prompt to sync: the exact stored tokens + the rest of the text.
     * @return tokens restored, 0 for a miss. */
    int kv_cache_try_load_text(session_slot *sl, const char *prompt_text,
                               const pulsar_text_span *prompt_spans, uint32_t prompt_n_spans,
                               const uint64_t *image_hashes, int n_images,
                               pulsar_tokens *effective_prompt, char **loaded_key_out, bool responses_protocol);
    /** kv_cache_try_load_text() for a request's prompt. */
    int kv_cache_try_load(session_slot *sl, const request *req, pulsar_tokens *effective_prompt, char **loaded_key_out);
    /** A chain whose restore the prefill then failed on: drop its last segment. */
    void kv_cache_discard_failed_chain(const char *key);
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
    /** Append only the newly-rendered suffix onto the live session, leaving the
     * committed prefix untouched. */
    bool append_rendered_suffix_to_live_session(session_slot *sl, const char *suffix,
                                                const pulsar_text_span *spans, uint32_t n_spans,
                                                int *tokens_appended, char *err, size_t errlen);
    /** Recover from a malformed tool-call block by continuing generation rather
     * than failing the request. Attempted at most once per request
     * (gen_state::dsml_recovery_attempted). */
    bool continue_after_invalid_dsml(session_slot *sl, const request *r, const thinking_state *thinking, const char *detail, int *tokens_appended, char *err, size_t errlen);
    /** Report a prefill failure to the client. Handles the case where SSE
     * headers already went out for the keepalive: the error must then be an
     * event in the open stream, not an HTTP status. */
    void send_prefill_failure_response(const job *j, const server_prefill_progress *progress, const char *ctx, const char *flags, const char *err);
    /** After a successful tool-call finish, make the live checkpoint match what the
     * next request will render.  Usually that is just the exact DSML remembered by
     * tool id.  If a client sends a tool call without an id we know, the fallback
     * renderer still builds valid DSML from JSON, and this function either rewrites
     * the short suffix in place or reloads an older disk checkpoint before replay.
     */
    void canonicalize_tool_checkpoint(session_slot *sl, const job *j, const char *ctx, uint64_t trace_id, const char *content, const char *reasoning, const tool_calls *calls);
    /** Shared failure epilogue for both prefill phases (the old duplicated blocks
     * after each pulsar_session_sync failure). Token vectors and the disk path are
     * freed centrally by gen_state_free.  `discard_loaded_entry`: the engine refused
     * the prefill on top of a snapshot this request loaded from disk, so that file is
     * suspect and is unlinked.  Every other failure -- a client disconnect, a bank
     * restore, a fused forward shared with other slots -- keeps the file (L261).
     */
    void gen_prefill_fail(session_slot *sl, bool discard_loaded_entry);
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
    /** Tear down the slot's generation state.
     *
     * Uninstalls every engine callback FIRST: a cancel/progress callback still
     * holding a freed gen_state pointer is a use-after-free the next engine
     * call would take. Safe on a slot with no gen_state. */
    void gen_state_free(session_slot *sl);
    /** Bind a dequeued job to the slot and resolve its prompt (the first quantum). */
    void generate_job_begin(session_slot *sl, job *j);
    /** Advance the job by one quantum. */
    void generate_job_step(session_slot *sl);
    /** Unbind: drain deferred client bytes, free the resumable state. */
    void generate_job_end(session_slot *sl);
    /** @name Live tool-state bindings
     *  A protocol tool loop is not "a new prompt with a long prefix": the
     *  client replays a tool RESULT plus a call id, and the authoritative
     *  prefix is the live KV -- including hidden reasoning the client never
     *  saw and cannot replay. These record what a slot's frontier holds so a
     *  following request can be bound back to the session that owns its
     *  conversation, and appended to rather than cold-prefilled.
     *  @{
     */
    /** Bind `calls` and the visible text to the slot's current frontier as a
     * Responses continuation point. */
    void responses_live_remember(session_slot *sl, const char *visible_text, const tool_calls *calls);
    /** Anthropic equivalent; Anthropic uses only the call-id side of the state. */
    void anthropic_live_remember(session_slot *sl, const tool_calls *calls);
    /** Drop the slot's Responses binding. */
    void responses_live_clear(session_slot *sl);
    /** Drop the slot's Anthropic binding. */
    void anthropic_live_clear(session_slot *sl);
    /** Is `id` bound on ANY slot? Answers "can this be served live at all", not
     * "which slot" -- a cheap pre-check before routing. */
    bool responses_live_has_call_id(const char *id);
    /** Anthropic equivalent of responses_live_has_call_id(). */
    bool anthropic_live_has_call_id(const char *id);
    /** Does `sl`'s binding hold ALL of `ids` at exactly `live_tokens`? Both
     * halves matter: a matching id set at a MOVED frontier is not a valid
     * continuation point. */
    bool responses_live_matches_request(const session_slot *sl, const stop_list *ids, int live_tokens);
    /** Anthropic equivalent of responses_live_matches_request(). */
    bool anthropic_live_matches_request(const session_slot *sl, const stop_list *ids, int live_tokens);
    /** Scheduler routing (worker thread): find the slot whose live binding holds
     * ALL of the request's continuation ids at that slot's current frontier, so
     * the job can be bound to the session that owns its conversation.
     */
    session_slot * live_slot_for_ids(const stop_list *ids, bool anthropic);
    /** live_slot_for_ids() for the Responses protocol. */
    session_slot * responses_live_slot_for_ids(const stop_list *ids);
    /** live_slot_for_ids() for the Anthropic protocol. */
    session_slot * anthropic_live_slot_for_ids(const stop_list *ids);
    /** @} */

    /** Is `id` present in tool memory? Takes `tool_mu`. */
    bool tool_memory_has_id(const char *id);
    /** Remember every call in `calls`, so a later replay can reproduce the
     * exact bytes the model produced for them. */
    void tool_memory_remember(const tool_calls *calls);
    /** Remember one call, recording whether it came from live generation or was
     * restored from a KV-file trailer. */
    void tool_memory_put_source(const char *id, const char *dsml, tool_memory_source source);
    /** tool_memory_put_source() with source TOOL_MEMORY_RAM. */
    void tool_memory_put(const char *id, const char *dsml);
    /** Fill in each replayed message's tool bytes from memory.
     * @param msgs   the replayed conversation, filled in place
     * @param stats  records what matched and what did not -- a miss is the
     *               reason a turn cold-prefills, so it is counted rather than
     *               swallowed. */
    void tool_memory_attach_to_messages(chat_msgs *msgs, tool_replay_stats *stats);
    /** Give every id-less call in `calls` a fresh id in `api`'s format.
     * Regenerates until the id collides with neither the batch nor anything in
     * tool memory, so a replayed conversation can never bind to the wrong call. */
    void assign_tool_call_ids(tool_calls *calls, api_style api);
    /** @name Request tracing
     *  All writes are serialised on `trace_mu`, so tracing is safe from any
     *  thread; `id` correlates every line belonging to one request.
     *  @{
     */
    /** Write raw bytes into the trace for request `id`. */
    void trace_piece(uint64_t id, const char *piece, size_t len);
    /** Write a formatted event line into the trace for request `id`. */
    void trace_event(uint64_t id, const char *fmt, ...);
    /** Record how the prompt was resolved: what the caches supplied, what the
     * tool replay matched, and which file was used. This is the record that
     * explains a cold prefill after the fact. */
    void trace_write_cache_diag(const trace_cache_diag *d, const tool_replay_stats *tool_replay, int cached, const char *cache_source, int disk_cached, const char *disk_path);
    /** Open a trace for `j`. @return the trace id every later call passes back. */
    uint64_t trace_begin(const job *j, int cached, int effective_prompt_tokens, const trace_cache_diag *cache_diag, const char *cache_source, int disk_cached, const char *disk_path);
    /** Close the trace: outcome, token counts, what the final parse produced. */
    void trace_finish(uint64_t id, const request *r, const char *final_finish, int completion, bool saw_tool_start, bool saw_tool_end, const char *parsed_content, const char *parsed_reasoning, const tool_calls *parsed_calls, double elapsed);
    /** @} */
    /** Append `j` to the queue and wake the worker. Takes `mu`.
     * @return false when the server is stopping and the job was not queued. */
    bool enqueue(job *j);
    /** Release listener, engine, session, and caches during shutdown. */
    void close_resources();
    /** Make `bank` the live bank on the shared session: lazily save the
     * outgoing bank's device views + host carry, restore the incoming one.
     * No-op when `bank` is already live. Worker thread only. */
    bool bank_switch(int bank);
    /** Evict exactly one idle victim so a fresh conversation gets a bank.
     * Live-tool owners are protected; LRU-superseded victims go first, else
     * plain LRU (worker_evict_one's picker). Reuses the proven eviction body
     * (snapshot + ledger release + bank reset). Worker thread only; returns
     * true when a bank was freed. */
    bool fresh_make_room();
    /** Bring a spilled bank's KV back from the spill directory.
     * @return false when the file is missing or unreadable, which forces the
     * conversation to replay from a checkpoint instead. */
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
    /** Proactive-eviction guard: spill banks to disk until resident KV is back
     * under guard_touched_budget. @param dec the decode set that must be kept
     * resident. @param n its length. */
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
    /** L282: service a slot the scheduler picked -- one step unless it is already GEN_DONE, then release it if it is
     *  GEN_DONE now.  The ONE statement for every lane: a slot that reached GEN_DONE without a step (a prefill the
     *  decode quantum abandoned) is released by the next pick, never skipped. */
    void worker_service_slot(session_slot *sl);
    /** Advance every slot in `dec` by one batched decode quantum.
     *
     * THE decode lane: all `n` slots step together through the shared multiseq
     * path, each row landing at its own bank's frontier. Since L118 there is no
     * classic per-slot alternative -- a single decoding slot is simply n == 1. */
    void worker_batched_decode_quantum(session_slot **dec, int n, int quantum_tokens);
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
    void worker_spec_batched_quantum(session_slot **dec, int n, int quantum_tokens);
    /** L260 fusion: queued prompts ride the spec lane's forwards when this is true
     * (pool mode; one Spark until the pair's fused frame lands). */
    bool fusion_enabled() const;
    /** L260 fusion: a prompt's first fused chunk -- install its bank once, start a
     * fresh conversation's bank at 0 (invalidate), and refuse (no_fuse) a prompt
     * that does not extend the bank's history; leaves no bank live. */
    bool fuse_prepare(session_slot *sl);
    /** L271 / L272 P1: a slot in the plain batch rejoins the spec-batched lane once it is the only
     *  decoder: its committed tokens reconcile onto the host checkpoint and its sampled feed token is
     *  fed and emitted, as the batch would have.  false = the slot failed (its phase says why). */
    bool batch_leave(session_slot *sl);
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
    /** Find or provision a slot able to serve a job needing `ctx` context.
     * @return the slot, or NULL with *refusal set. */
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
     * 2. Among free slots with enough context, the highest SCORE wins (L264):
     * the deepest grid checkpoint at or below the request's byte match, i.e.
     * where the sync will resume; the longer match breaks a tie.
     * 3. The winner continues in place unless that would discard meaningful
     * warm state of a DIFFERENT conversation (server_route_in_place); then a
     * fresh slot is preferred (after an eviction if that is what it takes),
     * and with the pool exhausted it falls back to the winner.
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
    /** Longest common token prefix between the slot's history and `prompt`. */
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
    /** The slot's committed KV position -- where a continuation would append. */
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
/** One queued request, and the handshake between its client thread and the
 * worker.
 *
 * OWNERSHIP IS ONE-WAY: only the worker pops and frees a job. The client
 * thread parks on `cv` until the worker signals `done`, even when its socket
 * has already died -- it flags `cancelled` and keeps waiting rather than
 * unlinking. Every queued-job helper depends on that invariant.
 */
struct job {
    int fd;        ///< client socket
    request req;   ///< the parsed request
    bool done;     ///< the worker has finished with this job; releases the client thread
    /** Set (under mu) by the client thread when its socket dies while the job
     * is still queued; the worker reaps the job pre-bind. The client thread
     * NEVER unlinks or frees — it stays parked on cv until the worker
     * signals done, preserving the worker-only pop/free invariant the
     * queued-job protect helpers depend on (see worker_try_bind). */
    bool cancelled;
    pthread_mutex_t mu;  ///< guards this job's own flags
    pthread_cond_t cv;   ///< signalled when `done` is set
    job *next;           ///< next job in the server's queue
    /** Which provisioning refusal this job has already been counted against, so
     * the /metrics counter records jobs blocked rather than bind retries (the
     * worker re-attempts the head every quantum). PROVISION_OK = not counted;
     * jobs are memset to zero at creation, so that is the natural initial
     * value. */
    provision_refusal refusal_counted;
};


/** A window of token ids around the point where a replayed prompt stopped
 * matching the live session.
 *
 * The single most useful artefact when a turn cold-prefills unexpectedly: it
 * shows the ids on BOTH sides at the divergence, so the cause -- a retokenised
 * boundary, an injected reminder, a stripped reasoning block -- is visible in
 * the trace instead of requiring a reproduction.
 */
typedef struct trace_cache_diag {
    bool valid;       ///< the window below was captured
    int old_pos;      ///< the live session's position before this request
    int prompt_len;   ///< length of the incoming prompt
    int common;       ///< tokens that matched; the divergence point
    int start;        ///< first position covered by the window
    int count;        ///< positions captured (<= TRACE_CACHE_WINDOW)
    int live_id[TRACE_CACHE_WINDOW];    ///< the live session's ids across the window
    int prompt_id[TRACE_CACHE_WINDOW];  ///< the incoming prompt's ids across the same window
} trace_cache_diag;

/** Userdata for the engine's prefill progress callback.
 *
 * Its ADDRESS is handed to the engine, so it must live at a stable location
 * for the whole prefill -- it is embedded in gen_state rather than passed by
 * value anywhere. Besides progress accounting it owns the SSE keepalive, since
 * the callback is the only code that runs during a long prefill.
 */
typedef struct server_prefill_progress {
    server *srv;         ///< owning server
    session_slot *slot;  ///< slot whose session is prefilling (worker thread)
    req_kind kind;       ///< which API shape the request arrived on
    int prompt_tokens;   ///< prompt length being prefilled
    int cached_tokens;   ///< of those, how many came from cache
    char ctx[48];        ///< context span string, for log lines
    const char *phase;   ///< current phase name, for log lines
    bool has_tools;      ///< the request declared tools
    bool responses_protocol;  ///< the request is on /responses
    double t0;           ///< wall-clock at prefill start
    double last_t;       ///< wall-clock of the last progress event, for interval rates
    int last_current;    ///< `current` at that event
    bool seen;           ///< at least one progress event has arrived
    /** SSE keepalive during long prefill: send HTTP/SSE headers ahead of
     * generation and emit a `:` comment line every few seconds so HTTP/TCP
     * idle timeouts on the client side don't close the connection while the
     * server is busy doing prefill. */
    int fd;                  ///< client socket, for the keepalive writes
    bool stream;             ///< the client asked for SSE
    bool headers_sent;       ///< SSE headers already went out (they precede generation)
    bool stream_failed;      ///< a keepalive write failed; the client is gone
    double last_keepalive;   ///< wall-clock of the last `:` comment line
} server_prefill_progress;

/** Tracks whether generation is inside a reasoning block.
 *
 * Fed the raw byte stream, so it keeps a rolling tail: a `</think>` tag can
 * arrive split across any number of tokens, and the boundary has to be found
 * without buffering the whole completion. */
typedef struct thinking_state {
    bool inside;   ///< currently inside a reasoning block
    char tail[8];  ///< Long enough for "</think>".
    int tail_len;  ///< bytes currently held in `tail`
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

/** L251: one Qwen generation's output side.  The ONE qwen_output_parser of the
 * generation is fed g->text's bytes as the stop-string scan releases them, and
 * its events become the response: reasoning and content deltas, and each
 * completed tool call, which is given its id here and kept in `calls` for the
 * final message.  A call reaches the client only when complete (TOOL_END): a
 * call the parser later rejects (ERROR) was never announced, so a stream never
 * carries a call the final message lacks. */
struct qwen_gen {
    qwen_output_parser parser;
    std::vector<qwen_out_event> ev;  ///< scratch, reused per feed
    size_t fed = 0;                  ///< bytes of g->text fed to the parser
    tool_calls calls = {};           ///< completed calls, ids assigned, in emission order
    bool finished = false;           ///< the end of the turn was fed (the finish did it)
    bool stream_ok = true;           ///< no client write failed while projecting
    std::string last_error;          ///< the last malformed-call report (the retry's detail)
    int undeclared = 0;              ///< calls dropped for naming an undeclared tool (L272)
    std::vector<qwen_out_event> pending;   ///< a forced seed's announcement, sent by the first feed
    bool open = false;               ///< the last of `calls` is announced and still being read (L284 P5)
    bool open_args = false;          ///< its argument object's "{" went out
    int wire_open = -1;              ///< its index on the stream
    int wire_n = 0;                  ///< calls announced on the stream (a broken one keeps its index)
    ~qwen_gen();                     ///< frees `calls` (parser_qwen.cpp)
};

/** Everything one in-flight generation needs, for the whole life of the
 * request.
 *
 * The server is a single worker time-slicing many requests, so a generation
 * cannot keep state on the stack: it is suspended and resumed at quantum
 * boundaries. This struct IS that stack frame. The block comments below mark
 * the lifetime regions, which differ -- some fields survive a `decode_again`
 * recovery attempt and some are reset by it, and confusing the two is how a
 * recovered generation ends up reporting the first attempt's numbers.
 */
struct gen_state {
    job *j;            ///< the request being generated
    gen_phase phase;   ///< resume point: which step runs on the next quantum

    /** prompt/cache resolution (owned by gen_begin, read by later phases) */
    /** Failure reason for the error paths.
     *
     * Wider than the 160-byte buffers the engine fills, because every
     * writer wraps one: "spec round begin failed: %s" and its three
     * siblings prepend 24-28 characters, so at 160 the engine's own
     * message lost its tail -- the END of a diagnostic, which is where
     * the specifics are.  gcc reported this as -Wformat-truncation on
     * four snprintf sites and nothing was reading warnings. */
    char err[256];
    pulsar_tokens effective_prompt;  ///< prompt after template rendering and any cache-driven rewrite
    const pulsar_tokens *prompt_for_sync;  ///< &j->req.prompt or &effective_prompt
    bool responses_protocol;          ///< request came in on the /responses API
    bool responses_live_continuation;   ///< resuming a live /responses stream rather than starting one
    bool anthropic_live_continuation;   ///< resuming a live Anthropic stream
    char *disk_cache_path;            ///< KV file backing this prompt, owned; NULL when none
    int prompt_tokens;                ///< prompt length actually prefilled
    double t0;                        ///< wall time the request began, for latency accounting
    double first_token_t;  ///< wall time the first output token was produced (TTFT); 0 until set. Request-lifetime: survives decode_again so it reflects the genuinely first token emitted.
    uint64_t trace_id;   ///< correlates every trace event for this request
    char ctx_span[48];   ///< human-readable context span, for logs ("1234-5678")
    char req_flags[64];  ///< compact flag string describing the request, for logs
    server_prefill_progress progress;  ///< stable address: callback userdata
    int cold_store_len;  ///< the shared-preamble cut (chat_anchor_pos, system prompt + tools before the task message) written as a "sys-prefix" entry after its own prefill phase; always < the prompt; 0 = no cold write
    pulsar_tokens cold_prefix;  ///< the token prefix the cold write covers

    /* prefill quantum policy (see gen_prefill_cancel_cb) */
    uint32_t prefill_min_suffix;  ///< 0 = interrupting is never exact
    int prefill_chunks_done;  ///< chunks completed in the current sync call
    int prefill_last_current;  ///< last `current` the progress callback reported
    int prefill_total;         ///< rows this prefill will evaluate in total

    /** response identity + per-protocol stream projections; these live across
     * quanta AND across decode_again recovery attempts */
    char id[96];                       ///< response id echoed to the client
    bool structured_stream;            ///< emit protocol-structured events, not raw text
    anthropic_stream anthropic_live;   ///< Anthropic SSE projection state
    openai_stream openai_live;         ///< OpenAI chat-completions SSE projection state
    responses_stream responses_live;   ///< /responses SSE projection state
    bool openai_live_chat;             ///< the OpenAI projection is in chat (not completion) shape
    bool responses_live_chat;          ///< the /responses projection is in chat shape
    /** L267: the request's protocol sink, whichever family's output parser drives it (set for a
     * streamed chat on any protocol: sink.text != NULL). */
    chat_sink sink;
    long responses_created_at;         ///< `created` timestamp, fixed at first emit so it is stable across quanta
    bool recovery_attempted;           ///< a malformed tool call already triggered the parser's one retry; do not loop
    /** Request-lifetime token count: accumulates across decode attempts (the
     * tool-error recovery continuation) so continued generations spend ONE
     * shared max_tokens budget. */
    int completion_total;    ///< tokens generated across ALL attempts; the shared max_tokens budget
    uint64_t rng;            ///< sampler state; per-request so concurrency cannot perturb a seeded run

    /** decode attempt state (reset by GEN_DECODE_INIT) */
    buf text;                 ///< generated text so far, this attempt
    logprob_ledger logprobs;  ///< per-token target distributions; see the type
    size_t plain_stream_pos;  ///< bytes of `text` already streamed to the client
    /** Where the stop-sequence scan resumes. A stop string can straddle a
     * chunk boundary, so the scan restarts behind the frontier rather than at
     * it, and this is that backed-up position. */
    size_t stop_scan_from;
    const char *finish;       ///< finish reason once known ("stop", "length", ...); NULL while running
    char *stop_sequence;      ///< the client stop sequence that ended generation, when one did; owned
    int completion;           ///< tokens generated this attempt
    int max_tokens;           ///< cap for this attempt
    int next_decode_log;      ///< token count at which to write the next decode log line
    double decode_t0;         ///< wall time decoding began, for tokens/s
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
    double last_decode_log_t;          ///< wall time of the last decode log line, for interval rates
    int last_decode_log_completion;    ///< token count at that line, for interval rates
    thinking_state thinking;           ///< reasoning-block tracking (the stop-string scan waits outside it)
    bool spec_enabled;          ///< speculative decoding is active for this request
    /** L272: an unnamed forced call's function name is being sampled from g->text[tool_name_from..):
     *  until the family's closer appears the sampler masks every token that would leave the declared
     *  names, and the slot holds speculation (lane 3 samples inside the engine, past any mask). */
    bool tool_name_constrained;
    size_t tool_name_from;
    /** L272 P3: the family's output parser and its state for this decode attempt (created at decode
     * init, destroyed with the next attempt or the request). */
    const server_output_parser_ops *parser;
    void *parser_st;

    /** Tier-2 batched-decode lane state (worker_batched_decode_quantum). A slot
     * becomes batch_active when it joins the shared multiseq lane; it stays
     * there until it finishes (no mid-conversation batched->classic switch, so
     * no stale-logits hazard). batch_feed_token is the next token to commit at
     * batch_feed_pos (the bank's KV frontier); batch_pending holds the tokens
     * committed via multiseq since the bank's last host-checkpoint save, to be
     * reconciled onto the checkpoint when the slot returns to a classic op
     * (finish/store). */
    bool batch_active;      ///< this slot is in the shared multiseq lane (it leaves at finish, or for the family's
                            ///< speculation once that lane can carry every decoder: server::batch_leave, L271/L284)
    bool batch_feed_valid;  ///< batch_feed_token/_pos hold a real pending commit
    int  batch_feed_token;  ///< next token to commit
    int  batch_feed_pos;    ///< position to commit it at (the bank's KV frontier)
    pulsar_tokens batch_pending;  ///< tokens committed via multiseq since the bank's last host checkpoint
    /** plan-34 inc 5: why this prefill slot is not fusable (NULL: it may fuse) -- a
     * fused step rejected its run as not-position-true, or fuse_prepare found its bank
     * needs a rewind / a resume below the frontier / a fresh compressor group; route it
     * CLASSIC.  Set once; the classic path handles it correctly.  The reason is what
     * the spec quantum's rider verdict prints (L284). */
    const char *no_fuse;
    /** L260 fusion: this prompt rides the spec lane's fused steps -- its bank was
     * installed once and found an exact extension point (or invalidated for a fresh
     * conversation).  A prompt that is not an extension of its bank's history is
     * marked no_fuse instead and prefills classically. */
    bool fuse_ready;

    /** deferred, non-blocking client writes (installed for send_all) */
    slot_writer writer;  ///< queues bytes so a slow client cannot block the worker
};

/* Emit the surface-appropriate keepalive if this slot has gone quiet (see
 * PULSAR_SERVER_HEARTBEAT_MS). Returns true if one was written. Non-blocking
 * and idempotent-by-clock, so it is safe to call every worker pass. */
bool gen_stream_heartbeat(gen_state *g);

/** A parsed HTTP request line and body. Method and path are fixed-size because
 * anything longer than these bounds is rejected rather than allocated for. */
typedef struct {
    char method[8];    ///< HTTP method, NUL-terminated
    char path[256];    ///< request path, NUL-terminated
    char *body;        ///< request body, owned; NULL when there is none
    size_t body_len;   ///< body length in bytes
} http_request;

/** Argument handed to a client thread at spawn: the server and the socket it
 * owns for the connection's lifetime. */
typedef struct {
    server *srv;  ///< the server
    int fd;       ///< the accepted socket this thread owns
} client_arg;

/** Startup configuration, resolved once from CLI flags and immutable
 * afterwards -- the worker reads it without locking on that basis. */
typedef struct {
    pulsar_engine_options engine;  ///< model/runtime options for the engine
    const char *host;              ///< listen address
    int port;                      ///< listen port
    int ctx_size;                  ///< per-bank context size
    int default_tokens;            ///< generation cap when a request does not set one
    const char *capture_path;      ///< --capture-requests: every accepted request body appended as a JSON line; NULL disables
    const char *trace_path;        ///< request trace file; NULL disables tracing
    const char *kv_disk_dir;       ///< directory for the on-disk KV cache
    bool kv_disk_disable;          ///< turn the disk cache off entirely
    uint64_t kv_disk_space_mb;     ///< disk-cache budget in MiB
    kv_cache_options kv_cache;     ///< checkpoint placement policy
} server_config;

/* ---- shared globals ---- */

extern volatile sig_atomic_t g_stop_requested;
extern volatile sig_atomic_t g_listen_fd;

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
/** Append a NUL-terminated string.  Nearly every renderer and emitter calls it, so its caller graph is
 *  past DOT_GRAPH_MAX_NODES and is not drawn.
 *  \hidecallergraph */
void buf_puts(buf *b, const char *s);
void buf_printf(buf *b, const char *fmt, ...);
char *buf_take(buf *b);
void buf_free(buf *b);
/* json_ws / json_lit / json_string[_n] / json_number / json_int / json_bool /
 * json_skip_value / json_raw_value are declared in pulsar_json.h (included
 * above) and defined in src/lib/pulsar_json.cpp: ONE scanner, shared with the
 * engine's safetensors reader rather than a second copy free to drift. */
size_t trim_truncated_dsml_close_tail(const char *raw, size_t start, size_t len);
char *json_minify_raw_value(const char *json);
/** Re-serialise one JSON value the way Python's json.dumps(value, ensure_ascii=False)
 * prints it after json.loads(): ", " and ": " separators, keys in document order,
 * strings escaped only for backslash, quote and controls (non-ASCII raw),
 * integers verbatim, floats as repr().  Malloc'd; the input verbatim when it
 * does not parse. */
char *json_python_dumps_raw_value(const char *json);
bool json_content(const char **p, char **out);
/** Decode a standard base64 payload (RFC 4648 alphabet, '=' padding, no
 * whitespace).  Returns a malloc'd buffer and sets `*out_len`, or NULL on
 * malformed input.  The server's ONE base64 decoder; used for inline data:
 * image URLs. */
uint8_t *base64_decode(const char *in, size_t in_len, size_t *out_len);
void random_tool_id(char *dst, size_t dstlen, api_style api);
/** `prefix` + 2*nbytes lowercase hex from the OS RNG; dies when no RNG is
 * available (ids must never be predictable).  The ONE id generator: tool-call,
 * Responses item and chat/completion ids all come from here (L192 item 7). */
void random_prefixed_id(char *dst, size_t dstlen, const char *prefix, size_t nbytes);
void tool_calls_free(tool_calls *calls);
void tool_calls_push(tool_calls *calls, tool_call tc);
void chat_msg_add_tool_call_id(chat_msg *m, const char *id);
/** Attach an encoded image file (`bytes` taken) whose placeholder sits at `placeholder_off` in content. */
void chat_msg_add_image(chat_msg *m, uint8_t *bytes, size_t len, size_t placeholder_off);
/** Fold each Anthropic message's parts back into the one user turn DeepSeek's template renders. */
void anthropic_fold_tool_results(chat_msgs *msgs);
void chat_msgs_free(chat_msgs *msgs);
void chat_msgs_push(chat_msgs *msgs, chat_msg msg);
void tool_schema_orders_free(tool_schema_orders *orders);
const tool_schema_order *tool_schema_orders_find(const tool_schema_orders *orders, const char *name);
/** Whether a parsed call names a tool the request declared (L272); when it does not, `detail` gets the
 * model-visible tool error: the name and the declared ones. */
bool tool_call_declared(const request *r, const char *name, char *detail, size_t detail_len);
void request_init(request *r, req_kind kind, int max_tokens);
void request_free(request *r);
bool parse_thinking_control_value(const char **p, bool *thinking_enabled);
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
bool anthropic_tools_supported(const char *tools_json, char *err, size_t errlen);
bool parse_tools_value(const char **p, char **out, tool_schema_orders *orders);
struct chat_conversation;
/** L284: every tool the conversation offers, one function schema a line -- the tools array (parse_tools_value,
 *  every protocol's dialect: functions, Responses namespaces and tool_search, Anthropic-shaped) then the schemas
 *  Responses input items loaded (tool_search output).  Appended to `out`; every family renders from this. */
void conversation_tool_schema_lines(const struct chat_conversation *c, buf *out);
/** L284: schema lines as an OpenAI tools array ([{"type":"function","function":{name, description, parameters}}]),
 *  for a template that renders OpenAI's shape (Qwen).  A line's parameters are `parameters` or, Anthropic-shaped,
 *  `input_schema`.  malloc'd; NULL + err when a line has no name. */
char *tool_schema_lines_openai_tools(const char *lines, size_t len, char *err, size_t errlen);
bool parse_messages(const char **p, chat_msgs *msgs, char *err, size_t errlen);
bool parse_anthropic_messages(const char **p, chat_msgs *msgs, char *err, size_t errlen);
/* Attach one inline image block to `msg` and write its placeholder into `out`.
 * The single authority behind the chat `image_url` and Responses `input_image`
 * readers. */
bool server_add_image_block(chat_msg *msg, const char *url, buf *out,
                            char *err, size_t errlen);
void chat_msg_clear_images(chat_msg *m);
bool parse_anthropic_system(const char **p, char **out);
void append_tool_result_text(buf *b, const char *s);
bool append_dsml_arguments_from_json(buf *b, const char *json, const tool_schema_order *order,
                                      bool v41 = true);
void append_json_object_or_empty(buf *b, const char *json);
void append_dsml_tool_calls_text(buf *b, const tool_calls *calls, bool v41 = true);
bool chat_history_uses_tool_context(const chat_msgs *msgs,
                                           const char *tool_schemas);
/* ---- how a chat turn renders: the ONE authority (prompt_render.cpp, L185) ----
 *
 * render_chat_prompt_text (the full replay), render_live_tool_tail (the
 * suffix appended to live KV for a tool-result continuation), the checkpoint
 * suffix builders in generate.cpp and the legacy /v1/completions template
 * all produce a turn's bytes through append_chat_msg / append_assistant_open /
 * append_assistant_turn_close below.  A checkpoint key that does not
 * byte-match the next request's render is a silent cold re-prefill of the
 * whole conversation (server_jobs.cpp remember_*_checkpoint), so there is
 * exactly one place that decides those bytes.  A checkpoint key names the
 * bytes of the LIVE TOKENS, which for a tool-call turn end before the EOS the
 * replay renders (append_assistant_turn_sampled, L196): the consumer
 * tokenises the request bytes after the key, so the key must end where the
 * sampled tokens end or the EOS is lost from the bank. */
/** Per-render state: the mode, the reasoning-replay facts of the message
 * list, and where the turn structure stands. */
typedef struct {
    bool v41;                 ///< the loaded model's template family (request::family->v41)
    const pulsar_dsml_syntax *dsml;  ///< the spelling this family WRITES (a row of the table)
    bool think;               ///< thinking mode enabled: assistant turns carry a think block
    bool tool_context;        ///< tools advertised or used in the history: reasoning replays on every turn
    int last_user_idx;        ///< index of the last user-side message; assistant turns after it replay reasoning
    bool pending_assistant;   ///< a user-side turn is open; the next assistant turn (or the tail) opens with the role marker
    bool user_turn_open;      ///< a user-side turn (text and/or tool results) is open; further user-side parts join it with "\n\n"
    /** V4 only: a tool-result message already opened its own turn, so further
     * results continue it instead of opening another `<｜User｜>`.  The V4.1 rule
     * merges every user-side part into one turn (user_turn_open), so this flag
     * is inert there -- but it is what V4's renderer keyed on, and keeping the
     * two flags separate is what lets both rules live in one function. */
    bool pending_tool_result;
} chat_render;
void chat_render_init(chat_render *r, const chat_msgs *msgs, bool tools_advertised,
                      pulsar_think_mode think_mode, bool v41);
/** Append message `i` of `msgs`, which must not be a system-REGION message
 * (the caller owns that decision; a system message given here renders in
 * place as an environment note). */
void append_chat_msg(buf *out, const chat_msgs *msgs, int i, chat_render *r);
/** The generation prefix: the assistant role marker and the think opener
 * (thinking mode) or the empty think block's close (not). */
void append_assistant_open(buf *out, bool think);
/** An assistant turn after its opener: `reasoning` (when non-NULL -- the
 * replayed block's body), the think close (when `close_think`), the visible
 * content, the DSML tool calls, EOS. */
void append_assistant_turn_close(buf *out, bool close_think, const char *reasoning,
                                 const char *content, const tool_calls *calls, bool v41 = true);
/** The same turn as the model SAMPLED it: a turn that ends in tool calls
 * stops at the closing tool_calls tag and has no EOS (the tail renders it);
 * a stop turn ends with the EOS it sampled.  This is what a checkpoint key
 * names -- the bytes of the live tokens -- while the replay is
 * append_assistant_turn_close (L196). */
void append_assistant_turn_sampled(buf *out, bool close_think, const char *reasoning,
                                   const char *content, const tool_calls *calls, bool v41 = true);
/** Close a render: an open user-side turn gets the generation prefix. */
void chat_render_finish(buf *out, const chat_render *r);
/** The live-KV continuation suffix for msgs[start..): EOS, the new user-side
 * messages, the generation prefix. */
char *render_live_tool_tail(const chat_msgs *msgs, int start, bool tools_advertised,
                            pulsar_think_mode think_mode, bool v41 = true);
/** As above, plus the tail's CLIENT-DATA ranges (L223) -- tool-result bodies and
 * any replayed assistant content; the EOS, the role markers and the generation
 * prefix are the renderer's own text. */
char *render_live_tool_tail_spans(const chat_msgs *msgs, int start, bool tools_advertised,
                                  pulsar_think_mode think_mode, bool v41,
                                  chat_text_span **spans_out, uint32_t *n_spans_out);
/** As render_chat_prompt_text, but also hands back the rendered text's
 * CLIENT-DATA ranges (see buf's span fields) so pulsar_tokenize_rendered_chat_spans
 * can keep a client from injecting a control token.  `spans_out`/`n_spans_out` may
 * be NULL, in which case this is exactly the old entry. */
char *render_chat_prompt_text_spans(const chat_msgs *msgs, const char *tool_schemas,
                                    const tool_schema_orders *tool_orders,
                                    pulsar_think_mode think_mode, bool v41,
                                    chat_text_span **spans_out, uint32_t *n_spans_out);
char *render_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas,
                                     const tool_schema_orders *tool_orders,
                                     pulsar_think_mode think_mode, bool v41 = true);
void responses_prepare_live_continuation(request *r,
                                                const chat_msgs *msgs);
void anthropic_prepare_live_continuation(request *r,
                                                const chat_msgs *msgs);
/** L267: the tool_choice a chat request asked for, whatever its protocol spelled it as. */
typedef enum {
    CHAT_TOOL_CHOICE_AUTO,    ///< the model decides (also: absent, null)
    CHAT_TOOL_CHOICE_NONE,    ///< render without the tools
    CHAT_TOOL_CHOICE_ANY,     ///< a call is required, any tool (OpenAI "required", Anthropic "any")
    CHAT_TOOL_CHOICE_NAMED,   ///< a call to request::forced_tool_name is required
} chat_tool_choice;

/** L267: one thinking / effort control as the client sent it: the request key and its raw JSON value.
 * The family reads the list in arrival order, each family by its own rules (chat_family.cpp). */
typedef struct {
    const char *key;   ///< "thinking", "think", "enable_thinking", "reasoning_effort", "output_config",
                       ///< "reasoning.effort" (Responses) or "chat_template_kwargs"; static
    char *raw;         ///< the value's JSON text, owned
} chat_control;

/** L284 P3: the thinking controls as sent, read by the ONE rule every family and protocol shares
 *  (chat_family.cpp): the switch, and the effort -- a row of the effort-name table or an integer. */
typedef struct chat_effort_ask {
    int thinking;   ///< -1 not sent, 0 off, 1 on (thinking / think / enable_thinking / the kwargs)
    int level;      ///< -1 no name sent; else the row of the effort-name table
    int value;      ///< an integer effort 1..100 (level -1 then); 0 = none sent
} chat_effort_ask;
/** Read the controls in arrival order (a later one replaces an earlier one's value).  false + err:
 *  a name outside the table, an integer outside 1..100, a chat_template_kwargs key no template takes. */
bool chat_effort_ask_read(const chat_control *controls, int n, chat_effort_ask *a, char *err, size_t errlen);
/** Thinking on or off: the explicit switch; else effort "none" is off; else model "deepseek-chat" (the
 *  DeepSeek API's non-thinking name, on any family) is off; else on.  The switch on with effort none
 *  contradicts itself: false + err. */
bool chat_effort_ask_thinking(const chat_effort_ask *a, const char *model, bool *on, char *err, size_t errlen);

/** L267: a chat request as its protocol parser read it -- what the client asked for, before any model
 * family reads it.  OpenAI chat, Anthropic Messages and Responses each produce one; the loaded family's
 * renderer (render_chat_conversation) makes the prompt from it.  The protocol-only parts of the request
 * (sampling, stream, stops, the API style, forced_tool_name, ...) are already in the request. */
typedef struct chat_conversation {
    chat_msgs msgs;                 ///< the conversation, system / instructions first when sent
    char *tools_raw;                ///< the tools array as sent; NULL when absent or null
    char *tool_schemas;             ///< the same tools, one function schema a line (parse_tools_value)
    buf loaded_tool_schemas;        ///< Responses: schemas carried by input items (tool_search output)
    chat_tool_choice tool_choice;   ///< what the client asked of the tools
    char *tool_choice_wire;         ///< the choice as the client spelled it (for a refusal); NULL = named
    chat_control *controls;         ///< thinking / effort controls in arrival order, owned
    int n_controls;                 ///< entries in controls
} chat_conversation;

void chat_conversation_free(chat_conversation *c);
/** The raw JSON text of member `key` of object `obj` (malloc'd), or NULL when absent. */
char *json_object_member_raw(const char *obj, const char *key);
/** An Anthropic SERVER tool entry (web_search_*): recognised only to be dropped. */
bool anthropic_server_tool_entry(const char *raw_tool_json);
/** Append one control (`key` static, `raw` taken). */
void chat_conversation_control(chat_conversation *c, const char *key, char *raw);
/** The loaded family's half of a chat request: thinking resolved from the controls, the protocol's
 * tool-result checks, then the prompt -- text, client spans, and with an engine its tokens and images.
 * `fmt` is the loaded model's (pulsar_engine_chat_format); `e` and `s` may be NULL (the
 * parse-without-engine test shape).  false = refused, with `err` naming why ("" = a malformed value). */
bool render_chat_conversation(pulsar_engine *e, pulsar_chat_format fmt, server *s, chat_conversation *c,
                              request *r, char *err, size_t errlen);
/** The protocol halves: each reads its wire format into `c` and the request's protocol fields.  The
 * request is initialised by the caller.  false = refused (`err` set, or "" for malformed JSON). */
bool parse_chat_conversation_openai(const char *body, chat_conversation *c, request *r, char *err, size_t errlen);
bool parse_chat_conversation_anthropic(const char *body, chat_conversation *c, request *r, char *err,
                                       size_t errlen);
bool parse_chat_conversation_responses(const char *body, chat_conversation *c, request *r, char *err,
                                       size_t errlen);
/** parse_chat_request up to and including the rendered prompt TEXT, plus --
 * when `e` is non-NULL -- the tokenisation and the image-span resolution.  The
 * engine is optional so the renderer gate can render request bodies with no
 * model loaded and compare the TEXT against the reference encoder; with an
 * engine, the reference's order is render, tokenise, then replace each image
 * placeholder with that image's sentinel block (the decoded pixels live in the
 * render's own message list, so that step cannot move out of this function). */
bool parse_chat_request_render(pulsar_engine *e, server *s, const char *body, int def_tokens,
                               request *r, char *err, size_t errlen);
bool parse_chat_request(pulsar_engine *e, server *s, const char *body, int def_tokens,
                               request *r, char *err, size_t errlen);
bool parse_anthropic_request(pulsar_engine *e, server *s, const char *body, int def_tokens,
                                    request *r, char *err, size_t errlen);
bool parse_responses_input(const char **p, chat_msgs *msgs,
                                  buf *loaded_tool_schemas,
                                  tool_schema_orders *orders,
                                  char *err, size_t errlen);
bool parse_responses_request(pulsar_engine *e, server *s, const char *body, int def_tokens,
                                    request *r, char *err, size_t errlen);
bool parse_completion_request(pulsar_engine *e, const char *body, int def_tokens,
                                     request *r, char *err, size_t errlen);
bool send_all(int fd, const void *p, size_t n);

/* Emit one `event: metrics` frame per change of *generation, plus a `: keepalive`
 * comment whenever nothing has changed for keepalive_ms, until the client goes
 * away or *stop becomes true. Return false only when a write failed (the
 * client is gone); a shutdown-driven exit returns true.
 *
 * Deliberately a free function taking no server and no engine: it is the part
 * with the interesting failure modes — a missed wakeup, a keepalive that never
 * fires, a shutdown that hangs the drain — and this signature is what lets all
 * three be tested over a socketpair without a GPU. */
bool metrics_stream_pump(int fd, pthread_mutex_t *mu, pthread_cond_t *cv,
                         const unsigned long long *generation, const bool *stop,
                         int keepalive_ms);
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
/* ---- the shared DSML tool-stream projection (genmsg.cpp) ---- */
bool dsml_tool_stream_init(dsml_tool_stream *ts, const char *raw, size_t raw_len, size_t pos);
void dsml_tool_stream_free(dsml_tool_stream *ts);
bool dsml_tool_stream_update(dsml_tool_stream *ts, chat_sink *k,
                             const char *raw, size_t raw_len);
bool dsml_tool_stream_finalize(dsml_tool_stream *ts, chat_sink *k,
                               const char *raw, size_t raw_len);
/** The call id of invocation `index`, generated on first use in the
 * protocol's id style and deduplicated against this stream's earlier ids and
 * (with a bound server) tool memory. */
const char *dsml_tool_stream_id(server *s, dsml_tool_stream *ts, int index, api_style api);
bool parse_generated_message_ex(const char *text, bool require_thinking_closed,
                                       char **content_out, char **reasoning_out,
                                       tool_calls *calls);
/** L284 P4: the rule every family's parser applies to a turn whose tool text is not a valid call
 *  (generate.cpp): retry allowed -- once, non-streaming, a chat with tools, forced or not; the retry; the
 *  turn as text (reasoning, then the answer's raw bytes, no call); the finish label (error / length as
 *  the generation ended, else tool_calls with calls, else stop). */
bool turn_tool_retry_allowed(const struct gen_state *g);
bool turn_tool_retry(server *s, struct session_slot *sl, struct gen_state *g, const char *detail, server_turn *out);
void turn_as_text(const struct gen_state *g, server_turn *out);
const char *turn_finish(const struct gen_state *g, int n_calls);
void append_json_object_string(buf *b, const char *json);
void append_tool_calls_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                   const tool_schema_orders *orders);
void append_tool_call_deltas_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                         const tool_schema_orders *orders);
bool http_response(int fd, int code, const char *type, const char *body);
bool http_error(int fd, int code, const char *msg);
/* Retry-After for a retryable 503, by cause: the box is going away
 * (shutting down, draining), or it is briefly full (connections, metric
 * streams).  The one authority for these values. */
enum { HTTP_RETRY_GOING_AWAY_S = 10, HTTP_RETRY_BUSY_S = 1 };
bool http_response_retry(int fd, int code, const char *type, const char *body,
                         int retry_after_s);
bool http_error_retry(int fd, int code, const char *msg, int retry_after_s);
bool http_error_anthropic(int fd, int code, const char *msg);
/** DeepSeek's call-block finder (kv_cache.cpp): every DSML spelling either template renders. */
const char *find_next_dsml_tool_block(const char *p, const char **end_out);
bool request_apply_forced_tool_prefill(request *r, char *err, size_t errlen);
/** L272: whether a token whose bytes are `tok` may follow `so_far` in an unnamed forced call's function
 *  name: the joined bytes stay a prefix of some declared name followed by `close`, or pass it with only
 *  whitespace after (a token may carry the closer and the newline).  An empty token is never allowed. */
bool tool_name_token_allowed(const char *so_far, size_t n_so_far, const char *tok, size_t n_tok, const char *open,
                             const tool_schema_orders *declared, const char *close);
/** The slot is sampling an unnamed forced call's function name (the closer has not appeared yet). */
bool gen_tool_name_open(const struct gen_state *g);
/** Mask a constrained slot's logits row to the declared names (server_jobs.cpp); false = nothing allowed
 *  (said), the row untouched. */
bool gen_mask_tool_name(struct server *s, struct gen_state *g, float *row, int width);
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
void openai_stream_start(const request *r, openai_stream *st);
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
/* L267: the protocol-out SINKS (chat_sink): every family's output parser drives the request's
 * protocol through one.  DeepSeek's walk (deepseek_stream_update) streams its DSML calls as they
 * decode and Qwen's parser hands a call over whole, both through the sink's generic tool events
 * (OpenAI and Anthropic stream them; Responses sends the calls with the finish).  The finish is the
 * protocol's, after the family's last text. */
void openai_sink_init(chat_sink *k, int fd, server *s, const request *r, const char *id, openai_stream *st);
bool openai_sse_finish(chat_sink *k, const tool_calls *calls, const char *finish, int prompt_tokens,
                       int completion_tokens);
void anthropic_sink_init(chat_sink *k, int fd, server *s, const request *r, const char *id,
                         anthropic_stream *st);
bool anthropic_sse_finish(chat_sink *k, const tool_calls *calls, const char *finish,
                          const char *stop_sequence, int completion_tokens);
void responses_sink_init(chat_sink *k, int fd, server *s, const request *r, const char *id,
                         responses_stream *st);
/** DeepSeek's one projection: walk the generated text [0, raw_len) from where it stopped and drive
 * the sink; `final` flushes what was held back. */
void deepseek_stream_walk_init(deepseek_stream_walk *w, const request *r);
void deepseek_stream_walk_free(deepseek_stream_walk *w);
bool deepseek_stream_update(deepseek_stream_walk *w, chat_sink *k, const char *raw, size_t raw_len,
                            bool final);
/** The tool-call ids a stream already showed the client, into the parsed calls that have none (they
 * must answer the client's tool_result / tool output ids). */
void apply_stream_tool_ids(tool_calls *calls, const dsml_tool_stream *ts);
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
/** The end of a Responses stream: the reasoning item closed; `recovered_tail` (a tool block the final
 * parse turned back into text, never streamed) as output_text; the message item closed; the
 * function_call items; response.completed. */
bool responses_sse_finish(int fd, const request *r, responses_stream *st, const char *recovered_tail,
                          size_t tail_len, const tool_calls *calls, const char *finish, int prompt_tokens,
                          int completion_tokens, long created_at);
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
                                     const tool_calls *calls, const char *id_prefix);
bool anthropic_final_response(int fd,
                                     const request *r, const char *id, const char *text,
                                     const char *reasoning, const tool_calls *calls, const char *finish,
                                     const char *stop_sequence,
                                     int prompt_tokens, int completion_tokens);
bool anthropic_sse_start_live(int fd, const request *r, const char *id,
                                     int prompt_tokens, anthropic_stream *st);
size_t text_stream_safe_limit(const char *raw, size_t start,
                                     size_t raw_len, bool has_tools,
                                     bool final);
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
kv_cache_options kv_cache_default_options(void);
void le_put32(uint8_t *p, uint32_t v);
void sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
bool id_list_contains(const stop_list *ids, const char *id);
void id_list_push_unique(stop_list *ids, const char *id);
void id_list_free(stop_list *ids);
void collect_tool_call_ids(const chat_msgs *msgs, stop_list *ids);
char *path_join(const char *dir, const char *name);
bool kv_cache_open(kv_disk_cache *kc, const char *dir, uint64_t budget_mb, uint32_t model_id,
                   kv_cache_options opt);
void kv_cache_close(kv_disk_cache *kc);
char *render_tokens_text(pulsar_engine *engine, const pulsar_tokens *tokens, size_t *out_len);
void tokens_copy_prefix(pulsar_tokens *dst, const pulsar_tokens *src, int n);
void build_prompt_from_exact_prefix_and_text_suffix(
        pulsar_engine *engine,
        const pulsar_tokens *exact_prefix,
        const char *suffix_text,
        const pulsar_text_span *spans,
        uint32_t n_spans,
        pulsar_tokens *out);
int kv_cache_sys_prefix_cut(const kv_disk_cache *kc, int anchor);
int kv_cache_chat_anchor_pos(const kv_disk_cache *kc, const pulsar_tokens *prompt, const pulsar_turn_markers *m);
/* Trivial-match classifier for the memory-token resolver (defined in
 * server_sched.cpp; unit-tested in server_tests.cpp). */
bool server_slot_match_is_trivial(int common, int slot_pos,
                                         int share_ceiling, int protect_floor);
/* L275: the position of the bank's last user marker among its prefilled tokens
 * when a completed exchange precedes it; -1 otherwise (defined in
 * server_sched.cpp; unit-tested in server_tests.cpp). */
int server_route_turn_anchor(const pulsar_tokens *bank, int prefilled, const pulsar_turn_markers *m);
/* L264 S3 + L275: does the best-scoring free bank take the request in place, or
 * is a fresh bank preferred.  `anchor` is server_route_turn_anchor's answer for
 * that bank (defined in server_sched.cpp; unit-tested in server_tests.cpp). */
bool server_route_in_place(int common, int score, int frontier, int prefilled, int protect_floor,
                           int anchor);
/* L273: an image request's cold phase -- `cut` (the sys-prefix cut) when every image begins at or past it, 0 when
 * one begins inside it (*inside = its index, else -1).  Defined in server_jobs.cpp; unit-tested in server_tests.cpp. */
int server_image_cold_cut(int cut, const pulsar_image_ref *images, int n_images, int *inside);
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
/* Log priced-vs-actual for a freshly created session and return the bytes the
 * ledger commits (the actual); 0 when they differ -- the price is the
 * allocator run dry, so a difference is an accounting hole and the caller
 * refuses to start (defined in server_sched.cpp). */
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
char *build_invalid_dsml_tool_error_suffix(const request *r,
                                                  const thinking_state *thinking,
                                                  const char *detail);
/** As above, but also hands back the suffix's CLIENT-DATA ranges (L223): the
 * reminder embeds the client's system region and a tool result's bytes, and a
 * suffix appended to the live session must not turn a client spelling into a
 * control token.  `spans_out`/`n_spans_out` may be NULL. */
char *build_invalid_dsml_tool_error_suffix_spans(const request *r,
                                                 const thinking_state *thinking,
                                                 const char *detail,
                                                 chat_text_span **spans_out,
                                                 uint32_t *n_spans_out);
char *build_tool_checkpoint_suffix(const request *r, const char *content,
                                          const char *reasoning, const tool_calls *calls);
/** As above, plus the suffix's CLIENT-DATA ranges (L223): the sampled content
 * and reasoning are client-replayed bytes; the DSML framing is the server's own
 * text and stays control text. */
char *build_tool_checkpoint_suffix_spans(const request *r, const char *content,
                                         const char *reasoning, const tool_calls *calls,
                                         chat_text_span **spans_out, uint32_t *n_spans_out);
char *build_responses_visible_assistant_suffix(const request *r,
                                                      const char *content,
                                                      const char *reasoning,
                                                      const tool_calls *calls);
char *build_responses_visible_assistant_suffix_spans(const request *r,
                                                     const char *content,
                                                     const char *reasoning,
                                                     const tool_calls *calls,
                                                     chat_text_span **spans_out,
                                                     uint32_t *n_spans_out);
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
