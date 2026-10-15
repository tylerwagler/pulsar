/* deepseek_dsml.h -- DeepSeek's DSML machinery, the declarations (L295 F5): the decode-time tracker, the
 * live tool-stream projection, the stream walk, the final parse, the call-block finder and the model-visible
 * tool error.  Only DeepSeek's modules include this (deepseek_dsml.cpp, deepseek_stream.cpp, parser_deepseek.cpp
 * and DeepSeek's hooks in chat_family.cpp) plus the unit tests; the shared server reaches all of it through
 * server_output_parser_ops and server_family_ops.  The tag literals and the syntax table are src/lib/pulsar_dsml.h;
 * the renderer that WRITES DSML is prompt_render.cpp. */
#ifndef PULSAR_SERVER_DEEPSEEK_DSML_H
#define PULSAR_SERVER_DEEPSEEK_DSML_H

#include "pulsar_server_internal.h"

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
 * ONE state machine (dsml_tool_stream_update / _finalize, deepseek_dsml.cpp) drives
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

size_t trim_truncated_dsml_close_tail(const char *raw, size_t start, size_t len);

const char *find_any_tool_start(const char *s);

const char *find_any_tool_end(const char *s);

bool complete_tool_call_inside_thinking(const char *text, size_t len, size_t *scan_from);

void observe_tool_markers(const char *scan, bool *saw_start,
                                 bool *saw_end, bool *orphan_end);

size_t trim_tool_separator_ws(const char *raw, size_t start, size_t limit);

const char *find_last_substr(const char *s, const char *needle);

/* ---- the live DSML tool-stream projection ---- */
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

/** DeepSeek's call-block finder: every DSML spelling either template renders. */
const char *find_next_dsml_tool_block(const char *p, const char **end_out);

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

/** DeepSeek's one projection: walk the generated text [0, raw_len) from where it stopped and drive
 * the sink; `final` flushes what was held back. */
void deepseek_stream_walk_init(deepseek_stream_walk *w, const request *r);
void deepseek_stream_walk_free(deepseek_stream_walk *w);
bool deepseek_stream_update(deepseek_stream_walk *w, chat_sink *k, const char *raw, size_t raw_len,
                            bool final);
/** The tool-call ids a stream already showed the client, into the parsed calls that have none (they
 * must answer the client's tool_result / tool output ids). */
void apply_stream_tool_ids(tool_calls *calls, const dsml_tool_stream *ts);

size_t text_stream_safe_limit(const char *raw, size_t start,
                                     size_t raw_len, bool has_tools,
                                     bool final);

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

#endif /* PULSAR_SERVER_DEEPSEEK_DSML_H */
