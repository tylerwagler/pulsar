/** @file
 *  pulsar-agent's shared internal declarations: the worker, UI, renderer, tool and session
 *  types and the functions the agent's translation units share. */
/* pulsar_agent_internal.h — internal shared declarations for the agent sources.
 * Produced by the multi-TU split of pulsar_agent.c; edit freely (the
 * generator is not part of the build). */
#ifndef PULSAR_AGENT_INTERNAL_H
#define PULSAR_AGENT_INTERNAL_H

#include "pulsar.h"
#include "pulsar_help.h"
#include "pulsar_kvtext.h"
#include "pulsar_segstore.h"
#include "pulsar_kvchain.h"
#include "pulsar_think_scan.hpp"
#include "pulsar_dsml.h"
#include "linenoise.h"

#include <errno.h>
#include <ctype.h>
#include <dirent.h>
#include <fnmatch.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <regex.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>


/* This is intentionally not in linenoise.h, but it is part of the existing
 * multiplexed editor implementation.  The agent uses it only to restore text
 * after Enter is pressed while the model is still busy. */
/** @brief Insert text at the cursor of a live linenoise edit (vendored linenoise).
 *  @param l    the edit state
 *  @param c    bytes to insert
 *  @param clen byte count
 *  @return 0 (text that does not fit is dropped), or -1 when a terminal write fails */
int linenoiseEditInsert(struct linenoiseState *l, const char *c, size_t clen);

/* ---- shared macros ---- */



/// Transcript tokens since the last system prompt (or reminder) after which the tool/system reminder is re-injected.
#define AGENT_SYSTEM_PROMPT_REMINDER_TOKENS 50000


/* Poor man's code highlighter inspired by antirez/kilo: a tiny language table
 * plus one line-oriented tokenizer for comments, strings, numbers, and
 * separator-bounded keywords.  This is deliberately not a full parser; it is
 * only for making fenced Markdown code readable in the terminal. */
/// Highlight class: plain code text.
#define AGENT_HL_NORMAL 0
/// Highlight class: a comment.
#define AGENT_HL_COMMENT 1
/// Highlight class: a primary keyword.
#define AGENT_HL_KEYWORD1 2
/// Highlight class: a secondary keyword (listed with a trailing '|' in the language table).
#define AGENT_HL_KEYWORD2 3
/// Highlight class: a string literal.
#define AGENT_HL_STRING 4
/// Highlight class: a numeric literal.
#define AGENT_HL_NUMBER 5

/// agent_syntax::flags: highlight numeric literals.
#define AGENT_SYNTAX_NUMBERS (1u<<0)
/// agent_syntax::flags: highlight quoted strings.
#define AGENT_SYNTAX_STRINGS (1u<<1)
/// agent_syntax::flags: backtick-quoted spans are strings too.
#define AGENT_SYNTAX_BACKTICK_STRINGS (1u<<2)
/// agent_syntax::flags: keywords match case-insensitively.
#define AGENT_SYNTAX_CASE_INSENSITIVE (1u<<3)


/// User turns /history and /switch replay when no count is given.
#define AGENT_HISTORY_DEFAULT_TURNS 3
/// The most user turns /history replays (larger requests are clamped).
#define AGENT_HISTORY_MAX_TURNS 200
/// Lines of one assistant reply shown in replayed history before it is truncated.
#define AGENT_HISTORY_ASSISTANT_MAX_LINES 80
/// Bytes of one assistant reply shown in replayed history before it is truncated.
#define AGENT_HISTORY_ASSISTANT_MAX_BYTES 12000


/// The largest file the file tools read into memory, bytes (larger files are refused).
#define AGENT_FILE_MAX_BYTES (16*1024*1024)
/// Lines the read / more tools return when the call names no count.
#define AGENT_READ_DEFAULT_LINES 500
/// Tokens a tool result must leave free in the context (for the reply) or the worker compacts first.
#define AGENT_TOOL_RESULT_RESERVE_TOKENS 1024
/// Bytes of an edit's streamed old text before the [upto] forcer may cut it short.
#define AGENT_EDIT_UPTO_MIN_PREFIX_BYTES 64
/// Non-empty lines of an edit's streamed old text before the [upto] forcer may cut it short.
#define AGENT_EDIT_UPTO_MIN_PREFIX_LINES 2
/// Compact once the transcript fills this percentage of the context.
#define AGENT_COMPACT_SOFT_PERCENT 85
/// Compact once fewer tokens than this stay free (capped at a quarter of the context).
#define AGENT_COMPACT_MIN_FREE_TOKENS 8192
/// The verbatim tail kept by compaction is at most ctx_size / this many tokens.
#define AGENT_COMPACT_TAIL_DIVISOR 10
/// The verbatim tail kept by compaction is at most this many tokens.
#define AGENT_COMPACT_TAIL_CAP_TOKENS 50000
/// The longest compaction summary the model may generate, tokens.
#define AGENT_COMPACT_SUMMARY_MAX_TOKENS 4096




/* ============================================================================
 * Asynchronous Bash Jobs
 * ============================================================================
 *
 * Bash commands are tracked jobs, not blocking one-shot calls.  Each job owns a
 * process, a pipe, and a secure /tmp output file.  The first observation is
 * head-biased so headers and early errors are visible; later progress updates
 * are tail-biased and report how much output was added since the previous
 * observation.
 */

/// Bytes of output a bash job's first observation shows from the head.
#define AGENT_BASH_HEAD_BYTES (8*1024)
/// Lines of output a bash job's first observation shows from the head.
#define AGENT_BASH_HEAD_LINES 100
/// Bytes at the end of a bash job's output file a tail read scans.
#define AGENT_BASH_TAIL_BYTES (32*1024)
/// Tail lines a later observation of a still-running bash job shows.
#define AGENT_BASH_PROGRESS_TAIL_LINES 4
/// Tail lines a later observation of a finished bash job shows.
#define AGENT_BASH_FINAL_TAIL_LINES 20

/// Initial size of the line editor's input buffer, bytes.
#define AGENT_INPUT_INITIAL_BUFLEN 4096
/// The line editor's input buffer may grow to this many bytes.
#define AGENT_INPUT_MAX_BUFLEN (1024*1024)
/// SGR that opens the status footer's style (grey background).
#define AGENT_STATUS_STYLE_START "\x1b[48;5;238;38;5;252m"
/// SGR that closes the status footer's style.
#define AGENT_STATUS_STYLE_END "\x1b[0m"
/// SGR for the filled part of the footer's progress bar.
#define AGENT_STATUS_BAR_FILL "\x1b[48;5;238;38;5;201;1m"
/// SGR for the queued-prompt preview in the footer.
#define AGENT_QUEUE_STYLE "\x1b[38;5;87;1m"
/// Minimum seconds between footer redraws while output streams.
#define AGENT_STATUS_REDRAW_INTERVAL_SEC 0.20
/// Cells of the footer's progress bar.
#define AGENT_PROGRESS_BAR_WIDTH 32
/// Buffer size for one rendered progress bar, escapes included, bytes.
#define AGENT_PROGRESS_BAR_MAX_BYTES 256

/* ---- shared types ---- */

/* ============================================================================
 * Configuration, Worker State, And Streaming Types
 * ============================================================================
 *
 * The agent is intentionally a single process: the UI thread owns terminal
 * input/output, while the worker thread owns the live DS4 session and KV state.
 * These types define the shared state and the small streaming state machines
 * used to render sampled assistant text and DSML tool calls as they arrive.
 */

/** What to generate and how to sample it: the agent's half of the engine
 * configuration (::pulsar_engine_options carries the model's half). */
typedef struct {
    const char *prompt;             ///< initial user prompt, or NULL to start interactive
    const char *system;             ///< system prompt prepended to the transcript
    const char *trace_path;         ///< file to write a generation trace to; NULL = no trace
    int n_predict;                  ///< generation cap in tokens per assistant turn
    int ctx_size;                   ///< session context size in tokens
    float temperature;              ///< sampling temperature; 0 selects argmax
    float top_p;                    ///< nucleus-sampling mass cutoff
    float min_p;                    ///< floor on candidate probability, relative to the top token
    uint64_t seed;                  ///< RNG seed for reproducible sampling
    pulsar_think_mode think_mode;   ///< whether the model may emit a reasoning block, and if so how it is shown
    bool think_mode_set;            ///< an explicit flag chose it (else the loaded family's default, after open)
} agent_generation_options;

/** The fully-resolved agent configuration, assembled once from CLI flags. */
typedef struct {
    pulsar_engine_options engine;   ///< model/runtime options handed to the engine
    agent_generation_options gen;   ///< prompt + sampling options for each turn
    const char *chdir_path;         ///< directory to run tools from; NULL = inherit the caller's cwd
    bool non_interactive;           ///< one-shot: run the prompt to completion and exit, no TUI
} agent_config;

/** Lifecycle state of the agent's single worker thread. Drives what the TUI
 * shows in the status line, so every long-running phase gets its own value
 * rather than being lumped into a generic "busy". */
typedef enum {
    AGENT_WORKER_IDLE,       ///< nothing in flight; waiting for a prompt
    AGENT_WORKER_PREFILL,    ///< evaluating prompt tokens (prefill_done/prefill_total apply)
    AGENT_WORKER_GENERATING, ///< decoding the assistant turn
    AGENT_WORKER_COMPACTING, ///< transcript exceeded the context; summarising it down
    AGENT_WORKER_SAVING,     ///< writing session state to the KV store
    AGENT_WORKER_ERROR,      ///< a turn failed; agent_status::error holds the message
    AGENT_WORKER_STOPPED,    ///< thread has exited; no further work will run
} agent_worker_state;

/** Worker progress published for the UI thread to render.
 *
 * Written by the worker, read by the renderer, so it is a snapshot rather than
 * a live view: every field is self-consistent only within one publication. */
typedef struct {
    agent_worker_state state;  ///< which phase the worker is in
    int prefill_done;          ///< prompt tokens evaluated so far this prefill
    int prefill_total;         ///< prompt tokens this prefill will evaluate in total
    /** Identifies WHICH prefill run the counters above belong to. A plain
     * monotonic counter (::agent_next_prefill_label); the renderer compares it
     * against the label it last drew to tell "this prefill advanced" from "a
     * new prefill started and the counters reset", which are indistinguishable
     * from prefill_done alone. */
    unsigned prefill_label;
    double prefill_tps;        ///< prefill throughput, tokens/s
    int generated;             ///< tokens emitted so far this assistant turn
    double gen_tps;            ///< decode throughput, tokens/s
    /** Sampling is currently forced to argmax. Set while the stream is inside
     * (or may be entering) a DSML tool-call marker, where sampling noise would
     * corrupt a structured token sequence. Scoped to one assistant round, so a
     * malformed reply, EOS, or Ctrl+C cannot leave sampling stuck greedy. */
    bool greedy_sampling;
    int ctx_used;              ///< context positions occupied by the transcript
    int ctx_size;              ///< total context positions available
    char error[256];           ///< message for AGENT_WORKER_ERROR; empty otherwise
} agent_status;

/// A tracked bash job (struct agent_bash_job, defined below).
typedef struct agent_bash_job agent_bash_job;

/** The agent's whole mutable world: engine handle, transcript, worker-thread
 * plumbing, and the UI-visible status.
 *
 * TWO THREADS touch this: the worker (which runs the engine) and the UI thread
 * (which renders and reads keys). Everything from `mu` down is shared state
 * guarded by that mutex; the engine handle and transcript are the worker's
 * alone. `cond` carries both directions of the handshake, so a wait on it must
 * always re-check its predicate.
 */
typedef struct {
    pulsar_engine *engine;      ///< the loaded model; worker thread only
    agent_config *cfg;          ///< resolved configuration, immutable after startup
    pulsar_session *session;    ///< KV session backing the transcript
    pulsar_tokens transcript;   ///< the conversation as tokens; source of truth for context use
    /** The turn being built: the messages since the last assistant turn (system notes, the user's text, tool
     *  results), rendered WHOLE into the transcript by the family's chat front when the assistant turn opens
     *  (agent_turn_flush).  `content` is owned. */
    pulsar_chat_message *turn;
    /** Messages in `turn`. */
    int turn_len;
    /** Slots allocated in `turn` (grows by doubling). */
    int turn_cap;

    char *cache_dir;            ///< directory holding saved sessions (`<sha>.session`) and `segments/`
    pulsar_segstore *kv;        ///< the KV segment store under cache_dir; NULL when unusable
    char session_sha[41];       ///< session identity: 40 hex chars + NUL
    char *session_title;        ///< human-readable session name for the picker
    uint64_t session_created_at;///< unix seconds when the session was first created
    bool user_activity;         ///< the user has typed something this session (gates autosave)
    bool session_dirty;         ///< transcript has changed since the last save

    pthread_t thread;           ///< the worker thread
    pthread_mutex_t mu;         ///< guards every field below, plus `status`
    pthread_cond_t cond;        ///< worker/UI handshake; always re-check the predicate
    int wake_fd[2];             ///< self-pipe so the UI's poll() wakes on worker events
    FILE *trace;                ///< generation trace stream, or NULL
    bool wake_pending;          ///< a wake byte is already in the pipe (coalesces writes)
    bool stop;                  ///< shutdown requested; every wait loop must exit
    bool interrupt;             ///< Ctrl+C: abandon the current turn, keep the session
    bool initialized;           ///< engine + session are up and the worker may accept work
    bool save_requested;        ///< UI asked for a session save at the next safe point
    bool compact_requested;     ///< UI asked for a transcript compaction

    /** Engine position where the current prefill began. Progress is reported as
     * `current - progress_base`, which subtracts the REUSED cached prefix so the
     * bar measures new work rather than replayed positions. */
    int progress_base;
    double progress_started_at; ///< wall-clock at prefill start, for the tokens/s rate
    /** Transcript length when the system prompt was last (re)stated. A reminder
     * is re-appended once the transcript has grown
     * ::AGENT_SYSTEM_PROMPT_REMINDER_TOKENS beyond it, so a long session cannot
     * drift away from its instructions. <= 0 means "never seen". */
    int last_system_prompt_reminder_at;
    char *cmd_text;             ///< slash-command text handed from UI to worker

    agent_status status;        ///< published progress snapshot the UI renders

    char *out;                  ///< assistant output accumulated this turn
    size_t out_len;             ///< bytes used in `out`
    size_t out_cap;             ///< bytes allocated for `out`

    /** Queued-user-message drain handshake.
     *
     * When a turn ends in a tool call, queued user messages must NOT preempt
     * the tool. The worker sets `_pending` and waits; the UI thread answers by
     * filling `_text` and setting `_answered`. The drain therefore happens only
     * after the tool result is appended, so the next model input carries both
     * the tool observation and the user's correction. */
    bool queued_user_drain_pending;
    bool queued_user_drain_answered;  ///< UI has filled `_text`; releases the worker's wait
    char *queued_user_drain_text;     ///< drained queue contents, owned by the worker once taken

    bool datetime_context_injected;   ///< the current date/time was already stated in this transcript

    /** Continuation cursor for a truncated file read, so the `more` tool can
     * resume where the last read stopped. `more_valid` is false when there is
     * nothing to continue. */
    char more_path[PATH_MAX];   ///< file the last read came from
    int more_next_line;         ///< 1-based line the next chunk starts at
    bool more_bare;             ///< reproduce raw bytes (no line-number prefixes), as the original read did
    bool more_valid;            ///< the cursor above refers to a real truncated read

    agent_bash_job *bash_jobs;  ///< linked list of background shell jobs
    int next_bash_job_id;       ///< id handed to the next background job
    bool raw_mode_needs_restore;///< the terminal is in raw mode and must be restored on exit
} agent_worker;

/** Fixed-size ring keeping the LAST N bytes of a stream.
 *
 * For output that must stay bounded but whose end is the interesting part --
 * the tail of a long tool result. `total` keeps counting past `cap`, so the
 * caller can report how much was dropped rather than silently truncating.
 */
typedef struct agent_tail_capture {
    char *buf;      ///< ring storage, owned
    size_t cap;     ///< ring capacity
    size_t start;   ///< index of the oldest retained byte
    size_t len;     ///< bytes currently retained (<= cap)
    size_t total;   ///< bytes ever appended, including those overwritten
    /** Append bytes, overwriting the oldest once the ring is full (allocates `buf` on first use).
     *  @param s bytes to append
     *  @param n byte count */
    void append(const char *s, size_t n);  ///< was agent_tail_capture_append
    /** Take the retained bytes and reset the capture (frees `buf`, zeroes every field).
     *  @param len receives the byte count returned; may be NULL
     *  @return a malloc'd NUL-terminated copy of the retained bytes, oldest first; caller frees */
    char *take(size_t *len);  ///< was agent_tail_capture_take
} agent_tail_capture;

/** Which run of held-back marker bytes the renderer is currently accumulating.
 *
 * `*` and `` ` `` cannot be emitted the moment they arrive: their meaning
 * depends on how many follow (one star is emphasis, two are bold; three
 * backticks open a fence). The renderer buffers the run and decides once it
 * ends. */
typedef enum {
    AGENT_MD_PENDING_NONE,      ///< not inside a marker run
    AGENT_MD_PENDING_STAR,      ///< accumulating '*' (emphasis / bold)
    AGENT_MD_PENDING_BACKTICK,  ///< accumulating '`' (inline code / fence)
} agent_markdown_pending;

/// A code-highlighter language (struct agent_syntax, defined below).
typedef struct agent_syntax agent_syntax;

/** Turns a stream of decoded tokens into styled terminal output.
 *
 * Everything here is INCREMENTAL: bytes are printed as they arrive, so the
 * renderer must carry the state a one-pass formatter would keep on the stack.
 * Three separate hold-back buffers exist because three different things can
 * arrive split across token boundaries -- a UTF-8 sequence, a markdown marker
 * run, and a possible DSML marker.
 *
 * Code blocks are shown raw as they stream, then the terminal row is repainted
 * with highlighting at end-of-line -- but only when the repaint is safe (simple
 * one-row ASCII lines). Long, tabbed, escaped or UTF-8 lines are left as
 * streamed and only advance the highlighter's state.
 */
typedef struct {
    pulsar_engine *engine;      ///< engine handle, for detokenising
    agent_worker *worker;       ///< owning worker (output routing, capture)
    bool format_markdown;       ///< apply markdown styling at all
    /** `<think>` tag state, fed by agent_stream_text: `think.in_think` selects the
     * dimmed reasoning style, `think.pending` holds a tag prefix split across
     * tokens. */
    pulsar_think_scanner think;
    bool color_open;            ///< an SGR colour sequence is open and must be closed
    bool use_color;             ///< the sink accepts ANSI colour
    bool last_output_newline;   ///< last byte written was '\n' (drives spacing decisions)
    bool wrote_visible_output;  ///< anything non-empty has been shown this turn

    bool md_bold;               ///< inside **bold**
    bool md_italic;             ///< inside *italic*
    bool md_inline_code;        ///< inside `inline code`
    bool md_code_block;         ///< inside a fenced code block
    bool md_fence_info;         ///< reading the info string on a fence opening line
    bool md_code_line_start;    ///< at the start of a code-block line (prefix not yet written)
    bool md_code_in_ml_comment; ///< highlighter state: inside a multi-line comment
    /** Highlighter is running as a DRY RUN: compute state, emit nothing.
     * Used to ask "would repainting this line change it?" without touching the
     * terminal. See renderer_code_scan_line(). */
    bool md_syntax_silent;
    /** Set by the highlighter whenever it produced a non-default colour. After
     * a silent pass this is the answer to "is a repaint worth it?"; the caller
     * saves and restores it around the dry run. */
    bool md_syntax_has_highlight;

    agent_markdown_pending md_pending;  ///< which marker run is being held back
    size_t md_pending_len;              ///< how many bytes of that run are held

    const agent_syntax *md_syntax;      ///< highlighter for the fence's language; NULL = none
    char md_fence_lang[32];             ///< language tag from the fence info string
    size_t md_fence_lang_len;           ///< bytes used in md_fence_lang
    const char *md_code_line_prefix;        ///< gutter string printed before each code line
    const char *md_code_line_prefix_color;  ///< SGR sequence for that gutter
    bool md_code_highlight_upto;        ///< repaint the streamed row once the line completes

    char *md_code_line;                 ///< the current code line, buffered for repaint
    size_t md_code_line_len;            ///< bytes used
    size_t md_code_line_cap;            ///< bytes allocated

    char utf8_pending[4];               ///< partial UTF-8 sequence split across tokens
    size_t utf8_pending_len;            ///< continuation bytes received so far
    size_t utf8_pending_need;           ///< total bytes this sequence requires

    agent_tail_capture *capture;        ///< optional tail recorder, for /copy and transcripts
} agent_token_renderer;

/** One parsed tool-call parameter. */
typedef struct {
    char *name;      ///< parameter name, owned
    char *value;     ///< parameter value as text, owned
    /** The value was written as a quoted string. Kept because it is the only
     * way to tell the string "true" from the boolean true once both are held
     * as text. */
    bool is_string;
} agent_tool_arg;

/** One tool invocation: a name plus its parsed parameters. */
typedef struct {
    char *name;             ///< tool name, owned
    agent_tool_arg *args;   ///< parameter array
    int argc;               ///< parameters present
    int argcap;             ///< parameters allocated
} agent_tool_call;

/** Growable list of tool calls parsed from one assistant turn. */
typedef struct {
    agent_tool_call *v;  ///< the calls
    int len;             ///< calls present
    int cap;             ///< calls allocated
} agent_tool_calls;

/** Where the DSML tool-call parser is in the stream.
 *
 * DSML arrives interleaved with ordinary prose, so the parser is a scanner
 * that spends most of its life in SEARCH and only commits once a real opening
 * marker appears. DONE and ERROR are terminal for the round. */
typedef enum {
    AGENT_DSML_SEARCH,      ///< scanning prose for an opening marker
    AGENT_DSML_STRUCTURAL,  ///< inside markup: tool names, parameter names, delimiters
    AGENT_DSML_PARAM_VALUE, ///< inside a parameter value, where prose is taken verbatim
    AGENT_DSML_DONE,        ///< a complete tool-call block was parsed
    AGENT_DSML_ERROR,       ///< malformed input; `error` holds the reason
} agent_dsml_state;

/** Incremental parser for DSML tool-call blocks.
 *
 * Fed one token at a time during generation, so every marker can arrive split
 * across token boundaries. The raw text is retained and parsed by OFFSET
 * (`parse_pos`) rather than consumed, which is what lets a partially-received
 * marker be re-examined once more bytes land.
 */
typedef struct {
    agent_dsml_state state;   ///< current scanner state
    char search_tail[64];     ///< trailing bytes kept in SEARCH so a marker split across tokens still matches
    size_t search_len;        ///< bytes held in search_tail

    char *raw;                ///< the full raw text seen this round
    size_t raw_len;           ///< bytes used
    size_t raw_cap;           ///< bytes allocated
    size_t parse_pos;         ///< how far into `raw` the parser has consumed

    agent_tool_call current;  ///< the call being assembled
    char *param_name;         ///< name of the parameter currently being read
    bool param_is_string;     ///< that parameter was opened as a quoted string
    size_t param_value_start; ///< offset in `raw` where the value began

    /** A PARTIAL closing marker is buffered at the tail of the current
     * parameter value -- the bytes so far match `</|DSML|` but the marker is
     * not complete. The sampler reads this to force argmax, so a structured
     * marker cannot be corrupted by sampling noise halfway through. */
    bool param_close_prefix;

    agent_tool_calls calls;   ///< completed calls for this round
    char error[160];          ///< reason for AGENT_DSML_ERROR
} agent_dsml_parser;

/** How a tool parameter should be DISPLAYED while it streams.
 *
 * Classification is presentational, not semantic: it decides whether the
 * visualiser shows the value inline, as a path, as a diff half, or as a
 * syntax-highlighted block. */
typedef enum {
    AGENT_TOOL_PARAM_NORMAL,       ///< plain scalar shown inline
    AGENT_TOOL_PARAM_PATH,         ///< a file path (rendered as a header)
    AGENT_TOOL_PARAM_OFFSET,       ///< a numeric offset/count, folded into the header line
    AGENT_TOOL_PARAM_CONTENT,      ///< a file body, shown as a highlighted block
    AGENT_TOOL_PARAM_DIFF_OLD,     ///< the `old` half of an edit, shown as removals
    AGENT_TOOL_PARAM_DIFF_NEW,     ///< the `new` half of an edit, shown as additions
    AGENT_TOOL_PARAM_BASH_COMMAND, ///< a shell command, shown as a command line
} agent_tool_param_kind;

/** Renders a tool call as it streams, before it has been fully parsed.
 *
 * The point is that the user sees the tool and its arguments taking shape
 * live, rather than a spinner followed by a wall of text. That means every
 * decision has to be made on a partial value, so the visualiser keeps its own
 * line-position state and cannot rely on the parser having finished.
 */
typedef struct {
    bool active;                        ///< a tool call is currently being visualised
    bool tool_announced;                ///< the tool's name header has been printed
    bool param_active;                  ///< inside a parameter value
    bool at_line_start;                 ///< the cursor is at column 0 of a fresh row
    bool last_output_newline;           ///< last byte emitted was '\n'

    agent_tool_param_kind param_kind;   ///< how the current parameter is displayed
    char tool_name[64];                 ///< name of the tool being visualised
    char param_name[64];                ///< name of the parameter being visualised
    char param_end_tail[64];            ///< trailing bytes held back while a closing marker may be forming
    size_t param_end_len;               ///< bytes held in param_end_tail

    bool read_style;                    ///< render with the compact `read` layout
    bool read_prefix_rendered;          ///< the read header (path + range) has been printed
    bool read_line_rendered;            ///< at least one read body line has been printed
    char read_path[512];                ///< `read` path argument, accumulated
    char read_start[32];                ///< `read` start-line argument, accumulated
    char read_max[32];                  ///< `read` max-lines argument, accumulated
    char read_whole[8];                 ///< `read` whole-file flag, accumulated

    char tool_path[512];                ///< path argument for non-read tools
    bool code_param_active;             ///< the current parameter is being syntax-highlighted
} agent_tool_visualizer;

/** Rolling tail matcher for a DSML opening marker.
 *
 * A marker can be split across any number of tokens, so the detector keeps the
 * last few bytes and re-tests on each arrival. Two instances exist per stream
 * -- one for prose, one for the thinking region -- because a marker inside
 * reasoning means something different from one in the reply. */
typedef struct {
    char tail[32];  ///< most recent bytes, enough to hold any partial marker
    size_t len;     ///< bytes currently held
} agent_dsml_marker_detector;

/** The top of the output pipeline: routes each token to the text renderer, the
 * DSML parser, and the tool visualiser, and decides which of them owns the
 * stream at any moment.
 *
 * This is where "is this prose or is this a tool call?" is answered, on
 * incomplete input, without ever un-printing anything already shown.
 */
typedef struct {
    agent_token_renderer *renderer;  ///< styled text output
    agent_dsml_parser *parser;       ///< tool-call parser fed the same bytes
    agent_tool_visualizer viz;       ///< live tool-call display

    bool dsml_active;                ///< a DSML block is being parsed
    /** The active DSML block was opened INSIDE the thinking region. It is
     * still displayed, but it is not a real tool call -- the model is
     * reasoning about a call, not making one. */
    bool dsml_ignored;
    /** Re-rendering saved transcript text rather than streaming live. Suppresses
     * cursor-control escapes, which must never leak into a pipe or a file. */
    bool replay;

    /** Bytes of a POSSIBLE opening marker held back from display. A lone '<'
     * is too common in prose to act on; past the second byte the held prefix is
     * specifically DSML-shaped, and sampling is forced to argmax. */
    char dsml_start_tail[64];
    size_t dsml_start_len;           ///< bytes held in dsml_start_tail

    agent_dsml_marker_detector plain_dsml;  ///< marker detector for the reply region
    agent_dsml_marker_detector think_dsml;  ///< marker detector for the reasoning region
    bool dsml_in_think;              ///< the detector that fired was the thinking one
    bool dsml_in_think_reported;     ///< that fact has already been surfaced (report once, not per token)
    bool post_think_gap;             ///< a blank line is owed after the reasoning block closes

    bool tool_preflight_error;       ///< the call was rejected before execution
    char tool_preflight_error_msg[256];  ///< why it was rejected, for display
} agent_stream_renderer;

/** Stops the model retyping a long `edit old=...` block it has already
 * uniquely anchored.
 *
 * While the old-text parameter streams, the next sampled token is inspected
 * BEFORE eval: once the emitted prefix already identifies exactly one place in
 * the file, and the model is still writing old text rather than closing the
 * parameter, the caller evaluates a complete "[upto]" marker line instead.
 * Saves the tokens, and the retyping is where drift creeps in. */
typedef struct {
    bool active;  ///< the emitted prefix is already a unique anchor
    bool done;    ///< the marker has been substituted for this parameter
} agent_edit_upto_forcer;

/** Growable byte buffer for the interactive input line. */
typedef struct {
    char *ptr;   ///< buffer, owned
    size_t len;  ///< bytes used
    size_t cap;  ///< bytes allocated
} agent_input_buf;

/** A language definition for the code-block syntax highlighter.
 *
 * Deliberately small: enough to colour keywords, strings, and comments in a
 * streamed terminal row, not a parser. */
struct agent_syntax {
    const char *name;        ///< canonical language name, matched against the fence info string
    const char *aliases;     ///< other accepted fence tags for this language
    const char **keywords;   ///< NULL-terminated keyword list
    const char *singleline_comments[3];  ///< up to three line-comment introducers
    const char *multiline_start;  ///< block-comment opener, or NULL
    const char *multiline_end;    ///< block-comment closer, or NULL
    unsigned flags;          ///< per-language lexer options (AGENT_SYNTAX_* bits)
    /** Whether a line comment starts at `p`.
     *  @param p position in the line
     *  @return the matching entry of singleline_comments, or NULL */
    const char *line_comment(const char *p) const;  ///< was agent_syntax_line_comment
    /** Whether one of `keywords` starts at `p` and ends at a separator.
     *  @param p        position in the line
     *  @param line_end end of the line (exclusive)
     *  @param out_len  receives the keyword's length on a match
     *  @param out_hl   receives AGENT_HL_KEYWORD2 for a secondary keyword (listed with a trailing '|'), else
     *                  AGENT_HL_KEYWORD1
     *  @return true on a match */
    bool match_keyword(const char *p, const char *line_end, size_t *out_len, int *out_hl) const;  ///< was agent_syntax_match_keyword
};

/** Growable output buffer with a size ceiling.
 *
 * `truncated` is the important field: tool output is capped so a runaway
 * command cannot blow out the context, and the flag is what lets the caller
 * say so rather than silently returning a short result. */
typedef struct {
    char *ptr;       ///< buffer, owned
    size_t len;      ///< bytes used
    size_t cap;      ///< bytes allocated
    bool truncated;  ///< the cap was hit; content is incomplete
} agent_buf;

/** A saved session as its `<sha>.session` file holds it (kvstore_session.cpp):
 *  the KV is not in it -- it lives in the segment store, found by the rendered
 *  text of `tokens`. */
typedef struct {
    char *title;             ///< session title, owned
    uint64_t created_at;     ///< unix seconds the session was created (part of its identity)
    uint64_t last_used;      ///< unix seconds of the last save
    uint32_t model_id;       ///< pulsar_engine_model_id the ids belong to
    pulsar_tokens tokens;    ///< the transcript's exact ids, owned
} agent_session_file;

/** Role marker attached to a line of rendered history, so the transcript view
 * can style turns without re-parsing the token stream. */
typedef enum {
    AGENT_HISTORY_MARK_NONE,      ///< continuation of the previous line
    AGENT_HISTORY_MARK_USER,      ///< first line of a user turn
    AGENT_HISTORY_MARK_ASSISTANT, ///< first line of an assistant turn
    AGENT_HISTORY_MARK_EOS,       ///< end-of-turn boundary
} agent_history_mark;

/** The loaded family's turn markers as they appear in rendered transcript text (pulsar_kvtext_render_tokens_text):
 * what the session title and /history scan for, so every family's transcript reads the same way (DeepSeek:
 * `<｜User｜>`, `<｜Assistant｜>`, `<｜end▁of▁sentence｜>`; Qwen: `<|im_start|>user`, `<|im_start|>assistant`,
 * `<|im_end|>`).  Built by agent_turn_marks_init; a marker the family does not spell is empty (len 0) and matches
 * nowhere. */
typedef struct {
    char user[64];            ///< the user-turn marker's text (pulsar_chat_turn_markers' user tokens)
    size_t user_len;          ///< bytes in `user`; 0 = not spelled
    char assistant[64];       ///< the assistant-turn marker's text (its assistant tokens)
    size_t assistant_len;     ///< bytes in `assistant`; 0 = not spelled
    char end[64];             ///< a closed assistant turn's end (pulsar_chat_end_assistant), whitespace trimmed
    size_t end_len;           ///< bytes in `end`; 0 = not spelled
} agent_turn_marks;

/** Rendered history as borrowed line pointers plus a parallel marker array.
 * Pointers are NOT owned -- they alias the rendered buffer. */
typedef struct {
    const char **v;           ///< line starts, borrowed
    agent_history_mark *mark; ///< role marker per line, same length as `v`
    int len;                  ///< lines present
    int cap;                  ///< lines allocated
} agent_history_ptrs;

/** One row in the session picker. */
typedef struct {
    char sha[41];            ///< session identity: 40 hex chars + NUL
    agent_session_file f;    ///< the file's contents, owned
} agent_session_list_item;


/** One line located inside a file buffer, as byte offsets.
 *
 * `content_end` and `end` differ by the line terminator, which is what lets a
 * read reproduce bytes EXACTLY (CRLF included, final newline or not) while
 * still being able to hand out the line without its ending. */
typedef struct {
    size_t start;        ///< offset of the first byte of the line
    size_t content_end;  ///< offset just past the last content byte, terminator excluded
    size_t end;          ///< offset just past the line terminator
} agent_line_span;

/** The line index of one file buffer. */
typedef struct {
    agent_line_span *v;  ///< spans, in file order
    int len;             ///< lines present
    int cap;             ///< lines allocated
} agent_line_spans;

/** State for one grep-style search across files. */
typedef struct {
    const char *query;    ///< the pattern, as given
    const char *glob;     ///< restrict to paths matching this glob; NULL = all
    regex_t regex;        ///< compiled pattern; valid only while `regex_ready`
    bool use_regex;       ///< treat `query` as a regex rather than a literal
    /** `regex` has been compiled and must be regfree()d. Separate from
     * `use_regex` because a regex search whose pattern failed to compile has
     * the first flag set and the second clear. */
    bool regex_ready;
    bool case_sensitive;  ///< match case exactly
    int context;          ///< lines of context to show around each hit
    int max_results;      ///< stop after this many hits
    int results;          ///< hits emitted so far
    agent_buf out;        ///< accumulated result text
} agent_search_ctx;

/** One background shell command.
 *
 * Output goes to a temp FILE rather than being held in memory, so a command
 * that prints megabytes cannot blow out the agent's heap or its context: the
 * model is shown a bounded head or tail and the path to the rest.
 */
struct agent_bash_job {
    int id;         ///< job id shown to the model (bash_status job=N)
    pid_t pid;      ///< child process id
    int pipe_fd;    ///< read end of the child's output pipe; -1 once closed
    int tmp_fd;     ///< write end into the temp output file; -1 once closed
    /** Always the mkstemp template "/tmp/pulsar_agent_output_XXXXXX" (27 chars +
     * NUL) from agent_bash_start — mkstemp only substitutes the X's, so it can
     * never lengthen.  Sized for exactly that, not PATH_MAX. */
    char path[32];
    char *cmd;            ///< the command line, owned
    double start_time;    ///< wall-clock at spawn, for elapsed_sec
    double timeout_sec;   ///< kill the job after this long; 0 = no timeout
    size_t bytes;         ///< output bytes captured so far
    int newline_count;    ///< newlines seen, for the line count without a rescan
    char last_byte;       ///< last byte captured; tells whether output ends in a newline

    /** @note WRITE-ONLY as of this writing. Both counters are assigned in
     * agent_bash_job::observation() and read NOWHERE. The intent was clearly an
     * incremental cursor -- report only output added since the last look -- but
     * only `observed_once` is actually consumed, and it selects HEAD (first
     * observation) vs TAIL (every later one). A repeated bash_status therefore
     * re-shows the same tail rather than just the new bytes. Either wire these
     * up or delete them; they are not currently load-bearing. */
    size_t observed_bytes;
    int observed_display_lines;  ///< @see observed_bytes -- also write-only

    bool observed_once;   ///< the job has been observed at least once; switches head display to tail
    int exit_status;      ///< waitpid status once `running` is false
    bool running;         ///< the child has not been reaped yet
    bool timed_out;       ///< the job was killed by its timeout
    struct agent_bash_job *next;  ///< next job in the worker's list
    agent_worker *worker;  ///< back-pointer for terminal state restoration

    /** @name Bash-job methods (C++ port)
     *  1:1 mirror of the agent_bash_* verb family; bodies keep the
     *  `auto *job = this` alias, logic verbatim.
     *  @{
     */
    /** Output lines captured so far (newline count, plus a trailing partial). */
    /** @return the line count; 0 before any output */
    int display_lines() const;
    /** Account for `n` freshly captured bytes (byte and newline counters). */
    /** @param s the captured bytes
     *  @param n byte count */
    void note_output(const char *s, size_t n);
    /** Release the job: close descriptors, unlink the temp file, free `cmd`. */
    void job_free();
    /** Read whatever the child has written into the pipe right now. */
    void drain();
    /** Record the child's exit `status` and mark the job no longer running. */
    /** @param status the waitpid status (exit_status becomes its exit code, 128 + signal, or -1) */
    void finalize(int status);
    /** Non-blocking progress step: drain output, reap the child if it exited,
     * and enforce the timeout. */
    void poll();
    /** Read up to `max_lines` lines / `max_bytes` bytes from the START of the
     * output.
     * @param max_lines    line cap
     * @param max_bytes    byte cap
     * @param lines_read   lines actually returned
     * @param byte_limited set when the BYTE cap, not the line cap, stopped the
     *                     read -- the caller reports truncation differently
     * @return newly allocated text, caller frees. */
    char *read_head(int max_lines, size_t max_bytes, int *lines_read, bool *byte_limited) const;
    /** Read the last `max_lines` lines of the output.
     * @return newly allocated text, caller frees. */
    /** @param max_lines line cap (read from the last AGENT_BASH_TAIL_BYTES of the output file) */
    char *read_tail_lines(int max_lines) const;
    /** Build the tool result describing this job: status line, then output.
     * The FIRST observation shows the head of the output; later ones show the
     * tail. @param mark_observed record that the model has now seen the job,
     * which is what flips head display to tail. */
    /** @return the tool result text, newly allocated; caller frees */
    char *observation(bool mark_observed);
    /** @} */
};

/** Prompts typed while the model was busy, drained in order at the next safe
 * point. Guarded by agent_worker::mu. */
typedef struct {
    char **v;    ///< queued prompt strings, owned
    size_t len;  ///< prompts queued
    size_t cap;  ///< slots allocated
} agent_prompt_queue;

/** The interactive input line, pinned below streaming output.
 *
 * The hard part is that the model is writing to the same terminal the user is
 * typing into. The editor claims a scroll region so generated text scrolls
 * ABOVE a fixed prompt row, and most of the geometry below exists to keep that
 * prompt where the user left it as output arrives and the terminal resizes.
 *
 * Input is read in non-blocking chunks by the outer event loop rather than by
 * linenoise, so two terminal protocols have to be handled here instead of
 * inside the editor: cursor-position reports and bracketed paste.
 */
typedef struct {
    struct linenoiseState edit;  ///< the underlying linenoise editor
    char *input;                 ///< the committed line, owned; NULL until Enter
    char prompt[160];            ///< prompt string drawn before the input
    char status[4096];           ///< status text drawn with the prompt

    int old_stdin_flags;         ///< saved stdin flags, restored on teardown
    bool active;                 ///< the editor is installed and drawing
    bool hidden;                 ///< temporarily hidden while something else owns the screen

    bool output_line_open;       ///< streamed output left the cursor mid-line
    bool prompt_below_output;    ///< the prompt sits below the output region
    int output_col;              ///< column the output cursor stopped at
    bool scroll_region;          ///< a terminal scroll region is installed
    int term_rows;               ///< terminal height as last measured
    int term_cols;               ///< terminal width as last measured
    int output_bottom;           ///< last row belonging to the output region
    int prompt_row;              ///< row the prompt is pinned to
    int reserved_rows;           ///< rows reserved below the output for prompt and status
    bool output_cursor_saved;    ///< the output cursor position is saved and must be restored
    bool output_at_scroll_boundary;  ///< output stopped exactly at the region's last row
    double last_prompt_redraw_time;  ///< wall-clock of the last redraw, to rate-limit repainting

    /** Partial cursor-position report (ESC[rows;colsR) held while its bytes
     * arrive. Since the event loop reads stdin in chunks, a CPR can be split;
     * incomplete bytes are buffered rather than fed to linenoise, which would
     * interpret them as keystrokes. */
    char cpr_buf[32];
    size_t cpr_len;              ///< bytes held in cpr_buf

    /** A bracketed-paste envelope is open: ESC[200~ was seen, ESC[201~ was
     * not. Bytes keep flowing into linenoise's queue, but the editor is NOT
     * stepped while this is set -- otherwise newlines inside a paste get
     * interpreted as Enter and submit a partial line. */
    bool paste_open;
    bool paste_start_pending;    ///< the tail is a PREFIX of ESC[200~; the marker may still be arriving
    char paste_tail[6];          ///< last few bytes, enough to recognise either paste marker
    size_t paste_tail_len;       ///< bytes held in paste_tail
} agent_editor;

/** Result of testing buffered bytes against the cursor-position-report shape. */
typedef enum {
    CPR_INVALID,   ///< cannot be a CPR; feed the bytes through as input
    CPR_PARTIAL,   ///< a valid PREFIX of a CPR; keep buffering
    CPR_COMPLETE,  ///< a complete CPR; consume it
} cpr_state;

/** What a yes/no prompt should do if it times out. */
typedef enum {
    AGENT_YES_NO_AUTO_NONE,  ///< never time out; wait for an answer
    AGENT_YES_NO_AUTO_NO,    ///< answer no on timeout
    AGENT_YES_NO_AUTO_YES,   ///< answer yes on timeout
} agent_yes_no_auto;

/** Timeout policy for an interactive yes/no prompt. */
typedef struct {
    int timeout_sec;                    ///< seconds to wait; 0 = wait forever
    agent_yes_no_auto timeout_answer;   ///< the answer to assume when it expires
} agent_yes_no_options;

/** What agent_maybe_save_before_exiting decided. */
typedef enum {
    AGENT_EXIT_CANCEL,
    AGENT_EXIT_CLEAN,
    AGENT_EXIT_NOW,
} agent_exit_save_result;

/* ---- shared globals ---- */

/** Set by agent_sigint_handler on SIGINT; the UI loop polls and clears it. */
extern volatile sig_atomic_t agent_sigint;
/** The live worker for linenoise's /switch completion callback (which takes no user data); NULL outside the
 *  interactive loop. */
extern agent_worker *agent_completion_worker;
/** Text appended to a DSML parse error's tool result, restating the tool-call syntax for the model. */
extern const char agent_dsml_syntax_reminder[];

/* ---- shared functions ---- */

/** SIGINT handler: sets agent_sigint (async-signal-safe; nothing else).
 *  @param sig the signal number (unused) */
void agent_sigint_handler(int sig);
/** malloc that exits the process on failure; n == 0 allocates 1 byte.
 *  @param n bytes
 *  @return the allocation, never NULL; caller frees */
void *agent_xmalloc(size_t n);
/** strdup that exits on allocation failure; NULL copies as "".
 *  @param s the string, or NULL
 *  @return a malloc'd copy; caller frees */
char *xstrdup(const char *s);
/** Copy exactly n bytes of s (NUL bytes included) plus a terminating NUL; exits on allocation failure.
 *  @param s the bytes (at least n readable)
 *  @param n byte count
 *  @return a malloc'd NUL-terminated copy; caller frees */
char *xstrndup(const char *s, size_t n);
/** realloc that exits the process on failure; n == 0 keeps 1 byte.
 *  @param ptr the block, or NULL
 *  @param n   the new size, bytes
 *  @return the (possibly moved) block, never NULL */
void *agent_xrealloc(void *ptr, size_t n);
/** write(2) until all n bytes are written, retrying EINTR; any other error drops the rest silently.
 *  @param fd the descriptor
 *  @param p  the bytes
 *  @param n  byte count */
void write_all(int fd, const char *p, size_t n);
/** Append bytes, growing the buffer by doubling (4 KiB first); keeps it NUL-terminated.
 *  @param b the buffer
 *  @param s bytes to append
 *  @param n byte count; 0 is a no-op */
void agent_input_buf_append(agent_input_buf *b, const char *s, size_t n);
/** Take the buffer's string and reset it to empty.
 *  @param b the buffer
 *  @return its NUL-terminated contents ("" when nothing was appended); caller frees */
char *agent_input_buf_take(agent_input_buf *b);
/** Free the buffer's storage and reset it to empty.
 *  @param b the buffer */
void agent_input_buf_free(agent_input_buf *b);
/** Whether `cmd` is one of the runtime's slash commands (/help, /save, /compact, /list, /quit, /exit, /new, or
 *  /switch, /del, /strip, /history with or without an argument).
 *  @param cmd the typed line
 *  @return true for a known command */
bool agent_slash_command_known(const char *cmd);
/** Monotonic clock (CLOCK_MONOTONIC).
 *  @return seconds, with nanosecond resolution */
double agent_now_sec(void);
/** Print pulsar-agent's help (pulsar_help_print, PULSAR_HELP_AGENT).
 *  @param fp    the stream
 *  @param topic a help topic, or NULL for the overview */
void usage(FILE *fp, const char *topic);
/** Parse pulsar-agent's command line over the agent's defaults (the merged drafter disabled, 100000-token
 *  context, 50000-token turn cap); exits with usage on a bad option.
 *  @param argc argument count
 *  @param argv arguments; the returned config points into them
 *  @return the resolved configuration */
agent_config parse_options(int argc, char **argv);
/** The built-in DSML tool prompt: trusted control text (pulsar_chat_message::trusted). */
/** @return the prompt text, malloc'd; caller frees */
char *agent_build_tools_prompt(void);
/** Record that the model has just seen the system prompt: the reminder clock restarts at the transcript's
 *  current length.  Worker thread.
 *  @param w the worker */
void agent_worker_note_system_prompt_seen(agent_worker *w);
/** Once per session, add a system note with the local date and time to the turn being built.  Worker thread.
 *  @param w the worker */
void agent_worker_maybe_append_datetime_context(agent_worker *w);
/** The full tool/system reminder is separate from DSML syntax errors: it is a
 * pressure-controlled refresh of the same trusted prompt shape used at startup,
 * added to the turn being built as system notes: the built-in prompt trusted
 * (DeepSeek keeps its DSML markers native control tokens), arbitrary -sys text
 * ordinary text.
 */
/** @param w the worker; worker thread */
void agent_worker_maybe_append_system_prompt_reminder(agent_worker *w);
/** Wake the UI thread after changing worker-visible state.  The byte in
 * wake_fd is level-triggered with wake_pending so bursts of sampled tokens do
 * not flood the pipe.
 */
/** @param w the worker; the caller holds w->mu */
void agent_wake_locked(agent_worker *w);
/** Queue rendered output for the UI thread.  The worker never writes directly
 * to the terminal, which keeps linenoise redraws serialized in one place.
 */
/** Takes w->mu; output is dropped if the queue cannot grow.
 *  @param w the worker
 *  @param s bytes to queue (copied)
 *  @param n byte count; 0 is a no-op */
void agent_publish(agent_worker *w, const char *s, size_t n);
/** printf-style agent_publish.
 *  @param w   the worker
 *  @param fmt printf format, then its arguments */
void agent_publishf(agent_worker *w, const char *fmt, ...);
/** Set the worker's state (clearing prefill_tps outside PREFILL and greedy_sampling outside GENERATING) and
 *  wake the UI.  Takes w->mu.
 *  @param w     the worker
 *  @param state the new state */
void agent_set_status(agent_worker *w, agent_worker_state state);
/** Put the worker in AGENT_WORKER_ERROR with `msg` (copied, truncated to status.error) and wake the UI.
 *  Takes w->mu.
 *  @param w   the worker
 *  @param msg the message; NULL = "unknown error" */
void agent_set_error(agent_worker *w, const char *msg);
/** Append one timestamped line to the trace file (no-op without one).  Takes w->mu.
 *  @param w   the worker
 *  @param fmt printf format, then its arguments */
void agent_trace(agent_worker *w, const char *fmt, ...);
/** Trace one token: its index, id, escaped text and hex bytes.  No-op without a trace file; takes w->mu.
 *  @param w        the worker
 *  @param token    the token id
 *  @param text     its decoded bytes
 *  @param text_len byte count
 *  @param index    its position, as the trace numbers it */
void agent_trace_token(agent_worker *w, int token, const char *text,
                              size_t text_len, int index);
/** Trace tokens[start..len) one line per token after a header line.  No-op without a trace file.
 *  @param w      the worker
 *  @param label  a name for the run
 *  @param tokens the tokens
 *  @param start  first index traced (clamped to [0, len]) */
void agent_trace_tokens(agent_worker *w, const char *label,
                               const pulsar_tokens *tokens, int start);
/** Trace a labelled, escaped text block.  No-op without a trace file; takes w->mu.
 *  @param w     the worker
 *  @param label the field name; NULL = "text"
 *  @param text  the bytes
 *  @param len   byte count */
void agent_trace_text(agent_worker *w, const char *label,
                             const char *text, size_t len);
/** The value of a tool call's named argument (the first one of that name).
 *  @param call the call
 *  @param name the argument name
 *  @return the value (borrowed from the call; "" for a valueless argument), or NULL when absent */
const char *agent_tool_arg_value(const agent_tool_call *call, const char *name);
/** Free everything the parser owns (raw copy, current call, parsed calls) and zero it.
 *  @param p the parser, or NULL */
void agent_dsml_parser_free(agent_dsml_parser *p);
/** Free the parser and leave it searching for the next tool-calls opener.
 *  @param p the parser */
void agent_dsml_parser_reset(agent_dsml_parser *p);
/** Put the parser in AGENT_DSML_ERROR with `msg` (copied, truncated to p->error).
 *  @param p   the parser
 *  @param msg the error */
void agent_dsml_set_error(agent_dsml_parser *p, const char *msg);
/** @brief Whether `tail` is (a prefix of) a parameter close tag in any DSML spelling.
 *  @param tail     the bytes after the parameter value so far
 *  @param len      byte count
 *  @param complete set true when `tail` is a whole close tag ending at its last byte
 *  @return true while `tail` could still be, or is, a close tag */
bool agent_dsml_parameter_close_tail(const char *tail, size_t len,
                                            bool *complete);
/** @brief Enter a tool-calls block: seed the raw copy with the canonical opener and parse structure from here.
 *  @param p the parser */
void agent_dsml_start(agent_dsml_parser *p);
/** Feed streamed model text: searches for an opener, then parses the block's invokes and parameters into
 *  p->calls; no-op once DONE or in ERROR.
 *  @param p the parser
 *  @param s the bytes
 *  @param n byte count */
void agent_dsml_feed(agent_dsml_parser *p, const char *s, size_t n);
/** Emit raw bytes: into r->capture when capturing, else to the UI via agent_publish.
 *  @param r the renderer
 *  @param s bytes
 *  @param n byte count */
void renderer_write(agent_token_renderer *r, const char *s, size_t n);
/** Write an SGR reset (when colour is on) and mark no colour open.
 *  @param r the renderer */
void renderer_reset_color(agent_token_renderer *r);
/** Re-emit the active text attributes (thinking grey, bold, italic, code) after something else reset the
 *  terminal; no-op when none are active or no colour is open.
 *  @param r the renderer */
void renderer_restore_text_attrs(agent_token_renderer *r);
/** The highlighter language for a file path, by extension (Dockerfile and Makefile map to sh).
 *  @param path the path, or NULL
 *  @return a static table entry; the generic language when nothing matches, never NULL */
const agent_syntax *agent_syntax_for_path(const char *path);
/** @return stdout's terminal width in columns, 80 when it is not a terminal */
int renderer_terminal_cols(void);
/** One byte of a code block: buffered into the current line (for highlighting) and echoed plainly; a newline
 *  emits the highlighted line.
 *  @param r the renderer
 *  @param c the byte */
void renderer_code_byte(agent_token_renderer *r, char c);
/** Start a highlighted code stream outside markdown (tool payloads such as write/edit content).
 *  @param r      the renderer
 *  @param syntax the language; NULL = generic */
void renderer_code_stream_begin(agent_token_renderer *r,
                                       const agent_syntax *syntax);
/** Set the gutter printed before each code line.
 *  @param r      the renderer
 *  @param prefix the gutter text (borrowed; must outlive the stream), or NULL
 *  @param color  its SGR sequence (borrowed), or NULL */
void renderer_code_stream_set_prefix(agent_token_renderer *r,
                                            const char *prefix,
                                            const char *color);
/** Enable or disable repainting a streamed code row when its line completes (md_code_highlight_upto).
 *  @param r       the renderer
 *  @param enabled the new setting */
void renderer_code_stream_set_upto_marker(agent_token_renderer *r,
                                                 bool enabled);
/** End the code stream: emit a pending non-blank line and clear the code-block state and gutter.
 *  @param r the renderer */
void renderer_code_end(agent_token_renderer *r);
/** One byte of assistant text: through the markdown state machine, or raw inside thinking / without markdown.
 *  @param r the renderer
 *  @param c the byte */
void renderer_write_char(agent_token_renderer *r, char c);
/** End the stream: flush pending markdown and UTF-8, reset colour, and close with a blank line when anything
 *  visible was written.
 *  @param r the renderer */
void renderer_finish(agent_token_renderer *r);
/** Switch to an SGR sequence after flushing pending text; NULL, "" or a reset mark no colour open.
 *  @param r   the renderer
 *  @param seq the escape sequence, or NULL */
void renderer_color(agent_token_renderer *r, const char *seq);
/** Write bytes verbatim (no markdown) after flushing pending text, tracking visible output and line start.
 *  @param r the renderer
 *  @param s bytes
 *  @param n byte count */
void renderer_plain(agent_token_renderer *r, const char *s, size_t n);
/** This is the single streaming display state machine for assistant output.  It
 * hides raw DSML as soon as the tool_calls marker is complete, lets the DSML
 * parser continue building executable calls, and paints semantic tool output
 * from parser state changes.  The sampled transcript remains unchanged: only
 * the terminal projection is rewritten.
 */
/** @param sr     the stream
 *  @param text   decoded bytes of the next token(s)
 *  @param len    byte count
 *  @param finish the reply ended: flush held bytes and close any open DSML display */
void agent_stream_text(agent_stream_renderer *sr, const char *text, size_t len, bool finish);
/** pulsar_session progress callback: on prefill_chunk / prefill_display events, updates status.prefill_done
 *  (relative to progress_base) and prefill_tps and wakes the UI.  Takes w->mu.
 *  @param ud      the agent_worker
 *  @param event   the event name
 *  @param current tokens processed so far (absolute position)
 *  @param total   unused */
void worker_progress_cb(void *ud, const char *event, int current, int total);
/** Whether an interrupt or stop is pending.  Takes w->mu.
 *  @param w the worker
 *  @return true to abandon the current operation */
bool worker_should_interrupt(agent_worker *w);
/** Ctrl+C is a latched request consumed by the worker.  Once an interrupted
 * operation has reached a stable append-only boundary and is about to publish
 * IDLE, the request must be acknowledged; otherwise the editor can observe an
 * idle worker with a stale interrupt still pending.
 */
/** @param w the worker; takes w->mu */
void worker_clear_interrupt(agent_worker *w);
/** Whether an error string is the session's "interrupted" (a cancel, not a failure).
 *  @param err the error, or NULL
 *  @return true for an interrupt */
bool agent_err_is_interrupted(const char *err);
/** pulsar_session cancel callback: worker_should_interrupt.
 *  @param ud the agent_worker
 *  @return true to cancel */
bool worker_cancel_session_cb(void *ud);
/** Append bytes, capping the buffer at 128 KiB: past it the rest is dropped and b->truncated set (then every
 *  later append is a no-op).  Keeps it NUL-terminated.
 *  @param b the buffer
 *  @param s bytes
 *  @param n byte count */
void agent_buf_append(agent_buf *b, const char *s, size_t n);
/** agent_buf_append of a NUL-terminated string.
 *  @param b the buffer
 *  @param s the string */
void agent_buf_puts(agent_buf *b, const char *s);
/** Take the buffer's string and reset it to empty.
 *  @param b the buffer
 *  @return its contents ("" when empty); caller frees */
char *agent_buf_take(agent_buf *b);
/** Whether two token vectors hold the same ids.
 *  @param a a vector (NULL compares unequal)
 *  @param b a vector (NULL compares unequal)
 *  @return true when equal */
bool agent_tokens_equal(const pulsar_tokens *a, const pulsar_tokens *b);
/** mkdir -p with mode 0700.
 *  @param path the directory
 *  @return true when it exists afterwards */
bool agent_mkdir_p(const char *path);
/** @return "$HOME/.ds4/kvcache" ("." for an unset HOME), malloc'd; caller frees */
char *agent_default_cache_dir(void);
/** The path of a session file: `<dir>/<sha>.session`.
 *  @param dir the cache directory
 *  @param sha the session identity, 40 hex chars + NUL
 *  @return the path, malloc'd; caller frees */
char *agent_session_path_for_sha(const char *dir, const char sha[41]);
/** A session's stable identity: SHA-1 hex of its title and creation time (not its transcript, so resaves keep
 *  the same file).
 *  @param title      the title, or NULL
 *  @param created_at unix seconds
 *  @param sha_out    receives 40 hex chars + NUL */
void agent_session_identity_sha(const char *title, uint64_t created_at,
                                       char sha_out[41]);
/** Forget the live session's identity (sha, title, creation time): the next save makes a new session.
 *  @param w the worker */
void agent_worker_clear_session_identity(agent_worker *w);
/** The KV segment store holds this much before evicting least-recently-used
 *  leaves -- the server's default budget. */
#define AGENT_KV_BUDGET_MB 65536
/** Free a session file's title and tokens and zero it.
 *  @param f the file contents */
void agent_session_file_free(agent_session_file *f);
/** Write a session file atomically (a temp file renamed over `path`).
 *  @param path    the destination
 *  @param f       the contents
 *  @param err     receives the reason on failure
 *  @param err_len its capacity
 *  @return true on success */
bool agent_session_file_write(const char *path, const agent_session_file *f, char *err, size_t err_len);
/** Reads and verifies (size, digest) a session file; false with `err` when it
 *  is not one. */
/** @param path    the file
 *  @param f       receives the contents (owned by the caller: agent_session_file_free); zeroed first
 *  @param err     receives the reason on failure
 *  @param err_len its capacity
 *  @return true when the file is a valid session */
bool agent_session_file_read(const char *path, agent_session_file *f, char *err, size_t err_len);
/** Opens w->kv at `<cache_dir>/segments`, keyed by model and routed format;
 *  false (said once) leaves sessions saving without cached KV. */
/** @param w the worker (cache_dir set)
 *  @return true when the store opened */
bool agent_kv_open(agent_worker *w);
/** Loads the deepest stored chain for `tokens`' rendered text into the live
 *  session; returns the tokens it covers.  The caller syncs `tokens` after. */
/** @param w      the worker
 *  @param tokens the transcript to restore
 *  @return tokens covered by the loaded chain; 0 when none (or no store) */
int agent_kv_load(agent_worker *w, const pulsar_tokens *tokens);
/** Extends the store's chain for the live session's history; a failure is
 *  reported (`what` names it), never fatal. */
/** @param w    the worker
 *  @param what names the persisted history in the failure message */
void agent_kv_persist(agent_worker *w, const char *what);
/** The system prompt's stored chain tip ("" when none). */
/** @param w   the worker
 *  @param tip receives the tip's 40 hex chars + NUL, or "" */
void agent_kv_system_tip(agent_worker *w, char tip[41]);
/** Render the transcript's head through the family's chat front (pulsar_chat_open): the built-in tool prompt as
 *  trusted text, then the configured -sys text.
 *  @param w   the worker
 *  @param out receives the head's tokens (appended) */
void agent_worker_build_system_tokens(agent_worker *w, pulsar_tokens *out);
/** Add a message to the turn being built (agent_worker::turn); `content` is copied. */
/** @param w       the worker
 *  @param role    the message role ("system", "user", "tool")
 *  @param content the text, copied
 *  @param trusted render as control text (pulsar_chat_message::trusted) */
void agent_turn_add(agent_worker *w, const char *role, const char *content, bool trusted);
/** The turn being built plus `extra` (n_extra messages after it), rendered as ONE turn onto `tokens`, with the
 *  assistant turn opened for `think_mode` -- what the next prefill will hold. */
/** @param w          the worker
 *  @param tokens     receives the rendering (appended)
 *  @param extra      messages after the turn, or NULL
 *  @param n_extra    entries in `extra`
 *  @param think_mode the think mode the assistant turn opens with */
void agent_turn_render(const agent_worker *w, pulsar_tokens *tokens, const pulsar_chat_message *extra, int n_extra,
                       pulsar_think_mode think_mode);
/** Render the turn being built onto the transcript with the assistant turn opened, and empty it. */
/** @param w          the worker
 *  @param think_mode the think mode the assistant turn opens with */
void agent_turn_flush(agent_worker *w, pulsar_think_mode think_mode);
/** Drop the turn being built (a new or switched session starts without one). */
/** @param w the worker */
void agent_turn_clear(agent_worker *w);
/** Publish a one-line system status (a marked, coloured line on a terminal); silent in non-interactive mode.
 *  @param w   the worker
 *  @param msg the text */
void agent_publish_system_status(agent_worker *w, const char *msg);
/** When a model turn finishes with a tool call, queued user messages should not
 * preempt that tool.  The worker asks the UI thread for the queue contents only
 * after the tool result is appended, so the next model input can contain both
 * the tool observation and the user's pending correction.
 */
/** Blocks the worker until the UI answers (worker_answer_queued_user_drain) or the worker is stopped.
 *  @param w the worker; worker thread
 *  @return the queued text (caller frees), or NULL when nothing was queued or the worker stopped */
char *worker_request_queued_user_drain(agent_worker *w);
/** UI thread: claim a pending queued-message request from the worker.  Takes w->mu.
 *  @param w the worker
 *  @return true when the worker is waiting for an answer */
bool worker_take_queued_user_drain_request(agent_worker *w);
/** UI thread: answer the worker's queued-message request and wake it.  Takes w->mu.
 *  @param w    the worker
 *  @param text the queued text (ownership passes to the worker), or NULL for none */
void worker_answer_queued_user_drain(agent_worker *w, char *text);
/** Synchronise the live session to a transcript (pulsar_session_sync): only the suffix past the live prefix is
 *  prefilled.  Interruptible (worker_cancel_session_cb).  Worker thread.
 *  @param w                the worker
 *  @param tokens           the transcript
 *  @param publish_progress show a prefill progress footer
 *  @param err              receives the reason on failure ("interrupted" on a cancel)
 *  @param err_len          its capacity
 *  @return 0, or nonzero on failure */
int agent_worker_sync_tokens(agent_worker *w, const pulsar_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len);
/** Start a new session at the system/tool prompt: its KV is the shortest chain
 * in the segment store, restored when the rendered prompt is unchanged and
 * prefilled and stored when not.  The store is per model and routed format, so
 * switching model families never restores incompatible KV.
 */
/** @param w       the worker; worker thread
 *  @param err     receives the reason on failure
 *  @param err_len its capacity
 *  @return true on success */
bool agent_worker_reset_to_sysprompt(agent_worker *w, char *err, size_t err_len);
/** Whether the session holds user activity beyond the system prompt.  Takes w->mu.
 *  @param w the worker
 *  @return true after user activity */
bool agent_worker_has_user_session(agent_worker *w);
/** Whether the session has user activity not yet saved.  Takes w->mu.
 *  @param w the worker
 *  @return true when dirty */
bool agent_worker_needs_save(agent_worker *w);
/** Save the live session now: sync the KV, persist its chain, write `<cache_dir>/<sha>.session` under its
 *  stable identity (titling and dating it on first save).  Worker thread, or with the worker idle.
 *  @param w          the worker
 *  @param sha_out    receives the identity (40 hex + NUL), or NULL
 *  @param tokens_out receives the transcript length saved, or NULL
 *  @param err        receives the reason on failure ("nothing to save" without user activity)
 *  @param err_len    its capacity
 *  @return true on success */
bool agent_worker_save_session_now(agent_worker *w, char sha_out[41],
                                          int *tokens_out,
                                          char *err, size_t err_len);
/** /save from the UI: agent_worker_save_session_now when the worker is idle, printing the saved sha.
 *  @param w       the worker
 *  @param err     receives the reason on failure ("model is busy" when not idle)
 *  @param err_len its capacity
 *  @return true on success */
bool agent_worker_save_session(agent_worker *w, char *err, size_t err_len);
/** A session title from a user prompt (normalised whitespace, cut to max_bytes).
 *  @param prompt    the prompt, or NULL
 *  @param max_bytes display budget; 0 = the full normalised title
 *  @return the title ("(empty user prompt)" for an empty one), malloc'd; caller frees */
char *agent_session_title_from_prompt(const char *prompt,
                                             size_t max_bytes);
/** The loaded family's turn markers as rendered text (agent_turn_marks), from the renderer's family-neutral API:
 *  pulsar_chat_turn_markers' tokens and pulsar_chat_end_assistant's, each rendered by pulsar_token_text.
 *  @param e the engine (its tokenizer)
 *  @param m receives the markers; one the family does not spell (or longer than its buffer) is left empty */
void agent_turn_marks_init(pulsar_engine *e, agent_turn_marks *m);
/** A session title from the first user turn of a rendered transcript: the text after the first user marker, up to
 *  the next assistant, user or end marker.
 *  @param marks     the loaded family's markers (agent_turn_marks_init)
 *  @param text      the rendered transcript
 *  @param text_len  byte count
 *  @param max_bytes display budget; 0 = the full normalised title
 *  @return the title ("(no user prompt)" when none is found), malloc'd; caller frees */
char *agent_session_title_from_text(const agent_turn_marks *marks, const char *text, size_t text_len,
                                    size_t max_bytes);
/** Print the live session's recent turns without touching the session.
 *  @param w          the worker (must be idle)
 *  @param user_turns user turns to show
 *  @param err        receives the reason on failure
 *  @param err_len    its capacity
 *  @return true on success */
bool agent_worker_show_history(agent_worker *w, int user_turns,
                                      char *err, size_t err_len);
/** Print resumable sessions from ~/.ds4/kvcache.  sysprompt.kv is intentionally
 * ignored because it is an implementation cache, not a user session.
 */
/** @param w the worker (its cache_dir is scanned) */
void agent_worker_list_sessions(agent_worker *w);
/** linenoise completion for /switch: saved session shas by recent use, filtered by the typed hex prefix.
 *  Reads agent_completion_worker.
 *  @param buf the line typed so far
 *  @param lc  receives the completions */
void agent_switch_completion_callback(const char *buf,
                                             linenoiseCompletions *lc);
/** Delete a saved session's file and release its unshared KV.
 *  @param w       the worker (must be idle)
 *  @param prefix  a unique prefix of the session's sha
 *  @param sha_out receives the deleted session's sha, or NULL
 *  @param err     receives the reason on failure
 *  @param err_len its capacity
 *  @return true on success */
bool agent_worker_delete_session(agent_worker *w, const char *prefix,
                                        char sha_out[41],
                                        char *err, size_t err_len);
/** Releases a saved session's cached KV that no other chain shares (never the
 *  system prompt's); the session stays and later prefills what was released.
 *  *bytes_out: what was freed. */
/** @param w         the worker (must be idle)
 *  @param prefix    a unique prefix of the session's sha
 *  @param sha_out   receives the session's sha, or NULL
 *  @param bytes_out receives the bytes freed, or NULL
 *  @param err       receives the reason on failure
 *  @param err_len   its capacity
 *  @return true on success */
bool agent_worker_strip_session(agent_worker *w, const char *prefix,
                                       char sha_out[41],
                                       uint64_t *bytes_out,
                                       char *err, size_t err_len);
/** Load a saved session into the live transcript (the deepest stored KV chain, then prefill of the rest) and
 *  replay its recent history.
 *  @param w             the worker (must be idle)
 *  @param prefix        a unique prefix of the session's sha
 *  @param history_turns user turns of history to replay
 *  @param err           receives the reason on failure
 *  @param err_len       its capacity
 *  @return true on success */
bool agent_worker_switch_session(agent_worker *w, const char *prefix,
                                        int history_turns,
                                        char *err, size_t err_len);
/** Parse a tool's timeout argument in seconds.
 *  @param s the text, or NULL
 *  @return seconds clamped to [1, 86400]; 3600 when absent, unparsable or <= 0 */
int agent_parse_timeout(const char *s);
/** Parse a whole-string decimal integer (trailing whitespace allowed).
 *  @param s   the text, or NULL
 *  @param def returned when absent or unparsable
 *  @param min lower clamp
 *  @param max upper clamp
 *  @return the clamped value, or def */
int agent_parse_int_default(const char *s, int def, int min, int max);
/** Parse true/yes/1 or false/no/0 (case-insensitive words).
 *  @param s   the text, or NULL
 *  @param def returned when absent or unrecognised
 *  @return the value */
bool agent_parse_bool_default(const char *s, bool def);
/** Free a span list and zero it.
 *  @param spans the list */
void agent_line_spans_free(agent_line_spans *spans);
/** Split a text buffer into line spans.  content_end excludes CR/LF so callers
 * can print or compare line content without newline spelling differences.
 */
/** @param data  the buffer
 *  @param len   byte count
 *  @param spans receives one span per line (appended; LF, CRLF and lone CR all end a line) */
void agent_split_lines(const char *data, size_t len, agent_line_spans *spans);
/** Read a whole file (at most AGENT_FILE_MAX_BYTES).
 *  @param path   the file
 *  @param data   receives the bytes, NUL-terminated, malloc'd; caller frees
 *  @param len    receives the byte count
 *  @param err    receives the reason on failure
 *  @param errlen its capacity
 *  @return 0, or -1 on failure */
int agent_read_file_bytes(const char *path, char **data, size_t *len,
                                 char *err, size_t errlen);
/** The lines an edit touched, for the edit result: 1-based old lines [start_line, end_line] holding the
 *  replaced bytes, and the change in line count.
 *  @param old_data     the file before the edit
 *  @param old_len      its byte count
 *  @param new_data     the file after the edit
 *  @param new_len      its byte count
 *  @param edit_offset  byte offset of the replaced span in old_data
 *  @param replaced_len bytes replaced
 *  @param start_line   receives the first touched line, or NULL
 *  @param end_line     receives the last touched line, or NULL
 *  @param delta        receives new line count - old line count, or NULL
 *  @return false when the old file had no lines (outputs untouched) */
bool agent_old_new_line_effect(const char *old_data, size_t old_len,
                                      const char *new_data, size_t new_len,
                                      size_t edit_offset, size_t replaced_len,
                                      int *start_line, int *end_line,
                                      int *delta);
/** The model-visible result of a successful edit: what changed, the line shift, and the post-edit context
 *  around the touched lines.
 *  @param path       the file
 *  @param start_line first touched old line, 1-based (<= 0: no line report)
 *  @param end_line   last touched old line
 *  @param delta      line-count change
 *  @param new_data   the file after the edit
 *  @param new_len    its byte count
 *  @param kind       how the edit was made, as the result names it
 *  @return the text, malloc'd; caller frees */
char *agent_edit_result(const char *path,
                                       int start_line, int end_line, int delta,
                                       const char *new_data, size_t new_len,
                                       const char *kind);
/** Whether the transcript plus the pending turn plus `result` as a tool message still leaves reserve_tokens
 *  free in the context.
 *  @param w              the worker
 *  @param result         the tool result
 *  @param reserve_tokens tokens to keep free
 *  @param tokens_out     receives the projected prompt length, or NULL
 *  @return true when it fits */
bool agent_tool_result_fits_context(agent_worker *w, const char *result,
                                           int reserve_tokens,
                                           int *tokens_out);
/** The read tool (path, start_line, max_lines, whole, raw).
 *  @param w    the worker (records the `more` continuation)
 *  @param call the call
 *  @return the result text, malloc'd; caller frees */
char *agent_tool_read(agent_worker *w, const agent_tool_call *call);
/** The more tool: continue the last read (count lines).
 *  @param w    the worker
 *  @param call the call
 *  @return the result text, malloc'd; caller frees */
char *agent_tool_more(agent_worker *w, const agent_tool_call *call);
/** The write tool: replace a file's contents (path, content).
 *  @param w    the worker (unused)
 *  @param call the call
 *  @return the result text, malloc'd; caller frees */
char *agent_tool_write(agent_worker *w, const agent_tool_call *call);
/** The list tool: a directory's entries (path; default ".").
 *  @param call the call
 *  @return the result text, malloc'd; caller frees */
char *agent_tool_list(const agent_tool_call *call);
/** Append the post-edit file around 1-based lines [anchor_start, anchor_end] (a few lines either side; a long
 *  edited span shows its head and tail) to an edit result.
 *  @param b            the result being built
 *  @param path         the file, for the header
 *  @param data         the file after the edit
 *  @param len          its byte count
 *  @param anchor_start first edited line
 *  @param anchor_end   last edited line */
void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end);
/** While an edit's old argument streams, whether the next sampled token should be replaced by an [upto]
 *  marker line because the emitted prefix is already a unique anchor in the file.
 *  @param forcer   the forcer's state (reset when the parser is outside edit/old)
 *  @param p        the DSML parser
 *  @param next_text the next token's text
 *  @param next_len  its byte count
 *  @return true to evaluate the marker instead of the token */
bool agent_edit_upto_forcer_should_replace(agent_edit_upto_forcer *forcer,
                                                  agent_dsml_parser *p,
                                                  const char *next_text,
                                                  size_t next_len);
/** Check, before the edit runs, that its old text matches the file uniquely (the [upto] rules included).
 *  @param w       the worker (unused)
 *  @param call    the edit call
 *  @param err     receives the reason on failure
 *  @param err_len its capacity
 *  @return true when it matches, or when no path is known yet */
bool agent_preflight_edit_old(agent_worker *w, const agent_tool_call *call,
                                     char *err, size_t err_len);
/** Old/new editing is intentionally conservative: exact old text must be unique.
 * For large replacements, old may contain one [upto] marker: the head must be
 * unique, and the tail must be unique after that head before the whole span is
 * replaced.
 */
/** @param w    the worker
 *  @param call the edit call (path, old, new)
 *  @return the result text, malloc'd; caller frees */
char *agent_tool_edit(agent_worker *w, const agent_tool_call *call);
/** Implement the search tool using either literal matching or POSIX regex. */
/** @param w    the worker
 *  @param call the search call
 *  @return the result text, malloc'd; caller frees */
char *agent_tool_search(agent_worker *w, const agent_tool_call *call);
/** Free every bash job (killing running ones) and empty the list.
 *  @param w the worker */
void agent_bash_jobs_free(agent_worker *w);
/** Find a job by id, or by pid when id <= 0.
 *  @param w   the worker
 *  @param id  the job id; <= 0 = match by pid
 *  @param pid the process id; used when id <= 0 (> 0 to match)
 *  @return the job (owned by the list), or NULL */
agent_bash_job *agent_bash_find_job(agent_worker *w, int id, pid_t pid);
/** Unlink a job from the list and free it.
 *  @param w      the worker
 *  @param target the job; no-op when not in the list */
void agent_bash_remove_job(agent_worker *w, agent_bash_job *target);
/** Spawn `cmd` under /bin/sh in its own process group, output to a pipe and a /tmp file, and add it to the
 *  worker's job list.
 *  @param w           the worker
 *  @param cmd         the shell command (copied)
 *  @param timeout_sec seconds before the job is killed
 *  @param err         receives the reason on failure
 *  @param err_len     its capacity
 *  @return the job (owned by the list), or NULL on failure */
agent_bash_job *agent_bash_start(agent_worker *w, const char *cmd,
                                        int timeout_sec, char *err, size_t err_len);
/** The bash / bash_status / bash_stop result: optionally stop the job (SIGTERM, then SIGKILL after 1 s), wait
 *  up to refresh_sec for it, then observe it and publish the observation.
 *  @param w              the worker
 *  @param job            the job
 *  @param wait           wait up to refresh_sec for it to finish
 *  @param refresh_sec    the wait, seconds
 *  @param stop           stop the job first
 *  @param remove_if_done free the job if it has finished
 *  @return the observation, malloc'd; caller frees */
char *agent_bash_job_tool_result(agent_worker *w, agent_bash_job *job,
                                        bool wait, int refresh_sec,
                                        bool stop, bool remove_if_done);
/** @param call a bash tool call
 *  @return its job argument, 0 when absent */
int agent_tool_job_id(const agent_tool_call *call);
/** @param call a bash tool call
 *  @return its pid argument, 0 when absent */
pid_t agent_tool_pid(const agent_tool_call *call);
/** Execute all tool calls from one DSML block, preserving per-call labels in the
 * combined result so the model can associate observations with calls.
 */
/** @param w     the worker
 *  @param calls the parsed calls
 *  @return the combined result, malloc'd; caller frees */
char *agent_execute_tool_calls(agent_worker *w, const agent_tool_calls *calls);
/** If compaction happens while a bash process is still alive, inject a small
 * tool-role reminder into the rebuilt transcript.  Otherwise the summary could
 * preserve the user's task but lose the fact that an external process still
 * needs status/wait/stop handling.
 */
/** @param w the worker
 *  @return the reminder text, malloc'd (caller frees), or NULL when no job exists */
char *agent_bash_jobs_compaction_observation(agent_worker *w);
/** Compact now: ask the model for a summary of the task state and rebuild the transcript as system prompt +
 *  summary + recent verbatim tail.  Any failure invalidates the live KV.  Worker thread.
 *  @param w       the worker
 *  @param reason  why, for the status line
 *  @param err     receives the reason on failure ("interrupted" on a cancel)
 *  @param err_len its capacity
 *  @return true on success (or an empty transcript) */
bool agent_worker_compact(agent_worker *w, const char *reason,
                                 char *err, size_t err_len);
/** agent_worker_compact when the transcript passes AGENT_COMPACT_SOFT_PERCENT or leaves too few tokens free.
 *  @param w       the worker
 *  @param reason  why, for the status line
 *  @param err     receives the reason on failure
 *  @param err_len its capacity
 *  @return true when no compaction was needed or it succeeded */
bool agent_worker_compact_if_needed(agent_worker *w, const char *reason,
                                           char *err, size_t err_len);
/** Evaluate a generated token, append it to the transcript, trace and stream it, and update the generation
 *  status.  Worker thread.
 *  @param w         the worker
 *  @param token     the token
 *  @param generated the turn's token count, incremented
 *  @param t0        the turn's start (agent_now_sec), for tokens/s
 *  @param stream    the display stream
 *  @param err       receives the reason on failure
 *  @param err_len   its capacity
 *  @return 0, or 1 when the eval failed */
int worker_accept_generated_token(agent_worker *w,
                                         int token,
                                         int *generated,
                                         double t0,
                                         agent_stream_renderer *stream,
                                         char *err,
                                         size_t err_len);
/** Tokenize `text` and accept its tokens as if generated (worker_accept_generated_token each).
 *  @param w          the worker
 *  @param text       the text to force
 *  @param max_tokens the turn's token cap
 *  @param generated  the turn's token count, incremented
 *  @param t0         the turn's start (agent_now_sec)
 *  @param stream     the display stream
 *  @param err        receives the reason on failure
 *  @param err_len    its capacity
 *  @return 0, or 1 when the text does not fit the cap or an eval failed */
int worker_force_generated_text(agent_worker *w,
                                       const char *text,
                                       int max_tokens,
                                       int *generated,
                                       double t0,
                                       agent_stream_renderer *stream,
                                       char *err,
                                       size_t err_len);
/** Ask the worker to save at its next stable point (busy /save).  Takes w->mu.
 *  @param w the worker */
void worker_request_save(agent_worker *w);
/** Ask the worker to compact at its next stable point (/compact).  Takes w->mu.
 *  @param w the worker */
void worker_request_compact(agent_worker *w);
/** Worker thread entry point.  The UI thread submits plain user text; this
 * thread owns all DS4 session mutation, tool execution, and compaction.
 */
/** @param arg the agent_worker
 *  @return NULL */
void *worker_main(void *arg);
/** Set or clear O_NONBLOCK on a descriptor.
 *  @param fd        the descriptor
 *  @param on        true to set it
 *  @param old_flags receives the previous flags, or NULL
 *  @return 0, or -1 on failure */
int set_nonblock(int fd, bool on, int *old_flags);
/** Check and clear the raw_mode_needs_restore flag under the worker mutex.
 * Returns true if the UI thread should verify/reapply linenoise raw mode.
 */
/** @param w the worker
 *  @return true when the UI should verify raw mode (the flag is cleared) */
bool worker_check_raw_mode_restore(agent_worker *w);
/** Read the wake pipe empty (non-blocking descriptor).
 *  @param fd its read end */
void drain_wake_fd(int fd);
/** Submit one user turn if the worker is idle.  Busy submissions are rejected so
 * the UI can keep the typed text editable instead of silently queueing it.
 */
/** @param w    the worker
 *  @param text the user text (copied)
 *  @return true when accepted */
bool worker_submit(agent_worker *w, const char *text);
/** Request interruption at the next model/tool polling point. */
/** @param w the worker */
void worker_interrupt(agent_worker *w);
/** Stop the worker thread. */
/** @param w the worker */
void worker_stop(agent_worker *w);
/** The UI thread consumes output in batches.  Taking ownership of w->out under
 * the mutex keeps terminal writes outside the lock while preserving order.
 */
/** @param w       the worker
 *  @param out     receives the queued output (caller frees; NULL when none), or NULL to leave it queued
 *  @param out_len receives its byte count (with out)
 *  @param status  receives a status snapshot, or NULL */
void worker_consume(agent_worker *w, char **out, size_t *out_len, agent_status *status);
/** Snapshot the worker's status (context use refreshed).  Takes w->mu.
 *  @param w      the worker
 *  @param status receives it */
void worker_get_status(agent_worker *w, agent_status *status);
/** Whether the worker is initialized and IDLE or in ERROR.  Takes w->mu.
 *  @param w the worker
 *  @return true when idle */
bool worker_is_idle(agent_worker *w);
/** Whether the worker finished initializing.  Takes w->mu.
 *  @param w      the worker
 *  @param status receives a status snapshot, or NULL
 *  @return true once initialized */
bool worker_is_initialized(agent_worker *w, agent_status *status);
/** @return true when stdout is a terminal */
bool stdout_is_tty(void);
/** The echo of a submitted user prompt ("* text", coloured on a terminal, then a blank line).
 *  @param text the prompt
 *  @return the echo, malloc'd; caller frees */
char *agent_format_user_prompt_echo(const char *text);
/** Print agent_format_user_prompt_echo to stdout.
 *  @param text the prompt */
void agent_echo_user_prompt(const char *text);
/** The editor's prompt string ("pulsar-agent> ").
 *  @param st  the status (unused)
 *  @param buf receives it
 *  @param len its capacity */
void build_prompt_text(const agent_status *st, char *buf, size_t len);
/** @return a fresh prefill label index (a process-wide counter; not thread-safe) */
unsigned agent_next_prefill_label(void);
/** Append a prompt.
 *  @param q    the queue
 *  @param text the prompt (copied; NULL = "") */
void agent_prompt_queue_push(agent_prompt_queue *q, const char *text);
/** Remove the oldest prompt.
 *  @param q the queue
 *  @return it (caller frees), or NULL when empty */
char *agent_prompt_queue_pop(agent_prompt_queue *q);
/** Put a prompt back at the front.
 *  @param q    the queue
 *  @param text the prompt; ownership passes to the queue */
void agent_prompt_queue_push_front(agent_prompt_queue *q, char *text);
/** Empty the queue into one user message (several prompts are numbered "Queued user message N:").
 *  @param q the queue
 *  @return the message (caller frees), or NULL when empty */
char *agent_prompt_queue_take_all(agent_prompt_queue *q);
/** The echoes of every queued prompt; the queue is left as it is.
 *  @param q the queue
 *  @return the text (caller frees), or NULL when empty */
char *agent_prompt_queue_take_all_echo(agent_prompt_queue *q);
/** Free every queued prompt and zero the queue.
 *  @param q the queue */
void agent_prompt_queue_free(agent_prompt_queue *q);
/** The editor footer: the status row, preceded by a preview of the first queued prompt when any.
 *  @param st    the worker status
 *  @param queue the prompt queue, or NULL
 *  @param cols  terminal width (at least 40 is assumed)
 *  @param buf   receives the text
 *  @param len   its capacity */
void build_footer_text(const agent_status *st, const agent_prompt_queue *queue,
                              int cols, char *buf, size_t len);
/** Drain stdin in nonblocking mode.  The outer event loop decides when queued
 * bytes are fed to linenoiseEditFeed().
 */
/** @param ed the editor */
void editor_read_stdin(agent_editor *ed);
/** Remove the first occurrence of `byte` from the editor's unread input.
 *  @param ed   the editor
 *  @param byte the byte
 *  @return true when one was found and removed */
bool editor_take_queued_byte(agent_editor *ed, unsigned char byte);
/** Consume a lone ESC held in the cursor-report buffer.
 *  @param ed the editor
 *  @return true when one was held (and is now dropped) */
bool editor_take_bare_escape(agent_editor *ed);
/** Replace the edit line's contents (showing the editor first if hidden).
 *  @param ed   the editor
 *  @param text the new text, or NULL / "" to clear */
void editor_replace_input(agent_editor *ed, const char *text);
/** Undo the scroll-region layout: reset attributes and the region, and leave the cursor on a fresh last line.
 *  @param ed the editor */
void editor_restore_terminal_layout(agent_editor *ed);
/** Start linenoise non-blocking with the status footer.
 *  @param ed      the editor
 *  @param prompt  the prompt (copied)
 *  @param status  the footer text (copied), or NULL
 *  @param initial text to pre-fill, or NULL
 *  @return 0, or -1 when linenoise could not start */
int editor_start(agent_editor *ed, const char *prompt,
                        const char *status, const char *initial);
/** Stop the live editor and restore stdin flags. */
/** @param ed the editor; no-op when not active */
void editor_stop(agent_editor *ed);
/** Restore the live prompt after output.  The primary path draws it in the
 * reserved bottom rows; the fallback path keeps the older one-row-below-output
 * trick for terminals where scroll regions are unavailable.
 */
/** @param ed the editor; no-op unless active and hidden */
void editor_show(agent_editor *ed);
/** Change the prompt and footer, redrawing only when something changed.
 *  @param ed     the editor
 *  @param prompt the prompt
 *  @param status the footer text, or NULL */
void editor_set_prompt_status(agent_editor *ed, const char *prompt,
                                     const char *status);
/** Write model/tool output without corrupting the live prompt (the terminal contract: in scroll-region mode
 *  output goes above the protected prompt rows; otherwise the prompt is hidden and redrawn).
 *  @param ed         the editor
 *  @param text       the output
 *  @param len        byte count
 *  @param prompt     the prompt to show afterwards
 *  @param status     the footer to show afterwards, or NULL
 *  @param force_show redraw the prompt now rather than at the throttled interval */
void editor_write_async(agent_editor *ed, const char *text, size_t len,
                               const char *prompt, const char *status,
                               bool force_show);
/** Ctrl+C while idle: clear the edit line and print a "press Ctrl+D to exit" hint.
 *  @param ed     the editor
 *  @param prompt the prompt
 *  @param status the footer, or NULL */
void editor_cancel_input_with_hint(agent_editor *ed,
                                          const char *prompt,
                                          const char *status);
/** Print the interactive commands and keys (/help). */
void runtime_help(void);
/** Write the startup banner (model and context) through editor_write_async.
 *  @param editor     the editor
 *  @param cfg        the configuration
 *  @param prompt     the prompt
 *  @param statusline the footer */
void editor_write_welcome_banner(agent_editor *editor,
                                        const agent_config *cfg,
                                        const char *prompt,
                                        const char *statusline);
/** Initialize the worker, cache directory, sysprompt checkpoint path, trace file,
 * and model thread.  After this returns, all DS4 session mutation happens on
 * the worker thread.
 */
/** @param w      the worker (zeroed here)
 *  @param engine the loaded engine (borrowed)
 *  @param cfg    the configuration (borrowed; must outlive the worker)
 *  @return 0, or -1 on failure */
int agent_worker_init(agent_worker *w, pulsar_engine *engine, agent_config *cfg);
/** Shut down the worker and release owned resources, including any live bash
 * process groups.
 */
/** @param w the worker */
void agent_worker_free(agent_worker *w);
/** Ask a y/n question on the terminal, optionally answering itself after a timeout.
 *  @param prompt    the question
 *  @param opts      the timeout policy, or NULL for none
 *  @param timed_out receives true when the timeout answered, or NULL
 *  @return the answer; false on EOF or a read error */
bool agent_prompt_yes_no_ex(const char *prompt,
                                   const agent_yes_no_options *opts,
                                   bool *timed_out);
/** Ask before discarding a dirty user session.  Fresh sessions that contain only
 * the system prompt are deliberately ignored.
 */
/** @param w the worker
 *  @return true to proceed (saved, declined, or confirmed after a failed save); false to stay */
bool agent_maybe_save_before_leaving_session(agent_worker *w);
/** Process exit is different from /new or /switch: once the terminal is already
 * restored, declining the save can terminate immediately and let the OS reclaim
 * model/GPU resources instead of waiting for orderly teardown.
 */
/** @param w the worker
 *  @return AGENT_EXIT_CLEAN (nothing to save, or saved), AGENT_EXIT_NOW (declined, or confirmed after a failed
 *          save), or AGENT_EXIT_CANCEL (do not exit) */
agent_exit_save_result agent_maybe_save_before_exiting(agent_worker *w);

/* ---- shared inline helpers ---- */


#endif /* PULSAR_AGENT_INTERNAL_H */
