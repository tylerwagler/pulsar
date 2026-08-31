/* pulsar_agent_internal.h — internal shared declarations for the agent sources.
 * Produced by the multi-TU split of pulsar_agent.c; edit freely (the
 * generator is not part of the build). */
#ifndef PULSAR_AGENT_INTERNAL_H
#define PULSAR_AGENT_INTERNAL_H

#include "pulsar.h"
#include "pulsar_help.h"
#include "pulsar_kvstore.h"
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
int linenoiseEditInsert(struct linenoiseState *l, const char *c, size_t clen);

/* ---- shared macros ---- */



#define AGENT_SYSTEM_PROMPT_REMINDER_TOKENS 50000


/* Poor man's code highlighter inspired by antirez/kilo: a tiny language table
 * plus one line-oriented tokenizer for comments, strings, numbers, and
 * separator-bounded keywords.  This is deliberately not a full parser; it is
 * only for making fenced Markdown code readable in the terminal. */
#define AGENT_HL_NORMAL 0
#define AGENT_HL_COMMENT 1
#define AGENT_HL_KEYWORD1 2
#define AGENT_HL_KEYWORD2 3
#define AGENT_HL_STRING 4
#define AGENT_HL_NUMBER 5

#define AGENT_SYNTAX_NUMBERS (1u<<0)
#define AGENT_SYNTAX_STRINGS (1u<<1)
#define AGENT_SYNTAX_BACKTICK_STRINGS (1u<<2)
#define AGENT_SYNTAX_CASE_INSENSITIVE (1u<<3)


#define AGENT_HISTORY_DEFAULT_TURNS 3
#define AGENT_HISTORY_MAX_TURNS 200
#define AGENT_HISTORY_ASSISTANT_MAX_LINES 80
#define AGENT_HISTORY_ASSISTANT_MAX_BYTES 12000


#define AGENT_FILE_MAX_BYTES (16*1024*1024)
#define AGENT_READ_DEFAULT_LINES 500
#define AGENT_TOOL_RESULT_RESERVE_TOKENS 1024
#define AGENT_EDIT_UPTO_MIN_PREFIX_BYTES 64
#define AGENT_EDIT_UPTO_MIN_PREFIX_LINES 2
#define AGENT_COMPACT_SOFT_PERCENT 85
#define AGENT_COMPACT_MIN_FREE_TOKENS 8192
#define AGENT_COMPACT_TAIL_DIVISOR 10
#define AGENT_COMPACT_TAIL_CAP_TOKENS 50000
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

#define AGENT_BASH_HEAD_BYTES (8*1024)
#define AGENT_BASH_HEAD_LINES 100
#define AGENT_BASH_TAIL_BYTES (32*1024)
#define AGENT_BASH_PROGRESS_TAIL_LINES 4
#define AGENT_BASH_FINAL_TAIL_LINES 20

#define AGENT_INPUT_INITIAL_BUFLEN 4096
#define AGENT_INPUT_MAX_BUFLEN (1024*1024)
#define AGENT_STATUS_STYLE_START "\x1b[48;5;238;38;5;252m"
#define AGENT_STATUS_STYLE_END "\x1b[0m"
#define AGENT_STATUS_BAR_FILL "\x1b[48;5;238;38;5;201;1m"
#define AGENT_QUEUE_STYLE "\x1b[38;5;87;1m"
#define AGENT_STATUS_REDRAW_INTERVAL_SEC 0.20
#define AGENT_PROGRESS_BAR_WIDTH 32
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

    char *cache_dir;            ///< directory holding persisted sessions
    char *sysprompt_path;       ///< file the system prompt was loaded from, if any
    char session_sha[41];       ///< session identity: 40 hex chars + NUL
    char *session_title;        ///< human-readable session name for the picker
    uint64_t session_created_at;///< unix seconds when the session was first created
    /** A pre-rename session file to unlink once the new one is safely written.
     * Deferred rather than deleted eagerly so a crash mid-save cannot leave the
     * session with no file at all. */
    char *legacy_session_path_to_delete;
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
    void append(const char *s, size_t n);  ///< was agent_tail_capture_append
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
    bool format_thinking;       ///< style reasoning blocks distinctly
    bool format_markdown;       ///< apply markdown styling at all
    bool in_think;              ///< currently inside a reasoning block
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

    char pending[16];                   ///< bytes withheld pending a formatting decision
    size_t pending_len;                 ///< bytes used in `pending`
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

    bool in_think;                   ///< inside the reasoning region
    bool dsml_active;                ///< a DSML block is being parsed
    /** The active DSML block was opened INSIDE the thinking region. It is
     * still displayed, but it is not a real tool call -- the model is
     * reasoning about a call, not making one. */
    bool dsml_ignored;
    /** Re-rendering saved transcript text rather than streaming live. Suppresses
     * cursor-control escapes, which must never leak into a pipe or a file. */
    bool replay;

    char pending[16];                ///< bytes withheld pending a routing decision
    size_t pending_len;              ///< bytes held in `pending`
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
    const char *line_comment(const char *p) const;  ///< was agent_syntax_line_comment
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

/** Session identity and title read back from a KV-store file. */
typedef struct {
    bool has_title_trailer;  ///< the file carries a title trailer record
    bool legacy_identity;    ///< identity came from the pre-trailer scheme (older file)
    char *title;             ///< session title, owned; NULL when absent
    uint64_t created_at;     ///< unix seconds the session was created
    char sha[41];            ///< session identity: 40 hex chars + NUL
} agent_kv_session_meta;

/** Role marker attached to a line of rendered history, so the transcript view
 * can style turns without re-parsing the token stream. */
typedef enum {
    AGENT_HISTORY_MARK_NONE,      ///< continuation of the previous line
    AGENT_HISTORY_MARK_USER,      ///< first line of a user turn
    AGENT_HISTORY_MARK_ASSISTANT, ///< first line of an assistant turn
    AGENT_HISTORY_MARK_EOS,       ///< end-of-turn boundary
} agent_history_mark;

/** Rendered history as borrowed line pointers plus a parallel marker array.
 * Pointers are NOT owned -- they alias the rendered buffer. */
typedef struct {
    const char **v;           ///< line starts, borrowed
    agent_history_mark *mark; ///< role marker per line, same length as `v`
    int len;                  ///< lines present
    int cap;                  ///< lines allocated
} agent_history_ptrs;

/** One row in the session picker: the store entry plus its resolved title. */
typedef struct {
    pulsar_kvstore_entry entry;  ///< the KV-store record
    char *title;                 ///< display title, owned
} agent_session_list_item;

/** A session remembered for tab-completion, ordered by recency. */
typedef struct {
    char sha[41];        ///< session identity: 40 hex chars + NUL
    uint64_t last_used;  ///< unix seconds of last use; the sort key
} agent_completion_session;

/** Growable list of completion candidates. */
typedef struct {
    agent_completion_session *v;  ///< the sessions
    int len;                      ///< entries present
    int cap;                      ///< entries allocated
} agent_completion_sessions;

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
    int display_lines() const;
    /** Account for `n` freshly captured bytes (byte and newline counters). */
    void note_output(const char *s, size_t n);
    /** Release the job: close descriptors, unlink the temp file, free `cmd`. */
    void job_free();
    /** Read whatever the child has written into the pipe right now. */
    void drain();
    /** Record the child's exit `status` and mark the job no longer running. */
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
    char *read_tail_lines(int max_lines) const;
    /** Build the tool result describing this job: status line, then output.
     * The FIRST observation shows the head of the output; later ones show the
     * tail. @param mark_observed record that the model has now seen the job,
     * which is what flips head display to tail. */
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

typedef enum {
    AGENT_EXIT_CANCEL,
    AGENT_EXIT_CLEAN,
    AGENT_EXIT_NOW,
} agent_exit_save_result;

/* ---- shared globals ---- */

extern volatile sig_atomic_t agent_sigint;
extern agent_worker *agent_completion_worker;
extern const char agent_dsml_syntax_reminder[];

/* ---- shared functions ---- */

void agent_sigint_handler(int sig);
void *agent_xmalloc(size_t n);
char *xstrdup(const char *s);
char *xstrndup(const char *s, size_t n);
void *agent_xrealloc(void *ptr, size_t n);
void write_all(int fd, const char *p, size_t n);
void agent_input_buf_append(agent_input_buf *b, const char *s, size_t n);
char *agent_input_buf_take(agent_input_buf *b);
void agent_input_buf_free(agent_input_buf *b);
bool agent_slash_command_known(const char *cmd);
double agent_now_sec(void);
void usage(FILE *fp, const char *topic);
agent_config parse_options(int argc, char **argv);
void log_context_memory(pulsar_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk);
pulsar_think_mode effective_think_mode(const agent_config *cfg);
void agent_append_system_prompt(pulsar_engine *engine, pulsar_tokens *tokens,
                                       const char *extra);
void agent_worker_note_system_prompt_seen(agent_worker *w);
void agent_worker_maybe_append_datetime_context(agent_worker *w);
/** The full tool/system reminder is separate from DSML syntax errors: it is a
 * pressure-controlled refresh of the same trusted prompt shape used at startup.
 * The built-in prompt is tokenized as rendered chat so DSML markers stay native
 * control tokens; arbitrary -sys text remains ordinary text.
 */
void agent_worker_maybe_append_system_prompt_reminder(agent_worker *w);
/** Wake the UI thread after changing worker-visible state.  The byte in
 * wake_fd is level-triggered with wake_pending so bursts of sampled tokens do
 * not flood the pipe.
 */
void agent_wake_locked(agent_worker *w);
/** Queue rendered output for the UI thread.  The worker never writes directly
 * to the terminal, which keeps linenoise redraws serialized in one place.
 */
void agent_publish(agent_worker *w, const char *s, size_t n);
void agent_publishf(agent_worker *w, const char *fmt, ...);
void agent_set_status(agent_worker *w, agent_worker_state state);
void agent_set_error(agent_worker *w, const char *msg);
void agent_trace(agent_worker *w, const char *fmt, ...);
void agent_trace_token(agent_worker *w, int token, const char *text,
                              size_t text_len, int index);
void agent_trace_tokens(agent_worker *w, const char *label,
                               const pulsar_tokens *tokens, int start);
void agent_trace_text(agent_worker *w, const char *label,
                             const char *text, size_t len);
bool bytes_has_prefix(const char *p, size_t n, const char *prefix);
bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix);
const char *agent_tool_arg_value(const agent_tool_call *call, const char *name);
void agent_dsml_parser_free(agent_dsml_parser *p);
void agent_dsml_parser_reset(agent_dsml_parser *p);
void agent_dsml_set_error(agent_dsml_parser *p, const char *msg);
bool agent_dsml_parameter_close_tail(const char *tail, size_t len,
                                            bool *complete);
void agent_dsml_start(agent_dsml_parser *p);
void agent_dsml_feed(agent_dsml_parser *p, const char *s, size_t n);
void renderer_write(agent_token_renderer *r, const char *s, size_t n);
void renderer_reset_color(agent_token_renderer *r);
void renderer_restore_text_attrs(agent_token_renderer *r);
const agent_syntax *agent_syntax_for_path(const char *path);
int renderer_terminal_cols(void);
void renderer_code_byte(agent_token_renderer *r, char c);
void renderer_code_stream_begin(agent_token_renderer *r,
                                       const agent_syntax *syntax);
void renderer_code_stream_set_prefix(agent_token_renderer *r,
                                            const char *prefix,
                                            const char *color);
void renderer_code_stream_set_upto_marker(agent_token_renderer *r,
                                                 bool enabled);
void renderer_code_end(agent_token_renderer *r);
void renderer_write_char(agent_token_renderer *r, char c);
void renderer_finish(agent_token_renderer *r);
void renderer_color(agent_token_renderer *r, const char *seq);
void renderer_plain(agent_token_renderer *r, const char *s, size_t n);
/** This is the single streaming display state machine for assistant output.  It
 * hides raw DSML as soon as the tool_calls marker is complete, lets the DSML
 * parser continue building executable calls, and paints semantic tool output
 * from parser state changes.  The sampled transcript remains unchanged: only
 * the terminal projection is rewritten.
 */
void agent_stream_text(agent_stream_renderer *sr, const char *text, size_t len, bool finish);
void worker_progress_cb(void *ud, const char *event, int current, int total);
bool worker_should_interrupt(agent_worker *w);
/** Ctrl+C is a latched request consumed by the worker.  Once an interrupted
 * operation has reached a stable append-only boundary and is about to publish
 * IDLE, the request must be acknowledged; otherwise the editor can observe an
 * idle worker with a stale interrupt still pending.
 */
void worker_clear_interrupt(agent_worker *w);
bool agent_err_is_interrupted(const char *err);
bool worker_cancel_session_cb(void *ud);
void agent_buf_append(agent_buf *b, const char *s, size_t n);
void agent_buf_puts(agent_buf *b, const char *s);
char *agent_buf_take(agent_buf *b);
bool agent_tokens_equal(const pulsar_tokens *a, const pulsar_tokens *b);
bool agent_mkdir_p(const char *path);
char *agent_default_cache_dir(void);
char *agent_kv_path_for_sha(const char *dir, const char sha[41]);
void agent_session_identity_sha(const char *title, uint64_t created_at,
                                       char sha_out[41]);
void agent_worker_clear_session_identity(agent_worker *w);
void agent_kv_session_meta_free(agent_kv_session_meta *m);
bool agent_kv_read_text(FILE *fp, uint32_t text_bytes,
                               char **text_out, char *err, size_t err_len);
bool agent_kv_write_title_trailer(FILE *fp, const char *title,
                                         char *err, size_t err_len);
bool agent_kv_read_title_trailer(FILE *fp, const pulsar_kvstore_entry *hdr,
                                        char **title_out,
                                        char *err, size_t err_len);
void agent_kv_identity_sha(const pulsar_kvstore_entry *hdr,
                                  const char *text, uint32_t text_bytes,
                                  const char *title,
                                  char sha_out[41]);
bool agent_kv_load_path(agent_worker *w, const char *path,
                               const char *expected_sha,
                               const char *expected_text,
                               size_t expected_text_len,
                               pulsar_tokens *loaded_tokens,
                               agent_kv_session_meta *meta_out,
                               char *err, size_t err_len);
void agent_worker_build_system_tokens(agent_worker *w, pulsar_tokens *out);
void agent_publish_system_status(agent_worker *w, const char *msg);
void agent_publishf_system_status(agent_worker *w, const char *fmt, ...);
/** When a model turn finishes with a tool call, queued user messages should not
 * preempt that tool.  The worker asks the UI thread for the queue contents only
 * after the tool result is appended, so the next model input can contain both
 * the tool observation and the user's pending correction.
 */
char *worker_request_queued_user_drain(agent_worker *w);
bool worker_take_queued_user_drain_request(agent_worker *w);
void worker_answer_queued_user_drain(agent_worker *w, char *text);
int agent_worker_sync_tokens(agent_worker *w, const pulsar_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len);
/** Start a new session at the system/tool prompt.  A fixed sysprompt.kv
 * checkpoint avoids paying this prefill cost repeatedly, but only when the
 * rendered prompt text still matches the file.  The same fixed path is shared
 * by Flash and Pro; agent_kv_load_path() checks the model id, so switching
 * model families rebuilds this cache instead of restoring incompatible KV.
 */
bool agent_worker_reset_to_sysprompt(agent_worker *w, char *err, size_t err_len);
bool agent_worker_has_user_session(agent_worker *w);
bool agent_worker_needs_save(agent_worker *w);
bool agent_worker_save_session_now(agent_worker *w, char sha_out[41],
                                          int *tokens_out,
                                          char *err, size_t err_len);
bool agent_worker_save_session(agent_worker *w, char *err, size_t err_len);
char *agent_session_title_from_prompt(const char *prompt,
                                             size_t max_bytes);
char *agent_session_title_from_text(const char *text, size_t text_len,
                                           size_t max_bytes);
bool agent_worker_show_history(agent_worker *w, int user_turns,
                                      char *err, size_t err_len);
/** Print resumable sessions from ~/.ds4/kvcache.  sysprompt.kv is intentionally
 * ignored because it is an implementation cache, not a user session.
 */
void agent_worker_list_sessions(agent_worker *w);
void agent_switch_completion_callback(const char *buf,
                                             linenoiseCompletions *lc);
bool agent_worker_delete_session(agent_worker *w, const char *prefix,
                                        char sha_out[41],
                                        char *err, size_t err_len);
bool agent_worker_strip_session(agent_worker *w, const char *prefix,
                                       char sha_out[41],
                                       uint32_t *tokens_out,
                                       char *err, size_t err_len);
bool agent_worker_switch_session(agent_worker *w, const char *prefix,
                                        int history_turns,
                                        char *err, size_t err_len);
int agent_parse_timeout(const char *s);
int agent_parse_int_default(const char *s, int def, int min, int max);
bool agent_parse_bool_default(const char *s, bool def);
void agent_line_spans_free(agent_line_spans *spans);
/** Split a text buffer into line spans.  content_end excludes CR/LF so callers
 * can print or compare line content without newline spelling differences.
 */
void agent_split_lines(const char *data, size_t len, agent_line_spans *spans);
int agent_read_file_bytes(const char *path, char **data, size_t *len,
                                 char *err, size_t errlen);
bool agent_old_new_line_effect(const char *old_data, size_t old_len,
                                      const char *new_data, size_t new_len,
                                      size_t edit_offset, size_t replaced_len,
                                      int *start_line, int *end_line,
                                      int *delta);
char *agent_edit_result(const char *path,
                                       int start_line, int end_line, int delta,
                                       const char *new_data, size_t new_len,
                                       const char *kind);
bool agent_tool_result_fits_context(agent_worker *w, const char *result,
                                           int reserve_tokens,
                                           int *tokens_out);
char *agent_tool_read(agent_worker *w, const agent_tool_call *call);
char *agent_tool_more(agent_worker *w, const agent_tool_call *call);
char *agent_tool_write(agent_worker *w, const agent_tool_call *call);
char *agent_tool_list(const agent_tool_call *call);
void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end);
bool agent_edit_upto_forcer_should_replace(agent_edit_upto_forcer *forcer,
                                                  agent_dsml_parser *p,
                                                  const char *next_text,
                                                  size_t next_len);
bool agent_preflight_edit_old(agent_worker *w, const agent_tool_call *call,
                                     char *err, size_t err_len);
/** Old/new editing is intentionally conservative: exact old text must be unique.
 * For large replacements, old may contain one [upto] marker: the head must be
 * unique, and the tail must be unique after that head before the whole span is
 * replaced.
 */
char *agent_tool_edit(agent_worker *w, const agent_tool_call *call);
/** Implement the search tool using either literal matching or POSIX regex. */
char *agent_tool_search(agent_worker *w, const agent_tool_call *call);
void agent_bash_jobs_free(agent_worker *w);
agent_bash_job *agent_bash_find_job(agent_worker *w, int id, pid_t pid);
void agent_bash_remove_job(agent_worker *w, agent_bash_job *target);
agent_bash_job *agent_bash_start(agent_worker *w, const char *cmd,
                                        int timeout_sec, char *err, size_t err_len);
char *agent_bash_job_tool_result(agent_worker *w, agent_bash_job *job,
                                        bool wait, int refresh_sec,
                                        bool stop, bool remove_if_done);
int agent_tool_job_id(const agent_tool_call *call);
pid_t agent_tool_pid(const agent_tool_call *call);
/** Execute all tool calls from one DSML block, preserving per-call labels in the
 * combined result so the model can associate observations with calls.
 */
char *agent_execute_tool_calls(agent_worker *w, const agent_tool_calls *calls);
/** If compaction happens while a bash process is still alive, inject a small
 * tool-role reminder into the rebuilt transcript.  Otherwise the summary could
 * preserve the user's task but lose the fact that an external process still
 * needs status/wait/stop handling.
 */
char *agent_bash_jobs_compaction_observation(agent_worker *w);
bool agent_worker_compact(agent_worker *w, const char *reason,
                                 char *err, size_t err_len);
bool agent_worker_compact_if_needed(agent_worker *w, const char *reason,
                                           char *err, size_t err_len);
int worker_accept_generated_token(agent_worker *w,
                                         int token,
                                         int *generated,
                                         double t0,
                                         agent_stream_renderer *stream,
                                         char *err,
                                         size_t err_len);
int worker_force_generated_text(agent_worker *w,
                                       const char *text,
                                       int max_tokens,
                                       int *generated,
                                       double t0,
                                       agent_stream_renderer *stream,
                                       char *err,
                                       size_t err_len);
void worker_request_save(agent_worker *w);
void worker_request_compact(agent_worker *w);
/** Worker thread entry point.  The UI thread submits plain user text; this
 * thread owns all DS4 session mutation, tool execution, and compaction.
 */
void *worker_main(void *arg);
int set_nonblock(int fd, bool on, int *old_flags);
/** Check and clear the raw_mode_needs_restore flag under the worker mutex.
 * Returns true if the UI thread should verify/reapply linenoise raw mode.
 */
bool worker_check_raw_mode_restore(agent_worker *w);
void drain_wake_fd(int fd);
/** Submit one user turn if the worker is idle.  Busy submissions are rejected so
 * the UI can keep the typed text editable instead of silently queueing it.
 */
bool worker_submit(agent_worker *w, const char *text);
/** Request interruption at the next model/tool polling point. */
void worker_interrupt(agent_worker *w);
/** Stop the worker thread. */
void worker_stop(agent_worker *w);
/** The UI thread consumes output in batches.  Taking ownership of w->out under
 * the mutex keeps terminal writes outside the lock while preserving order.
 */
void worker_consume(agent_worker *w, char **out, size_t *out_len, agent_status *status);
void worker_get_status(agent_worker *w, agent_status *status);
bool worker_is_idle(agent_worker *w);
bool worker_is_initialized(agent_worker *w, agent_status *status);
bool stdout_is_tty(void);
char *agent_format_user_prompt_echo(const char *text);
void agent_echo_user_prompt(const char *text);
void build_prompt_text(const agent_status *st, char *buf, size_t len);
unsigned agent_next_prefill_label(void);
void agent_prompt_queue_push(agent_prompt_queue *q, const char *text);
char *agent_prompt_queue_pop(agent_prompt_queue *q);
void agent_prompt_queue_push_front(agent_prompt_queue *q, char *text);
char *agent_prompt_queue_take_all(agent_prompt_queue *q);
char *agent_prompt_queue_take_all_echo(agent_prompt_queue *q);
void agent_prompt_queue_free(agent_prompt_queue *q);
void build_footer_text(const agent_status *st, const agent_prompt_queue *queue,
                              int cols, char *buf, size_t len);
/** Drain stdin in nonblocking mode.  The outer event loop decides when queued
 * bytes are fed to linenoiseEditFeed().
 */
void editor_read_stdin(agent_editor *ed);
bool editor_take_queued_byte(agent_editor *ed, unsigned char byte);
bool editor_take_bare_escape(agent_editor *ed);
void editor_replace_input(agent_editor *ed, const char *text);
void editor_restore_terminal_layout(agent_editor *ed);
int editor_start(agent_editor *ed, const char *prompt,
                        const char *status, const char *initial);
/** Stop the live editor and restore stdin flags. */
void editor_stop(agent_editor *ed);
/** Restore the live prompt after output.  The primary path draws it in the
 * reserved bottom rows; the fallback path keeps the older one-row-below-output
 * trick for terminals where scroll regions are unavailable.
 */
void editor_show(agent_editor *ed);
void editor_set_prompt_status(agent_editor *ed, const char *prompt,
                                     const char *status);
void editor_write_async(agent_editor *ed, const char *text, size_t len,
                               const char *prompt, const char *status,
                               bool force_show);
void editor_cancel_input_with_hint(agent_editor *ed,
                                          const char *prompt,
                                          const char *status);
void runtime_help(void);
void editor_write_welcome_banner(agent_editor *editor,
                                        const agent_config *cfg,
                                        const char *prompt,
                                        const char *statusline);
/** Initialize the worker, cache directory, sysprompt checkpoint path, trace file,
 * and model thread.  After this returns, all DS4 session mutation happens on
 * the worker thread.
 */
int agent_worker_init(agent_worker *w, pulsar_engine *engine, agent_config *cfg);
/** Shut down the worker and release owned resources, including any live bash
 * process groups.
 */
void agent_worker_free(agent_worker *w);
bool agent_prompt_yes_no_ex(const char *prompt,
                                   const agent_yes_no_options *opts,
                                   bool *timed_out);
/** Ask before discarding a dirty user session.  Fresh sessions that contain only
 * the system prompt are deliberately ignored.
 */
bool agent_maybe_save_before_leaving_session(agent_worker *w);
/** Process exit is different from /new or /switch: once the terminal is already
 * restored, declining the save can terminate immediately and let the OS reclaim
 * model/GPU resources instead of waiting for orderly teardown.
 */
agent_exit_save_result agent_maybe_save_before_exiting(agent_worker *w);

/* ---- shared inline helpers ---- */


#endif /* PULSAR_AGENT_INTERNAL_H */
