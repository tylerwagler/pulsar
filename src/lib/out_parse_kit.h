/** @file
 *  The output-parser kit: one incremental reader of think / text / tool-call sections for every family. */
/* out_parse_kit.h -- the output-parser kit (L295 F5): ONE reading of sampled text for every family whose
 * output is think / text / tool-call sections delimited by literal markers.
 *
 * A family's output parser is a CONFIGURATION, not a parser: an out_grammar (its markers, the transition
 * table between four states, how each text section is reported) plus the reader of its tool-call body (an
 * out_call_body: Qwen's XML parameters, or the shared JSON {"name", "args"} body).  out_parser runs it,
 * incrementally (any chunking of the input gives the same events), and emits the generic events the
 * server's output-parser ops project into every protocol (parser_kit.cpp): reasoning and content deltas,
 * and each tool call as it is read -- TOOL_BEGIN when its name is known, TOOL_ARGS fragments of its
 * arguments object (the fragments plus "}" are the object), TOOL_END at its close, or ERROR naming what
 * was malformed (that call is dropped).
 *
 * The machine (vLLM's ParserEngine shape: vllm/parser/engine/parser_engine_config.py):
 *   - in each state only the markers with a transition FROM that state are structure; any other text,
 *     another family's marker spelling included, is the state's text;
 *   - a marker split across feeds waits (the held bytes are text if the turn ends there);
 *   - CONTENT and REASONING report text under their section policy; HEADER (a message header -- the name
 *     between a role marker and a content-kind marker) buffers its text, which a transition either drops or
 *     sends as content (OUT_ACT_HEADER_TEXT), and the turn's end sends as content;
 *   - entering CALL opens a call and leaving it closes one; the call's bytes from the marker that opened it
 *     (or from the text of the header that preceded it -- past the header's own role marker, so a call's key
 *     reads the same whether it is the turn's first block, after the generation prompt's marker, or a later
 *     one) to the marker that closed it are the tool memory's key. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

/** One thing the parser recognised. */
struct out_event {
    /// What an event reports; which of the fields below it fills is noted per kind.
    enum kind_t {
        REASONING,   ///< `text` is a reasoning delta
        CONTENT,     ///< `text` is a content delta
        TOOL_BEGIN,  ///< call `index` named `name` opened
        TOOL_ARGS,   ///< `text` is the next fragment of call `index`'s arguments object, its final "}" excluded
        TOOL_END,    ///< call `index` complete: `name`, `arguments` (a JSON object's text)
        ERROR,       ///< `text` says what was malformed; the call it concerns is dropped
    } kind;
    /** @var out_event::kind
    *  This event's kind. */
    /// The delta (REASONING, CONTENT), the arguments fragment (TOOL_ARGS) or the diagnostic (ERROR).
    std::string text;
    /// The call's 0-based index within the turn (TOOL_*, ERROR about a call); -1 for a text delta.
    int index = -1;
    /// The call's function name (TOOL_BEGIN, TOOL_END).
    std::string name;
    /// The call's whole arguments object as JSON text (TOOL_END only).
    std::string arguments;
};

/** One complete tool call the parser read. */
struct out_call {
    /// The function the call names.
    std::string name;
    std::string arguments;   ///< a JSON object's text
};

/** Where the decode is. */
enum out_state { OUT_HEADER, OUT_CONTENT, OUT_REASONING, OUT_CALL };

/** How a text section is reported (the events and the final reading alike). */
enum out_text_policy {
    /** Python's str.strip() of each section: leading whitespace dropped, interior whitespace held until the
     *  next non-space proves it interior, trailing whitespace dropped at the section's end (Qwen: its
     *  template strips both on replay). */
    OUT_TEXT_STRIP,
    /** The turn's text with its trailing whitespace dropped (str.rstrip() of all of it): whitespace is held
     *  until more text proves it interior, across sections, and dropped at the turn's end (Inkling's
     *  reasoning, vLLM's strip_trailing_reasoning_whitespace). */
    OUT_TEXT_RSTRIP,
    /** The turn's text verbatim, except text that is nothing but whitespace when a call opens or the turn
     *  ends: it is held until a non-space arrives, dropped when a call opens while it is all the text so far,
     *  and at the turn's end sent only if no call was read (Inkling's content, vLLM's
     *  drop_whitespace_only_content_before_tools). */
    OUT_TEXT_KEEP_UNLESS_BLANK,
    /** The turn's text verbatim until a call opens; from there each section str.strip()'d: whitespace is held until
     *  more text proves it interior, sent at the turn's end when no call opened, dropped where a call opens; after a
     *  call, a section's leading whitespace is dropped too (vLLM's strip_content_whitespace_with_tools as far as a
     *  stream can follow it: a turn with calls reads its content stripped, except leading whitespace already sent
     *  before the first call; L296 Nemotron-H's content). */
    OUT_TEXT_TRIM_AT_CALLS,
};

/** A transition's extra action. */
enum : uint32_t {
    OUT_ACT_HEADER_TEXT = 1u,   ///< leaving HEADER: its buffered text is content (else it is dropped)
};

/// One edge of an out_grammar's state machine: in state `from`, marker `marker` moves the reading to `to`.
struct out_transition {
    /// The state the marker is structure in.
    out_state from;
    int marker;         ///< index into out_grammar::markers
    /// The state the marker moves the reading to.
    out_state to;
    uint32_t act;       ///< OUT_ACT_* bits
};

/** A family's output grammar. */
struct out_grammar {
    /// The family's name, for diagnostics.
    const char *name;
    /// The literal marker strings; out_transition::marker indexes this list.
    std::vector<std::string> markers;
    /// The machine's edges; a marker is structure only in a state that has an edge on it.
    std::vector<out_transition> transitions;
    /// How OUT_REASONING text is reported.
    out_text_policy reasoning;
    /// How OUT_CONTENT text (and HEADER text sent as content) is reported.
    out_text_policy content;
    /** The marker that opens a call, as the diagnostics name it ("the turn ended inside <tool_call>") and as
     *  the server finds where a turn's calls start in its text. */
    const char *call_open;
    /** Markers removed wherever they appear (the tokenizer's other special tokens: vLLM drops a special
     *  token that is not one of the grammar's terminals). */
    std::vector<std::string> dropped;
};

/** The reader of a call's body: the bytes between the marker that opened the call and the marker that
 * closes it.  The kit owns the call's index, the events' order and the call list; the reader owns the
 * body's syntax. */
class out_call_body {
public:
    virtual ~out_call_body() = default;
    /** A new call opened. */
    virtual void open() = 0;
    /** The body grew to `b` (all of it so far).  `held` = bytes after it the kit holds as a possible closer;
     *  `closing` = the closer was found and `b` is complete.  Announce the call (TOOL_BEGIN) once its name is
     *  known and stream TOOL_ARGS fragments as they become final. */
    /// @param b       the call's body so far (all of it, from just past the opening marker)
    /// @param held    bytes after `b` that the kit holds back as a possible closing marker (0 when `closing`)
    /// @param closing true once the closing marker was found: `b` is the whole body
    /// @param index   the call's index within the turn, for the events
    /// @param ev      receives the TOOL_BEGIN / TOOL_ARGS events
    virtual void grow(const std::string &b, size_t held, bool closing, int index, std::vector<out_event> *ev) = 0;
    /** The next byte is a call's PAYLOAD (a string value), not its structure (the shared decode rule, L284
     *  P6: structure decodes greedily, payload is sampled). */
    /// @return true when the next byte is payload (sampled), false when it is structure (greedy)
    virtual bool in_payload() const = 0;
    /** The call closed with body `b`: the call (true; TOOL_BEGIN first if grow never announced it), or false
     *  with `why` (what is missing or wrong) and `at` (the byte of the body where reading stopped). */
    /// @param b     the call's complete body
    /// @param index the call's index within the turn
    /// @param ev    receives TOOL_BEGIN if grow never sent it
    /// @param call  receives the call (name and arguments object) on success
    /// @param why   on failure, a static string naming what is missing or wrong
    /// @param at    on failure, the byte offset in `b` where reading stopped
    /// @return true when the body is a well-formed call
    virtual bool close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call, const char **why,
                       size_t *at) = 0;
};

/** The shared JSON call body: one object `{"name": NAME, "args": {...}}` (other members ignored; no "args"
 * = {}).  The call is announced when "name" closes; the args object streams verbatim as it arrives (whole
 * UTF-8 sequences, its final "}" held back), and the call's arguments are that object's bytes as sampled.
 * The payload is any string inside the args value. */
class out_json_call_body : public out_call_body {
public:
    void open() override;
    void grow(const std::string &b, size_t held, bool closing, int index, std::vector<out_event> *ev) override;
    bool in_payload() const override { return payload_; }
    bool close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call, const char **why,
               size_t *at) override;

private:
    /// Advance the scanner over `b`'s bytes past pos_: the depth-1 member keys, "name"'s string value, and
    /// where the "args" object starts (args_lo_) and ends (args_hi_).
    /// @param b the call's body so far
    void scan(const std::string &b);
    /// Send the args object's bytes not yet streamed as one TOOL_ARGS fragment: whole UTF-8 sequences only
    /// (unless `closing`), its final "}" held back.  Nothing before TOOL_BEGIN.
    /// @param b       the call's body so far
    /// @param closing the body is complete
    /// @param index   the call's index, for the event
    /// @param ev      receives the fragment
    void stream_args(const std::string &b, bool closing, int index, std::vector<out_event> *ev);

    size_t pos_ = 0;            ///< bytes of the body scanned
    int depth_ = 0;             ///< object / array depth at pos_
    bool in_str_ = false;       ///< inside a string literal
    bool esc_ = false;          ///< the previous byte was a backslash in it
    size_t str_lo_ = 0;         ///< where the open string literal starts (its quote)
    std::string key_;           ///< the last depth-1 string closed (a member key until its value starts)
    bool want_value_ = false;   ///< a depth-1 ':' was seen: the next value is key_'s
    std::string value_key_;     ///< the member whose value is being read at depth 1
    bool name_done_ = false;    ///< "name"'s string value closed
    std::string name_;          ///< its decoded text
    size_t args_lo_ = std::string::npos;  ///< where the "args" object starts
    size_t args_hi_ = std::string::npos;  ///< one past its closing "}"
    size_t args_sent_ = 0;      ///< args bytes streamed (from args_lo_)
    bool began_ = false;        ///< TOOL_BEGIN sent
    /// The next byte is inside a string at depth >= 2 (the args value), with no closer held: payload.
    bool payload_ = false;
    bool held_ = false;         ///< the kit holds a possible closer
};

/** The parser: one per assistant turn (one decode attempt). */
class out_parser {
public:
    out_parser() = default;
    virtual ~out_parser() = default;
    out_parser &operator=(const out_parser &) = delete;   ///< it points at its configuration's body
    /** L296: a copy of the reading so far -- its own body in the same state, the same grammar -- for a scratch
     *  probe that reads ahead without touching this one (the server's per-position tool-call override). */
    /// @return a new parser (caller owns it) with this one's reading and its own copy of the body
    virtual out_parser *clone() const = 0;
    /** `g` and `body` outlive the parser; `initial` is the state the prompt left the turn in.  Resets every
     *  reading: one init per turn. */
    /// @param g       the family's grammar (not owned)
    /// @param body    the family's call-body reader (not owned)
    /// @param initial the state the turn starts in
    void init(const out_grammar *g, out_call_body *body, out_state initial);
    /// @return the grammar init() bound
    const out_grammar &grammar() const { return *g_; }
    /// Read the next `n` sampled bytes.  Any chunking of the stream yields the same events; bytes that could
    /// start a marker are held until the next feed (or finish()).
    /// @param p  the bytes
    /// @param n  their count
    /// @param ev receives the events these bytes complete
    void feed(const char *p, size_t n, std::vector<out_event> *ev);
    /** End of the turn (a stop token, or the length cap). */
    /// @param ev receives the held text, the sections' final text and an ERROR if the turn ended inside a call
    void finish(std::vector<out_event> *ev);

    /// @return all reasoning text reported so far (the REASONING deltas concatenated)
    const std::string &reasoning() const { return reasoning_; }
    /// @return all content text reported so far (the CONTENT deltas concatenated)
    const std::string &content() const { return content_; }
    /// @return the well-formed calls read so far, in order (malformed ones are dropped)
    const std::vector<out_call> &calls() const { return calls_; }
    /// @return the number of ERROR events (malformed or unterminated calls) so far
    int errors() const { return errors_; }
    /** The parser sits inside an open call. */
    /// @return true while state is OUT_CALL
    bool in_tool_call() const { return state_ == OUT_CALL; }
    /** The next byte is an open call's payload (out_call_body::in_payload). */
    /// @return true when the next byte is sampled payload rather than greedy structure
    bool in_payload() const { return state_ == OUT_CALL && body_->in_payload(); }
    /** The turn's calls as SAMPLED: [lo, hi) in the fed stream, from where the first call's bytes start to
     * the end of the last call that closed (a malformed one among them included -- these are the bytes the
     * live KV holds).  false until a call was read. */
    /// @param lo receives the stream offset where the first call's bytes start
    /// @param hi receives the stream offset one past the last closed call's closer
    /// @return true when at least one well-formed call was read (then `*lo`, `*hi` are set)
    bool raw_span(size_t *lo, size_t *hi) const;

protected:
    /** A configuration's clone: the base's reading copied, then pointed at the copy's own body. */
    out_parser(const out_parser &) = default;
    /// Point a clone at its own body (a configuration's clone() calls this after copying).
    /// @param body the clone's body, owned by the clone
    void rebind_body(out_call_body *body) { body_ = body; }

private:
    /// One reported text stream's policy state: [0] content, [1] reasoning.
    struct section {           /* one reported text stream */
        bool started = false;  ///< STRIP: a non-space codepoint went out; KEEP_UNLESS_BLANK: one arrived
        std::string ws;        ///< whitespace held back (STRIP, RSTRIP), or all the text (KEEP_UNLESS_BLANK)
        std::string partial;   ///< an incomplete UTF-8 sequence
    };
    /// Report state `st`'s text under its section's policy, a codepoint at a time (a split UTF-8 sequence
    /// waits in section::partial; an invalid byte is text).  HEADER text is buffered in header_ instead.
    /// @param st the state the text belongs to
    /// @param p  the bytes
    /// @param n  their count
    /// @param ev receives the deltas
    void emit_text(out_state st, const char *p, size_t n, std::vector<out_event> *ev);
    /// A section of state `st` ends (the state is left): drop a dangling partial sequence; STRIP resets.
    /// @param st the state being left (a no-op unless CONTENT or REASONING)
    void end_section(out_state st);
    /// The turn ended: send what `st`'s policy still holds (KEEP_UNLESS_BLANK with no call read,
    /// TRIM_AT_CALLS with no call opened), drop the rest.
    /// @param st OUT_CONTENT or OUT_REASONING
    /// @param ev receives the final delta
    void finish_section(out_state st, std::vector<out_event> *ev);
    /// Append `text` to the reading and emit it as one delta (nothing when empty).
    /// @param reasoning true: a REASONING delta; false: CONTENT
    /// @param text      the delta (moved from)
    /// @param ev        receives the event
    void send(bool reasoning, std::string &&text, std::vector<out_event> *ev);
    /** The earliest marker in hold_ with a transition from the current state (`*tr`) or dropped everywhere
     *  (`*tr` NULL): its offset (npos if none) and length, and the bytes at hold_'s end that could still
     *  start one. */
    /// @param tr   receives the transition the marker takes, or NULL for a dropped marker (or none found)
    /// @param len  receives the marker's length (0 when none)
    /// @param keep receives how many bytes at hold_'s end are a proper prefix of some candidate marker
    /// @return the marker's offset in hold_, or std::string::npos
    size_t find_marker(const out_transition **tr, size_t *len, size_t *keep) const;
    /// Close the open call: the body's verdict becomes TOOL_END (and calls_) or ERROR; index_ advances either way.
    /// @param ev receives the event
    void close_call(std::vector<out_event> *ev);

    /// The bound grammar (not owned).
    const out_grammar *g_ = nullptr;
    /// The bound call-body reader (not owned; a clone's own copy after rebind_body).
    out_call_body *body_ = nullptr;
    /// The machine's current state.
    out_state state_ = OUT_CONTENT;
    std::string hold_;         ///< bytes not yet classified
    size_t fed_ = 0;           ///< bytes fed so far; hold_ is the stream's unclassified tail
    std::string header_;       ///< HEADER's buffered text
    size_t header_at_ = 0;     ///< stream offset of the current header's text (past the marker that opened it)
    bool block_seen_ = false;  ///< a call opened: raw_lo_ is set
    size_t raw_lo_ = 0;        ///< stream offset of the first call's bytes
    size_t raw_hi_ = 0;        ///< stream offset past the last call's closer
    section sec_[2];           ///< [0] content, [1] reasoning
    std::string block_;        ///< the open call's body
    /// The next call's index (counts malformed calls too).
    int index_ = 0;
    /// ERROR events sent this turn.
    int errors_ = 0;
    /// The reasoning and content text reported so far.
    std::string reasoning_, content_;
    /** @var out_parser::content_
    *  The content text reported so far. */
    /// The well-formed calls read so far.
    std::vector<out_call> calls_;
};
