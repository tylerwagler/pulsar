/** @file
 *  The XML tool-call body of the output-parser kit (Qwen's and Nemotron-H's templates). */
/* xml_call_body.h -- one tool-call body format of the output-parser kit (out_parse_kit.h), shared by the families
 * whose templates write it: Qwen 3.8 and Nemotron-H (L296).  Moved out of qwen_chat.h (L295 F8 step 1) so a family
 * that writes this format does not include another family's frontend.  The methods are in out_parse_kit.cpp. */
#pragma once

#include <string>
#include <vector>

#include "out_parse_kit.h"
#include "pyjson.h"

/** The XML tool-call body (Qwen's template, and Nemotron-H's, L296): XML parameters inside one
 * \<tool_call\> block.
 * @verbatim
   <function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n ... </function>
   @endverbatim
 * A parameter VALUE is the text between the tags minus one leading and one trailing newline (the template
 * writes exactly those).  Arguments are rebuilt as JSON.  The template wrote strings raw and every other
 * value with tojson, so a value is JSON exactly when (a) the tool schema does not declare the property a
 * string (out_tool_param_type: the declared type is or includes string, read through unions and local
 * references) and (b) the text parses AND is the tojson spelling of what it parses to; anything else is
 * the text as a string.  Either way the value re-renders to the bytes the model sampled.
 *
 * L284 P5: a call streams as it is read -- TOOL_BEGIN when its name is complete, TOOL_ARGS as each
 * parameter CLOSES (a value's JSON spelling is known only then: the schema and the whole text decide it),
 * TOOL_END (or ERROR) at its close.  The fragments plus "}" are the arguments; a repeated key streams twice
 * (a JSON reader keeps the last, as `arguments` does).
 *
 * L284 P6: the payload is inside a parameter value the schema declares a string, or inside a JSON string
 * literal of any other value -- past the value's leading newline (the template's) and not on a partial
 * closer ("</" onward).
 *
 * L429: every per-feed reading resumes where the last one stopped (the body only grows between open() calls), so a
 * feed costs its new bytes and not the body so far: the call's name, the parameter the stream waits on
 * (param_resume), the payload scan and holds_closer's scan (two cursors over one tokenizer, scan_values).  close()
 * reads the whole body once, from its start, and is the authority on the call. */
class xml_call_body : public out_call_body {
public:
    /** The request's tools (the schema that types arguments); shared with the parser's copies (L430). */
    out_tools tools;
    /** How the template spelt a non-string SCALAR: false = tojson (Qwen: true / false / null), true = Python's str()
     *  (L296 Nemotron-H's `args_value | string`: True / False / None; numbers spell the same either way; mappings and
     *  lists are tojson in both).  A value is typed only from its template's own spelling. */
    bool python_scalars = false;

    void open() override;
    void grow(const std::string &b, GenBytes held, bool closing, int index, std::vector<out_event> *ev) override;
    bool in_payload() const override { return pay_.value && pay_.payload; }
    bool holds_closer(const std::string &b) override;
    bool close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call, const char **why,
               BodyOff *at) override;

private:
    /// Advance pay_ over the body's new bytes and set pay_.payload for the next byte.
    /// @param b    the call's body so far
    /// @param held bytes the kit holds back as a possible "</tool_call>"
    void scan_payload(const std::string &b, GenBytes held);
    struct value_scan;
    /// Advance a value scan over the body's new bytes: where parameter values open ("<parameter=KEY>") and
    /// close (the first "</parameter>" after it), and inside one, whether a JSON string literal is open.  The
    /// one tokenizer the payload scan and holds_closer share.
    /// @param vs the scan's state (its own cursor)
    /// @param b  the call's body so far
    void scan_values(value_scan *vs, const std::string &b) const;
    /// Where the stream's reading of the one parameter it waits on stands (L429): what read_param found so far, so
    /// its next reading of the same parameter in a longer body starts there.
    struct param_resume {
        BodyOff key_end = BodyOff{0u};   ///< the offset of the key's '>', once found (0: not yet; a key never ends at 0)
        BodyOff seen = BodyOff{0u};      ///< no key end (before key_end is known) or value closer (after) starts before it
    };
    /// Read one "<parameter=KEY>\nVALUE\n</parameter>" of function `fn` at b[*i] (the reading close() and the
    /// stream share).
    /// @param b   the call's body
    /// @param i   in: where the parameter starts; out: past it (on 1)
    /// @param fn  the call's function name (its schema types the value)
    /// @param key receives the parameter's name
    /// @param v   receives the typed value
    /// @param why receives what is missing or wrong (static string)
    /// @param r   NULL (close(): a whole reading), or the stream's resume state for THIS parameter: read on 0,
    ///            updated on 0 -- the same answers as a whole reading, without rescanning what a 0 already covered
    /// @return 1 = read, 0 = not complete yet, -1 = not a parameter
    int read_param(const std::string &b, BodyOff *i, const std::string &fn, std::string *key, pyjson_value *v,
                   const char **why, param_resume *r) const;
    /// Emit a TOOL_ARGS fragment for each parameter that closed since the last call.
    /// @param b     the call's body so far
    /// @param index the call's index, for the events
    /// @param ev    receives the fragments
    void stream_params(const std::string &b, int index, std::vector<out_event> *ev);

    bool began_ = false;       ///< TOOL_BEGIN sent for the open call
    BodyOff head_ws_ = BodyOff{0u};     ///< leading whitespace of the body skipped so far (before began_)
    BodyOff head_seen_ = BodyOff{0u};   ///< no '>' or newline of "<function=NAME>" before this offset (before began_)
    std::string fn_;           ///< the open call's function name, once complete
    BodyOff arg_pos_ = BodyOff{0u};   ///< body offset past the last parameter streamed (TOOL_ARGS) and the whitespace after it
    param_resume arg_resume_;  ///< the reading of the parameter at arg_pos_ so far
    int arg_n_ = 0;            ///< parameters streamed for the open call
    bool arg_stop_ = false;    ///< the open call stopped streaming: its next bytes are not a parameter
    /// An incremental scan of the open call's parameter values (scan_values).
    struct value_scan {
        BodyOff pos = BodyOff{0u};        ///< bytes of the body classified
        BodyOff key_seen = BodyOff{0u};   ///< an open "<parameter=" tag at pos has no '>' before this offset
        bool value = false;    ///< inside a parameter value
        BodyOff vstart = BodyOff{0u};     ///< where its text starts (past the tag)
        bool is_string = false;///< the schema declares it a string
        bool json_str = false; ///< inside a JSON string literal of a non-string value
        bool esc = false;      ///< the previous byte was a backslash in that literal
        bool payload = false;  ///< the result for the next byte (scan_payload's; pay_ only)
    };
    value_scan pay_;           ///< the scan in_payload reads: advanced by every grow of an announced call
    value_scan closer_;        ///< holds_closer's scan: advanced to each closing marker found
};
