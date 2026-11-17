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
 * closer ("</" onward). */
class xml_call_body : public out_call_body {
public:
    /** The request's tools (the schema that types arguments). */
    pyjson_value tools;
    /** How the template spelt a non-string SCALAR: false = tojson (Qwen: true / false / null), true = Python's str()
     *  (L296 Nemotron-H's `args_value | string`: True / False / None; numbers spell the same either way; mappings and
     *  lists are tojson in both).  A value is typed only from its template's own spelling. */
    bool python_scalars = false;

    void open() override;
    void grow(const std::string &b, size_t held, bool closing, int index, std::vector<out_event> *ev) override;
    bool in_payload() const override { return pay_.value && pay_.payload; }
    bool holds_closer(const std::string &b) const override;
    bool close(const std::string &b, int index, std::vector<out_event> *ev, out_call *call, const char **why,
               size_t *at) override;

private:
    /// Advance pay_ over the body's new bytes and set pay_.payload for the next byte.
    /// @param b    the call's body so far
    /// @param held bytes the kit holds back as a possible "</tool_call>"
    void scan_payload(const std::string &b, size_t held);
    /// Read one "<parameter=KEY>\nVALUE\n</parameter>" of function `fn` at b[*i] (the reading close() and the
    /// stream share).
    /// @param b   the call's body
    /// @param i   in: where the parameter starts; out: past it (on 1)
    /// @param fn  the call's function name (its schema types the value)
    /// @param key receives the parameter's name
    /// @param v   receives the typed value
    /// @param why receives what is missing or wrong (static string)
    /// @return 1 = read, 0 = not complete yet, -1 = not a parameter
    int read_param(const std::string &b, size_t *i, const std::string &fn, std::string *key, pyjson_value *v,
                   const char **why) const;
    /// Emit a TOOL_ARGS fragment for each parameter that closed since the last call.
    /// @param b     the call's body so far
    /// @param index the call's index, for the events
    /// @param ev    receives the fragments
    void stream_params(const std::string &b, int index, std::vector<out_event> *ev);

    /** @var pay_
    *  Where the open block's decode is (in_payload), scanned incrementally. */
    bool began_ = false;       ///< TOOL_BEGIN sent for the open call
    std::string fn_;           ///< the open call's function name, once complete
    size_t arg_pos_ = 0;       ///< body offset past the last parameter streamed (TOOL_ARGS)
    int arg_n_ = 0;            ///< parameters streamed for the open call
    bool arg_stop_ = false;    ///< the open call stopped streaming: its next bytes are not a parameter
    /// The incremental payload scanner's state for the open call.
    struct payload_scan {
        size_t pos = 0;        ///< bytes of the body classified
        bool value = false;    ///< inside a parameter value
        size_t vstart = 0;     ///< where its text starts (past the tag)
        bool is_string = false;///< the schema declares it a string
        bool json_str = false; ///< inside a JSON string literal of a non-string value
        bool esc = false;      ///< the previous byte was a backslash in that literal
        bool payload = false;  ///< the result for the next byte
    } pay_;
};
