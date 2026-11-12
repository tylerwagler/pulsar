/** @file
 *  The Qwen3.8-Flash-Next output parser: the output-parser kit's Qwen grammar and its XML tool-call body. */
/* qwen_output.cpp -- the Qwen3.8-Flash-Next output parser: the output-parser kit's Qwen configuration (its
 * grammar) and Qwen's XML tool-call body.  See qwen_chat.h. */
#include "qwen_chat.h"

#include <stdio.h>
#include <string.h>

const out_grammar &qwen_output_grammar(bool thinking) {
    enum { THINK_END, CALL_OPEN, CALL_CLOSE };
    /* thinking off: the turn never opened reasoning, so `</think>` in the content is text */
    static const out_grammar off = {
        /* .name        = */ "qwen",
        /* .markers     = */ {"</think>", "<tool_call>", "</tool_call>"},
        /* .transitions = */ {
            {OUT_REASONING, THINK_END, OUT_CONTENT, 0},
            {OUT_CONTENT, CALL_OPEN, OUT_CALL, 0},
            {OUT_CALL, CALL_CLOSE, OUT_CONTENT, 0},
        },
        /* .reasoning   = */ OUT_TEXT_STRIP,
        /* .content     = */ OUT_TEXT_STRIP,
        /* .call_open   = */ "<tool_call>",
        /* .dropped     = */ {},
    };
    /* thinking on (L344): the same, and a stray `</think>` after the reasoning closed is dropped (Nemotron's rule) */
    static const out_grammar on = {
        /* .name        = */ "qwen",
        /* .markers     = */ {"</think>", "<tool_call>", "</tool_call>"},
        /* .transitions = */ {
            {OUT_REASONING, THINK_END, OUT_CONTENT, 0},
            {OUT_CONTENT, THINK_END, OUT_CONTENT, 0},
            {OUT_CONTENT, CALL_OPEN, OUT_CALL, 0},
            {OUT_CALL, CALL_CLOSE, OUT_CONTENT, 0},
        },
        /* .reasoning   = */ OUT_TEXT_STRIP,
        /* .content     = */ OUT_TEXT_STRIP,
        /* .call_open   = */ "<tool_call>",
        /* .dropped     = */ {},
    };
    return thinking ? on : off;
}

bool qwen_output_parser::init(bool thinking, const char *tools_json, char *err, size_t errlen) {
    body_.tools = pyjson_value();
    if (tools_json) {
        char jerr[160] = "not a JSON array";
        if (!pyjson_parse(tools_json, strlen(tools_json), &body_.tools, jerr, sizeof jerr) ||
            body_.tools.kind != pyjson_value::ARR) {
            if (err && errlen) snprintf(err, errlen, "tools_not_array: %s", jerr);
            return false;
        }
    }
    out_parser::init(&qwen_output_grammar(thinking), &body_, thinking ? OUT_REASONING : OUT_CONTENT);
    return true;
}

out_parser *qwen_output_parser::clone() const {
    qwen_output_parser *p = new qwen_output_parser(*this);
    p->rebind_body(&p->body_);
    return p;
}
