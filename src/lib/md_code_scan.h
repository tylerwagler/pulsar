/** @file
 *  The Markdown code scanner: whether the next byte of a text lies inside an inline code span or a fenced block. */
/* md_code_scan.h -- L419: is this point of a reply inside Markdown code?
 *
 * A model that EXPLAINS tool calling quotes its own call markup -- `<tool_call>` in backticks, a whole example
 * call in a fenced block.  A marker quoted that way is prose, not a call.  This scanner answers the one question
 * both readers of a turn ask at a call-opening marker -- the output parser (out_parse_kit.cpp) and the grammar gate
 * (constrain.cpp) -- from the section's bytes BEFORE the marker alone, so the two agree by construction and the
 * gate can decide at the token that completes the marker, with no lookahead.
 *
 * The rules (CommonMark's code spans and fenced blocks, narrowed where a prefix cannot decide):
 *   - an INLINE SPAN opens at a run of N backticks and closes at the next run of exactly N backticks on the SAME
 *     LINE.  It never crosses a newline: a stray backtick ("the ` key") therefore quotes at most the rest of its
 *     own line, and a call opened on a later line -- where every XML template opens one -- is read as a call.
 *     (CommonMark lets a span run to the end of its paragraph, but only when its closer exists, which a prefix
 *     cannot know; a span that wraps a line is read as text, today's reading.)
 *   - a FENCE opens at a run of three or more backticks or tildes that starts its line (after spaces or tabs only;
 *     any indentation, so a fence nested in a list item counts) and whose line -- the info string -- holds no
 *     backtick when the fence is backticks (such a line is an inline span instead).  It closes at a line that is
 *     only a run of the same character at least as long, then spaces or tabs.  A fence that is never closed stays
 *     open to the end of the section, as CommonMark's does to the end of the document.
 *   - a closing run directly followed by the marker (no newline between) has closed the fence: a call opened right
 *     after a block, on its last line, is a call.
 *   - tildes open fences only; there is no inline tilde span.  Indented code blocks are not code here.
 *
 * One byte at a time, constant state, linear in the text: any chunking of the same bytes gives the same answers.
 * The state is a plain value (the parser's clone and the gate's rollback copy it). */
#pragma once

#include <stddef.h>
#include <stdint.h>

/** The scanner: feed() a section's bytes in order, ask in_code() where a marker would start. */
class md_code_scan {
public:
    /** A new section starts (its first byte starts a line). */
    void reset() { *this = md_code_scan(); }

    /** The section's next `n` bytes. */
    /// @param p the bytes
    /// @param n their count
    void feed(const char *p, size_t n) {
        for (size_t i = 0; i < n; i++) step(p[i]);
    }

    /** Whether a marker starting at the next byte lies inside an inline span or a fenced block.  The marker's own
     *  first byte is taken to end a backtick or tilde run in progress (no marker starts with either). */
    /// @return true when the next byte is quoted code
    bool in_code() const {
        md_code_scan s = *this;
        s.end_run();
        return s.mode_ == INLINE || s.mode_ == FENCE_INFO || (s.mode_ == FENCE && !s.closing_);
    }

private:
    /// Where the next byte is.
    enum mode_t : uint8_t {
        TEXT,        ///< prose
        INLINE,      ///< inside an inline span opened by open_ backticks on this line
        FENCE_INFO,  ///< on a fence's opening line, past its run (a backtick here makes it an inline span instead)
        FENCE,       ///< inside a fenced block opened by open_ of fence_
    };

    /// The run of run_ x run_n_ ended: what it opens or closes.
    void end_run() {
        if (!run_n_) return;
        const uint32_t n = run_n_;
        const bool tick = run_ == '`';
        run_n_ = 0;
        switch (mode_) {
        case TEXT:
            if (n >= 3 && run_lead_) {
                mode_ = FENCE_INFO;
                fence_ = run_;
                open_ = n;
            } else if (tick) {
                mode_ = INLINE;
                open_ = n;
            }
            break;
        case FENCE_INFO:
            if (!tick || fence_ != '`') break;
            mode_ = INLINE;   /* a backtick in a backtick fence's info string: the line is an inline span */
            [[fallthrough]];
        case INLINE:
            if (tick && n == open_) mode_ = TEXT;
            break;
        case FENCE:
            if (run_ == fence_ && run_lead_ && n >= open_) closing_ = true;
            break;
        }
    }

    /// The section's next byte.
    /// @param c the byte
    void step(char c) {
        if (run_n_ && c == run_) {
            run_n_++;
            return;
        }
        end_run();
        if (c == '`' || c == '~') {
            run_ = c;
            run_n_ = 1;
            run_lead_ = lead_;
            lead_ = false;
            closing_ = false;   /* a second run on a closing line: not a closer */
        } else if (c == '\n') {
            if (mode_ == INLINE || (mode_ == FENCE && closing_)) mode_ = TEXT;
            else if (mode_ == FENCE_INFO) mode_ = FENCE;
            closing_ = false;
            lead_ = true;
        } else if (c != ' ' && c != '\t' && c != '\r') {
            lead_ = false;
            closing_ = false;   /* text after a closing run: not a closer */
        }
    }

    mode_t mode_ = TEXT;      ///< where the next byte is
    char fence_ = 0;          ///< the open fence's character (FENCE_INFO, FENCE)
    char run_ = 0;            ///< the character of the run in progress
    bool run_lead_ = false;   ///< the run in progress started its line (after spaces or tabs only)
    bool lead_ = true;        ///< the current line holds only spaces or tabs so far
    bool closing_ = false;    ///< FENCE: this line is a closing run and spaces so far
    uint32_t run_n_ = 0;      ///< the length of the run in progress (0 = none)
    uint32_t open_ = 0;       ///< the length of the run that opened the span or the fence
};
