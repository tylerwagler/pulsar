/** @file
 *  The generation path's host work as it read before L430 and L431, kept for the differential tests and the bench. */
/* gen_host_ref.h -- host work of the generation path as it stood at b1a217d6, the commit before L430 and L431 made
 * its cost independent of the text generated so far:
 *
 *   gen_release_ref                 server_jobs.cpp's gen_release (L431): the stop-string scan's resume point moved
 *                                   only when the longest stop string was longer than one byte, so a list of
 *                                   one-byte strings searched the whole text at every token
 *   gen_spec_greedy_rows_text_ref   server_jobs.cpp's gen_spec_greedy_rows_text: every round it copied all of g->text
 *   gen_turn_draft_prefix_text_ref  and gen_turn_draft_prefix_text            and asked the parser for a probe that
 *   kit_probe_*_ref                 parser_kit.cpp's probe: a clone() of the output parser -- the turn's reported
 *                                   text, the open call's body, the read-ahead copy the same -- with its statics
 *                                   kit_read_ahead / kit_read_limit
 *
 * The code is that commit's, but for the names (`_ref`), `inline`, and the dispatch a renamed copy needs: a kit
 * family's probe is the three kit_probe_*_ref here, any other parser's (DeepSeek's tracker copy, which L430 did not
 * touch) is the family's own.  The walks release bytes by gen_release_ref, as they did.
 *
 * It is the REFERENCE of two instruments and is not to be edited with the live code:
 *   src/server/server_tests.cpp  test_l430_round_walks_match_reference: the live walks must give this one's answers
 *                                at every round of thousands of random turns, and leave the attempt as it was;
 *                                test_l431_stop_scan_matches_reference: the live release must give this one's at
 *                                every token of thousands of random streams;
 *   tests/gen_host_bench.cpp     what a round's copies and a token's scan cost.
 * Delete it with those when the walks' or the release's answers change on purpose (the reference would then be wrong).
 *
 * Included after the server's sources by a unit that compiles the server whole (pulsar_test.cpp, gen_host_bench.cpp). */
#pragma once

#include <string.h>

#include <vector>

/// server_jobs.cpp's gen_release at b1a217d6: the emit path's byte release -- the reasoning tracker fed, the stop
/// strings looked for from the resume point, the bytes the stream may send.
/// @param stops             the request's stop strings
/// @param text              the generated text, `piece` already appended (NUL-terminated)
/// @param len               its length, bytes
/// @param piece             the token's bytes
/// @param piece_len         their length
/// @param plain_stream_pos  bytes already released
/// @param thinking          the reasoning tracker, fed `piece`
/// @param stop_scan_from    the stop scan's resume offset, advanced (only when the longest stop is longer than a byte)
/// @param hit_stop          set when a stop string matched
/// @param stop_pos          receives its offset
/// @param stop_len          receives its length
/// @return how far into `text` the stream may send, bytes
inline size_t gen_release_ref(const stop_list *stops, const char *text, size_t len, const char *piece, size_t piece_len,
                              size_t plain_stream_pos, thinking_state *thinking, size_t *stop_scan_from,
                              bool *hit_stop, size_t *stop_pos, size_t *stop_len) {
    const bool was_thinking = thinking->inside;
    thinking->feed(piece, piece_len);
    if (was_thinking && !thinking->inside) {
        const size_t close_base = len - piece_len;
        if (*stop_scan_from < close_base) *stop_scan_from = close_base;
    }
    *hit_stop = !thinking->inside && stop_list_find_from(stops, text, *stop_scan_from, stop_pos, stop_len);
    size_t stream_len = *hit_stop ? *stop_pos : stop_list_stream_safe_len(stops, len);
    if (stream_len > len) stream_len = len;
    stream_len = utf8_stream_safe_len(text, plain_stream_pos, stream_len, *hit_stop);
    if (!*hit_stop && stops->max_len > 1) {
        const size_t hold = stops->max_len - 1;
        *stop_scan_from = len > hold ? len - hold : 0;
    }
    return stream_len;
}

/// parser_kit.cpp's kit_read_ahead at b1a217d6 (L428: the read-ahead copy, fed a byte at a time up to the first
/// declared call's close and the family's call_tail).
/// @param q     the kit state (the attempt's, or a probe's)
/// @param g     the attempt (its request: single_tool_call, the declared tools, the family's call_tail)
/// @param text  the generated text
/// @param len   its length
inline void kit_read_ahead_ref(kit_gen *q, const gen_state *g, const char *text, size_t len) {
    const request *r = &g->j->req;
    if (!r->single_tool_call || r->kind != REQ_CHAT || !r->has_tools || q->cut) return;
    if (!q->ahead) {
        q->ahead = q->parser->clone();
        q->ahead_fed = q->fed;
    }
    const char *tail = r->family->call_tail;
    std::vector<out_event> ev;
    while (!q->cut && (q->ahead_fed < len || (q->call_closed && !tail[q->tail_at]))) {
        if (q->call_closed) {
            if (!tail[q->tail_at]) {   /* the call and the template's bytes after it: the turn */
                q->cut = true;
                q->cut_at = q->ahead_fed;
            } else if (text[q->ahead_fed] == tail[q->tail_at]) {
                q->ahead_fed++;
                q->tail_at++;
            } else {                   /* not the template's bytes: the turn is the call */
                q->cut = true;
                q->cut_at = q->call_end;
            }
            continue;
        }
        ev.clear();
        q->ahead->feed(text + q->ahead_fed, 1, &ev);
        q->ahead_fed++;
        for (const out_event &e : ev) {
            char undeclared[512];
            if (e.kind != out_event::TOOL_END ||
                !tool_call_declared(r, e.name.c_str(), undeclared, sizeof(undeclared)))
                continue;
            q->call_closed = true;
            q->call_end = q->ahead_fed;
        }
    }
}

/// parser_kit.cpp's kit_read_limit at b1a217d6.
/// @param q      the kit state (the attempt's, or a probe's), kit_read_ahead_ref run
/// @param upto   bytes of the text the stop-string scan released
/// @param final  the turn's end is being fed
/// @return how many of them the parser reads
inline size_t kit_read_limit_ref(const kit_gen *q, size_t upto, bool final) {
    if (q->cut) return upto < q->cut_at ? upto : q->cut_at;
    if (q->call_closed && !final && upto > q->call_end) return q->call_end;
    return upto;
}

/// parser_kit.cpp's parser_kit_probe_new at b1a217d6: a kit_gen around a clone() of the parser (and of its read-ahead).
/// @param st  the kit state (a kit_gen, from parser_kit_start)
/// @return a new probe (kit_probe_free_ref)
inline void *kit_probe_new_ref(const void *st) {
    const kit_gen *q = (const kit_gen *)st;
    kit_gen *p = new kit_gen();
    p->parser = q->parser->clone();
    p->fed = q->fed;
    /* L428: the read-ahead with it, so the probe says where the turn ends as the state will (turn_complete) */
    if (q->ahead) p->ahead = q->ahead->clone();
    p->ahead_fed = q->ahead_fed;
    p->call_closed = q->call_closed;
    p->call_end = q->call_end;
    p->tail_at = q->tail_at;
    p->cut = q->cut;
    p->cut_at = q->cut_at;
    return p;
}

/// parser_kit.cpp's parser_kit_probe_feed at b1a217d6.
/// @param probe  the probe
/// @param g      the attempt
/// @param text   the generated text with the round's tokens so far
/// @param len    its length
/// @param upto   bytes the stop-string scan released
inline void kit_probe_feed_ref(void *probe, const gen_state *g, const char *text, size_t len, size_t upto) {
    kit_gen *p = (kit_gen *)probe;
    kit_read_ahead_ref(p, g, text, len);
    upto = kit_read_limit_ref(p, upto, false);
    if (upto <= p->fed) return;
    p->ev.clear();
    p->parser->feed(text + p->fed, upto - p->fed, &p->ev);
    p->fed = upto;
}

/// parser_kit.cpp's parser_kit_probe_free at b1a217d6.
/// @param probe  the probe (kit_probe_new_ref)
inline void kit_probe_free_ref(void *probe) { delete (kit_gen *)probe; }

/// The probe a walk at b1a217d6 got from g->parser->probe_new: the kit's clone for a kit family, else the family's.
/// @param g  the attempt
/// @return the probe (gen_probe_free_ref)
inline void *gen_probe_new_ref(const gen_state *g) {
    return g->parser->probe_new == parser_kit_probe_new ? kit_probe_new_ref(g->parser_st)
                                                        : g->parser->probe_new(g->parser_st);
}

/// g->parser->probe_feed at b1a217d6.
/// @param probe  the probe (gen_probe_new_ref)
/// @param g      the attempt
/// @param text   the walk's copy of the generated text with the round's tokens so far
/// @param len    its length
/// @param upto   bytes the stop-string scan released
inline void gen_probe_feed_ref(void *probe, const gen_state *g, const char *text, size_t len, size_t upto) {
    if (g->parser->probe_new == parser_kit_probe_new) kit_probe_feed_ref(probe, g, text, len, upto);
    else g->parser->probe_feed(probe, g, text, len, upto);
}

/// g->parser->probe_free at b1a217d6.
/// @param probe  the probe (gen_probe_new_ref)
/// @param g      the attempt
inline void gen_probe_free_ref(void *probe, const gen_state *g) {
    if (g->parser->probe_new == parser_kit_probe_new) kit_probe_free_ref(probe);
    else g->parser->probe_free(probe);
}

/// server_jobs.cpp's gen_tool_override_varies at b1a217d6.
/// @param g  the generation state
/// @return true when the per-position override must be computed
inline bool gen_tool_override_varies_ref(const gen_state *g) {
    const request *req = &g->j->req;
    float temperature, top_p, min_p;
    int top_k;
    gen_resolve_sampling(req, &temperature, &top_k, &top_p, &min_p);
    return req->kind == REQ_CHAT && req->has_tools && g->parser_st && temperature > 0.0f;
}

/// server_jobs.cpp's gen_spec_greedy_rows_text at b1a217d6: the walk over a copy of the text and a cloned parser.
/// @param g             the request's generation state
/// @param pieces        each token's text (NULL = a stop token)
/// @param piece_lens    their lengths, bytes
/// @param n             how many
/// @param base_emitted  the base was already emitted
/// @return the greedy-row bit set
inline uint32_t gen_spec_greedy_rows_text_ref(const gen_state *g, const char *const *pieces, const size_t *piece_lens,
                                              int n, bool base_emitted) {
    if (!gen_tool_override_varies_ref(g)) return 0u;
    const request *req = &g->j->req;
    void *probe = gen_probe_new_ref(g);
    buf text;
    memset(&text, 0, sizeof text);
    if (g->text.len) buf_append(&text, g->text.ptr, g->text.len);
    thinking_state thinking = g->thinking;
    size_t stop_scan_from = g->stop_scan_from;
    size_t plain_stream_pos = g->plain_stream_pos;
    bool greedy = g->parser->in_tool_call(probe, g);
    uint32_t bits = base_emitted && greedy ? 1u : 0u;   /* row 0 follows a streamed base: the state as it stands */
    int row = base_emitted ? 1 : 0;
    bool live = true;
    for (int i = 0; i < n && row < 32; i++, row++) {
        if (live && !pieces[i]) live = false;
        if (live) {
            buf_append(&text, pieces[i], piece_lens[i]);
            size_t stop_pos = 0, stop_len = 0;
            bool hit_stop = false;
            const size_t upto = gen_release_ref(&req->stops, text.ptr, text.len, pieces[i], piece_lens[i],
                                                plain_stream_pos, &thinking, &stop_scan_from, &hit_stop, &stop_pos,
                                                &stop_len);
            if (hit_stop) {
                live = false;
            } else {
                if (gen_streams_plain(g) && upto > plain_stream_pos) plain_stream_pos = upto;
                gen_probe_feed_ref(probe, g, text.ptr, text.len, upto);
                greedy = g->parser->in_tool_call(probe, g);
            }
        }
        if (greedy) bits |= 1u << row;
    }
    buf_free(&text);
    gen_probe_free_ref(probe, g);
    return bits;
}

/// server_jobs.cpp's gen_turn_draft_prefix_text at b1a217d6.
/// @param g           the request's generation state
/// @param pieces      the base's text, then each draft's (NULL = a stop token)
/// @param piece_lens  their lengths, bytes
/// @param n           the drafts (pieces holds n + 1)
/// @return the usable prefix length, 0..n
inline int gen_turn_draft_prefix_text_ref(const gen_state *g, const char *const *pieces, const size_t *piece_lens,
                                          int n) {
    const request *req = &g->j->req;
    if (n <= 0 || !req->single_tool_call || req->kind != REQ_CHAT || !req->has_tools || !g->parser_st) return n;
    void *probe = gen_probe_new_ref(g);
    buf text;
    memset(&text, 0, sizeof text);
    if (g->text.len) buf_append(&text, g->text.ptr, g->text.len);
    thinking_state thinking = g->thinking;
    size_t stop_scan_from = g->stop_scan_from;
    size_t plain_stream_pos = g->plain_stream_pos;
    int keep = n;
    for (int i = 0; i <= n; i++) {   /* pieces[0] is the base, pieces[i] draft i - 1 */
        if (!pieces[i]) break;
        buf_append(&text, pieces[i], piece_lens[i]);
        size_t stop_pos = 0, stop_len = 0, end = 0;
        bool hit_stop = false;
        const size_t upto = gen_release_ref(&req->stops, text.ptr, text.len, pieces[i], piece_lens[i],
                                            plain_stream_pos, &thinking, &stop_scan_from, &hit_stop, &stop_pos,
                                            &stop_len);
        if (hit_stop) break;
        if (gen_streams_plain(g) && upto > plain_stream_pos) plain_stream_pos = upto;
        gen_probe_feed_ref(probe, g, text.ptr, text.len, upto);
        if (g->parser->turn_complete(probe, g, &end)) {
            keep = i;
            break;
        }
    }
    buf_free(&text);
    gen_probe_free_ref(probe, g);
    return keep;
}
