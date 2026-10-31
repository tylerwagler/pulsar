/** @file tokenizer.cpp
 *  @brief DeepSeek's byte-level BPE tokenizer: vocabulary tables, BPE merge and chat tokenization. */
#include "pulsar_engine_internal.h"
#include "pulsar_utf8.h"
#include <string>



/** The smallest power of two >= n (1 for n <= 1).  @param n the value  @return the power of two */
static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}



/** Allocate an empty open-addressing table sized for `expected` entries (capacity a power of two >= 2x + 16).
 *  @param t        the table
 *  @param expected entries it will hold */
static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->used = 0;
    t->entry = (str_i32_entry *)xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}



/** Free a table's entries and zero it (keys are borrowed, not freed).  @param t the table */
static void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}



/** Insert or overwrite `key` (linear probing; the table never grows, so it must be sized for every key).
 *  @param t     the table
 *  @param key   the key; borrowed, must outlive the table
 *  @param value the value */
static void table_put(str_i32_table *t, pulsar_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;

    while (t->entry[i].used) {
        if (pulsar_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }

    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}



/// @param t     the table
/// @param ptr   the key's bytes
/// @param len   their count
/// @param value receives the value when found
/// @return true when found
static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (t->cap == 0) return false;

    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(ptr, len) & mask;

    while (t->entry[i].used) {
        pulsar_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}



/** Append a token, growing the vector by doubling (from 64).  @param tv the vector  @param token the token id */
void token_vec_push(token_vec *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = (int *)xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}



/** Free the vector's storage and zero it.  @param tv the vector */
void token_vec_free(token_vec *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}



/** Append a token.  @param tv the token list  @param token the token id */
void pulsar_tokens_push(pulsar_tokens *tv, int token) {
    token_vec_push(tv, token);
}



/** Free the list's storage and zero it.  @param tv the token list */
void pulsar_tokens_free(pulsar_tokens *tv) {
    token_vec_free(tv);
}



/** Make `dst` a copy of `src`, reusing dst's storage when it is large enough.  @param dst the destination
 *  @param src the source */
void pulsar_tokens_copy(pulsar_tokens *dst, const pulsar_tokens *src) {
    /* Reserve once and memcpy: the push loop this replaced called xrealloc O(n)
     * times on the first copy into an empty vector, and the spec bank carry
     * copies a whole 100k-token history several times per round (L219).  The
     * growth policy is unchanged so allocation behavior does not shift. */
    dst->len = 0;
    if (src->len <= 0) return;
    if (dst->cap < src->len) {
        while (dst->cap < src->len) dst->cap = dst->cap ? dst->cap * 2 : 16;
        dst->v = (int *)xrealloc(dst->v, (size_t)dst->cap * sizeof(dst->v[0]));
    }
    memcpy(dst->v, src->v, (size_t)src->len * sizeof(dst->v[0]));
    dst->len = src->len;
}



/** Does `tokens` begin with `prefix`?  @param tokens the list  @param prefix the prefix
 *  @return true when it does (an empty prefix always) */
bool pulsar_tokens_starts_with(const pulsar_tokens *tokens, const pulsar_tokens *prefix) {
    if (prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}












/** Write `cp` as UTF-8 at *p and advance *p (1-4 bytes; no validation).  @param p the write cursor
 *  @param cp the codepoint */
static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}



/** GPT-2's byte-to-unicode map: printable bytes map to themselves, the rest to 256 + their rank.
 *  @param b the byte  @return the codepoint */
static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return b;
    }

    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}



/* GPT-2 byte-level BPE first maps raw bytes to printable Unicode codepoints
 * so merges can operate on UTF-8 strings without losing byte identity. */
/// @param in      the raw bytes
/// @param out_len receives the encoded length in bytes
/// @return the NUL-terminated encoding, xmalloc'd (the caller frees it)
static char *byte_encode(pulsar_str in, uint64_t *out_len) {
    char *out = (char *)xmalloc((size_t)in.len * 4 + 1);
    char *p = out;

    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}



/* Decode the UTF-8 sequence at s[pos] (strict rule: pulsar_utf8.h).  An
 * ill-formed byte -- bad lead, bad or truncated continuation, Table 3-7
 * violation -- is its own one-byte unit and decodes to its byte value.  Sets
 * *next just past the consumed bytes. */
/// @param s    the text
/// @param len  its length in bytes
/// @param pos  the position decoded, < len
/// @param next receives the position after the unit
/// @return the codepoint (an ill-formed byte's value)
static uint32_t utf8_decode_at(const char *s, uint64_t len, uint64_t pos, uint64_t *next) {
    const unsigned char *p = (const unsigned char *)s + pos;
    const int n = utf8_seq_ok(p, (size_t)(len - pos));
    if (n <= 1) {
        *next = pos + 1;
        return p[0];
    }
    uint32_t cp = p[0] & (0x7fu >> n);
    for (int i = 1; i < n; i++) cp = (cp << 6) | (p[i] & 0x3fu);
    *next = pos + (uint64_t)n;
    return cp;
}



/** A heap copy of [ptr, ptr + len) (not NUL-terminated).  @param ptr the bytes  @param len their count
 *  @return the copy (its ptr xmalloc'd; the owner frees it) */
static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = (char *)xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}



/* Look up the merge rank for two adjacent BPE symbols. */
int pulsar_vocab::bpe_rank(const owned_str *a, const owned_str *b) const {
    const auto *vocab = this;
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : (char *)xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);

    if (buf != stack) free(buf);
    return rank;
}



/* Apply byte-level BPE to one regex-like pre-tokenized piece and emit token ids. */
void pulsar_vocab::bpe_emit_piece(pulsar_str raw_piece, token_vec *out) const {
    const auto *vocab = this;
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);

    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = (owned_str *)xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_seq_ok((const unsigned char *)encoded + off, (size_t)(encoded_len - off));
        if (n == 0) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = (owned_str *)xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = vocab->bpe_rank(&sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }

        if (best_i < 0) break;

        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = (char *)xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;

        for (int j = best_i + 1; j + 1 < n_sym; j++) {
            sym[j] = sym[j + 1];
        }
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            token_vec_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    token_vec_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
}



/** The position after the UTF-8 unit at pos (an ill-formed byte is one unit).  @param s the text
 *  @param len its length  @param pos the position, < len  @return the next position */
static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    const int n = utf8_seq_ok((const unsigned char *)s + pos, (size_t)(len - pos));
    return pos + (uint64_t)(n ? n : 1);
}



/** ASCII A-Z / a-z.  @param c the byte  @return true for a letter */
static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}



/** ASCII 0-9.  @param c the byte  @return true for a digit */
static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}



/** ASCII space, \\t, \\n, \\r, \\v, \\f.  @param c the byte  @return true for whitespace */
static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}



/** \\n or \\r.  @param c the byte  @return true for a newline byte */
static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}



/** The JoyAI pre-tokenizer's ASCII punctuation / symbol class.  @param c the byte  @return true when in it */
static bool joyai_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}



/** CJK unified ideographs (U+4E00-U+9FA5), hiragana or katakana.  @param cp the codepoint
 *  @return true when in those ranges */
static bool utf8_is_cjk_hira_kata(uint32_t cp) {
    return (cp >= 0x4e00 && cp <= 0x9fa5) ||
           (cp >= 0x3040 && cp <= 0x309f) ||
           (cp >= 0x30a0 && cp <= 0x30ff);
}



/** Does a "letter" start at pos (an ASCII letter, or any non-ASCII byte; see below)?  @param s the text
 *  @param len its length (unused)  @param pos the position  @return true for a letter-like unit */
static bool joyai_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    (void)len;
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return ascii_alpha(c);

    /*
     * The JoyAI tokenizer maps Unicode letters into a collapsed regex alphabet before
     * applying the JoyAI pre-tokenizer.  The prompts we care about are mostly
     * ASCII, but treating non-ASCII non-control bytes as letters preserves the
     * useful behavior for ordinary UTF-8 text such as Italian accents.  CJK and
     * kana are isolated by the JoyAI pre-tokenizer before the generic letter
     * rule, below.
     */
    return true;
}



/** Skip a run of letter-like units.  @param s the text  @param len its length  @param pos the run's start
 *  @return the position after the run */
static uint64_t joyai_consume_letters(const char *s, uint64_t len, uint64_t pos) {
    while (pos < len && joyai_letter_like_at(s, len, pos)) {
        pos = next_utf8_char(s, len, pos);
    }
    return pos;
}



/** Does a CJK / kana character start at pos?  @param s the text  @param len its length  @param pos the position
 *  @return true when it does */
static bool joyai_cjk_at(const char *s, uint64_t len, uint64_t pos) {
    if ((uint8_t)s[pos] < 128) return false;
    uint64_t next = pos;
    uint32_t cp = utf8_decode_at(s, len, pos, &next);
    return utf8_is_cjk_hira_kata(cp);
}



/*
 * DeepSeek V4 Flash declares tokenizer.ggml.pre = "joyai-llm".  The split
 * below mirrors the JoyAI BPE pre-tokenizer for the cases this model
 * uses in normal text and source-code prompts:
 *
 *   \p{N}{1,3}
 *   [CJK/Hiragana/Katakana]+
 *   [P/S][A-Za-z]+
 *   [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
 *    ?[\p{P}\p{S}]+[\r\n]*
 *   \s*[\r\n]+
 *   \s+(?!\S)
 *   \s+
 *
 * The punctuation rule intentionally keeps trailing newlines in the same BPE
 * word (for example ">;\n").  Splitting those newlines separately changes the
 * token stream for code prompts and produces wrong long-context logits.
 */
/* JoyAI/DeepSeek pre-tokenization.  The split shape matters: different pieces
 * lead to different BPE merges even when the final text bytes are identical. */
void pulsar_vocab::bpe_tokenize_text(const char *text, token_vec *out) const {
    const auto *vocab = this;
    const uint64_t len = strlen(text);
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];

        if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                pos++;
                ndigits++;
            }
        } else if (joyai_cjk_at(text, len, pos)) {
            do {
                pos = next_utf8_char(text, len, pos);
            } while (pos < len && joyai_cjk_at(text, len, pos));
        } else if (joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   ascii_alpha((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && ascii_alpha((uint8_t)text[pos])) pos++;
        } else if (joyai_letter_like_at(text, len, pos)) {
            pos = joyai_consume_letters(text, len, pos);
        } else if (!ascii_newline(c) &&
                   !joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   joyai_letter_like_at(text, len, pos + 1)) {
            pos++;
            pos = joyai_consume_letters(text, len, pos);
        } else if (c == ' ' &&
                   pos + 1 < len &&
                   joyai_ascii_punct_symbol((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (joyai_ascii_punct_symbol(c)) {
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1 &&
                       (joyai_letter_like_at(text, len, p) ||
                        joyai_ascii_punct_symbol((uint8_t)text[p]))) {
                /*
                 * JoyAI lets a single leading space join the following word or
                 * punctuation run.  For "    int", the pre-tokenizer therefore emits
                 * "   " then " int", not "    " then "int".
                 */
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, len, pos);
        }

        if (pos == start) pos = next_utf8_char(text, len, pos);
        vocab->bpe_emit_piece((pulsar_str){ text + start, pos - start }, out);
    }
}



int pulsar_vocab::vocab_lookup(const char *text) const {
    const auto *vocab = this;
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        fprintf(stderr, "pulsar: required tokenizer token is missing: %s\n", text);
        pulsar_load_refuse();   /* L272: load time only (vocab_load); the load fails at its stage boundary */
        return -1;
    }
    return token;
}



/* The optional counterpart: an artifact legitimately lacks the image
 * placeholder, so its absence is a fact (no image path), not a refusal. */
int pulsar_vocab::vocab_find(const char *text) const {
    const auto *vocab = this;
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) return -1;
    return token;
}



/* Load token strings, special token ids, and merge ranks from GGUF metadata. */
void pulsar_vocab::vocab_load(const pulsar_model *model) {
    auto *vocab = this;
    memset(vocab, 0, sizeof(*vocab));

    pulsar_array_ref tokens;
    pulsar_array_ref merges;
    if (!model_get_array(model, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != PULSAR_META_STRING ||
        tokens.len > INT32_MAX) {
        pulsar_die("GGUF tokenizer token table is missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.merges", &merges) ||
        merges.type != PULSAR_META_STRING) {
        pulsar_die("GGUF tokenizer merge table is missing or invalid");
    }

    vocab->n_vocab = (int)tokens.len;
    vocab->token = (pulsar_str *)xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);

    pulsar_cursor c = cursor_at(model, tokens.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        if (!cursor_string(&c, &vocab->token[i])) pulsar_die(c.error);
        table_put(&vocab->token_to_id, vocab->token[i], i);
    }

    table_init(&vocab->merge_rank, merges.len);
    c = cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        pulsar_str merge;
        if (!cursor_string(&c, &merge)) pulsar_die(c.error);
        table_put(&vocab->merge_rank, merge, (int)i);
    }

    vocab->bos_id       = vocab->vocab_lookup("<｜begin▁of▁sentence｜>");
    vocab->eos_id       = vocab->vocab_lookup("<｜end▁of▁sentence｜>");
    vocab->user_id      = vocab->vocab_lookup("<｜User｜>");
    vocab->assistant_id = vocab->vocab_lookup("<｜Assistant｜>");
    /* The System marker is V4.1's (the template that writes it is), and the
     * 0731 artifacts built before that encoding landed do not carry it -- their
     * vocab is otherwise complete.  Required-vs-optional is therefore a
     * two-profile decision, not a format check: requiring it makes the unified
     * engine refuse a 0731 artifact it served the day before.  Optional here,
     * with the requirement enforced where the marker is actually written
     * (chat_tmpl_push_system_marker). */
    vocab->system_id    = vocab->vocab_find("<｜System｜>");
    vocab->think_start_id = vocab->vocab_lookup("<think>");
    vocab->think_end_id = vocab->vocab_lookup("</think>");
    vocab->dsml_id = vocab->vocab_lookup("｜DSML｜");
    /* The image placeholder is resolved BY STRING, never a literal id: an
     * artifact without the token leaves image_id < 0 and cannot serve images,
     * and one that has it gets whichever id its own table assigned.  This is
     * the optional lookup -- a text-only artifact must still load. */
    vocab->image_id = vocab->vocab_find(PULSAR_IMAGE_PLACEHOLDER);
}



void pulsar_vocab::vocab_free() {
    auto *vocab = this;
    free(vocab->token);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}



/* The V4.1 conversation lead-in after BOS: the System token when the
 * conversation opens with the effort line (thinking on) or with system text,
 * then the effort line.  One authority for the CLI path; the server renderer
 * writes the same bytes (prompt_render.cpp). */
/* --- L185: the chat template's TEXT rules, ONE authority --------------------
 *
 * The server renders the template as TEXT and tokenises it once, so a message's
 * tokens span the text RUN between two markers: `user text` + `\n\n` +
 * `<tool_result>` + body + `</tool_result>` is ONE BPE run, and a mid-conversation
 * system message drags its `<system-reminder>` wrapper into the run around it.
 * The twin used to tokenise each piece separately, which is why it disagreed
 * with the renderer on four measured shapes.
 *
 * The twin now appends the same TEXT and RE-TOKENISES the open run -- the tokens
 * after the last template marker -- whenever text joins it.  The run's text is
 * recovered from the tokens themselves (vocab_token_text is the inverse the KV
 * keys already rely on), so the twin carries NO state of its own: the rules
 * below are the only authority, and token surgery by a caller (the CLI REPL's
 * head swap, compaction's transcript swap) cannot desynchronise a state
 * copy because there is none.
 *
 * The flush is eager: a run is tokenised as soon as its text is known, and
 * joining more text re-tokenises it.  That can move tokens inside the OPEN run,
 * which is exactly what the renderer does too -- a caller that has already synced
 * that prefix sees a shorter common prefix and re-prefills from there. */

/** The ids that bound a text run: every marker this template emits. */
/// @param v  the vocab
/// @param id the token id
/// @return true for a BOS / EOS / role / think marker
static bool chat_tmpl_marker(const pulsar_vocab *v, int id) {
    return id == v->bos_id || id == v->eos_id || id == v->system_id ||
           id == v->user_id || id == v->assistant_id ||
           id == v->think_start_id || id == v->think_end_id;
}

/** Where the open text run starts: just after the last marker.  @param v the vocab  @param t the tokens
 *  @return the run's first index (0 when no marker) */
static int chat_tmpl_run_start(const pulsar_vocab *v, const pulsar_tokens *t) {
    for (int i = t->len - 1; i >= 0; i--)
        if (chat_tmpl_marker(v, t->v[i])) return i + 1;
    return 0;
}

/** The last marker's id.  @param v the vocab  @param t the tokens  @return the id, -1 when none */
static int chat_tmpl_last_marker(const pulsar_vocab *v, const pulsar_tokens *t) {
    for (int i = t->len - 1; i >= 0; i--)
        if (chat_tmpl_marker(v, t->v[i])) return t->v[i];
    return -1;
}

/** Has a user or assistant message been appended yet?  A system message before
 * one is the LEADING system region (plain content); after one it is a
 * mid-conversation note (the wrapper rule applies). */
/// @param v the vocab
/// @param t the tokens
/// @return true once a user or assistant marker is present
static bool chat_tmpl_saw_turn(const pulsar_vocab *v, const pulsar_tokens *t) {
    for (int i = 0; i < t->len; i++)
        if (t->v[i] == v->user_id || t->v[i] == v->assistant_id) return true;
    return false;
}

/** The text of tokens [start, len), recovered from the vocab.  @param v the vocab  @param t the tokens
 *  @param start the first index  @param n_out receives the byte length (may be NULL)
 *  @return the NUL-terminated text, xmalloc'd (the caller frees it) */
static char *chat_tmpl_run_text(const pulsar_vocab *v, const pulsar_tokens *t,
                                int start, size_t *n_out) {
    size_t n = 0;
    for (int i = start; i < t->len; i++) {
        size_t l = 0;
        char *p = vocab_token_text(v, t->v[i], &l);
        n += l;
        free(p);
    }
    char *out = (char *)xmalloc(n + 1);
    size_t at = 0;
    for (int i = start; i < t->len; i++) {
        size_t l = 0;
        char *p = vocab_token_text(v, t->v[i], &l);
        memcpy(out + at, p, l);
        at += l;
        free(p);
    }
    out[at] = '\0';
    if (n_out) *n_out = at;
    return out;
}

/** Append a marker id.  @param t the tokens  @param id the marker */
static void chat_tmpl_push_marker(pulsar_tokens *t, int id) {
    token_vec_push((token_vec *)t, id);
}

/** Push the System MARKER, refusing loudly if this artifact has no such token.
 *
 * The marker is the V4.1 template's (see encode_chat_lead_in), and V4.1's own
 * vocab always carries it -- but the 0731 artifacts built before the V4.1
 * encoding landed do NOT, and their vocab is otherwise complete (checked: the
 * three shipped 0731 GGUFs differ in exactly this one marker).  Making the
 * LOOKUP required therefore refused artifacts the engine served happily the day
 * before, which is a two-profile regression, not a format check: at f5bea7aa
 * `v5mx4-0731-srcfmt-v1-reapfix-lt.gguf` ran; one commit later it exited at load
 * with "required tokenizer token is missing".
 *
 * So the lookup is optional (like the image placeholder) and the requirement
 * lives HERE, at the point of use, where the family is already known: a V4.1
 * prompt rendered against a vocab without the marker would silently drop it --
 * exactly the silent-template-drift class the render-bytes gate exists for --
 * so it refuses by name instead. */
/// @param v the vocab
/// @param t the tokens (the marker appended; exits the process when the vocab has none)
static void chat_tmpl_push_system_marker(const pulsar_vocab *v, pulsar_tokens *t) {
    if (v->system_id < 0) {
        fprintf(stderr, "pulsar: this artifact's vocabulary has no System marker, which the V4.1 chat "
                        "template requires -- refusing to render a prompt without it\n");
        exit(1);
    }
    chat_tmpl_push_marker(t, v->system_id);
}

/** Append TEXT to the open run and re-tokenise the run -- the one primitive that
 * makes the twin's BPE boundary the renderer's. */
/// @param v    the vocab
/// @param t    the tokens
/// @param text NUL-terminated text (NULL or empty: nothing)
static void chat_tmpl_append_text(const pulsar_vocab *v, pulsar_tokens *t, const char *text) {
    if (!text || !text[0]) return;
    const int start = chat_tmpl_run_start(v, t);
    size_t run = 0;
    char *have = chat_tmpl_run_text(v, t, start, &run);
    const size_t add = strlen(text);
    char *joined = (char *)xmalloc(run + add + 1);
    memcpy(joined, have, run);
    memcpy(joined + run, text, add + 1);
    free(have);
    t->len = start;                          /* drop the run, re-tokenise it joined */
    v->tokenize_span(joined, run + add, (token_vec *)t);
    free(joined);
}

/** chat_tmpl_escape_emit's user data: where the escaped tool-result pieces are appended (both borrowed). */
struct chat_tmpl_escape_sink {
    /// the vocab that tokenises each piece
    const pulsar_vocab *vocab;
    /// the token run the pieces are appended to (its open run, re-tokenised joined)
    pulsar_tokens *tokens;
};

/** pulsar_tool_result_escape's sink: append one escaped piece to the open run.  @param ud a chat_tmpl_escape_sink
 *  @param bytes the piece  @param n its length */
static void chat_tmpl_escape_emit(void *ud, const char *bytes, size_t n) {
    auto *sink = (chat_tmpl_escape_sink *)ud;
    char *piece = (char *)xmalloc(n + 1);
    memcpy(piece, bytes, n);
    piece[n] = '\0';
    chat_tmpl_append_text(sink->vocab, sink->tokens, piece);
    free(piece);
}

/** Append a tool-result body with the sentinel escaped (pulsar_tool_result_escape
 * is the ONE authority for that rule), into the open run. */
/// @param v       the vocab
/// @param t       the tokens
/// @param content the tool result's body
static void chat_tmpl_append_tool_body(const pulsar_vocab *v, pulsar_tokens *t,
                                       const char *content) {
    chat_tmpl_escape_sink sink = { v, t };
    pulsar_tool_result_escape(content, chat_tmpl_escape_emit, &sink);
}

/** Does the open run end with the tool-result wrapper's close?  That is the
 * renderer's `pending_tool_result` fact, derived instead of stored. */
/// @param v the vocab
/// @param t the tokens
/// @return true when the open run's text ends with "</tool_result>"
static bool chat_tmpl_run_ends_tool_result(const pulsar_vocab *v, const pulsar_tokens *t) {
    static const char end[] = "</tool_result>";
    const size_t endlen = sizeof(end) - 1;
    const int start = chat_tmpl_run_start(v, t);
    size_t run = 0;
    char *text = chat_tmpl_run_text(v, t, start, &run);
    const bool yes = run >= endlen && memcmp(text + run - endlen, end, endlen) == 0;
    free(text);
    return yes;
}


/** The chat's lead-in: the System marker (V4.1, when there is an effort line or a system message) and the effort
 *  line's text in the open run.
 *  @param vocab      the vocab
 *  @param has_system a system message follows
 *  @param think_mode the reasoning effort
 *  @param out        the tokens appended to */
static void encode_chat_lead_in(const pulsar_vocab *vocab, bool has_system,
                                pulsar_think_mode think_mode, token_vec *out) {
    /* The effort line is the FAMILY's spelling (the same fact as the marker
     * below: V4.1 marks the region and writes the numeric line; 0731 does
     * neither at low and writes its own texts at high/max) -- L239: the CLI's
     * 0731 prompt carried V4.1's numeric line, as the server's did. */
    const char *effort_prefix = pulsar_think_effort_prefix_family(think_mode, PULSAR_CHAT_SYSTEM_MARKER);
    /* The MARKER is the template's, and the template is the model family's:
     * V4.1 marks its lead-in system region, 0731 does not.  Writing it for both
     * cost V4 exactly one prompt token -- enough to change the answer. */
    if (PULSAR_CHAT_SYSTEM_MARKER && (effort_prefix[0] || has_system))
        chat_tmpl_push_system_marker(vocab, out);
    /* The effort line is TEXT, and the system region joins it unchanged (the
     * renderer writes marker + effort + region as one run), so it goes into the
     * open run rather than being tokenised on its own. */
    chat_tmpl_append_text(vocab, out, effort_prefix);
}



/* The tokenizer entries serve a family with a tokenizer table (pulsar_family::tokenizer); the chat front's entries
 * (chat_open / chat_turn / chat_end_assistant) a family that also declares its chat front (family.h
 * PULSAR_FAMILY_CAP_CHAT) -- L295: a family's tokenizer can be wired before its chat renderer is (token ids and
 * text in and out without the turn-by-turn chat front).  These entries return no error, so a family without
 * what they need ends the process here, by name, instead of tokenizing with an empty vocabulary; the front ends
 * refuse such an engine at startup (pulsar_engine_has_tokenizer, which asks for both) before they can reach this. */
/** The engine's tokenizer, or exit(1) by name when its family has none.  @param e the engine
 *  @param op names the entry  @return the tokenizer ops */
static const pulsar_family_tokenizer *tokenizer_require(const pulsar_engine *e, const char *op) {
    if (e && e->family->tokenizer) return e->family->tokenizer;
    fprintf(stderr, "pulsar: %s has no tokenizer table (%s) -- exiting\n", e ? e->family->name : "(no engine)", op);
    exit(1);
}
/** The engine's tokenizer for a chat entry, or exit(1) when the family has no chat front.  @param e the engine
 *  @param op names the entry  @return the tokenizer ops */
static const pulsar_family_tokenizer *chat_require(const pulsar_engine *e, const char *op) {
    if (e && pulsar_family_require(e, PULSAR_FAMILY_CAP_CHAT, op)) return tokenizer_require(e, op);
    fprintf(stderr, "pulsar: %s has no chat front in this build -- exiting\n", e ? e->family->name : "(no engine)");
    exit(1);
}

bool pulsar_engine_has_tokenizer(const pulsar_engine *e) {
    return !e || ((e->family->caps & PULSAR_FAMILY_CAP_CHAT) != 0 && e->family->tokenizer);
}

bool pulsar_token_is_stop(pulsar_engine *e, int token) {
    return tokenizer_require(e, "pulsar_token_is_stop")->is_stop(e, token);
}



/** Tokenize plain text (no chat rendering) through the family's tokenizer (exits without one).
 *  @param e    the engine
 *  @param text the text
 *  @param out  receives the tokens (appended) */
void pulsar_tokenize_text(pulsar_engine *e, const char *text, pulsar_tokens *out) {
    tokenizer_require(e, "pulsar_tokenize_text")->encode_text(e, text, out);
}



bool pulsar_vocab::special_token_at(const char *p, int *token, size_t *len) const {
    const auto *vocab = this;
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"<｜begin▁of▁sentence｜>", vocab->bos_id},
        {"<｜end▁of▁sentence｜>",   vocab->eos_id},
        {"<｜User｜>",              vocab->user_id},
        {"<｜Assistant｜>",         vocab->assistant_id},
        {"<｜System｜>",            vocab->system_id},
        {"<think>",                vocab->think_start_id},
        {"</think>",               vocab->think_end_id},
        {"｜DSML｜",                vocab->dsml_id},
        {PULSAR_IMAGE_PLACEHOLDER, vocab->image_id},
    };

    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        if (specials[i].token < 0) continue;   /* absent in this artifact's vocab */
        size_t n = strlen(specials[i].text);
        if (!strncmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}



void pulsar_vocab::tokenize_span(const char *p, size_t n, token_vec *out) const {
    const auto *vocab = this;
    if (!n) return;
    char *tmp = (char *)xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    vocab->bpe_tokenize_text(tmp, out);
    free(tmp);
}



void pulsar_vocab::tokenize_rendered_chat_vocab(const char *text,
                                                token_vec *out) const {
    const auto *vocab = this;
    if (!text) text = "";

    const char *span = text;
    const char *p = text;
    while (*p) {
        int token = -1;
        size_t len = 0;
        if (vocab->special_token_at(p, &token, &len)) {
            vocab->tokenize_span(span, (size_t)(p - span), out);
            token_vec_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    vocab->tokenize_span(span, (size_t)(p - span), out);
}



/* L223: as above, but a byte inside one of the n half-open [lo, hi) ranges is
 * PLAIN TEXT -- special-token matching is suppressed there.  The renderer records
 * those ranges for every byte it copied from client data, so a client typing
 * `<|Assistant|>` or `｜DSML｜` into a message gets ordinary text tokens instead of
 * a control token.  Ranges are ascending and disjoint; a range is clamped to the
 * string's end. */
void pulsar_vocab::tokenize_rendered_chat_spans_vocab(const char *text,
                                                      const pulsar_text_span *spans,
                                                      uint32_t n_spans,
                                                      token_vec *out) const {
    const auto *vocab = this;
    if (!text) text = "";
    const char *base = text;
    const char *end_of_text = text + strlen(text);
    const char *span = text;
    const char *p = text;
    uint32_t si = 0;
    while (*p) {
        const size_t off = (size_t)(p - base);
        while (si < n_spans && (off >= spans[si].hi || spans[si].hi <= spans[si].lo)) si++;
        if (si < n_spans && off >= spans[si].lo) {
            const char *end = base + spans[si].hi;
            if (end > end_of_text) end = end_of_text;   /* clamp to the string */
            vocab->tokenize_span(span, (size_t)(p - span), out);   /* flush the pending plain run */
            vocab->tokenize_span(p, (size_t)(end - p), out);       /* client text: BPE only */
            p = span = end;
            si++;
            continue;
        }
        int token = -1;
        size_t len = 0;
        if (vocab->special_token_at(p, &token, &len)) {
            vocab->tokenize_span(span, (size_t)(p - span), out);
            token_vec_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    vocab->tokenize_span(span, (size_t)(p - span), out);
}

/** Tokenize text that is already a rendered chat (its marker strings become marker ids) through the family's
 *  tokenizer (exits without one).
 *  @param e    the engine
 *  @param text the rendered text
 *  @param out  receives the tokens (appended) */
void pulsar_tokenize_rendered_chat(pulsar_engine *e, const char *text, pulsar_tokens *out) {
    tokenizer_require(e, "pulsar_tokenize_rendered_chat")->encode_rendered(e, text, NULL, 0u, out);
}

void pulsar_tokenize_rendered_chat_spans(pulsar_engine *e, const char *text,
                                         const pulsar_text_span *spans, uint32_t n_spans,
                                         pulsar_tokens *out) {
    const pulsar_family_tokenizer *t = tokenizer_require(e, "pulsar_tokenize_rendered_chat_spans");
    t->encode_rendered(e, text, n_spans && spans ? spans : NULL, n_spans && spans ? n_spans : 0u, out);
}

pulsar_text_span *pulsar_text_spans_slice(const pulsar_text_span *spans, uint32_t n_spans,
                                          size_t skip, size_t len, uint32_t *n_out) {
    if (n_out) *n_out = 0;
    if (!spans || !n_spans || !len) return NULL;
    pulsar_text_span *out = (pulsar_text_span *)xmalloc((size_t)n_spans * sizeof *out);
    const size_t end = skip + len;
    uint32_t k = 0;
    for (uint32_t i = 0; i < n_spans; i++) {
        size_t lo = spans[i].lo, hi = spans[i].hi;
        if (hi <= skip || lo >= end) continue;
        if (lo < skip) lo = skip;
        if (hi > end) hi = end;
        if (hi <= lo) continue;
        out[k].lo = (uint32_t)(lo - skip);
        out[k].hi = (uint32_t)(hi - skip);
        k++;
    }
    if (!k) { free(out); return NULL; }
    if (n_out) *n_out = k;
    return out;
}



void pulsar_chat_open(pulsar_engine *e, pulsar_tokens *tokens, const char *trusted, const char *system,
                      pulsar_think_mode think_mode) {
    chat_require(e, "pulsar_chat_open")->chat_open(e, tokens, trusted, system, think_mode);
}

void pulsar_chat_append_turn(pulsar_engine *e, pulsar_tokens *tokens, const pulsar_chat_message *msgs, int n,
                             bool generation_prompt, pulsar_think_mode think_mode) {
    chat_require(e, "pulsar_chat_append_turn")->chat_turn(e, tokens, msgs, n, generation_prompt, think_mode);
}

void pulsar_chat_end_assistant(pulsar_engine *e, pulsar_tokens *tokens) {
    chat_require(e, "pulsar_chat_end_assistant")->chat_end_assistant(e, tokens);
}

void pulsar_encode_chat_prompt(pulsar_engine *e, const char *system, const char *prompt, pulsar_think_mode think_mode,
                               pulsar_tokens *out) {
    pulsar_chat_open(e, out, NULL, system, think_mode);
    const pulsar_chat_message user = {"user", prompt ? prompt : "", false};
    pulsar_chat_append_turn(e, out, &user, 1, true, think_mode);
}



size_t pulsar_tool_result_escape(const char *s,
                                 void (*emit)(void *ud, const char *bytes, size_t n),
                                 void *ud) {
    /* Tool output is plain data inside <tool_result>...</tool_result>.  Preserve
     * literal '<', '>' and '&' so shell output and file snippets stay intact, but
     * escape the exact closing sentinel so a malicious or accidental tool payload
     * cannot terminate the wrapper early.  ONE authority (L185): the server's
     * renderer and the engine's token-level twin both walk this. */
    static const char end[] = "</tool_result>";
    static const char entity[] = "&lt;";
    const size_t endlen = sizeof(end) - 1;
    size_t total = 0;
    const char *span = s ? s : "";
    const char *p = span;
    while (*p) {
        if (!strncmp(p, end, endlen)) {
            const size_t n = (size_t)(p - span);
            if (n) { emit(ud, span, n); total += n; }
            emit(ud, entity, sizeof(entity) - 1);
            total += sizeof(entity) - 1;
            p++;
            span = p;
        } else {
            p++;
        }
    }
    if (p != span) { emit(ud, span, (size_t)(p - span)); total += (size_t)(p - span); }
    return total;
}







/* Front-end control text in DeepSeek's markers (pulsar_chat_message::trusted), tokenised as rendered chat into a
 * piece of its own: joining it to the open run would re-tokenise its markers (the DSML ids) as plain text. */
/// @param vocab  the vocab
/// @param tokens the tokens appended to
/// @param text   the trusted control text, tokenised as rendered chat
static void ds4_chat_append_trusted(pulsar_vocab *vocab, pulsar_tokens *tokens, const char *text) {
    vocab->tokenize_rendered_chat_vocab(text, (token_vec *)tokens);
}

/* One message of a DeepSeek turn: the renderer's rules, appended to the run they join. */
/// @param e       the engine
/// @param tokens  the tokens appended to
/// @param role    the message's role ("system", "user", "assistant", "tool", ...)
/// @param content the message's text
/// @param trusted the content is front-end control text in DeepSeek's markers
static void ds4_chat_append_message(pulsar_engine *e, pulsar_tokens *tokens, const char *role, const char *content,
                                    bool trusted) {
    pulsar_vocab *vocab = &e->vocab;
    const bool v41 = pulsar_engine_chat_v41(e);
    if (!role) role = "user";
    if (!content) content = "";

    if (!strcmp(role, "system") || !strcmp(role, "developer")) {
        /* The renderer's system rule: a system message BEFORE any user/assistant
         * turn is the leading system region (plain content, joined into the
         * lead-in's run); after one it is a mid-conversation note -- V4.1 marks
         * it in place with the System token, V4 wraps it in a user turn. */
        if (!chat_tmpl_saw_turn(vocab, tokens)) {
            if (trusted) ds4_chat_append_trusted(vocab, tokens, content);
            else chat_tmpl_append_text(vocab, tokens, content);
        } else if (v41) {
            chat_tmpl_push_system_marker(vocab, tokens);
            if (trusted) ds4_chat_append_trusted(vocab, tokens, content);
            else chat_tmpl_append_text(vocab, tokens, content);
        } else {
            chat_tmpl_push_marker(tokens, vocab->user_id);
            if (trusted) {
                std::string note = std::string("<system-reminder>\n") + content + "\n</system-reminder>";
                ds4_chat_append_trusted(vocab, tokens, note.c_str());
            } else {
                chat_tmpl_append_text(vocab, tokens, "<system-reminder>\n");
                chat_tmpl_append_text(vocab, tokens, content);
                chat_tmpl_append_text(vocab, tokens, "\n</system-reminder>");
            }
        }
    } else if (trusted) {
        pulsar_die("pulsar_chat_append_turn: only a system message carries trusted control text");
    } else if (!strcmp(role, "tool") || !strcmp(role, "function")) {
        /* V4.1 joins consecutive user-SIDE messages (user text and tool results in
         * any order) into one turn; V4 opens a fresh turn for each user message
         * and for the first tool result after one, and joins after a tool result. */
        const bool join = v41 ? chat_tmpl_last_marker(vocab, tokens) == vocab->user_id
                              : chat_tmpl_run_ends_tool_result(vocab, tokens);
        if (!join) chat_tmpl_push_marker(tokens, vocab->user_id);
        else chat_tmpl_append_text(vocab, tokens, "\n\n");
        chat_tmpl_append_text(vocab, tokens, "<tool_result>");
        chat_tmpl_append_tool_body(vocab, tokens, content);
        chat_tmpl_append_text(vocab, tokens, "</tool_result>");
    } else if (!strcmp(role, "assistant")) {
        /* Not reachable from this tree (no caller passes "assistant": the agent
         * and the CLI append a SAMPLED turn's tokens, and the server owns the
         * replay rules).  The old branch guessed a stripped replay and could not
         * be right; refuse instead of guessing. */
        pulsar_die("pulsar_chat_append_turn: an assistant turn cannot be "
                   "appended here -- append the sampled tokens, or render it "
                   "through the server's renderer (L185)");
    } else {
        const bool join = v41 && chat_tmpl_last_marker(vocab, tokens) == vocab->user_id;
        if (!join) chat_tmpl_push_marker(tokens, vocab->user_id);
        else chat_tmpl_append_text(vocab, tokens, "\n\n");
        chat_tmpl_append_text(vocab, tokens, content);
    }
}



/* ---- the DeepSeek family's chat front, turn by turn (pulsar.h pulsar_chat_open, L284 P14) ---------------- */

/* BOS, the lead-in (the System marker and effort line), then the leading system region: trusted control text
 * as a rendered piece of its own, and client text joining the open run (the effort line's, when there is no
 * trusted text -- as the renderer writes it). */
/// pulsar_family_tokenizer::chat_open for DeepSeek.
/// @param e          the engine
/// @param tokens     the tokens appended to
/// @param trusted    trusted control text for the system region, or NULL
/// @param system     the client's system message, or NULL
/// @param think_mode the reasoning effort
static void ds4_chat_open(pulsar_engine *e, pulsar_tokens *tokens, const char *trusted, const char *system,
                          pulsar_think_mode think_mode) {
    pulsar_vocab *vocab = &e->vocab;
    const bool has_system = system && system[0];
    chat_tmpl_push_marker(tokens, vocab->bos_id);
    encode_chat_lead_in(vocab, has_system, think_mode, (token_vec *)tokens);
    if (trusted && trusted[0]) ds4_chat_append_trusted(vocab, tokens, trusted);
    if (!has_system) return;
    if (trusted && trusted[0]) vocab->tokenize_span(system, strlen(system), (token_vec *)tokens);
    else chat_tmpl_append_text(vocab, tokens, system);
}

/** pulsar_family_tokenizer::chat_turn for DeepSeek: the messages, then (generation_prompt) the assistant marker and
 *  the think open / close marker.
 *  @param e                 the engine
 *  @param tokens            the tokens appended to
 *  @param msgs              [n] the messages
 *  @param n                 messages
 *  @param generation_prompt open the assistant's turn
 *  @param think_mode        the reasoning effort (on: the think-start marker, off: the think-end marker) */
static void ds4_chat_turn(pulsar_engine *e, pulsar_tokens *tokens, const pulsar_chat_message *msgs, int n,
                          bool generation_prompt, pulsar_think_mode think_mode) {
    for (int i = 0; i < n; i++) ds4_chat_append_message(e, tokens, msgs[i].role, msgs[i].content, msgs[i].trusted);
    if (!generation_prompt) return;
    chat_tmpl_push_marker(tokens, e->vocab.assistant_id);
    chat_tmpl_push_marker(tokens, pulsar_think_mode_enabled(think_mode) ?
                          e->vocab.think_start_id : e->vocab.think_end_id);
}

/* the sampled turn's stop token is the template's turn end */
/// pulsar_family_tokenizer::chat_end_assistant for DeepSeek.
/// @param e      the engine
/// @param tokens the tokens appended to (EOS)
static void ds4_chat_end_assistant(pulsar_engine *e, pulsar_tokens *tokens) {
    chat_tmpl_push_marker(tokens, e->vocab.eos_id);
}



/* Print a decoded piece as a quoted string: printable ASCII and well-formed
 * UTF-8 verbatim, \n \r \t \" \\ as escapes, every other byte as \xNN. */
void pulsar_dump_piece_quoted(FILE *fp, const char *s, size_t n) {
    fputc('"', fp);
    for (size_t i = 0; i < n;) {
        const unsigned char c = (unsigned char)s[i];
        const int seq = utf8_seq_ok((const unsigned char *)s + i, n - i);
        if (seq > 1) {
            fwrite(s + i, 1, (size_t)seq, fp);
            i += (size_t)seq;
            continue;
        }
        switch (c) {
        case '\n': fputs("\\n", fp); break;
        case '\r': fputs("\\r", fp); break;
        case '\t': fputs("\\t", fp); break;
        case '"':  fputs("\\\"", fp); break;
        case '\\': fputs("\\\\", fp); break;
        default:
            if (c >= 0x20 && c < 0x7f) fputc(c, fp);
            else fprintf(fp, "\\x%02x", c);
        }
        i++;
    }
    fputc('"', fp);
}



/* --dump-tokens: the ids, then one line per token with the DECODED bytes (what
 * pulsar_token_text hands the server and the CLI) followed by the raw GPT-2
 * byte-encoded vocab string, so a tokenization mismatch can be read against
 * either form. */
/// @param fp     the output stream
/// @param vocab  the vocab
/// @param tokens the tokens (ids outside the vocab are listed but get no line)
void dump_tokens_fp(FILE *fp, const pulsar_vocab *vocab, const token_vec *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) {
        if (i) fprintf(fp, ", ");
        fprintf(fp, "%d", tokens->v[i]);
    }
    fprintf(fp, "]\n");

    for (int i = 0; i < tokens->len; i++) {
        int id = tokens->v[i];
        if (id >= 0 && id < vocab->n_vocab) {
            size_t n = 0;
            char *piece = vocab_token_text(vocab, id, &n);
            fprintf(fp, "%6d  ", id);
            pulsar_dump_piece_quoted(fp, piece, n);
            fprintf(fp, "  raw=%.*s\n", (int)vocab->token[id].len, vocab->token[id].ptr);
            free(piece);
        }
    }
}








/** The inverse of gpt2_byte_to_codepoint.  @param cp the codepoint  @return the byte, -1 when cp is no byte's */
static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }

    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
            continue;
        }
        if (cp == 256 + n) return (int)b;
        n++;
    }
    return -1;
}



/** Is the vocab string a literal special token (it contains U+FF5C, the fullwidth bar DeepSeek's markers use), whose
 *  text is its own bytes rather than GPT-2 byte-encoded?  @param s the vocab string  @return true when it is */
static bool vocab_token_is_literal_special(pulsar_str s) {
    const unsigned char bar[] = {0xef, 0xbd, 0x9c}; /* U+FF5C fullwidth vertical bar. */
    if (s.len < sizeof(bar)) return false;
    for (uint64_t i = 0; i + sizeof(bar) <= s.len; i++) {
        if (!memcmp(s.ptr + i, bar, sizeof(bar))) return true;
    }
    return false;
}



/** A token's text through the family's tokenizer (exits without one).
 *  @param e     the engine
 *  @param token the token id
 *  @param len   receives the length in bytes (0 bytes = a control token)
 *  @return a malloc'd NUL-terminated copy (the caller frees it) */
char *pulsar_token_text(pulsar_engine *e, int token, size_t *len) {
    return tokenizer_require(e, "pulsar_token_text")->token_text(e, token, len);
}



char *vocab_token_text(const pulsar_vocab *vocab, int token, size_t *len) {
    if (token < 0 || token >= vocab->n_vocab) {
        if (len) *len = 0;
        char *out = (char *)xmalloc(1);
        out[0] = '\0';
        return out;
    }

    pulsar_str s = vocab->token[token];
    char *out = (char *)xmalloc((size_t)s.len + 1);
    if (vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }

    size_t n = 0;
    uint64_t pos = 0;
    while (pos < s.len) {
        uint32_t cp = utf8_decode_at(s.ptr, s.len, pos, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}



/** The family's end-of-sequence id (exits without a tokenizer).  @param e the engine  @return the id */
int pulsar_token_eos(pulsar_engine *e) {
    return tokenizer_require(e, "pulsar_token_eos")->eos(e);
}



/* L272 B5: the turn markers the server's prefix anchors scan for.  DeepSeek names a turn with one
 * special token; Qwen's template spells it as <|im_start|> plus the role word, which the checkpoint's
 * BPE holds as one token (Qwen3.8-Flash-Next: "user", "assistant").  A tokenizer that splits either
 * word leaves the anchors off, said once -- never a marker that matches at the wrong place. */
/// @param e   the engine
/// @param out receives the markers (zeroed first)
/// @return false when the family's tokenizer cannot name both markers (the anchors stay off)
bool pulsar_chat_turn_markers(pulsar_engine *e, pulsar_turn_markers *out) {
    const pulsar_family_tokenizer *t = tokenizer_require(e, "pulsar_chat_turn_markers");
    memset(out, 0, sizeof(*out));
    return t->turn_markers(e, out);
}

int pulsar_turn_marker_at(const pulsar_turn_markers *m, const int *v, int n, int i) {
    if (!m || !v || i < 0) return 0;
    if (m->n_user > 0 && i + m->n_user <= n && v[i] == m->user[0] &&
        (m->n_user < 2 || v[i + 1] == m->user[1]))
        return 1;
    if (m->n_assistant > 0 && i + m->n_assistant <= n && v[i] == m->assistant[0] &&
        (m->n_assistant < 2 || v[i + 1] == m->assistant[1]))
        return 2;
    return 0;
}

int pulsar_token_think_close(pulsar_engine *e) {
    return tokenizer_require(e, "pulsar_token_think_close")->think_close(e);
}



/* The request's sampler stream (L288 sdet): a Weyl state.  The KEY of the token `ahead` places past the stream's
 * next undrawn one is mix(state + (ahead + 1) * STEP) -- a function of (seed, token index) only, so a lane reads a
 * token's key without consuming anything, and a token's draw is the same whichever lane makes it, whatever was
 * drafted around it (session_spec.cpp's walk is why that matters).  Every committed token advances the state one
 * STEP (pulsar_sample_advance).
 *
 * STEP is deliberately NOT SplitMix64's golden-ratio gamma: two seeds that differ by k steps share their streams
 * shifted by k tokens, and seeds spaced by that gamma are a common idiom -- the spec-sampling oracle seeds its
 * trajectories gamma * (t + 1), which made trajectory t's token i + 1 draw at trajectory t + 1's token i key and
 * inflated its chi-squares with correlated samples.  An arbitrary odd constant puts any seed family nobody builds
 * from it ~2^63 steps apart. */
/** The sampler stream's Weyl step (see above). */
#define SAMPLE_STREAM_STEP 0xa0761d6478bd642fULL

/** SplitMix64's output mix.  @param z the state  @return the mixed value */
static uint64_t sample_mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

uint64_t pulsar_sample_key(uint64_t rng, uint32_t ahead) {
    return sample_mix64(rng + ((uint64_t)ahead + 1u) * SAMPLE_STREAM_STEP);
}

void pulsar_sample_advance(uint64_t *rng, uint32_t n) {
    *rng += (uint64_t)n * SAMPLE_STREAM_STEP;
}



/* =========================================================================
 * Engine API and Process Lock.
 * =========================================================================
 *
 * The public entry points acquire the single instance lock, open the GGUF with
 * the backend-appropriate mmap policy, and expose tokenized prompt operations
 * to the CLI and server.
 */

/** The backend's name for messages.  @param backend the backend  @return "cuda", or "unknown" */
const char *pulsar_backend_name(pulsar_backend backend) {
    switch (backend) {
    case PULSAR_BACKEND_CUDA:  return "cuda";
    }
    return "unknown";
}



/* ---- the DeepSeek family's tokenizer table (family.h pulsar_family_tokenizer, L272 P2) ---------------- */

/** pulsar_family_tokenizer::encode_text for DeepSeek: plain BPE.  @param e the engine  @param text the text
 *  (NULL = "")  @param out receives the tokens */
static void ds4_tok_encode_text(pulsar_engine *e, const char *text, pulsar_tokens *out) {
    e->vocab.bpe_tokenize_text(text ? text : "", out);
}

/// pulsar_family_tokenizer::encode_rendered for DeepSeek.
/// @param e       the engine
/// @param text    the rendered text
/// @param spans   [n_spans] the trusted spans, or NULL (the whole text is rendered chat)
/// @param n_spans spans
/// @param out     receives the tokens
static void ds4_tok_encode_rendered(pulsar_engine *e, const char *text, const pulsar_text_span *spans,
                                    uint32_t n_spans, pulsar_tokens *out) {
    if (!spans || !n_spans) e->vocab.tokenize_rendered_chat_vocab(text, out);
    else e->vocab.tokenize_rendered_chat_spans_vocab(text, spans, n_spans, out);
}

/** pulsar_family_tokenizer::is_stop: EOS only.  @param e the engine  @param token the id  @return true for EOS */
static bool ds4_tok_is_stop(pulsar_engine *e, int token) { return token == e->vocab.eos_id; }
/** pulsar_family_tokenizer::eos.  @param e the engine  @return the EOS id */
static int ds4_tok_eos(pulsar_engine *e) { return e->vocab.eos_id; }
/** pulsar_family_tokenizer::token_text (vocab_token_text).  @param e the engine  @param token the id
 *  @param len receives the length  @return the malloc'd text */
static char *ds4_tok_token_text(pulsar_engine *e, int token, size_t *len) { return vocab_token_text(&e->vocab, token, len); }
/** pulsar_family_tokenizer::think_close.  @param e the engine  @return the think-end marker id */
static int ds4_tok_think_close(pulsar_engine *e) { return e->vocab.think_end_id; }

/* DeepSeek names a turn with one special token */
/** pulsar_family_tokenizer::turn_markers: one special token each.  @param e the engine
 *  @param out receives the markers  @return true, always */
static bool ds4_tok_turn_markers(pulsar_engine *e, pulsar_turn_markers *out) {
    out->user[0] = e->vocab.user_id;
    out->n_user = 1;
    out->assistant[0] = e->vocab.assistant_id;
    out->n_assistant = 1;
    return true;
}

/** pulsar_family_tokenizer::dump (dump_tokens_fp).  @param e the engine  @param fp the stream
 *  @param tokens the tokens */
static void ds4_tok_dump(pulsar_engine *e, FILE *fp, const pulsar_tokens *tokens) { dump_tokens_fp(fp, &e->vocab, tokens); }

/** pulsar_family_tokenizer::n_tokens.  @param e the engine  @return the vocab size */
static int ds4_tok_n_tokens(pulsar_engine *e) { return e->vocab.n_vocab; }

const pulsar_family_tokenizer k_ds4_tokenizer = {
    /* .encode_text              = */ ds4_tok_encode_text,
    /* .encode_rendered          = */ ds4_tok_encode_rendered,
    /* .chat_open                = */ ds4_chat_open,
    /* .chat_turn                = */ ds4_chat_turn,
    /* .chat_end_assistant       = */ ds4_chat_end_assistant,
    /* .is_stop                  = */ ds4_tok_is_stop,
    /* .eos                      = */ ds4_tok_eos,
    /* .token_text               = */ ds4_tok_token_text,
    /* .think_close              = */ ds4_tok_think_close,
    /* .turn_markers             = */ ds4_tok_turn_markers,
    /* .dump                     = */ ds4_tok_dump,
    /* .n_tokens                 = */ ds4_tok_n_tokens,
};
