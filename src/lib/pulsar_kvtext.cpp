/* The rendered-text and small format helpers pulsar-server and pulsar-agent
 * share (pulsar_kvtext.h). */
#include "pulsar_kvtext.h"

#include "sha1.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace pulsar {
namespace {

[[noreturn]] void kv_die(const char *msg) {
    fprintf(stderr, "pulsar-kvtext: %s\n", msg);
    exit(1);
}

void *kv_xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) kv_die("out of memory");
    return p;
}

char *kv_xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = static_cast<char *>(kv_xrealloc(NULL, n + 1));
    memcpy(p, s, n + 1);
    return p;
}

/* Growable NUL-terminated byte builder. Deliberately malloc-backed, not
 * std::string: take() hands the buffer to C callers that free() it. */
class KvBuf {
public:
    KvBuf() = default;
    KvBuf(const KvBuf &) = delete;
    KvBuf &operator=(const KvBuf &) = delete;
    ~KvBuf() { free(ptr_); }

    size_t len() const { return len_; }
    const char *data() const { return ptr_; }

    void append(const void *p, size_t n) {
        reserve(n);
        memcpy(ptr_ + len_, p, n);
        len_ += n;
        ptr_[len_] = '\0';
    }

    void putc(char c) { append(&c, 1); }
    void puts(const char *s) { append(s, strlen(s)); }

    /* Transfer ownership to the caller (who free()s it). */
    char *take() {
        if (!ptr_) return kv_xstrdup("");
        char *p = ptr_;
        ptr_ = NULL;
        len_ = 0;
        cap_ = 0;
        return p;
    }

private:
    void reserve(size_t add) {
        if (add > SIZE_MAX - len_ - 1) kv_die("buffer overflow");
        size_t need = len_ + add + 1;
        if (need <= cap_) return;
        size_t cap = cap_ ? cap_ * 2 : 256;
        while (cap < need) cap *= 2;
        ptr_ = static_cast<char *>(kv_xrealloc(ptr_, cap));
        cap_ = cap;
    }

    char *ptr_ = nullptr;
    size_t len_ = 0;
    size_t cap_ = 0;
};

} // namespace
} // namespace pulsar

void pulsar_kvtext_le_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

uint32_t pulsar_kvtext_le_get32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

void pulsar_kvtext_sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    pulsar::Sha1::bytes_hex(ptr, len, out);
}

char *pulsar_kvtext_path_join(const char *dir, const char *name) {
    pulsar::KvBuf b;
    b.puts(dir);
    if (b.len() == 0 || b.data()[b.len() - 1] != '/') b.putc('/');
    b.puts(name);
    return b.take();
}

char *pulsar_kvtext_render_tokens_text(pulsar_engine *engine,
                                     const pulsar_tokens *tokens,
                                     size_t *out_len) {
    return pulsar_history_text(engine, tokens, out_len);
}

bool pulsar_kvtext_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len) {
    return prefix_len <= text_len &&
           (prefix_len == 0 || memcmp(text, prefix, prefix_len) == 0);
}

void pulsar_kvtext_tokens_copy_prefix(pulsar_tokens *dst, const pulsar_tokens *src, int n) {
    dst->len = 0;
    if (!src) return;
    if (n > src->len) n = src->len;
    for (int i = 0; i < n; i++) pulsar_tokens_push(dst, src->v[i]);
}

bool pulsar_kvtext_text_ends_with_live(pulsar_engine *engine, const char *text,
                                     size_t text_len, const pulsar_tokens *tokens) {
    const int eos = pulsar_token_eos(engine);
    size_t eos_len = 0;
    char *eos_text = pulsar_token_text(engine, eos, &eos_len);
    const bool text_eos = eos_len > 0 && text_len >= eos_len &&
                          memcmp(text + text_len - eos_len, eos_text, eos_len) == 0;
    free(eos_text);
    const bool live_eos = tokens && tokens->len > 0 && tokens->v[tokens->len - 1] == eos;
    return text_eos == live_eos;
}

void pulsar_kvtext_build_prompt_from_exact_prefix_and_text_suffix(
        pulsar_engine *engine,
        const pulsar_tokens *exact_prefix,
        const char *suffix_text,
        const pulsar_text_span *spans,
        uint32_t n_spans,
        pulsar_tokens *out) {
    pulsar_tokens_copy(out, exact_prefix);

    pulsar_tokens suffix = {};
    /* The suffix may start with DS4 chat markers such as <｜User｜> or
     * </think>, so use the rendered-chat tokenizer, not plain text BPE -- but
     * its CLIENT-DATA ranges stay plain text (L223). */
    pulsar_tokenize_rendered_chat_spans(engine, suffix_text ? suffix_text : "",
                                        spans, n_spans, &suffix);
    for (int i = 0; i < suffix.len; i++) pulsar_tokens_push(out, suffix.v[i]);
    pulsar_tokens_free(&suffix);
}
