#include "pulsar_kvstore.h"

#include "sha1.hpp"

/* The agent's session-file format and the rendered-text helpers shared with
 * the server.  See pulsar_kvstore.h for the file layout. */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define KV_CACHE_MAGIC0 'K'
#define KV_CACHE_MAGIC1 'V'
#define KV_CACHE_MAGIC2 'C'
#define KV_CACHE_VERSION 1u
/* Header byte 20 carries the graph-payload ABI.  It is separate from the outer
 * file version because the KVC envelope can remain stable while the serialized
 * pulsar_session internals become unsafe to restore across runtime changes. */
#define KV_CACHE_PAYLOAD_ABI 2u

namespace pulsar {
namespace {

[[noreturn]] void kv_die(const char *msg) {
    fprintf(stderr, "pulsar-kvstore: %s\n", msg);
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

void kv_le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

uint64_t kv_le_get64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

} // namespace
} // namespace pulsar

uint8_t pulsar_kvstore_reason_code(const char *reason) {
    if (!reason) return PULSAR_KVSTORE_REASON_UNKNOWN;
    if (!strcmp(reason, "agent-system")) return PULSAR_KVSTORE_REASON_AGENT_SYSTEM;
    if (!strcmp(reason, "agent-session")) return PULSAR_KVSTORE_REASON_AGENT_SESSION;
    return PULSAR_KVSTORE_REASON_UNKNOWN;
}

void pulsar_kvstore_le_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

uint32_t pulsar_kvstore_le_get32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

void pulsar_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    pulsar::Sha1::bytes_hex(ptr, len, out);
}

bool pulsar_kvstore_sha_hex_name(const char *name, char sha[41]) {
    if (strlen(name) != 43 || strcmp(name + 40, ".kv")) return false;
    for (int i = 0; i < 40; i++) {
        if (!isxdigit((unsigned char)name[i])) return false;
        sha[i] = (char)tolower((unsigned char)name[i]);
    }
    sha[40] = '\0';
    return true;
}

char *pulsar_kvstore_path_join(const char *dir, const char *name) {
    pulsar::KvBuf b;
    b.puts(dir);
    if (b.len() == 0 || b.data()[b.len() - 1] != '/') b.putc('/');
    b.puts(name);
    return b.take();
}

void pulsar_kvstore_entry_free(pulsar_kvstore_entry *e) {
    free(e->path);
    memset(e, 0, sizeof(*e));
}

void pulsar_kvstore_fill_header(uint8_t h[PULSAR_KVSTORE_FIXED_HEADER],
                             uint8_t model_id, uint8_t quant_bits,
                             uint8_t reason, uint8_t ext_flags,
                             uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes) {
    memset(h, 0, PULSAR_KVSTORE_FIXED_HEADER);
    h[0] = KV_CACHE_MAGIC0;
    h[1] = KV_CACHE_MAGIC1;
    h[2] = KV_CACHE_MAGIC2;
    h[3] = KV_CACHE_VERSION;
    h[4] = quant_bits;
    h[5] = reason;
    h[6] = ext_flags;
    h[7] = model_id;
    pulsar_kvstore_le_put32(h + 8, tokens);
    pulsar_kvstore_le_put32(h + 12, hits);
    pulsar_kvstore_le_put32(h + 16, ctx_size);
    h[20] = KV_CACHE_PAYLOAD_ABI;
    pulsar::kv_le_put64(h + 24, created_at);
    pulsar::kv_le_put64(h + 32, last_used);
    pulsar::kv_le_put64(h + 40, payload_bytes);
}

bool pulsar_kvstore_read_header(FILE *fp, pulsar_kvstore_entry *e,
                             uint32_t *text_bytes) {
    uint8_t h[PULSAR_KVSTORE_FIXED_HEADER];
    if (fread(h, 1, sizeof(h), fp) != sizeof(h)) return false;
    if (h[0] != KV_CACHE_MAGIC0 || h[1] != KV_CACHE_MAGIC1 ||
        h[2] != KV_CACHE_MAGIC2 || h[3] != KV_CACHE_VERSION) return false;
    if (h[20] != KV_CACHE_PAYLOAD_ABI) return false;
    e->quant_bits = h[4];
    e->reason = (h[5] == PULSAR_KVSTORE_REASON_AGENT_SYSTEM ||
                 h[5] == PULSAR_KVSTORE_REASON_AGENT_SESSION) ?
                h[5] : (uint8_t)PULSAR_KVSTORE_REASON_UNKNOWN;
    e->ext_flags = h[6];
    e->model_id = h[7];
    e->tokens = pulsar_kvstore_le_get32(h + 8);
    e->hits = pulsar_kvstore_le_get32(h + 12);
    e->ctx_size = pulsar_kvstore_le_get32(h + 16);
    e->created_at = pulsar::kv_le_get64(h + 24);
    e->last_used = pulsar::kv_le_get64(h + 32);
    e->payload_bytes = pulsar::kv_le_get64(h + 40);
    uint8_t tb[4];
    if (fread(tb, 1, sizeof(tb), fp) != sizeof(tb)) return false;
    *text_bytes = pulsar_kvstore_le_get32(tb);
    e->text_bytes = *text_bytes;
    return e->tokens != 0 && pulsar_kvstore_quant_bits_valid(e->quant_bits);
}

bool pulsar_kvstore_quant_bits_valid(int quant_bits) {
    return quant_bits == 2 || quant_bits == 4 ||
           quant_bits == 24 || quant_bits == 25 || quant_bits == 26;
}

bool pulsar_kvstore_read_entry_file(const char *path, const char sha[41],
                                 pulsar_kvstore_entry *out) {
    struct stat st;
    if (stat(path, &st) != 0 ||
        st.st_size < (off_t)(PULSAR_KVSTORE_FIXED_HEADER + 4))
        return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    pulsar_kvstore_entry e = {};
    uint32_t text_bytes = 0;
    bool ok = pulsar_kvstore_read_header(fp, &e, &text_bytes);
    fclose(fp);
    if (!ok) return false;
    const uint64_t fixed = PULSAR_KVSTORE_FIXED_HEADER + 4ull;
    if (UINT64_MAX - fixed < (uint64_t)text_bytes ||
        UINT64_MAX - fixed - (uint64_t)text_bytes < e.payload_bytes)
        return false;
    const uint64_t expected = fixed + (uint64_t)text_bytes + e.payload_bytes;
    if ((uint64_t)st.st_size < expected) return false;
    memcpy(e.sha, sha, 41);
    e.path = pulsar::kv_xstrdup(path);
    e.file_size = (uint64_t)st.st_size;
    *out = e;
    return true;
}

char *pulsar_kvstore_render_tokens_text(pulsar_engine *engine,
                                     const pulsar_tokens *tokens,
                                     size_t *out_len) {
    pulsar::KvBuf b;
    for (int i = 0; i < tokens->len; i++) {
        size_t len = 0;
        char *piece = pulsar_token_text(engine, tokens->v[i], &len);
        b.append(piece, len);
        free(piece);
    }
    if (out_len) *out_len = b.len();
    return b.take();
}

bool pulsar_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len) {
    return prefix_len <= text_len &&
           (prefix_len == 0 || memcmp(text, prefix, prefix_len) == 0);
}

void pulsar_kvstore_tokens_copy_prefix(pulsar_tokens *dst, const pulsar_tokens *src, int n) {
    dst->len = 0;
    if (!src) return;
    if (n > src->len) n = src->len;
    for (int i = 0; i < n; i++) pulsar_tokens_push(dst, src->v[i]);
}

bool pulsar_kvstore_text_ends_with_live(pulsar_engine *engine, const char *text,
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

void pulsar_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
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
