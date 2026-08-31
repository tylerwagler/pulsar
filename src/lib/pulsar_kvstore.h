#ifndef PULSAR_KVSTORE_H
#define PULSAR_KVSTORE_H

#include "pulsar.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>


#define PULSAR_KVSTORE_FIXED_HEADER 48u
/* 64 GiB: one full-context Claude Code snapshot is ~3 GiB, so the old 4 GiB
 * default held ONE conversation and every new one evicted the last
 * (disk-cache-full churn, observed 2026-08); checkpoints are regenerable
 * cache on a store with hundreds of GB free, so size for ~20 warm
 * conversations, not for scarcity. */
#define PULSAR_KVSTORE_DEFAULT_MB 65536
#define PULSAR_KVSTORE_HIT_HALF_LIFE_SECONDS (6ull * 60ull * 60ull)

#define PULSAR_KVSTORE_EXT_TOOL_MAP          (1u << 0)
#define PULSAR_KVSTORE_EXT_RESPONSES_VISIBLE (1u << 1)
#define PULSAR_KVSTORE_EXT_THINKING_VISIBLE  (1u << 2)
#define PULSAR_KVSTORE_EXT_SESSION_TITLE     (1u << 3)

typedef enum {
    PULSAR_KVSTORE_REASON_UNKNOWN   = 0,
    PULSAR_KVSTORE_REASON_COLD      = 1,
    PULSAR_KVSTORE_REASON_CONTINUED = 2,
    PULSAR_KVSTORE_REASON_EVICT     = 3,
    PULSAR_KVSTORE_REASON_SHUTDOWN  = 4,
    PULSAR_KVSTORE_REASON_AGENT_SYSTEM  = 5,
    PULSAR_KVSTORE_REASON_AGENT_SESSION = 6,
    /* Truncated shared-preamble checkpoint (system prompt + tools, cut before
     * the first task message): the one snapshot every NEW conversation with
     * the same preamble can text-prefix restore from. Same concept as
     * AGENT_SYSTEM but written by the server's cold-anchor path. */
    PULSAR_KVSTORE_REASON_SYS_PREFIX    = 7,
} pulsar_kvstore_reason;

/** Category of a message handed to the store's log callback, so the host can
 * route cache chatter separately from real warnings. */
typedef enum {
    PULSAR_KVSTORE_LOG_DEFAULT,  ///< ordinary informational message
    PULSAR_KVSTORE_LOG_KVCACHE,  ///< cache hit/miss/store activity
    PULSAR_KVSTORE_LOG_WARNING,  ///< something went wrong but was survivable
} pulsar_kvstore_log_type;

typedef struct {
    /** The file name is the rendered byte prefix, not the token sequence. The
     * payload still carries the exact tokens and graph state; the hash only
     * answers "does this checkpoint represent the bytes at the front of the
     * incoming prompt?" */
    char sha[41];        ///< hash of the rendered byte prefix: 40 hex chars + NUL
    char *path;          ///< backing file, owned
    uint8_t quant_bits;  ///< KV quantisation the payload was written with
    /** Stored in header byte 7.  Flash is 0 for backward compatibility with
     * older cache files where this reserved byte was always written as zero. */
    uint8_t model_id;
    uint8_t reason;          ///< ::pulsar_kvstore_reason this checkpoint was written for
    uint32_t tokens;         ///< prompt tokens the checkpoint covers
    uint32_t hits;           ///< times this entry has been restored; the eviction recency signal
    uint32_t ctx_size;       ///< context size the payload was captured at
    uint8_t ext_flags;       ///< which optional trailers the file carries
    uint64_t created_at;     ///< unix seconds the entry was written
    uint64_t last_used;      ///< unix seconds of the last restore
    uint64_t payload_bytes;  ///< bytes of graph/KV payload
    uint64_t text_bytes;     ///< bytes of stored prefix text
    uint64_t file_size;      ///< total file size, the unit the budget is spent in
} pulsar_kvstore_entry;

/** Store-placement policy: WHERE a checkpoint is cut, and how often.
 *
 * The cut position matters more than it looks. A checkpoint that ends a few
 * tokens into volatile text is useless -- the next request diverges right
 * there and reuses nothing -- so cuts are pulled back off the frontier and
 * landed on an alignment boundary.
 */
typedef struct {
    int min_tokens;                ///< below this many tokens a checkpoint is not worth writing
    int cold_max_tokens;           ///< cap on a cold checkpoint's length
    int continued_interval_tokens; ///< write a continued checkpoint every this many new tokens
    /** Back the cut off the frontier by this many tokens. The last few tokens
     * before the frontier are the least stable part of a prompt; a checkpoint
     * ending inside them rarely matches the next request. */
    int boundary_trim_tokens;
    int boundary_align_tokens;     ///< round the cut down to a multiple of this
    /** Extra margin below the chat anchor for a sys-prefix checkpoint, to clear
     * harness-injected preamble jitter that would otherwise vary the cut. */
    int sys_prefix_margin_tokens;
} pulsar_kvstore_options;

/** An on-disk checkpoint store: a directory, a byte budget, and the index of
 * what is in it. */
typedef struct {
    bool enabled;                  ///< the store is in use; everything is a no-op when false
    char *dir;                     ///< directory holding the files, owned
    uint64_t budget_bytes;         ///< total size ceiling; eviction keeps the store under it
    bool reject_different_quant;   ///< refuse entries whose quant_bits differ from the running engine
    pulsar_kvstore_options opt;    ///< placement policy
    int continued_last_store_tokens;  ///< token count at the last continued write; the interval baseline
    pulsar_kvstore_entry *entry;   ///< the index
    int len;                       ///< entries present
    int cap;                       ///< entries allocated
    const char *log_name;          ///< prefix identifying this store in log lines
    void *log_ud;                  ///< opaque userdata handed back to `log`
    void (*log)(void *ud, pulsar_kvstore_log_type type, const char *msg);  ///< log sink; NULL to discard
} pulsar_kvstore;

/** What the eviction pass must NOT throw away: the entry the caller is about
 * to use. Passed in so a store can make room for an incoming write without
 * evicting the very checkpoint that write depends on. */
typedef struct {
    const char *text;             ///< prefix text of the protected entry
    size_t text_len;              ///< its length in bytes
    uint8_t model_id;             ///< model the caller is running
    uint8_t quant_bits;           ///< KV quantisation the caller is running
    uint32_t ctx_size;            ///< context size the caller is running
    bool reject_different_quant;  ///< treat a quant mismatch as unusable, hence evictable
} pulsar_kvstore_eviction_context;

/** Lets a caller append its own data to a checkpoint file.
 *
 * The store owns the payload; anything ELSE a restore needs -- the server's
 * tool map, for instance -- rides along as a trailer written through these
 * hooks. The store never interprets it, it only reserves the space and calls
 * back at the right offset.
 */
typedef struct {
    void *ud;           ///< opaque userdata passed to every hook
    uint8_t ext_flag;   ///< ext_flags bit set on files carrying this trailer
    /** Report the trailer's serialized size for `text`, so the store can size
     * the file before writing. */
    bool (*serialized_size)(void *ud, const char *text, uint64_t *bytes_out);
    /** Write the trailer at the current position of `fp`. */
    bool (*write)(void *ud, FILE *fp, const char *text, uint64_t *written_bytes);
    /** Read the trailer back. @return entries loaded, 0 for none. */
    int (*load)(void *ud, FILE *fp, const void *wanted);
    const void *load_wanted;  ///< passed as `wanted` to the load hook
} pulsar_kvstore_trailer_hooks;

/** What a restore actually produced, reported back to the caller. */
typedef struct {
    int tokens;            ///< prompt tokens restored; 0 means nothing was loaded
    uint32_t text_bytes;   ///< bytes of prefix text restored
    uint8_t quant_bits;    ///< KV quantisation the file was written with
    uint8_t ext_flags;     ///< trailers the file carried
    double load_ms;        ///< wall-clock spent loading, for the cache-hit metrics
    char *path;            ///< file that satisfied the restore, owned by the caller
} pulsar_kvstore_load_result;

pulsar_kvstore_options pulsar_kvstore_default_options(void);
uint8_t pulsar_kvstore_reason_code(const char *reason);
const char *pulsar_kvstore_key_kind(uint8_t ext_flags);

bool pulsar_kvstore_open(pulsar_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, pulsar_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, pulsar_kvstore_log_type type, const char *msg),
                      void *log_ud);
void pulsar_kvstore_close(pulsar_kvstore *kc);
void pulsar_kvstore_entry_free(pulsar_kvstore_entry *e);

char *pulsar_kvstore_render_tokens_text(pulsar_engine *engine,
                                     const pulsar_tokens *tokens,
                                     size_t *out_len);
bool pulsar_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len);
void pulsar_kvstore_tokens_copy_prefix(pulsar_tokens *dst, const pulsar_tokens *src, int n);
void pulsar_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        pulsar_engine *engine,
        const pulsar_tokens *exact_prefix,
        const char *suffix_text,
        pulsar_tokens *out);

int pulsar_kvstore_store_len(const pulsar_kvstore *kc, int tokens);
int pulsar_kvstore_chat_anchor_pos(const pulsar_kvstore *kc,
                                const pulsar_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id);
int pulsar_kvstore_sys_prefix_cut(const pulsar_kvstore *kc, int anchor);
int pulsar_kvstore_continued_store_target(const pulsar_kvstore *kc, int live_tokens);
void pulsar_kvstore_note_store(pulsar_kvstore *kc, int tokens);
int pulsar_kvstore_suppress_continued_store(pulsar_kvstore *kc, int tokens);
void pulsar_kvstore_restore_suppressed_continued(pulsar_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens);

bool pulsar_kvstore_file_size_fits(const pulsar_kvstore *kc,
                                uint64_t text_bytes,
                                uint64_t payload_bytes,
                                uint64_t trailer_bytes,
                                uint64_t *file_bytes_out,
                                uint64_t *required_bytes_out);
double pulsar_kvstore_entry_eviction_score(const pulsar_kvstore_entry *e,
                                        const pulsar_tokens *live,
                                        uint64_t now,
                                        const pulsar_kvstore_eviction_context *incoming);
void pulsar_kvstore_evict(pulsar_kvstore *kc, const pulsar_tokens *live,
                       uint64_t extra_bytes,
                       const pulsar_kvstore_eviction_context *incoming);
int pulsar_kvstore_find_text_prefix(pulsar_kvstore *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size);

bool pulsar_kvstore_store_live_prefix_text(pulsar_kvstore *kc,
                                        pulsar_engine *engine,
                                        pulsar_session *session,
                                        const pulsar_tokens *tokens,
                                        int store_len,
                                        const char *reason,
                                        const char *cache_text_override,
                                        uint8_t cache_text_ext,
                                        const char *cache_text_key,
                                        const pulsar_kvstore_trailer_hooks *hooks,
                                        char *err,
                                        size_t err_len);
int pulsar_kvstore_try_load_text(pulsar_kvstore *kc,
                              pulsar_engine *engine,
                              pulsar_session *session,
                              const char *prompt_text,
                              pulsar_tokens *effective_prompt,
                              pulsar_kvstore_load_result *result,
                              const pulsar_kvstore_trailer_hooks *hooks,
                              bool responses_protocol);
void pulsar_kvstore_load_result_free(pulsar_kvstore_load_result *result);

bool pulsar_kvstore_read_header(FILE *fp, pulsar_kvstore_entry *e,
                             uint32_t *text_bytes);
bool pulsar_kvstore_read_entry_file(const char *path, const char sha[41],
                                 pulsar_kvstore_entry *out);
void pulsar_kvstore_fill_header(uint8_t h[PULSAR_KVSTORE_FIXED_HEADER],
                             uint8_t model_id, uint8_t quant_bits,
                             uint8_t reason, uint8_t ext_flags,
                             uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes);
bool pulsar_kvstore_touch_file(const char *path, uint32_t hits);
bool pulsar_kvstore_sha_hex_name(const char *name, char sha[41]);
void pulsar_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
char *pulsar_kvstore_path_join(const char *dir, const char *name);
void pulsar_kvstore_le_put32(uint8_t *p, uint32_t v);
uint32_t pulsar_kvstore_le_get32(const uint8_t *p);


#endif
