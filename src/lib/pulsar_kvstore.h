#ifndef PULSAR_KVSTORE_H
#define PULSAR_KVSTORE_H

/* The agent's session-file format, plus the rendered-text helpers the agent
 * and the server share.
 *
 * A session file ("KVC" envelope) is: a 48-byte fixed header, the rendered
 * text byte count + rendered text, the engine payload written by
 * pulsar_session_save_payload() (absent in a "stripped" file), and optional
 * agent trailers flagged in the header's ext_flags.  pulsar-agent writes and
 * reads these for its explicit sessions and system-prompt checkpoints; the
 * server's disk KV cache lives on the segment store (pulsar_segstore.h) and
 * uses only the text/token helpers and the small format primitives here. */

#include "pulsar.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>


#define PULSAR_KVSTORE_FIXED_HEADER 48u

/* ext_flags bits.  Bits 0..2 were the retired server cache's trailers (tool
 * map, responses-visible and thinking-visible keys); old files may still carry
 * them, so they are not reused. */
#define PULSAR_KVSTORE_EXT_SESSION_TITLE     (1u << 3)
/** The AGENT writes its session/sysprompt files with an extra trailer holding
 * the exact token ids the rendered text renders.  It is what lets a
 * payload-less ("stripped") file be restored WITHOUT re-tokenising its text,
 * which cannot tell a control token from its literal spelling (L223) and may
 * re-merge BPE across boundaries. */
#define PULSAR_KVSTORE_EXT_AGENT_TOKENS      (1u << 4)

/** Header byte 5: why the file was written.  The values are on disk; codes
 * 1..4 and 7 belonged to the retired server cache and read back as UNKNOWN. */
typedef enum {
    PULSAR_KVSTORE_REASON_UNKNOWN       = 0,
    PULSAR_KVSTORE_REASON_AGENT_SYSTEM  = 5,
    PULSAR_KVSTORE_REASON_AGENT_SESSION = 6,
} pulsar_kvstore_reason;

/** One session file's header, as read back: what it holds and where it lives. */
typedef struct {
    /** The file name is the hash of the rendered byte prefix, not of the token
     * sequence; the payload still carries the exact tokens and graph state. */
    char sha[41];        ///< hash of the rendered byte prefix: 40 hex chars + NUL
    char *path;          ///< backing file, owned
    uint8_t quant_bits;  ///< routed-format discriminator the payload was written with
    /** Stored in header byte 7.  Flash is 0 for backward compatibility with
     * older files where this reserved byte was always written as zero. */
    uint8_t model_id;
    uint8_t reason;          ///< ::pulsar_kvstore_reason this file was written for
    uint32_t tokens;         ///< prompt tokens the file covers
    uint32_t hits;           ///< restore count carried in the header
    uint32_t ctx_size;       ///< context size the payload was captured at
    uint8_t ext_flags;       ///< which optional trailers the file carries
    uint64_t created_at;     ///< unix seconds the file was written
    uint64_t last_used;      ///< unix seconds of the last write/restore
    uint64_t payload_bytes;  ///< bytes of graph/KV payload (0 = stripped)
    uint64_t text_bytes;     ///< bytes of stored prefix text
    uint64_t file_size;      ///< total file size
} pulsar_kvstore_entry;

/** The reason code for "agent-system" / "agent-session"; UNKNOWN otherwise. */
uint8_t pulsar_kvstore_reason_code(const char *reason);
void pulsar_kvstore_entry_free(pulsar_kvstore_entry *e);

char *pulsar_kvstore_render_tokens_text(pulsar_engine *engine,
                                     const pulsar_tokens *tokens,
                                     size_t *out_len);
bool pulsar_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len);
/** L196: does `text` end where `tokens` end?  A checkpoint's text names the
 * bytes of its tokens, so it ends with the EOS piece exactly when the last
 * token is the EOS.  A tool-call turn stops at the closing tool_calls tag
 * without sampling one; a key that carried the EOS anyway made every consumer
 * tokenise the request bytes AFTER it, so the EOS never entered the bank and
 * the live history byte-diverged from every later render at that point.
 * Store and load both refuse a disagreeing pair. */
bool pulsar_kvstore_text_ends_with_live(pulsar_engine *engine, const char *text,
                                     size_t text_len, const pulsar_tokens *tokens);
void pulsar_kvstore_tokens_copy_prefix(pulsar_tokens *dst, const pulsar_tokens *src, int n);
/** `exact_prefix`'s tokens, then `suffix_text` tokenised as RENDERED chat --
 * except the bytes inside `spans`/`n_spans` (the suffix's CLIENT-DATA ranges),
 * which are plain text (L223), so a client spelling in the suffix cannot become
 * a control token.  `spans` may be NULL when the suffix is entirely the
 * server's own text. */
void pulsar_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        pulsar_engine *engine,
        const pulsar_tokens *exact_prefix,
        const char *suffix_text,
        const pulsar_text_span *spans,
        uint32_t n_spans,
        pulsar_tokens *out);

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
bool pulsar_kvstore_sha_hex_name(const char *name, char sha[41]);
void pulsar_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
char *pulsar_kvstore_path_join(const char *dir, const char *name);
void pulsar_kvstore_le_put32(uint8_t *p, uint32_t v);
uint32_t pulsar_kvstore_le_get32(const uint8_t *p);
/** The accepted values of the routed-format discriminator in a session-file
 *  header (pulsar_engine_routed_quant_bits): 2 = the IQ2 tier, 4 = the MXFP4
 *  tier, 20 + k2 = an EXL3 rate in half-bit units (24, 25, 26).  The header
 *  reader refuses anything else. */
bool pulsar_kvstore_quant_bits_valid(int quant_bits);


#endif
