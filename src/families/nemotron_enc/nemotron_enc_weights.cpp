/* nemotron_enc_weights.cpp -- the Nemotron-3.5 encoders' weights and tokenizer (nemotron_enc.h section 2).
 *
 * Every tensor the shape declares, bound by its bind record through the core's role binder (src/engine/role_bind.h),
 * its format admitted through the core's front doors (weight_format.cpp), its bytes priced.  The forward hands its
 * linears raw bf16 rows (pulsar_act_kind ROWS) on the PROMPT arms only (an encoder has no decode), so every linear --
 * q / k / v / o, gate / up / down -- takes what the registry serves a dense linear at bf16 rows (an EXL3 dense-arm
 * rate, BF16; a device without the format refuses at open, L321 S0c).  Norms and the embedding are read as stored,
 * bf16; the Reranker's score head too, widened here to f32 once for the host's dot. */
/** @file nemotron_enc_weights.cpp
 *  @brief The Nemotron-3.5 encoders' weights bound by role, the score head widened, and the tokenizer load. */
#include "nemotron_enc.h"
#include "lib/bpe_tokenizer.h"

#include <string.h>
#include <sys/stat.h>

#include <string>
#include <vector>

namespace {

/// The binder's owner name and every refusal's prefix.
const char *const OWNER = PULSAR_NEMOTRON_ENC_ARCH;
/// The trunk layers' block name in role keys.
const char *const LAYERS = "layers";

/// Read as stored: bf16 only (norms, the embedding, the score head).
const pulsar_role_admit A_BF16 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_BF16), false, PULSAR_ROLE_DENSE, 0};
/// A dense linear: whatever the registry serves at bf16 rows.
const pulsar_role_admit A_LINEAR = {0, true, PULSAR_ROLE_DENSE, PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};

/** Read a whole file.
 *  @param path  the file
 *  @param out   receives its bytes (cleared first)
 *  @return false when the file cannot be opened or a read fails */
bool read_text(const std::string &path, std::string *out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[1 << 16];
    size_t n;
    out->clear();
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
    const bool ok = !ferror(f);
    fclose(f);
    return ok;
}

/** Tokenize raw text with no client spans (the tokenizer's self-check).
 *  @param t     the tokenizer
 *  @param text  NUL-terminated UTF-8
 *  @param ids   receives the ids (cleared first)
 *  @return the tokenizer's verdict (false = refused) */
bool encode(const bpe_tokenizer *t, const char *text, std::vector<int> *ids) {
    char err[128];
    ids->clear();
    return bpe_tokenizer_encode(t, text, strlen(text), NULL, 0, ids, err, sizeof err);
}

}  // namespace

/** @param m  the container
 *  @param w  the family state: its shape is read; the tensors, the census and (Rerank) score_f32 are written
 *  @return true = bound; false = refused (said) */
bool pulsar_nemotron_enc_bind(const pulsar_model *m, pulsar_nemotron_enc_model *w) {
    const pulsar_nemotron_enc_shape *s = &w->shape;
    pulsar_role_binder *b = pulsar_role_binder_open(m, OWNER);
    const uint64_t E = s->n_embd, V = s->n_vocab, F = s->n_ff;
    const uint64_t Q = (uint64_t)s->n_head * s->head_dim, KV = (uint64_t)s->n_head_kv * s->head_dim;
    w->embed = pulsar_role_bind(b, pulsar_role_top("embed"), A_BF16, 2, E, V);
    w->output_norm = pulsar_role_bind(b, pulsar_role_top("output_norm"), A_BF16, 1, E);
    if (s->kind == PULSAR_ENCODER_RERANK) w->score = pulsar_role_bind(b, pulsar_role_top("score"), A_BF16, 2, E, 1);
    for (uint32_t il = 0; il < s->n_layer; il++) {
        auto k = [&](const char *role) { return pulsar_role_in(role, LAYERS, (int32_t)il); };
        pulsar_nemotron_enc_layer *L = &w->layer[il];
        L->attn_norm = pulsar_role_bind(b, k("attn_norm"), A_BF16, 1, E);
        L->attn_q = pulsar_role_bind(b, k("attn_q"), A_LINEAR, 2, E, Q);
        L->attn_k = pulsar_role_bind(b, k("attn_k"), A_LINEAR, 2, E, KV);
        L->attn_v = pulsar_role_bind(b, k("attn_v"), A_LINEAR, 2, E, KV);
        L->attn_o = pulsar_role_bind(b, k("attn_o"), A_LINEAR, 2, Q, E);
        L->ffn_norm = pulsar_role_bind(b, k("ffn_norm"), A_BF16, 1, E);
        L->ffn_gate = pulsar_role_bind(b, k("ffn_gate"), A_LINEAR, 2, E, F);
        L->ffn_up = pulsar_role_bind(b, k("ffn_up"), A_LINEAR, 2, E, F);
        L->ffn_down = pulsar_role_bind(b, k("ffn_down"), A_LINEAR, 2, F, E);
    }
    /* the census: every directory entry declared, every byte priced (an extra -- an embedder with a score head, a
     * reranker without one -- refused by name) */
    if (!pulsar_role_binder_close(b, &w->census)) return false;
    if (w->score) {
        const uint16_t *h = (const uint16_t *)tensor_data(m, w->score);
        w->score_f32 = (float *)xmalloc(E * sizeof(float));
        for (uint64_t i = 0; i < E; i++) {
            const uint32_t u = (uint32_t)h[i] << 16;
            memcpy(&w->score_f32[i], &u, 4);
        }
    }
    return true;
}

/** @param w           the family state: its shape is read; tok is written
 *  @param model_path  the container's directory, or a shard path inside it
 *  @return true = loaded and checked (tekken, the U+200C normaliser, n_vocab ids); false = refused (said) */
bool pulsar_nemotron_enc_load_tokenizer(pulsar_nemotron_enc_model *w, const char *model_path) {
    std::string dir = model_path ? model_path : "";
    struct stat st;
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        const size_t slash = dir.find_last_of('/');
        dir = slash == std::string::npos ? "." : dir.substr(0, slash);
    }
    std::string tok;
    const std::string tp = dir + "/tokenizer.json";
    if (!read_text(tp, &tok)) {
        fprintf(stderr, "pulsar: %s: the container carries no %s -- the builder (tools/container/nemotron_enc.py) "
                        "writes the checkpoint's tokenizer.json beside the shards; refusing\n", OWNER, tp.c_str());
        return false;
    }
    /* the checkpoints ship no generation_config.json: the loader's one stop id is the config's eos (an encoder never
     * stops; the container's nemotron_enc.eos_token_id is text_config's, pinned by the shape) */
    char cfg[64], err[256] = "";
    snprintf(cfg, sizeof cfg, "{\"eos_token_id\": %u}", w->shape.eos_id);
    w->tok = bpe_tokenizer_load_hf(tok.data(), tok.size(), cfg, strlen(cfg), err, sizeof(err));
    if (!w->tok) {
        fprintf(stderr, "pulsar: %s: tokenizer refused: %s\n", OWNER, err);
        return false;
    }
    if (strcmp(bpe_tokenizer_pattern(w->tok)->name, "tekken") != 0) {
        fprintf(stderr, "pulsar: %s: the tokenizer's pre-tokenizer is \"%s\", not this family's \"tekken\" -- refusing\n",
                OWNER, bpe_tokenizer_pattern(w->tok)->name);
        return false;
    }
    /* the encoders' row of tekken -- the one whose normaliser drops the template's U+200C after '[' / '<' -- by what it
     * does: without it every client '[' would carry a stray token */
    std::vector<int> marked, plain;
    if (!encode(w->tok, "[\xe2\x80\x8c<\xe2\x80\x8c", &marked) || !encode(w->tok, "[<", &plain) || marked != plain) {
        fprintf(stderr, "pulsar: %s: the tokenizer does not drop the template's U+200C after '[' / '<' (not the "
                        "encoders' normaliser) -- refusing\n", OWNER);
        return false;
    }
    if ((uint32_t)bpe_tokenizer_n_tokens(w->tok) != w->shape.n_vocab) {
        fprintf(stderr, "pulsar: %s: the tokenizer's %d ids are not the embedding's %u rows -- refusing\n", OWNER,
                bpe_tokenizer_n_tokens(w->tok), w->shape.n_vocab);
        return false;
    }
    fprintf(stderr, "pulsar: %s: tokenizer %d ids (HF BPE, tekken + the U+200C Replace pair) from %s\n", OWNER,
            bpe_tokenizer_n_tokens(w->tok), tp.c_str());
    return true;
}
