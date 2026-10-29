/* nemotron_parse_weights.cpp -- Parse's weights, bound by role through the core's binder (src/engine/role_bind.h), and
 * its decode table (nemotron_parse.h section 2).
 *
 * The formats: every Linear (the ViT's, the neck's convolutions as Linears, the decoder's) and the tied embedding /
 * head f16 -- the f16 rows arm (pulsar_cuda_f16_rows.h), at f16 operand rows (the decoder's attention output: bf16
 * rows); the embedding gathered from the same f16 table; every norm,
 * bias, the class tokens and the position field f32 as stored (the position field read on the host).
 *
 * The tokenizer: Parse's tokenizer.json is a Nougat / Galactica BPE (NFKC, a digit / punctuation pre-tokenizer, the
 * ByteLevel decoder) that the engine's BPE engine does not implement -- and Parse never needs it to: its prompt is
 * added tokens only, and its output only needs DECODING.  So this reads the two tables decoding needs, the vocabulary
 * and the added tokens, and decodes as tokenizers does (Tokenizer::decode with skip_special_tokens false, the
 * ByteLevel decode_chain, then from_utf8_lossy; tokenizer_config.json's clean_up_tokenization_spaces is false):
 * every token's string -- an added token's content or the vocabulary's -- becomes the bytes its characters stand for
 * in the GPT-2 byte map, or, when one character has no byte (a space, a newline), its UTF-8 as is; the bytes of the
 * whole sequence are then read as UTF-8, each maximal invalid subsequence one U+FFFD. */
#include "nemotron_parse.h"
#include "lib/pyjson.h"

#include <string.h>
#include <sys/stat.h>

namespace {

const char *const OWNER = PULSAR_NEMOTRON_PARSE_ARCH;

const pulsar_role_admit A_F32 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_F32), false, PULSAR_ROLE_DENSE, 0};
const pulsar_role_admit A_F16 = {PULSAR_ROLE_PLAIN(PULSAR_TENSOR_F16), false, PULSAR_ROLE_DENSE, 0};
const pulsar_role_admit A_LINEAR = {0, true, PULSAR_ROLE_DENSE,
                                    PULSAR_ACTS(PULSAR_ACT_ROWS_F16) | PULSAR_ACTS(PULSAR_ACT_ROWS_BF16)};

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

/* GPT-2's byte map, reversed: code point -> byte, -1 for a code point that stands for no byte */
void byte_of_cp(int map[512]) {
    for (int i = 0; i < 512; i++) map[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        const bool kept = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        map[kept ? b : 256 + n++] = b;
    }
}

/* one UTF-8 code point at s[i..], or -1 */
int next_cp(const std::string &s, size_t *i) {
    const unsigned char c = (unsigned char)s[*i];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (!n || *i + n > s.size()) return -1;
    int cp = n == 1 ? c : c & (0x7f >> n);
    for (int k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[*i + k] & 0x3f);
    *i += n;
    return cp;
}

/* a token's string through ByteLevel's decode_chain */
std::string byte_level(const std::string &t, const int map[512]) {
    std::string out;
    for (size_t i = 0; i < t.size();) {
        const int cp = next_cp(t, &i);
        if (cp < 0 || cp >= 512 || map[cp] < 0) return t;
        out.push_back((char)map[cp]);
    }
    return out;
}

}  // namespace

bool pulsar_nemotron_parse_bind(const pulsar_model *m, pulsar_nemotron_parse_model *w) {
    const pulsar_nemotron_parse_shape *s = &w->shape;
    pulsar_role_binder *b = pulsar_role_binder_open(m, OWNER);
    const uint64_t D = s->vit_width, HS = (uint64_t)s->vit_heads * s->vit_head_dim, M = s->vit_mlp;
    const uint64_t N = s->neck_width, E = s->width, F = s->ffn, KV = (uint64_t)s->kv_heads * (s->width / s->heads);
    auto top = [&](const char *r) { return pulsar_role_top(r); };
    const uint64_t P = (uint64_t)s->patch * s->patch * 3u;
    w->patch = pulsar_role_bind(b, top("vit_patch"), A_LINEAR, 2, P, D);
    w->cls = pulsar_role_bind(b, top("vit_cls"), A_F32, 2, D, s->n_cls);
    w->pos = pulsar_role_bind(b, top("vit_pos"), A_F32, 2, D, (uint64_t)s->pos_grid * s->pos_grid);
    for (uint32_t il = 0; il < s->vit_depth; il++) {
        auto k = [&](const char *r) { return pulsar_role_in(r, "vit", (int32_t)il); };
        pulsar_vit_block &B = w->vit[il];
        B.norm1 = pulsar_role_bind(b, k("vit_norm1"), A_F32, 1, D);
        B.norm1_b = pulsar_role_bind(b, k("vit_norm1_bias"), A_F32, 1, D);
        B.qkv = pulsar_role_bind(b, k("vit_qkv"), A_LINEAR, 2, D, 3u * HS);
        B.qkv_b = pulsar_role_bind(b, k("vit_qkv_bias"), A_F32, 1, 3u * HS);
        B.proj = pulsar_role_bind(b, k("vit_proj"), A_LINEAR, 2, HS, D);
        B.proj_b = pulsar_role_bind(b, k("vit_proj_bias"), A_F32, 1, D);
        B.norm2 = pulsar_role_bind(b, k("vit_norm2"), A_F32, 1, D);
        B.norm2_b = pulsar_role_bind(b, k("vit_norm2_bias"), A_F32, 1, D);
        B.fc1 = pulsar_role_bind(b, k("vit_fc1"), A_LINEAR, 2, D, M);
        B.fc1_b = pulsar_role_bind(b, k("vit_fc1_bias"), A_F32, 1, M);
        B.fc2 = pulsar_role_bind(b, k("vit_fc2"), A_LINEAR, 2, M, D);
        B.fc2_b = pulsar_role_bind(b, k("vit_fc2_bias"), A_F32, 1, D);
    }
    w->neck_proj = pulsar_role_bind(b, top("neck_proj"), A_LINEAR, 2, D, N);
    w->neck_proj_b = pulsar_role_bind(b, top("neck_proj_bias"), A_F32, 1, N);
    w->neck_norm1 = pulsar_role_bind(b, top("neck_norm1"), A_F32, 1, N);
    w->neck_norm1_b = pulsar_role_bind(b, top("neck_norm1_bias"), A_F32, 1, N);
    w->neck_merge = pulsar_role_bind(b, top("neck_merge"), A_LINEAR, 2, (uint64_t)s->merge * N, N);
    w->neck_norm2 = pulsar_role_bind(b, top("neck_norm2"), A_F32, 1, N);
    w->neck_norm2_b = pulsar_role_bind(b, top("neck_norm2_bias"), A_F32, 1, N);
    w->summary = pulsar_role_bind(b, top("neck_summary"), A_LINEAR, 2, 2u * D, N);
    w->summary_b = pulsar_role_bind(b, top("neck_summary_bias"), A_F32, 1, N);
    w->neck_norm3 = pulsar_role_bind(b, top("neck_norm3"), A_F32, 1, N);
    w->neck_norm3_b = pulsar_role_bind(b, top("neck_norm3_bias"), A_F32, 1, N);
    w->embed = pulsar_role_bind(b, top("embed"), A_F16, 2, E, s->vocab);
    w->embed_norm = pulsar_role_bind(b, top("embed_norm"), A_F32, 1, E);
    w->embed_norm_b = pulsar_role_bind(b, top("embed_norm_bias"), A_F32, 1, E);
    w->output_norm = pulsar_role_bind(b, top("output_norm"), A_F32, 1, E);
    w->output_norm_b = pulsar_role_bind(b, top("output_norm_bias"), A_F32, 1, E);
    for (uint32_t il = 0; il < s->n_layer; il++) {
        auto k = [&](const char *r) { return pulsar_role_in(r, "layers", (int32_t)il); };
        pulsar_nemotron_parse_dec_layer &L = w->layer[il];
        L.attn_norm = pulsar_role_bind(b, k("attn_norm"), A_F32, 1, E);
        L.attn_norm_b = pulsar_role_bind(b, k("attn_norm_bias"), A_F32, 1, E);
        L.q = pulsar_role_bind(b, k("attn_q"), A_LINEAR, 2, E, E);
        L.q_b = pulsar_role_bind(b, k("attn_q_bias"), A_F32, 1, E);
        L.k = pulsar_role_bind(b, k("attn_k"), A_LINEAR, 2, E, KV);
        L.k_b = pulsar_role_bind(b, k("attn_k_bias"), A_F32, 1, KV);
        L.v = pulsar_role_bind(b, k("attn_v"), A_LINEAR, 2, E, KV);
        L.v_b = pulsar_role_bind(b, k("attn_v_bias"), A_F32, 1, KV);
        L.o = pulsar_role_bind(b, k("attn_o"), A_LINEAR, 2, E, E);
        L.o_b = pulsar_role_bind(b, k("attn_o_bias"), A_F32, 1, E);
        L.ffn_norm = pulsar_role_bind(b, k("ffn_norm"), A_F32, 1, E);
        L.ffn_norm_b = pulsar_role_bind(b, k("ffn_norm_bias"), A_F32, 1, E);
        L.up = pulsar_role_bind(b, k("ffn_up"), A_LINEAR, 2, E, F);
        L.up_b = pulsar_role_bind(b, k("ffn_up_bias"), A_F32, 1, F);
        L.down = pulsar_role_bind(b, k("ffn_down"), A_LINEAR, 2, F, E);
        L.down_b = pulsar_role_bind(b, k("ffn_down_bias"), A_F32, 1, E);
    }
    if (!pulsar_role_binder_close(b, &w->census)) return false;
    /* the position field on the host: the page geometry resamples it (nemotron_parse_image.cpp) */
    const float *pf = (const float *)tensor_data(m, w->pos);
    w->pos_field.assign(pf, pf + (size_t)s->pos_grid * s->pos_grid * D);
    return true;
}

bool pulsar_nemotron_parse_load_vocab(pulsar_nemotron_parse_model *w, const char *model_path) {
    std::string dir = model_path ? model_path : "";
    struct stat st;
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        const size_t slash = dir.find_last_of('/');
        dir = slash == std::string::npos ? "." : dir.substr(0, slash);
    }
    std::string js;
    const std::string tp = dir + "/tokenizer.json";
    pyjson_value doc;
    char err[256] = "";
    if (!read_text(tp, &js) || !pyjson_parse(js.data(), js.size(), &doc, err, sizeof err)) {
        fprintf(stderr, "pulsar: %s: %s is missing or not JSON (%s) -- refusing\n", OWNER, tp.c_str(), err);
        return false;
    }
    const pyjson_value *model = doc.get("model"), *decoder = doc.get("decoder"), *added = doc.get("added_tokens");
    const pyjson_value *vocab = model ? model->get("vocab") : NULL, *type = model ? model->get("type") : NULL,
                       *dtype = decoder ? decoder->get("type") : NULL;
    if (!vocab || vocab->kind != pyjson_value::OBJ || !type || type->s != "BPE" || !dtype || dtype->s != "ByteLevel" ||
        !added || added->kind != pyjson_value::ARR) {
        fprintf(stderr, "pulsar: %s: %s is not a BPE vocabulary with a ByteLevel decoder and added tokens -- the "
                        "decode table this family reads -- refusing\n", OWNER, tp.c_str());
        return false;
    }
    pulsar_nemotron_parse_vocab *v = new pulsar_nemotron_parse_vocab();
    w->vocab = v;   /* unload frees it, whatever this verdict */
    std::vector<std::string> str(w->shape.vocab);
    std::vector<bool> have(w->shape.vocab, false);
    bool ok = true;
    auto put = [&](long long id, const std::string &t) {
        if (id < 0 || id >= (long long)w->shape.vocab || have[(size_t)id]) {
            ok = false;
            return;
        }
        str[(size_t)id] = t;
        have[(size_t)id] = true;
    };
    for (const auto &kv : vocab->o) put(kv.second.kind == pyjson_value::INT ? atoll(kv.second.num.c_str()) : -1, kv.first);
    for (const pyjson_value &a : added->a) {
        const pyjson_value *id = a.get("id"), *content = a.get("content");
        if (!id || !content || id->kind != pyjson_value::INT || content->kind != pyjson_value::STR) {
            ok = false;
            continue;
        }
        const long long i = atoll(id->num.c_str());
        if (i >= 0 && i < (long long)w->shape.vocab && have[(size_t)i] && str[(size_t)i] == content->s) {
            v->added[content->s] = (int)i;   /* an added token the vocabulary also holds, as itself */
            continue;
        }
        put(i, content->s);
        v->added[content->s] = (int)i;
    }
    /* the embedding is padded past the tokenizer's last id: those ids have no token, and decode skips them (as
     * tokenizers' id_to_token gives none) */
    size_t missing = 0;
    for (bool h : have) missing += !h;
    if (!ok) {
        fprintf(stderr, "pulsar: %s: %s: an id outside 0 .. %u or one id with two tokens -- refusing\n", OWNER,
                tp.c_str(), w->shape.vocab);
        return false;
    }
    int map[512];
    byte_of_cp(map);
    v->bytes.resize(str.size());
    for (size_t i = 0; i < str.size(); i++) v->bytes[i] = byte_level(str[i], map);
    if (!pulsar_nemotron_parse_encode_prompt(v, w->shape.prompt, &w->prompt_ids, err, sizeof err)) {
        fprintf(stderr, "pulsar: %s: the default prompt: %s -- refusing\n", OWNER, err);
        return false;
    }
    fprintf(stderr, "pulsar: %s: decode table %zu ids (%zu added, %zu padding ids with no token) from %s; default prompt "
                    "%zu ids\n", OWNER, v->bytes.size(), v->added.size(), missing, tp.c_str(), w->prompt_ids.size());
    return true;
}

bool pulsar_nemotron_parse_encode_prompt(const pulsar_nemotron_parse_vocab *v, const char *text, std::vector<int> *ids,
                                          char *err, size_t errlen) {
    ids->clear();
    const std::string t = text ? text : "";
    for (size_t i = 0; i < t.size();) {
        size_t best = 0;
        int id = -1;
        for (size_t n = 1; n <= 64 && i + n <= t.size(); n++) {   /* the longest added token at i */
            const auto it = v->added.find(t.substr(i, n));
            if (it != v->added.end()) { best = n; id = it->second; }
        }
        if (!best) {
            if (err) snprintf(err, errlen, "byte %zu of the prompt begins no added token (a Parse prompt is its task "
                                           "tokens, e.g. <predict_bbox><output_markdown>)", i);
            return false;
        }
        ids->push_back(id);
        i += best;
    }
    if (ids->empty()) {
        if (err) snprintf(err, errlen, "an empty prompt");
        return false;
    }
    return true;
}

std::string pulsar_nemotron_parse_decode(const pulsar_nemotron_parse_vocab *v, const int *ids, size_t n) {
    std::string raw;
    for (size_t i = 0; i < n; i++)
        if (ids[i] >= 0 && (size_t)ids[i] < v->bytes.size()) raw += v->bytes[(size_t)ids[i]];
    /* from_utf8_lossy: valid sequences kept, each maximal invalid subsequence one U+FFFD */
    std::string out;
    for (size_t i = 0; i < raw.size();) {
        const unsigned char c = (unsigned char)raw[i];
        size_t n = c < 0x80 ? 1 : c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        size_t good = n ? 1 : 0;
        for (size_t k = 1; k < n && i + k < raw.size(); k++) {
            const unsigned char d = (unsigned char)raw[i + k];
            unsigned char lo = 0x80, hi = 0xbf;   /* the second byte's range depends on the first (Unicode Table 3-7) */
            if (k == 1) {
                if (c == 0xe0) lo = 0xa0;
                else if (c == 0xed) hi = 0x9f;
                else if (c == 0xf0) lo = 0x90;
                else if (c == 0xf4) hi = 0x8f;
            }
            if (d < lo || d > hi) break;
            good = k + 1;
        }
        if (n && good == n) {
            out.append(raw, i, n);
            i += n;
        } else {
            out += "\xef\xbf\xbd";
            i += good ? good : 1;
        }
    }
    return out;
}
