/* qwen_tokenizer.cpp -- see qwen_tokenizer.h. */
#include "qwen_tokenizer.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <queue>
#include <unordered_map>

#include "pulsar_utf8.h"
#include "pyjson.h"

namespace {
#include "qwen_unicode_tables.inc"
}

/* ---- what the loader accepts: the checkpoint's pipeline, exactly ---------- */

/* The pre-tokenizer regex this file implements (split_pieces below), as the
 * JSON string decodes.  Any other pattern is a different tokenizer. */
static const char kQwenPattern[] =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";

static const char kByteLevel[] =
    R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})";

struct qwen_tokenizer {
    int n_tokens = 0;
    std::vector<std::string> bytes;       ///< id -> decoded bytes
    int byte_id[256];                     ///< raw byte -> its single-byte token
    std::unordered_map<uint64_t, std::pair<int, int>> merges;  ///< (a<<32|b) -> (rank, merged id)
    struct added { std::string text; int id; };
    std::vector<added> added_tokens;      ///< longest first
    bool added_first[256] = {};           ///< first bytes that can open an added token
    std::vector<int> stop_ids;
};

namespace {

bool refuse(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        char msg[400];
        vsnprintf(msg, sizeof msg, fmt, ap);
        va_end(ap);
        snprintf(err, errlen, "qwen tokenizer: %s", msg);
    }
    return false;
}

/* GPT-2 byte-level alphabet: printable Latin-1 bytes map to themselves, the
 * other 68 to U+0100.. in byte order. */
void byte_level_map(uint32_t cp_of[256]) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        cp_of[b] = keep ? (uint32_t)b : (uint32_t)(256 + n++);
    }
}

uint32_t utf8_decode(const unsigned char *p, int n) {
    switch (n) {
    case 1: return p[0];
    case 2: return ((uint32_t)(p[0] & 0x1f) << 6) | (p[1] & 0x3f);
    case 3: return ((uint32_t)(p[0] & 0x0f) << 12) | ((uint32_t)(p[1] & 0x3f) << 6) | (p[2] & 0x3f);
    default: return ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3f) << 12) |
                    ((uint32_t)(p[2] & 0x3f) << 6) | (p[3] & 0x3f);
    }
}

void utf8_put(std::string *s, uint32_t cp) {
    if (cp < 0x80) {
        s->push_back((char)cp);
    } else if (cp < 0x800) {
        s->push_back((char)(0xc0 | (cp >> 6)));
        s->push_back((char)(0x80 | (cp & 0x3f)));
    } else if (cp < 0x10000) {
        s->push_back((char)(0xe0 | (cp >> 12)));
        s->push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
        s->push_back((char)(0x80 | (cp & 0x3f)));
    } else {
        s->push_back((char)(0xf0 | (cp >> 18)));
        s->push_back((char)(0x80 | ((cp >> 12) & 0x3f)));
        s->push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
        s->push_back((char)(0x80 | (cp & 0x3f)));
    }
}

/* ---- Unicode lookups -------------------------------------------------- */

template <size_t N>
uint32_t range_lookup(const uint32_t (&t)[N][3], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < t[mid][0]) hi = mid;
        else if (cp > t[mid][1]) lo = mid + 1;
        else return t[mid][2];
    }
    return 0;
}

enum : uint32_t { CL_L = 1, CL_M = 2, CL_N = 4, CL_S = 8 };

uint32_t cls(uint32_t cp) { return range_lookup(qwen_uni_class, cp); }
uint32_t ccc(uint32_t cp) { return cp < 0x300 ? 0 : range_lookup(qwen_uni_ccc, cp); }

constexpr uint32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
constexpr uint32_t LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = LCount * NCount;

void decompose_one(uint32_t cp, std::vector<uint32_t> *out) {
    if (cp >= SBase && cp < SBase + SCount) {
        const uint32_t s = cp - SBase;
        out->push_back(LBase + s / NCount);
        out->push_back(VBase + (s % NCount) / TCount);
        if (s % TCount) out->push_back(TBase + s % TCount);
        return;
    }
    size_t lo = 0, hi = sizeof qwen_uni_decomp / sizeof qwen_uni_decomp[0];
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < qwen_uni_decomp[mid][0]) hi = mid;
        else if (cp > qwen_uni_decomp[mid][0]) lo = mid + 1;
        else {
            const uint32_t off = qwen_uni_decomp[mid][1], n = qwen_uni_decomp[mid][2];
            for (uint32_t i = 0; i < n; i++) out->push_back(qwen_uni_decomp_pool[off + i]);
            return;
        }
    }
    out->push_back(cp);
}

/* The primary composite of (a, b), or 0. */
uint32_t compose(uint32_t a, uint32_t b) {
    if (a >= LBase && a < LBase + LCount && b >= VBase && b < VBase + VCount)
        return SBase + ((a - LBase) * VCount + (b - VBase)) * TCount;
    if (a >= SBase && a < SBase + SCount && (a - SBase) % TCount == 0 && b > TBase && b < TBase + TCount)
        return a + (b - TBase);
    size_t lo = 0, hi = sizeof qwen_uni_comp / sizeof qwen_uni_comp[0];
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        const uint32_t *r = qwen_uni_comp[mid];
        if (a < r[0] || (a == r[0] && b < r[1])) hi = mid;
        else if (a > r[0] || b > r[1]) lo = mid + 1;
        else return r[2];
    }
    return 0;
}

/* NFC in place.  Below U+0300 no codepoint decomposes to anything but itself
 * recomposed, carries a combining class, or composes with a predecessor, so a
 * run entirely below it is already NFC. */
void nfc(std::vector<uint32_t> *cps) {
    bool trivial = true;
    for (uint32_t cp : *cps)
        if (cp >= 0x300) { trivial = false; break; }
    if (trivial) return;
    std::vector<uint32_t> d;
    d.reserve(cps->size() + 8);
    for (uint32_t cp : *cps) decompose_one(cp, &d);
    /* canonical ordering: stable sort of every run of non-starters */
    for (size_t i = 0; i < d.size();) {
        if (!ccc(d[i])) { i++; continue; }
        size_t j = i;
        while (j < d.size() && ccc(d[j])) j++;
        std::stable_sort(d.begin() + (long)i, d.begin() + (long)j,
                         [](uint32_t x, uint32_t y) { return ccc(x) < ccc(y); });
        i = j;
    }
    /* canonical composition */
    cps->clear();
    long starter = -1;
    long last_ccc = -1;   /* ccc of the last char kept after the starter; -1 = adjacent */
    for (uint32_t cp : d) {
        const long c = (long)ccc(cp);
        if (starter >= 0 && !(last_ccc != -1 && (last_ccc == 0 || last_ccc >= c))) {
            const uint32_t comp = compose((*cps)[(size_t)starter], cp);
            if (comp) {
                (*cps)[(size_t)starter] = comp;
                continue;
            }
        }
        if (c == 0) {
            starter = (long)cps->size();
            last_ccc = -1;
        } else {
            last_ccc = c;
        }
        cps->push_back(cp);
    }
}

/* ---- the pre-tokenizer regex ------------------------------------------ *
 * (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
 *   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * Oniguruma tries the alternatives in order at each position; this returns the
 * end of the first that matches at i (every codepoint is covered by one of
 * them, so the split has no gaps).  Graded over every scalar value against
 * tokenizers' own Split. */

uint32_t ascii_lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

size_t match_at(const std::vector<uint32_t> &s, const std::vector<uint32_t> &cl, size_t i) {
    const size_t n = s.size();
    const uint32_t c = s[i];
    /* (?i:'s|'t|'re|'ve|'m|'ll|'d): Onig's case folding also takes U+017F
     * (LONG S) for s -- measured, and the only non-ASCII fold that lands here.
     * (Qwen's vocabulary has no token holding U+017F's bytes, so this changes
     * the pieces but never the ids: no gate can see it, the split is still HF's.) */
    if (c == '\'' && i + 1 < n) {
        const uint32_t a = ascii_lower(s[i + 1]);
        const uint32_t b = i + 2 < n ? ascii_lower(s[i + 2]) : 0;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
        if (a == 's' || a == 't' || a == 'm' || a == 'd' || s[i + 1] == 0x17f) return i + 2;
    }
    auto lm_run = [&](size_t k) {
        while (k < n && (cl[k] & (CL_L | CL_M))) k++;
        return k;
    };
    /* [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ */
    if (c != '\r' && c != '\n' && !(cl[i] & (CL_L | CL_N)) && i + 1 < n && (cl[i + 1] & (CL_L | CL_M)))
        return lm_run(i + 1);
    if (cl[i] & (CL_L | CL_M)) return lm_run(i);
    /* \p{N} */
    if (cl[i] & CL_N) return i + 1;
    /*  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* */
    {
        size_t k = i;
        if (s[k] == ' ') k++;
        if (k < n && !(cl[k] & (CL_S | CL_L | CL_M | CL_N))) {
            while (k < n && !(cl[k] & (CL_S | CL_L | CL_M | CL_N))) k++;
            while (k < n && (s[k] == '\r' || s[k] == '\n')) k++;
            return k;
        }
    }
    /* \s*[\r\n]+ : up to the LAST CR/LF of the whitespace run */
    size_t j = i;
    while (j < n && (cl[j] & CL_S)) j++;
    for (size_t q = j; q > i; q--)
        if (s[q - 1] == '\r' || s[q - 1] == '\n') return q;
    /* \s+(?!\S) then \s+ */
    if (j > i) {
        if (j == n || j - 1 == i) return j;
        return j - 1;
    }
    return i + 1;   /* unreachable: every codepoint class is covered above */
}

}  // namespace

/* ---- BPE: tokenizers' Word::merge_all, verbatim in behaviour ------------ */
namespace {

struct sym {
    int id;
    long prev;
    long next;
    bool gone;
};

struct merge_item {
    int rank;
    size_t pos;
    int new_id;
    bool operator<(const merge_item &o) const {   /* max-heap pops lowest rank, then lowest pos */
        if (rank != o.rank) return rank > o.rank;
        return pos > o.pos;
    }
};

void bpe(const qwen_tokenizer *t, const unsigned char *p, size_t n, std::vector<int> *out) {
    if (n == 1) {
        out->push_back(t->byte_id[p[0]]);
        return;
    }
    std::vector<sym> w(n);
    for (size_t i = 0; i < n; i++)
        w[i] = {t->byte_id[p[i]], (long)i - 1, i + 1 < n ? (long)i + 1 : -1, false};
    std::priority_queue<merge_item> q;
    auto find = [&](int a, int b) -> const std::pair<int, int> * {
        auto it = t->merges.find(((uint64_t)(uint32_t)a << 32) | (uint32_t)b);
        return it == t->merges.end() ? NULL : &it->second;
    };
    for (size_t i = 0; i + 1 < n; i++)
        if (const auto *m = find(w[i].id, w[i + 1].id)) q.push({m->first, i, m->second});
    while (!q.empty()) {
        const merge_item top = q.top();
        q.pop();
        sym &cur = w[top.pos];
        if (cur.gone || cur.next < 0) continue;
        const size_t next_pos = (size_t)cur.next;
        const sym right = w[next_pos];
        const auto *m = find(cur.id, right.id);
        if (!m || m->second != top.new_id) continue;   /* an expired entry */
        cur.id = top.new_id;
        cur.next = right.next;
        w[next_pos].gone = true;
        if (right.next >= 0) w[(size_t)right.next].prev = (long)top.pos;
        if (cur.prev >= 0) {
            if (const auto *pm = find(w[(size_t)cur.prev].id, cur.id))
                q.push({pm->first, (size_t)cur.prev, pm->second});
        }
        if (cur.next >= 0) {
            if (const auto *nm = find(cur.id, w[(size_t)cur.next].id))
                q.push({nm->first, top.pos, nm->second});
        }
    }
    for (size_t i = 0; i < n;) {
        out->push_back(w[i].id);
        if (w[i].next < 0) break;
        i = (size_t)w[i].next;
    }
}

/* One run between added tokens: NFC, split, BPE. */
void encode_run(const qwen_tokenizer *t, const char *text, size_t len, std::vector<int> *out) {
    if (!len) return;
    std::vector<uint32_t> cps;
    cps.reserve(len);
    for (size_t i = 0; i < len;) {
        const int k = utf8_seq_ok((const unsigned char *)text + i, len - i);   /* validated by the caller */
        cps.push_back(utf8_decode((const unsigned char *)text + i, k));
        i += (size_t)k;
    }
    nfc(&cps);
    std::vector<uint32_t> cl(cps.size());
    for (size_t i = 0; i < cps.size(); i++) cl[i] = cls(cps[i]);
    std::string piece;
    for (size_t i = 0; i < cps.size();) {
        const size_t j = match_at(cps, cl, i);
        piece.clear();
        for (size_t k = i; k < j; k++) utf8_put(&piece, cps[k]);
        bpe(t, (const unsigned char *)piece.data(), piece.size(), out);
        i = j;
    }
}

bool expect_dump(const pyjson_value *v, const char *expect, const char *what, char *err, size_t errlen) {
    std::string got;
    if (v) pyjson_dump(*v, &got);
    else got = "(missing)";
    if (got != expect)
        return refuse(err, errlen, "%s is %.200s; this tokenizer implements exactly %.200s", what,
                      got.c_str(), expect);
    return true;
}

}  // namespace

qwen_tokenizer *qwen_tokenizer_load(const char *tokenizer_json, size_t tokenizer_len,
                                    const char *generation_config_json, size_t generation_len,
                                    char *err, size_t errlen) {
    pyjson_value doc;
    char jerr[200];
    if (!pyjson_parse(tokenizer_json, tokenizer_len, &doc, jerr, sizeof jerr)) {
        refuse(err, errlen, "tokenizer.json: %s", jerr);
        return NULL;
    }
    /* The pipeline, section by section: anything else is a different tokenizer. */
    std::string pre_expect = R"({"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": )";
    pyjson_dump_string(kQwenPattern, strlen(kQwenPattern), &pre_expect);
    pre_expect += R"(}, "behavior": "Isolated", "invert": false}, )";
    pre_expect += kByteLevel;
    pre_expect += "]}";
    if (!expect_dump(doc.get("normalizer"), R"({"type": "NFC"})", "the normalizer", err, errlen) ||
        !expect_dump(doc.get("pre_tokenizer"), pre_expect.c_str(), "the pre-tokenizer", err, errlen) ||
        !expect_dump(doc.get("decoder"), kByteLevel, "the decoder", err, errlen) ||
        !expect_dump(doc.get("post_processor"), kByteLevel, "the post-processor", err, errlen))
        return NULL;
    const pyjson_value *model = doc.get("model");
    if (!model || model->kind != pyjson_value::OBJ) {
        refuse(err, errlen, "tokenizer.json has no model object");
        return NULL;
    }
    static const struct { const char *key, *value; } model_expect[] = {
        {"type", "\"BPE\""}, {"dropout", "null"}, {"unk_token", "null"},
        {"continuing_subword_prefix", "\"\""}, {"end_of_word_suffix", "\"\""},
        {"fuse_unk", "false"}, {"byte_fallback", "false"}, {"ignore_merges", "false"},
    };
    for (const auto &e : model_expect) {
        char what[64];
        snprintf(what, sizeof what, "model.%s", e.key);
        if (!expect_dump(model->get(e.key), e.value, what, err, errlen)) return NULL;
    }
    const pyjson_value *vocab = model->get("vocab");
    const pyjson_value *merges = model->get("merges");
    const pyjson_value *added = doc.get("added_tokens");
    if (!vocab || vocab->kind != pyjson_value::OBJ || !merges || merges->kind != pyjson_value::ARR ||
        !added || added->kind != pyjson_value::ARR) {
        refuse(err, errlen, "tokenizer.json lacks model.vocab / model.merges / added_tokens");
        return NULL;
    }

    qwen_tokenizer *t = new qwen_tokenizer();
    uint32_t cp_of[256];
    byte_level_map(cp_of);
    std::unordered_map<uint32_t, int> byte_of_cp;
    for (int b = 0; b < 256; b++) byte_of_cp[cp_of[b]] = b;

    std::unordered_map<std::string, int> id_of;
    id_of.reserve(vocab->o.size() * 2);
    int max_id = -1;
    std::vector<std::pair<int, std::string>> entries;
    entries.reserve(vocab->o.size() + added->a.size());
    for (const auto &kv : vocab->o) {
        if (kv.second.kind != pyjson_value::INT) { refuse(err, errlen, "vocab id is not an int"); delete t; return NULL; }
        const int id = atoi(kv.second.num.c_str());
        id_of[kv.first] = id;
        /* byte-level text -> raw bytes */
        std::string raw;
        const unsigned char *s = (const unsigned char *)kv.first.data();
        for (size_t i = 0; i < kv.first.size();) {
            const int k = utf8_seq_ok(s + i, kv.first.size() - i);
            auto it = k ? byte_of_cp.find(utf8_decode(s + i, k)) : byte_of_cp.end();
            if (it == byte_of_cp.end()) {
                refuse(err, errlen, "vocab token %d is not byte-level text", id);
                delete t;
                return NULL;
            }
            raw.push_back((char)it->second);
            i += (size_t)k;
        }
        entries.emplace_back(id, std::move(raw));
        max_id = std::max(max_id, id);
    }
    for (const auto &a : added->a) {
        const pyjson_value *id = a.get("id"), *content = a.get("content");
        if (!id || id->kind != pyjson_value::INT || !content || content->kind != pyjson_value::STR) {
            refuse(err, errlen, "added_tokens entry lacks id/content");
            delete t;
            return NULL;
        }
        for (const char *flag : {"single_word", "lstrip", "rstrip", "normalized"}) {
            const pyjson_value *f = a.get(flag);
            if (!f || f->kind != pyjson_value::BOOL || f->b) {
                refuse(err, errlen, "added token %s: %s is not false (this tokenizer matches added tokens "
                       "on raw text, whole, with no stripping)", content->s.c_str(), flag);
                delete t;
                return NULL;
            }
        }
        if (content->s.empty()) { refuse(err, errlen, "empty added token"); delete t; return NULL; }
        const int iid = atoi(id->num.c_str());
        entries.emplace_back(iid, content->s);
        t->added_tokens.push_back({content->s, iid});
        t->added_first[(unsigned char)content->s[0]] = true;
        max_id = std::max(max_id, iid);
    }
    t->n_tokens = max_id + 1;
    t->bytes.assign((size_t)t->n_tokens, std::string());
    std::vector<bool> seen((size_t)t->n_tokens, false);
    for (auto &e : entries) {
        if (e.first < 0 || seen[(size_t)e.first]) {
            refuse(err, errlen, "token id %d is negative or assigned twice", e.first);
            delete t;
            return NULL;
        }
        seen[(size_t)e.first] = true;
        t->bytes[(size_t)e.first] = std::move(e.second);
    }
    for (int i = 0; i < t->n_tokens; i++)
        if (!seen[(size_t)i]) { refuse(err, errlen, "token id space has a hole at %d", i); delete t; return NULL; }
    std::stable_sort(t->added_tokens.begin(), t->added_tokens.end(),
                     [](const qwen_tokenizer::added &x, const qwen_tokenizer::added &y) {
                         return x.text.size() > y.text.size();
                     });
    /* The span guard drops a marker that touches client bytes and moves on; that
     * equals HF-with-the-guard only if no shorter added token could match at the
     * same place, i.e. no added token is a prefix of another.  Asserted here. */
    for (const auto &x : t->added_tokens)
        for (const auto &y : t->added_tokens)
            if (x.id != y.id && y.text.size() < x.text.size() && !x.text.compare(0, y.text.size(), y.text)) {
                refuse(err, errlen, "added token %s is a prefix of %s (the span guard assumes none is)",
                       y.text.c_str(), x.text.c_str());
                delete t;
                return NULL;
            }
    for (int b = 0; b < 256; b++) {
        std::string s;
        utf8_put(&s, cp_of[b]);
        auto it = id_of.find(s);
        if (it == id_of.end()) { refuse(err, errlen, "byte 0x%02x has no token", b); delete t; return NULL; }
        t->byte_id[b] = it->second;
    }
    t->merges.reserve(merges->a.size() * 2);
    for (size_t r = 0; r < merges->a.size(); r++) {
        const pyjson_value &m = merges->a[r];
        const size_t sp = m.kind == pyjson_value::STR ? m.s.find(' ') : std::string::npos;
        if (sp == std::string::npos || m.s.find(' ', sp + 1) != std::string::npos) {
            refuse(err, errlen, "merge %zu is not one \"a b\" string", r);
            delete t;
            return NULL;
        }
        const std::string a = m.s.substr(0, sp), b = m.s.substr(sp + 1);
        auto ia = id_of.find(a), ib = id_of.find(b), im = id_of.find(a + b);
        if (ia == id_of.end() || ib == id_of.end() || im == id_of.end()) {
            refuse(err, errlen, "merge %zu (%s) names a token the vocab lacks", r, m.s.c_str());
            delete t;
            return NULL;
        }
        /* a later duplicate pair overwrites, as tokenizers' HashMap collect does */
        t->merges[((uint64_t)(uint32_t)ia->second << 32) | (uint32_t)ib->second] = {(int)r, im->second};
    }

    pyjson_value gen;
    if (!pyjson_parse(generation_config_json, generation_len, &gen, jerr, sizeof jerr)) {
        refuse(err, errlen, "generation_config.json: %s", jerr);
        delete t;
        return NULL;
    }
    const pyjson_value *eos = gen.get("eos_token_id");
    std::vector<const pyjson_value *> eos_ids;
    if (eos && eos->kind == pyjson_value::INT) eos_ids.push_back(eos);
    else if (eos && eos->kind == pyjson_value::ARR)
        for (const auto &x : eos->a) eos_ids.push_back(&x);
    if (eos_ids.empty()) {
        refuse(err, errlen, "generation_config.json has no eos_token_id");
        delete t;
        return NULL;
    }
    for (const pyjson_value *x : eos_ids) {
        const int id = x->kind == pyjson_value::INT ? atoi(x->num.c_str()) : -1;
        bool is_added = false;
        for (const auto &a : t->added_tokens) is_added |= a.id == id;
        if (!is_added) {
            refuse(err, errlen, "eos_token_id %d is not an added token", id);
            delete t;
            return NULL;
        }
        t->stop_ids.push_back(id);
    }
    return t;
}

void qwen_tokenizer_free(qwen_tokenizer *t) { delete t; }

int qwen_tokenizer_n_tokens(const qwen_tokenizer *t) { return t->n_tokens; }

bool qwen_tokenizer_encode(const qwen_tokenizer *t, const char *text, size_t len,
                           const pulsar_text_span *spans, uint32_t n_spans,
                           std::vector<int> *out, char *err, size_t errlen) {
    for (size_t i = 0; i < len;) {
        const int k = utf8_seq_ok((const unsigned char *)text + i, len - i);
        if (!k) return refuse(err, errlen, "input is not UTF-8 at byte %zu", i);
        i += (size_t)k;
    }
    for (uint32_t i = 0; i < n_spans; i++)
        if (spans[i].lo > spans[i].hi || (i + 1 < n_spans && spans[i].hi > spans[i + 1].lo))
            return refuse(err, errlen, "client spans are not ascending and disjoint (span %u)", i);
    size_t run = 0;
    uint32_t si = 0;
    for (size_t pos = 0; pos < len;) {
        if (!t->added_first[(unsigned char)text[pos]]) { pos++; continue; }
        const qwen_tokenizer::added *hit = NULL;
        for (const auto &a : t->added_tokens) {   /* longest first: leftmost-longest */
            if (a.text.size() <= len - pos && !memcmp(text + pos, a.text.data(), a.text.size())) {
                hit = &a;
                break;
            }
        }
        if (hit) {
            /* L223: a match that touches a client range is client text, not a marker */
            while (si < n_spans && spans[si].hi <= pos) si++;
            if (si < n_spans && spans[si].lo < pos + hit->text.size()) hit = NULL;
        }
        if (!hit) { pos++; continue; }
        encode_run(t, text + run, pos - run, out);
        out->push_back(hit->id);
        pos += hit->text.size();
        run = pos;
    }
    encode_run(t, text + run, len - run, out);
    return true;
}

const char *qwen_tokenizer_token_bytes(const qwen_tokenizer *t, int id, size_t *len) {
    if (id < 0 || id >= t->n_tokens) return NULL;
    if (len) *len = t->bytes[(size_t)id].size();
    return t->bytes[(size_t)id].data();
}

int qwen_tokenizer_added_id(const qwen_tokenizer *t, const char *text) {
    for (const auto &a : t->added_tokens)
        if (a.text == text) return a.id;
    return -1;
}

const std::vector<int> &qwen_tokenizer_stop_ids(const qwen_tokenizer *t) { return t->stop_ids; }
