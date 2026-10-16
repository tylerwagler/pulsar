/* bpe_tokenizer.cpp -- see bpe_tokenizer.h. */
#include "bpe_tokenizer.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <queue>
#include <string_view>
#include <unordered_map>

#include "pulsar_utf8.h"
#include "pyjson.h"

namespace {
#include "bpe_unicode_tables.inc"
}

struct bpe_tokenizer {
    const bpe_pretok_pattern *pattern = nullptr;
    bool nfc = false;                     ///< runs are NFC-normalised first (HF's NFC normaliser)
    int n_tokens = 0;
    std::vector<std::string> bytes;       ///< id -> decoded bytes
    int byte_id[256];                     ///< raw byte -> its single-byte token
    /** HF merges: (a<<32|b) -> (rank, merged id).  Empty for a tiktoken configuration. */
    std::unordered_map<uint64_t, std::pair<int, int>> merges;
    /** tiktoken ranks: a BPE token's bytes -> its id (= its rank); the views point into `bytes`, which is
     *  never resized after the map is built.  Empty for an HF configuration. */
    std::unordered_map<std::string_view, int> rank_of;
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
        snprintf(err, errlen, "bpe tokenizer: %s", msg);
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

/* the class flags of bpe_unicode_tables.inc */
enum : uint32_t {
    CL_L = 1, CL_M = 2, CL_N = 4, CL_S = 8,
    CL_LU = 16, CL_LL = 32, CL_LT = 64, CL_LM = 128, CL_LO = 256,
};

uint32_t cls(uint32_t cp) { return range_lookup(bpe_uni_class, cp); }
uint32_t ccc(uint32_t cp) { return cp < 0x300 ? 0 : range_lookup(bpe_uni_ccc, cp); }

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
    size_t lo = 0, hi = sizeof bpe_uni_decomp / sizeof bpe_uni_decomp[0];
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < bpe_uni_decomp[mid][0]) hi = mid;
        else if (cp > bpe_uni_decomp[mid][0]) lo = mid + 1;
        else {
            const uint32_t off = bpe_uni_decomp[mid][1], n = bpe_uni_decomp[mid][2];
            for (uint32_t i = 0; i < n; i++) out->push_back(bpe_uni_decomp_pool[off + i]);
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
    size_t lo = 0, hi = sizeof bpe_uni_comp / sizeof bpe_uni_comp[0];
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        const uint32_t *r = bpe_uni_comp[mid];
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

/* ---- the pre-tokenizer patterns (the registry's matchers) -------------- *
 * Oniguruma tries the alternatives in order at each position; each matcher
 * returns the end of the first that matches at i (every codepoint is covered
 * by one of them, so the split has no gaps).  Each is graded over every scalar
 * value against tokenizers' own Split. */

uint32_t ascii_lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

/* (?i:'s|'t|'re|'ve|'m|'ll|'d) at i: its end, or i.  Onig's case folding also
 * takes U+017F (LONG S) for s -- measured, and the only non-ASCII fold that
 * lands here. */
size_t contraction_at(const uint32_t *s, size_t n, size_t i) {
    if (i >= n || s[i] != '\'' || i + 1 >= n) return i;
    const uint32_t a = ascii_lower(s[i + 1]);
    const uint32_t b = i + 2 < n ? ascii_lower(s[i + 2]) : 0;
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    if (a == 's' || a == 't' || a == 'm' || a == 'd' || s[i + 1] == 0x17f) return i + 2;
    return i;
}

/* \s*[\r\n]+ | \s+(?!\S) | \s+ at i (the three whitespace alternatives both patterns end with): the end, or
 * i when i is not whitespace.  \s*[\r\n]+ ends at the LAST CR/LF of the whitespace run. */
size_t whitespace_at(const uint32_t *s, const uint32_t *cl, size_t n, size_t i) {
    size_t j = i;
    while (j < n && (cl[j] & CL_S)) j++;
    for (size_t q = j; q > i; q--)
        if (s[q - 1] == '\r' || s[q - 1] == '\n') return q;
    if (j > i) {
        if (j == n || j - 1 == i) return j;
        return j - 1;
    }
    return i;
}

/* qwen35 -- Qwen3.5 / 3.8:
 * (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
 *   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * (Qwen's vocabulary has no token holding U+017F's bytes, so the contraction's
 * long-s fold changes the pieces but never the ids: no gate can see it, the
 * split is still HF's.) */
size_t qwen35_match_at(const uint32_t *s, const uint32_t *cl, size_t n, size_t i) {
    const uint32_t c = s[i];
    if (const size_t e = contraction_at(s, n, i); e > i) return e;
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
    if (const size_t e = whitespace_at(s, cl, n, i); e > i) return e;
    return i + 1;   /* unreachable: every codepoint class is covered above */
}

/* o200k -- tiktoken's o200k_base split (Inkling):
 * [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
 *   |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
 *   |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * The two word alternatives backtrack: the optional prefix is tried taken first, then not; the greedy
 * upper-class run gives back one codepoint at a time until the lower class can start (Lm, Lo and M are in
 * both classes). */
size_t o200k_match_at(const uint32_t *s, const uint32_t *cl, size_t n, size_t i) {
    constexpr uint32_t UP = CL_LU | CL_LT | CL_LM | CL_LO | CL_M;   /* [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] */
    constexpr uint32_t LO = CL_LL | CL_LM | CL_LO | CL_M;           /* [\p{Ll}\p{Lm}\p{Lo}\p{M}] */
    const bool prefix = s[i] != '\r' && s[i] != '\n' && !(cl[i] & (CL_L | CL_N));
    auto run = [&](size_t k, uint32_t mask) {
        while (k < n && (cl[k] & mask)) k++;
        return k;
    };
    /* UP* LO+ from k: 0 = no match */
    auto word_lower = [&](size_t k) -> size_t {
        const size_t j0 = run(k, UP);
        for (size_t j = j0 + 1; j-- > k;)
            if (j < n && (cl[j] & LO)) return contraction_at(s, n, run(j, LO));
        return 0;
    };
    /* UP+ LO* from k: 0 = no match */
    auto word_upper = [&](size_t k) -> size_t {
        if (k >= n || !(cl[k] & UP)) return 0;
        return contraction_at(s, n, run(run(k, UP), LO));
    };
    if (prefix && i + 1 < n)
        if (const size_t e = word_lower(i + 1)) return e;
    if (const size_t e = word_lower(i)) return e;
    if (prefix && i + 1 < n)
        if (const size_t e = word_upper(i + 1)) return e;
    if (const size_t e = word_upper(i)) return e;
    /* \p{N}{1,3} */
    if (cl[i] & CL_N) {
        size_t k = i + 1;
        while (k < n && k < i + 3 && (cl[k] & CL_N)) k++;
        return k;
    }
    /*  ?[^\s\p{L}\p{N}]+[\r\n/]* */
    {
        size_t k = i;
        if (s[k] == ' ') k++;
        if (k < n && !(cl[k] & (CL_S | CL_L | CL_N))) {
            while (k < n && !(cl[k] & (CL_S | CL_L | CL_N))) k++;
            while (k < n && (s[k] == '\r' || s[k] == '\n' || s[k] == '/')) k++;
            return k;
        }
    }
    if (const size_t e = whitespace_at(s, cl, n, i); e > i) return e;
    return i + 1;   /* unreachable: every codepoint class is covered above */
}

const char kQwen35Pattern[] =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
const char kO200kPattern[] =
    R"re([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";

const bpe_pretok_pattern k_patterns[] = {
    {"qwen35", kQwen35Pattern, qwen35_match_at},
    {"o200k", kO200kPattern, o200k_match_at},
};

}  // namespace

const bpe_pretok_pattern *bpe_pretok_by_name(const char *name) {
    for (const bpe_pretok_pattern &p : k_patterns)
        if (name && !strcmp(p.name, name)) return &p;
    return nullptr;
}

const bpe_pretok_pattern *bpe_pretok_by_regex(const char *regex) {
    for (const bpe_pretok_pattern &p : k_patterns)
        if (regex && !strcmp(p.regex, regex)) return &p;
    return nullptr;
}

/* ---- BPE: tokenizers' Word::merge_all (HF merges) and tiktoken's byte_pair_merge (ranks), ONE loop ---- *
 * Both merge the adjacent pair of lowest rank first, leftmost among equals; they differ only in what a
 * pair's rank is -- an HF merge's index, or the rank of the pair's concatenated bytes. */
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

/* The merge of the symbols at w[l] and w[r] (adjacent; r's bytes end at `end`): (rank, merged id), or
 * false. */
bool pair_rank(const bpe_tokenizer *t, const unsigned char *p, const std::vector<sym> &w, size_t l, size_t r,
               size_t end, int *rank, int *id) {
    if (t->rank_of.empty()) {
        auto it = t->merges.find(((uint64_t)(uint32_t)w[l].id << 32) | (uint32_t)w[r].id);
        if (it == t->merges.end()) return false;
        *rank = it->second.first;
        *id = it->second.second;
        return true;
    }
    auto it = t->rank_of.find(std::string_view((const char *)p + l, end - l));
    if (it == t->rank_of.end()) return false;
    *rank = *id = it->second;
    return true;
}

void bpe(const bpe_tokenizer *t, const unsigned char *p, size_t n, std::vector<int> *out) {
    if (n == 1) {
        out->push_back(t->byte_id[p[0]]);
        return;
    }
    if (!t->rank_of.empty()) {   /* a piece that is a token is that token */
        auto it = t->rank_of.find(std::string_view((const char *)p, n));
        if (it != t->rank_of.end()) {
            out->push_back(it->second);
            return;
        }
    }
    std::vector<sym> w(n);
    for (size_t i = 0; i < n; i++)
        w[i] = {t->byte_id[p[i]], (long)i - 1, i + 1 < n ? (long)i + 1 : -1, false};
    /* where the symbol at `pos` ends: the next live symbol's start */
    auto end_of = [&](size_t pos) { return w[pos].next < 0 ? n : (size_t)w[pos].next; };
    std::priority_queue<merge_item> q;
    auto push = [&](size_t l) {
        if (w[l].next < 0) return;
        const size_t r = (size_t)w[l].next;
        int rank = 0, id = 0;
        if (pair_rank(t, p, w, l, r, end_of(r), &rank, &id)) q.push({rank, l, id});
    };
    for (size_t i = 0; i + 1 < n; i++) push(i);
    while (!q.empty()) {
        const merge_item top = q.top();
        q.pop();
        sym &cur = w[top.pos];
        if (cur.gone || cur.next < 0) continue;
        const size_t next_pos = (size_t)cur.next;
        int rank = 0, id = 0;
        if (!pair_rank(t, p, w, top.pos, next_pos, end_of(next_pos), &rank, &id) || id != top.new_id)
            continue;   /* an expired entry */
        const sym right = w[next_pos];
        cur.id = top.new_id;
        cur.next = right.next;
        w[next_pos].gone = true;
        if (right.next >= 0) w[(size_t)right.next].prev = (long)top.pos;
        if (cur.prev >= 0) push((size_t)cur.prev);
        push(top.pos);
    }
    for (size_t i = 0; i < n;) {
        out->push_back(w[i].id);
        if (w[i].next < 0) break;
        i = (size_t)w[i].next;
    }
}

/* One run between added tokens: NFC (when the configuration has it), split, BPE. */
void encode_run(const bpe_tokenizer *t, const char *text, size_t len, std::vector<int> *out) {
    if (!len) return;
    std::vector<uint32_t> cps;
    cps.reserve(len);
    for (size_t i = 0; i < len;) {
        const int k = utf8_seq_ok((const unsigned char *)text + i, len - i);   /* validated by the caller */
        cps.push_back(utf8_decode((const unsigned char *)text + i, k));
        i += (size_t)k;
    }
    if (t->nfc) nfc(&cps);
    std::vector<uint32_t> cl(cps.size());
    for (size_t i = 0; i < cps.size(); i++) cl[i] = cls(cps[i]);
    std::string piece;
    for (size_t i = 0; i < cps.size();) {
        const size_t j = t->pattern->match_at(cps.data(), cl.data(), cps.size(), i);
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

/* An added token: its flags must be the ones this engine implements (matched on raw text, whole, no
 * stripping). */
bool check_added_flags(const pyjson_value &a, const std::string &content, char *err, size_t errlen) {
    for (const char *flag : {"single_word", "lstrip", "rstrip", "normalized"}) {
        const pyjson_value *f = a.get(flag);
        if (!f || f->kind != pyjson_value::BOOL || f->b)
            return refuse(err, errlen, "added token %s: %s is not false (this tokenizer matches added tokens "
                          "on raw text, whole, with no stripping)", content.c_str(), flag);
    }
    return !content.empty() || refuse(err, errlen, "empty added token");
}

/* What both loaders finish with, over (id, bytes) entries and the added tokens already in t: the id space
 * (no holes, no id twice), the added tokens longest first (no one a prefix of another: the span guard's
 * assumption), the single-byte tokens, and the stop ids from the eos config. */
bool finish_table(bpe_tokenizer *t, std::vector<std::pair<int, std::string>> *entries, int max_id,
                  const char *eos_config_json, size_t eos_len, char *err, size_t errlen) {
    t->n_tokens = max_id + 1;
    t->bytes.assign((size_t)t->n_tokens, std::string());
    std::vector<bool> seen((size_t)t->n_tokens, false);
    for (auto &e : *entries) {
        if (e.first < 0 || seen[(size_t)e.first])
            return refuse(err, errlen, "token id %d is negative or assigned twice", e.first);
        seen[(size_t)e.first] = true;
        t->bytes[(size_t)e.first] = std::move(e.second);
    }
    for (int i = 0; i < t->n_tokens; i++)
        if (!seen[(size_t)i]) return refuse(err, errlen, "token id space has a hole at %d", i);
    std::stable_sort(t->added_tokens.begin(), t->added_tokens.end(),
                     [](const bpe_tokenizer::added &x, const bpe_tokenizer::added &y) {
                         return x.text.size() > y.text.size();
                     });
    /* The span guard drops a marker that touches client bytes and moves on; that
     * equals HF-with-the-guard only if no shorter added token could match at the
     * same place, i.e. no added token is a prefix of another.  Asserted here. */
    for (const auto &x : t->added_tokens)
        for (const auto &y : t->added_tokens)
            if (x.id != y.id && y.text.size() < x.text.size() && !x.text.compare(0, y.text.size(), y.text))
                return refuse(err, errlen, "added token %s is a prefix of %s (the span guard assumes none is)",
                              y.text.c_str(), x.text.c_str());
    for (const auto &a : t->added_tokens) t->added_first[(unsigned char)a.text[0]] = true;

    pyjson_value gen;
    char jerr[200];
    if (!pyjson_parse(eos_config_json, eos_len, &gen, jerr, sizeof jerr))
        return refuse(err, errlen, "eos config: %s", jerr);
    const pyjson_value *eos = gen.get("eos_token_id");
    std::vector<const pyjson_value *> eos_ids;
    if (eos && eos->kind == pyjson_value::INT) eos_ids.push_back(eos);
    else if (eos && eos->kind == pyjson_value::ARR)
        for (const auto &x : eos->a) eos_ids.push_back(&x);
    if (eos_ids.empty()) return refuse(err, errlen, "the eos config has no eos_token_id");
    for (const pyjson_value *x : eos_ids) {
        const int id = x->kind == pyjson_value::INT ? atoi(x->num.c_str()) : -1;
        bool is_added = false;
        for (const auto &a : t->added_tokens) is_added |= a.id == id;
        if (!is_added) return refuse(err, errlen, "eos_token_id %d is not an added token", id);
        t->stop_ids.push_back(id);
    }
    return true;
}

/* Standard base64 (tiktoken's rank file); false on anything else. */
bool base64_decode(const char *p, size_t n, std::string *out) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    out->clear();
    if (n % 4) return false;
    for (size_t i = 0; i < n; i += 4) {
        int v[4];
        int pad = 0;
        for (int k = 0; k < 4; k++) {
            if (p[i + k] == '=' && i + 4 == n && k >= 2) { v[k] = 0; pad++; continue; }
            if (pad || (v[k] = val(p[i + k])) < 0) return false;
        }
        const uint32_t w = ((uint32_t)v[0] << 18) | ((uint32_t)v[1] << 12) | ((uint32_t)v[2] << 6) | (uint32_t)v[3];
        out->push_back((char)(w >> 16));
        if (pad < 2) out->push_back((char)((w >> 8) & 0xff));
        if (pad < 1) out->push_back((char)(w & 0xff));
    }
    return true;
}

}  // namespace

bpe_tokenizer *bpe_tokenizer_load_hf(const char *tokenizer_json, size_t tokenizer_len, const char *eos_config_json,
                                     size_t eos_len, char *err, size_t errlen) {
    pyjson_value doc;
    char jerr[200];
    if (!pyjson_parse(tokenizer_json, tokenizer_len, &doc, jerr, sizeof jerr)) {
        refuse(err, errlen, "tokenizer.json: %s", jerr);
        return NULL;
    }
    /* The pipeline, section by section: anything else is a different tokenizer.  The pre-tokenizer's
     * regex names its pattern in the registry. */
    static const char kByteLevel[] =
        R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})";
    const pyjson_value *pre = doc.get("pre_tokenizer");
    const pyjson_value *steps = pre ? pre->get("pretokenizers") : NULL;
    const pyjson_value *split = steps && steps->kind == pyjson_value::ARR && !steps->a.empty() ? &steps->a[0] : NULL;
    const pyjson_value *pat = split && split->get("pattern") ? split->get("pattern")->get("Regex") : NULL;
    const bpe_pretok_pattern *pattern = pat && pat->kind == pyjson_value::STR ? bpe_pretok_by_regex(pat->s.c_str())
                                                                              : NULL;
    if (!pattern) {
        refuse(err, errlen, "the pre-tokenizer's Split regex is not a registered pattern (bpe_pretok_*)");
        return NULL;
    }
    std::string pre_expect = R"({"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": )";
    pyjson_dump_string(pattern->regex, strlen(pattern->regex), &pre_expect);
    pre_expect += R"(}, "behavior": "Isolated", "invert": false}, )";
    pre_expect += kByteLevel;
    pre_expect += "]}";
    if (!expect_dump(doc.get("normalizer"), R"({"type": "NFC"})", "the normalizer", err, errlen) ||
        !expect_dump(pre, pre_expect.c_str(), "the pre-tokenizer", err, errlen) ||
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

    bpe_tokenizer *t = new bpe_tokenizer();
    t->pattern = pattern;
    t->nfc = true;
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
        if (!check_added_flags(a, content->s, err, errlen)) { delete t; return NULL; }
        const int iid = atoi(id->num.c_str());
        entries.emplace_back(iid, content->s);
        t->added_tokens.push_back({content->s, iid});
        max_id = std::max(max_id, iid);
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
    if (!finish_table(t, &entries, max_id, eos_config_json, eos_len, err, errlen)) {
        delete t;
        return NULL;
    }
    return t;
}

bpe_tokenizer *bpe_tokenizer_load_tiktoken(const char *ranks, size_t ranks_len, const char *pattern,
                                           const char *tokenizer_config_json, size_t config_len,
                                           const char *eos_config_json, size_t eos_len, char *err, size_t errlen) {
    const bpe_pretok_pattern *pat = bpe_pretok_by_name(pattern);
    if (!pat) {
        refuse(err, errlen, "pre-tokenizer pattern \"%s\" is not registered", pattern ? pattern : "(null)");
        return NULL;
    }
    pyjson_value cfg;
    char jerr[200];
    if (!pyjson_parse(tokenizer_config_json, config_len, &cfg, jerr, sizeof jerr)) {
        refuse(err, errlen, "tokenizer_config.json: %s", jerr);
        return NULL;
    }
    const pyjson_value *decoder = cfg.get("added_tokens_decoder");
    if (!decoder || decoder->kind != pyjson_value::OBJ) {
        refuse(err, errlen, "tokenizer_config.json has no added_tokens_decoder object");
        return NULL;
    }
    bpe_tokenizer *t = new bpe_tokenizer();
    t->pattern = pat;
    t->nfc = false;
    std::vector<std::pair<int, std::string>> entries;
    int max_id = -1;
    /* the rank file: "base64 rank\n" per token */
    for (size_t i = 0, line = 1; i < ranks_len; line++) {
        const char *eol = (const char *)memchr(ranks + i, '\n', ranks_len - i);
        const size_t end = eol ? (size_t)(eol - ranks) : ranks_len;
        const char *sp = (const char *)memchr(ranks + i, ' ', end - i);
        std::string bytes;
        char *rest = NULL;
        const long rank = sp ? strtol(sp + 1, &rest, 10) : -1;
        if (!sp || !base64_decode(ranks + i, (size_t)(sp - ranks) - i, &bytes) || bytes.empty() ||
            rest != ranks + end || rank < 0 || rank > INT32_MAX) {
            refuse(err, errlen, "rank file line %zu is not \"base64 rank\"", line);
            delete t;
            return NULL;
        }
        entries.emplace_back((int)rank, std::move(bytes));
        max_id = std::max(max_id, (int)rank);
        i = end + (eol ? 1 : 0);
    }
    /* the special tokens: each is the rank file's bytes at its id */
    std::vector<bool> special;
    for (const auto &kv : decoder->o) {
        const pyjson_value *content = kv.second.get("content");
        char *rest = NULL;
        const long id = strtol(kv.first.c_str(), &rest, 10);
        if (!content || content->kind != pyjson_value::STR || kv.first.empty() || *rest || id < 0 || id > max_id) {
            refuse(err, errlen, "added_tokens_decoder entry \"%s\" lacks a content or names no rank", kv.first.c_str());
            delete t;
            return NULL;
        }
        if (!check_added_flags(kv.second, content->s, err, errlen)) { delete t; return NULL; }
        t->added_tokens.push_back({content->s, (int)id});
        if (special.size() <= (size_t)id) special.resize((size_t)id + 1, false);
        special[(size_t)id] = true;
    }
    if (!finish_table(t, &entries, max_id, eos_config_json, eos_len, err, errlen)) {
        delete t;
        return NULL;
    }
    for (const auto &a : t->added_tokens)
        if (t->bytes[(size_t)a.id] != a.text) {
            refuse(err, errlen, "added token %s is not the rank file's bytes at id %d", a.text.c_str(), a.id);
            delete t;
            return NULL;
        }
    /* the BPE vocabulary: every rank that is not a special token (HF keeps added tokens out of its model) */
    t->rank_of.reserve((size_t)t->n_tokens * 2);
    for (int id = 0; id < t->n_tokens; id++)
        if ((size_t)id >= special.size() || !special[(size_t)id])
            t->rank_of.emplace(std::string_view(t->bytes[(size_t)id]), id);
    for (int b = 0; b < 256; b++) {
        const char c = (char)b;
        auto it = t->rank_of.find(std::string_view(&c, 1));
        if (it == t->rank_of.end()) { refuse(err, errlen, "byte 0x%02x has no token", b); delete t; return NULL; }
        t->byte_id[b] = it->second;
    }
    return t;
}

void bpe_tokenizer_free(bpe_tokenizer *t) { delete t; }

const bpe_pretok_pattern *bpe_tokenizer_pattern(const bpe_tokenizer *t) { return t->pattern; }

int bpe_tokenizer_n_tokens(const bpe_tokenizer *t) { return t->n_tokens; }

bool bpe_tokenizer_encode(const bpe_tokenizer *t, const char *text, size_t len, const pulsar_text_span *spans,
                          uint32_t n_spans, std::vector<int> *out, char *err, size_t errlen) {
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
        const bpe_tokenizer::added *hit = NULL;
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

const char *bpe_tokenizer_token_bytes(const bpe_tokenizer *t, int id, size_t *len) {
    if (id < 0 || id >= t->n_tokens) return NULL;
    if (len) *len = t->bytes[(size_t)id].size();
    return t->bytes[(size_t)id].data();
}

int bpe_tokenizer_added_id(const bpe_tokenizer *t, const char *text) {
    for (const auto &a : t->added_tokens)
        if (a.text == text) return a.id;
    return -1;
}

const std::vector<int> &bpe_tokenizer_stop_ids(const bpe_tokenizer *t) { return t->stop_ids; }
