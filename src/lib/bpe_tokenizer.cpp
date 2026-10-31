/** @file
 *  The byte-level BPE tokenizer engine: added-token cut, NFC, pre-tokenizer patterns, merges and decode. */
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

/// A loaded byte-level BPE vocabulary (HF tokenizer.json or tiktoken ranks): the opaque handle of
/// bpe_tokenizer.h.  Immutable once a loader returns it.
struct bpe_tokenizer {
    /// The pre-tokenizer split this vocabulary runs (one of k_patterns).
    const bpe_pretok_pattern *pattern = nullptr;
    bool nfc = false;                     ///< runs are NFC-normalised first (HF's NFC normaliser)
    bool drop_zwnj = false;               ///< a U+200C right after '[' or '<' is dropped first (L321: HF's Replace pair)
    /// Size of the id space: ids are 0 .. n_tokens-1, every one assigned (finish_table asserts no holes).
    int n_tokens = 0;
    std::vector<std::string> bytes;       ///< id -> decoded bytes
    int byte_id[256];                     ///< raw byte -> its single-byte token
    /** HF merges: (a<<32|b) -> (rank, merged id).  Empty for a tiktoken configuration. */
    std::unordered_map<uint64_t, std::pair<int, int>> merges;
    /** tiktoken ranks: a BPE token's bytes -> its id (= its rank); the views point into `bytes`, which is
     *  never resized after the map is built.  Empty for an HF configuration. */
    std::unordered_map<std::string_view, int> rank_of;
    /** HF model.vocab with ignore_merges: a vocab token's bytes -> its id (views into `bytes`, as rank_of's). */
    std::unordered_map<std::string_view, int> vocab_of;
    /** The whole-piece lookup a piece takes before any merge (tokenizers' ignore_merges, tiktoken's encoder
     *  lookup): &rank_of (tiktoken), &vocab_of (HF with ignore_merges), or NULL (HF without). */
    const std::unordered_map<std::string_view, int> *whole = nullptr;
    /// An added (special) token: matched on the raw text, whole, before any split.
    /// `text` is its bytes, `id` its token id.
    struct added { std::string text; int id; };
    /** @var bpe_tokenizer::added::text
    *  The token's bytes. */
    /** @var bpe_tokenizer::added::id
    *  Its token id. */
    std::vector<added> added_tokens;      ///< longest first
    bool added_first[256] = {};           ///< first bytes that can open an added token
    /// The eos ids from the eos config (generation_config.json's eos_token_id); each is an added token.
    std::vector<int> stop_ids;
};

namespace {

/// Write "bpe tokenizer: <fmt...>" into `err` (when given) and fail.
/// @param err    the caller's error buffer, or NULL
/// @param errlen its size in bytes
/// @param fmt    printf format of the reason
/// @return false, always
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
/// @param cp_of receives, for each raw byte, the codepoint that stands for it in the vocabulary's text
void byte_level_map(uint32_t cp_of[256]) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        cp_of[b] = keep ? (uint32_t)b : (uint32_t)(256 + n++);
    }
}

/// Decode one UTF-8 sequence already validated as `n` bytes long.
/// @param p the sequence's lead byte
/// @param n its length in bytes (1..4)
/// @return the codepoint
uint32_t utf8_decode(const unsigned char *p, int n) {
    switch (n) {
    case 1: return p[0];
    case 2: return ((uint32_t)(p[0] & 0x1f) << 6) | (p[1] & 0x3f);
    case 3: return ((uint32_t)(p[0] & 0x0f) << 12) | ((uint32_t)(p[1] & 0x3f) << 6) | (p[2] & 0x3f);
    default: return ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3f) << 12) |
                    ((uint32_t)(p[2] & 0x3f) << 6) | (p[3] & 0x3f);
    }
}

/// Append `cp`'s UTF-8 encoding to `s` (no validation: `cp` is a scalar value from a decode).
/// @param s  the string appended to
/// @param cp the codepoint
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

/// Binary search of a sorted table of inclusive ranges {lo, hi, value}.
/// @param t  the table (bpe_unicode_tables.inc), sorted, ranges disjoint
/// @param cp the codepoint looked up
/// @return the value of the range holding `cp`, or 0
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

/// @param cp a codepoint
/// @return its CL_* class flags (0 for an unlisted codepoint)
uint32_t cls(uint32_t cp) { return range_lookup(bpe_uni_class, cp); }
/// @param cp a codepoint
/// @return its canonical combining class (0 = a starter)
uint32_t ccc(uint32_t cp) { return cp < 0x300 ? 0 : range_lookup(bpe_uni_ccc, cp); }

/// Hangul syllable algorithmic (de)composition constants (Unicode ch. 3.12): first syllable, leading
/// consonant, vowel and trailing consonant bases.
constexpr uint32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
/// Hangul jamo counts: leading, vowel, trailing (incl. none), per-leading (N) and syllable (S) totals.
constexpr uint32_t LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = LCount * NCount;

/** @var LBase
*  First leading consonant jamo (U+1100). */
/** @var VBase
*  First vowel jamo (U+1161). */
/** @var TBase
*  One before the first trailing consonant jamo (U+11A7: trailing index 0 = none). */
/** @var VCount
*  Vowel jamo count. */
/** @var TCount
*  Trailing consonant count, "none" included. */
/** @var NCount
*  Syllables per leading consonant (VCount * TCount). */
/** @var SCount
*  Precomposed syllable count (LCount * NCount). */
/// Append `cp`'s full canonical decomposition (Hangul algorithmically, the rest from bpe_uni_decomp).
/// @param cp  the codepoint
/// @param out receives the decomposition (`cp` itself when it has none)
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
/// @param a a starter
/// @param b the codepoint following it
/// @return their primary composite, or 0
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
/// @param cps the codepoints, replaced by their NFC form
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

/// @param c a codepoint
/// @return `c` with ASCII A-Z lowered; anything else unchanged
uint32_t ascii_lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

/* (?i:'s|'t|'re|'ve|'m|'ll|'d) at i: its end, or i.  Onig's case folding also
 * takes U+017F (LONG S) for s -- measured, and the only non-ASCII fold that
 * lands here. */
/// @param s the run's codepoints
/// @param n their count
/// @param i the position tried
/// @return the contraction's end, or `i` when none matches
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
/// @param s  the run's codepoints
/// @param cl their CL_* class flags
/// @param n  their count
/// @param i  the position tried
/// @return the match's end, or `i`
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
/// @param s  the run's codepoints
/// @param cl their CL_* class flags
/// @param n  their count
/// @param i  the position the next piece starts at (< n)
/// @return the piece's end (> i)
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

/* The cased-word family of splits: o200k (tiktoken's o200k_base, Inkling) and tekken (Mistral's Tekken,
 * Nemotron-H) share every alternative but two -- whether a word takes a trailing contraction C, and how many
 * digits \p{N} takes (D):
 * [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+C
 *   |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*C
 *   |\p{N}{1,D}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * o200k: C = (?i:'s|'t|'re|'ve|'m|'ll|'d)?, D = 3; tekken: no C, D = 1.
 * The two word alternatives backtrack: the optional prefix is tried taken first, then not; the greedy
 * upper-class run gives back one codepoint at a time until the lower class can start (Lm, Lo and M are in
 * both classes). */
/// @tparam CONTRACTIONS the word alternatives take a trailing contraction (o200k)
/// @tparam DIGITS       the most digits one \\p{N} piece takes
/// @param s  the run's codepoints
/// @param cl their CL_* class flags
/// @param n  their count
/// @param i  the position the next piece starts at (< n)
/// @return the piece's end (> i)
template <bool CONTRACTIONS, size_t DIGITS>
size_t cased_word_match_at(const uint32_t *s, const uint32_t *cl, size_t n, size_t i) {
    constexpr uint32_t UP = CL_LU | CL_LT | CL_LM | CL_LO | CL_M;   /* [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] */
    constexpr uint32_t LO = CL_LL | CL_LM | CL_LO | CL_M;           /* [\p{Ll}\p{Lm}\p{Lo}\p{M}] */
    const bool prefix = s[i] != '\r' && s[i] != '\n' && !(cl[i] & (CL_L | CL_N));
    auto run = [&](size_t k, uint32_t mask) {
        while (k < n && (cl[k] & mask)) k++;
        return k;
    };
    auto tail = [&](size_t k) { return CONTRACTIONS ? contraction_at(s, n, k) : k; };
    /* UP* LO+ from k: 0 = no match */
    auto word_lower = [&](size_t k) -> size_t {
        const size_t j0 = run(k, UP);
        for (size_t j = j0 + 1; j-- > k;)
            if (j < n && (cl[j] & LO)) return tail(run(j, LO));
        return 0;
    };
    /* UP+ LO* from k: 0 = no match */
    auto word_upper = [&](size_t k) -> size_t {
        if (k >= n || !(cl[k] & UP)) return 0;
        return tail(run(run(k, UP), LO));
    };
    if (prefix && i + 1 < n)
        if (const size_t e = word_lower(i + 1)) return e;
    if (const size_t e = word_lower(i)) return e;
    if (prefix && i + 1 < n)
        if (const size_t e = word_upper(i + 1)) return e;
    if (const size_t e = word_upper(i)) return e;
    /* \p{N}{1,D} */
    if (cl[i] & CL_N) {
        size_t k = i + 1;
        while (k < n && k < i + DIGITS && (cl[k] & CL_N)) k++;
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

/* o200k -- tiktoken's o200k_base split (Inkling): contractions, \p{N}{1,3} */
/// @param s  the run's codepoints
/// @param cl their CL_* class flags
/// @param n  their count
/// @param i  the position the next piece starts at (< n)
/// @return the piece's end (> i)
size_t o200k_match_at(const uint32_t *s, const uint32_t *cl, size_t n, size_t i) {
    return cased_word_match_at<true, 3>(s, cl, n, i);
}

/* tekken -- Mistral's Tekken split (Nemotron-H): no contractions, \p{N} one digit.  tokenizer_config.json's
 * fix_mistral_regex names this same pattern (transformers' replacement string is byte-equal to the stored one), so
 * the flag changes nothing. */
/// @param s  the run's codepoints
/// @param cl their CL_* class flags
/// @param n  their count
/// @param i  the position the next piece starts at (< n)
/// @return the piece's end (> i)
size_t tekken_match_at(const uint32_t *s, const uint32_t *cl, size_t n, size_t i) {
    return cased_word_match_at<false, 1>(s, cl, n, i);
}

/// The qwen35 Split regex exactly as tokenizer.json spells it (bpe_pretok_by_regex matches on it).
const char kQwen35Pattern[] =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
/// The o200k Split regex exactly as the checkpoint spells it.
const char kO200kPattern[] =
    R"re([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";

/// The tekken Split regex exactly as tokenizer.json spells it.
const char kTekkenPattern[] =
    R"re([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";

/// The registry: every pre-tokenizer split this engine implements (name, regex, matcher).
const bpe_pretok_pattern k_patterns[] = {
    {"qwen35", kQwen35Pattern, qwen35_match_at},
    {"o200k", kO200kPattern, o200k_match_at},
    {"tekken", kTekkenPattern, tekken_match_at},
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

/// One symbol of a piece during BPE, stored at the byte offset it starts at; the live symbols form a
/// doubly linked list through prev / next.
struct sym {
    /// The symbol's token id.
    int id;
    /// The previous live symbol's start offset, or -1.
    long prev;
    /// The next live symbol's start offset, or -1 (this symbol's bytes end there).
    long next;
    /// Merged into its left neighbour: no longer in the list.
    bool gone;
};

/// A candidate merge in the priority queue (entries go stale; bpe() re-checks before applying one).
struct merge_item {
    /// The merge's rank (lower merges first).
    int rank;
    /// Start offset of the pair's left symbol.
    size_t pos;
    /// The merged token's id.
    int new_id;
    /// @param o the other entry
    /// @return true when this entry pops after `o` (higher rank, or same rank and later position)
    bool operator<(const merge_item &o) const {   /* max-heap pops lowest rank, then lowest pos */
        if (rank != o.rank) return rank > o.rank;
        return pos > o.pos;
    }
};

/* The merge of the symbols at w[l] and w[r] (adjacent; r's bytes end at `end`): (rank, merged id), or
 * false. */
/// @param t    the tokenizer
/// @param p    the piece's bytes
/// @param w    the piece's symbols
/// @param l    the left symbol's offset
/// @param r    the right symbol's offset
/// @param end  one past the right symbol's last byte
/// @param rank receives the merge's rank
/// @param id   receives the merged token's id
/// @return true when the pair merges
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

/// BPE one pre-tokenized piece: lowest-rank adjacent pair first, leftmost among equals, until none merges
/// (a piece that is whole in `t->whole` is that token without merging).
/// @param t   the tokenizer
/// @param p   the piece's bytes
/// @param n   their count (>= 1)
/// @param out receives the piece's token ids
void bpe(const bpe_tokenizer *t, const unsigned char *p, size_t n, std::vector<int> *out) {
    if (n == 1) {
        out->push_back(t->byte_id[p[0]]);
        return;
    }
    if (t->whole) {   /* a piece that is a token is that token */
        auto it = t->whole->find(std::string_view((const char *)p, n));
        if (it != t->whole->end()) {
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
/// @param t    the tokenizer
/// @param text the run's bytes (valid UTF-8: the caller validated it)
/// @param len  their count
/// @param out  receives the run's token ids
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
    if (t->drop_zwnj) {
        /* the Nemotron-3.5 encoders' normaliser: Replace "[\u200c" -> "[" then "<\u200c" -> "<" (HF's non-overlapping
         * left-to-right scans; neither can make a match for the other), i.e. a U+200C goes when the code point
         * BEFORE it in the run as written is '[' or '<' */
        size_t o = 0;
        for (size_t i = 0; i < cps.size(); i++)
            if (!(cps[i] == 0x200Cu && i > 0 && (cps[i - 1] == '[' || cps[i - 1] == '<'))) cps[o++] = cps[i];
        cps.resize(o);
    }
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

/// Refuse unless tokenizer.json section `v` dumps (pyjson_dump) to exactly `expect`.
/// @param v      the section, or NULL when absent (dumps as "(missing)")
/// @param expect the one spelling this engine implements
/// @param what   the section's name, for the message
/// @param err    the error buffer
/// @param errlen its size
/// @return true when it matches
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
/// @param a       the added token's tokenizer.json object
/// @param content its text
/// @param err     the error buffer
/// @param errlen  its size
/// @return true when single_word, lstrip, rstrip and normalized are all false and the text is not empty
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
/// @param t               the tokenizer being built (its added_tokens already filled)
/// @param entries         every (id, bytes) of the vocabulary, added tokens included; moved from
/// @param max_id          the largest id among them
/// @param eos_config_json the eos config's JSON text (generation_config.json)
/// @param eos_len         its length in bytes
/// @param err             the error buffer
/// @param errlen          its size
/// @return true when the table is complete and consistent
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
/// @param p   the base64 text
/// @param n   its length (a multiple of 4)
/// @param out receives the decoded bytes
/// @return true when `p` is valid standard base64 ('=' padding only at the end)
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

/* The HF pipelines the loader implements, one per registered pattern it serves: every tokenizer.json section
 * around the Split, as the checkpoint's file spells it (pyjson_dump form).  Options that move only offsets (a
 * ByteLevel step's trim_offsets; the post-processor's and decoder's add_prefix_space / use_regex, which read
 * offsets and decode text, never ids) are part of the spelling, so a file is one row or refused -- the gates grade
 * each row's ids against HF tokenizers. */
/// One implemented HF pipeline: the spelling of each tokenizer.json section around the Split.
struct hf_pipeline {
    const char *pattern;         ///< the registered pattern's name
    const char *normalizer;      ///< the normaliser (NFC or none)
    bool nfc;                    ///< ... as the engine runs it
    const char *byte_level;      ///< the ByteLevel step after the Split
    /// The decoder section's spelling.
    const char *decoder;
    /// The post_processor section's spelling.
    const char *post_processor;
    const char *affix;           ///< model.continuing_subword_prefix and .end_of_word_suffix
    bool ignore_merges;          ///< a piece found whole in the vocab is that token (tokenizers' BPE::ignore_merges)
    bool drop_zwnj;              ///< the normaliser is the Replace pair dropping U+200C after '[' / '<' (L321)
};
/// The implemented pipelines: a tokenizer.json must spell one row exactly or it is refused.
const hf_pipeline k_hf_pipelines[] = {
    {"qwen35", R"({"type": "NFC"})", true,
     R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})",
     R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})",
     R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})", "\"\"", false,
     false},
    {"tekken", "null", false,
     R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false})",
     R"({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true})",
     R"({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true})", "null", true,
     false},
    /* L321: the Nemotron-3.5 encoders (Embed / Rerank): tekken with a normaliser that drops a zero-width non-joiner
     * after '[' or '<' (their chat template writes one into client text so no added token can form there) */
    {"tekken",
     "{\"type\": \"Sequence\", \"normalizers\": [{\"type\": \"Replace\", \"pattern\": {\"String\": \"[\xe2\x80\x8c\"}, "
     "\"content\": \"[\"}, {\"type\": \"Replace\", \"pattern\": {\"String\": \"<\xe2\x80\x8c\"}, \"content\": \"<\"}]}",
     false,
     R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false})",
     R"({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true})",
     R"({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true})", "null", true,
     true},
};

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
     * regex names its pattern in the registry, and the pattern names the one pipeline around it this engine
     * implements (k_hf_pipelines). */
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
    /* A pattern may carry more than one row (L321: tekken with and without the encoders' Replace pair): the
     * file's normaliser picks among them; a pattern none of whose rows spells it is refused below, by section. */
    const hf_pipeline *pl = NULL;
    std::string norm_got;
    if (doc.get("normalizer")) pyjson_dump(*doc.get("normalizer"), &norm_got);
    for (const hf_pipeline &x : k_hf_pipelines)
        if (!strcmp(x.pattern, pattern->name) && (!pl || norm_got == x.normalizer)) pl = &x;
    if (!pl) {
        refuse(err, errlen, "pattern \"%s\" has no HF pipeline row (k_hf_pipelines)", pattern->name);
        return NULL;
    }
    std::string pre_expect = R"({"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": )";
    pyjson_dump_string(pattern->regex, strlen(pattern->regex), &pre_expect);
    pre_expect += R"(}, "behavior": "Isolated", "invert": false}, )";
    pre_expect += pl->byte_level;
    pre_expect += "]}";
    if (!expect_dump(doc.get("normalizer"), pl->normalizer, "the normalizer", err, errlen) ||
        !expect_dump(pre, pre_expect.c_str(), "the pre-tokenizer", err, errlen) ||
        !expect_dump(doc.get("decoder"), pl->decoder, "the decoder", err, errlen) ||
        !expect_dump(doc.get("post_processor"), pl->post_processor, "the post-processor", err, errlen))
        return NULL;
    const pyjson_value *model = doc.get("model");
    if (!model || model->kind != pyjson_value::OBJ) {
        refuse(err, errlen, "tokenizer.json has no model object");
        return NULL;
    }
    const struct { const char *key, *value; } model_expect[] = {
        {"type", "\"BPE\""}, {"dropout", "null"}, {"unk_token", "null"},
        {"continuing_subword_prefix", pl->affix}, {"end_of_word_suffix", pl->affix},
        {"fuse_unk", "false"}, {"byte_fallback", "false"}, {"ignore_merges", pl->ignore_merges ? "true" : "false"},
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
    t->nfc = pl->nfc;
    t->drop_zwnj = pl->drop_zwnj;
    uint32_t cp_of[256];
    byte_level_map(cp_of);
    std::unordered_map<uint32_t, int> byte_of_cp;
    for (int b = 0; b < 256; b++) byte_of_cp[cp_of[b]] = b;

    std::unordered_map<std::string, int> id_of;
    id_of.reserve(vocab->o.size() * 2);
    int max_id = -1;
    std::vector<std::pair<int, std::string>> entries;
    entries.reserve(vocab->o.size() + added->a.size());
    std::unordered_map<int, size_t> vocab_at;   /* a vocab id -> its entry */
    vocab_at.reserve(vocab->o.size() * 2);
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
        vocab_at[id] = entries.size();
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
        /* an added token the vocab also holds (Nemotron-H's 1000 specials) is ONE entry: the same bytes, or refused */
        auto in_vocab = vocab_at.find(iid);
        if (in_vocab != vocab_at.end()) {
            if (entries[in_vocab->second].second != content->s) {
                refuse(err, errlen, "added token %s (id %d) is not the vocab's bytes at that id", content->s.c_str(), iid);
                delete t;
                return NULL;
            }
        } else {
            entries.emplace_back(iid, content->s);
        }
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
        /* tokenizers writes a merge as one "a b" string (byte-level text has no raw space) or an [a, b] pair */
        std::string a, b;
        if (m.kind == pyjson_value::STR) {
            const size_t sp = m.s.find(' ');
            if (sp != std::string::npos && m.s.find(' ', sp + 1) == std::string::npos) {
                a = m.s.substr(0, sp);
                b = m.s.substr(sp + 1);
            }
        } else if (m.kind == pyjson_value::ARR && m.a.size() == 2 && m.a[0].kind == pyjson_value::STR &&
                   m.a[1].kind == pyjson_value::STR) {
            a = m.a[0].s;
            b = m.a[1].s;
        }
        if (a.empty() || b.empty()) {
            refuse(err, errlen, "merge %zu is neither one \"a b\" string nor an [a, b] pair", r);
            delete t;
            return NULL;
        }
        auto ia = id_of.find(a), ib = id_of.find(b), im = id_of.find(a + b);
        if (ia == id_of.end() || ib == id_of.end() || im == id_of.end()) {
            refuse(err, errlen, "merge %zu (%s %s) names a token the vocab lacks", r, a.c_str(), b.c_str());
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
    if (pl->ignore_merges) {   /* the vocab's tokens, whole (views into t->bytes, fixed from here on) */
        t->vocab_of.reserve(vocab_at.size() * 2);
        for (const auto &v : vocab_at) t->vocab_of.emplace(std::string_view(t->bytes[(size_t)v.first]), v.first);
        t->whole = &t->vocab_of;
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
    t->whole = &t->rank_of;
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
