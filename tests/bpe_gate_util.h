/* bpe_gate_util.h -- what every tokenizer / chat gate shares (L295 F5): the check counters, the golden
 * readers, and the tokenizer sections graded against HF tokenizers' goldens -- T1 adversarial strings, T2
 * the client-span guard, T3 every scalar value in a per-gate context, T4 a corpus.  One gate per
 * translation unit includes it. */
#pragma once

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include "../src/lib/bpe_tokenizer.h"
#include "../src/lib/pyjson.h"
#include "../src/lib/sha1.hpp"

static int g_fail = 0;
static int g_checks = 0;

static void fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "  FAIL: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    g_fail++;
}

#define CHECK(cond, ...)                 \
    do {                                 \
        g_checks++;                      \
        if (!(cond)) fail(__VA_ARGS__);  \
    } while (0)

static bool read_file(const std::string &path, std::string *out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[1 << 16];
    size_t n;
    out->clear();
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    fclose(f);
    return true;
}

static pyjson_value load_json(const std::string &path) {
    std::string text;
    if (!read_file(path, &text)) {
        fprintf(stderr, "qwen_chat_gate: cannot read %s\n", path.c_str());
        exit(2);
    }
    pyjson_value v;
    char err[200];
    if (!pyjson_parse(text.data(), text.size(), &v, err, sizeof err)) {
        fprintf(stderr, "qwen_chat_gate: %s: %s\n", path.c_str(), err);
        exit(2);
    }
    return v;
}

static std::string sha1_hex(const std::string &s) {
    char hex[41];
    pulsar::Sha1::bytes_hex(s.data(), s.size(), hex);
    return hex;
}

static std::string ids_text(const std::vector<int> &ids) {
    std::string s;
    char b[16];
    for (int id : ids) {
        snprintf(b, sizeof b, "%d,", id);
        s += b;
    }
    return s;
}

static std::vector<int> json_ids(const pyjson_value &v) {
    std::vector<int> ids;
    for (const auto &x : v.a) ids.push_back(atoi(x.num.c_str()));
    return ids;
}

static std::vector<pulsar_text_span> json_spans(const pyjson_value &v) {
    std::vector<pulsar_text_span> spans;
    for (const auto &x : v.a) spans.push_back({(uint32_t)atoi(x.a[0].num.c_str()), (uint32_t)atoi(x.a[1].num.c_str())});
    return spans;
}

static std::vector<int> encode(const bpe_tokenizer *t, const std::string &text,
                               const std::vector<pulsar_text_span> *spans = NULL) {
    std::vector<int> ids;
    char err[200];
    if (!bpe_tokenizer_encode(t, text.data(), text.size(), spans ? spans->data() : NULL,
                               spans ? (uint32_t)spans->size() : 0, &ids, err, sizeof err))
        fail("encode refused: %s", err);
    return ids;
}

static std::string first_diff(const std::vector<int> &a, const std::vector<int> &b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) i++;
    char m[160];
    snprintf(m, sizeof m, "len %zu vs %zu, first difference at %zu (%d vs %d)", a.size(), b.size(), i,
             i < a.size() ? a[i] : -1, i < b.size() ? b[i] : -1);
    return m;
}

/* ---- the tokenizer sections ----------------------------------------------------- */

static void t1_adversarial(const bpe_tokenizer *t, const std::string &dir) {
    const pyjson_value v = load_json(dir + "/tok_adversarial.json");
    int bad = 0;
    for (const auto &c : v.a) {
        const std::string &text = c.get("text")->s;
        const std::vector<int> want = json_ids(*c.get("ids"));
        const std::vector<int> got = encode(t, text);
        g_checks++;
        if (got != want) {
            std::string q;
            pyjson_dump_string(text.data(), std::min<size_t>(text.size(), 80), &q);
            if (bad++ < 10) fail("T1 %s: %s", q.c_str(), first_diff(got, want).c_str());
        }
    }
    printf("T1 adversarial strings: %zu, %d mismatched\n", v.a.size(), bad);
}

static void t2_spans(const bpe_tokenizer *t, const std::string &dir) {
    const pyjson_value v = load_json(dir + "/tok_spans.json");
    int bad = 0, guarded = 0;
    for (const auto &c : v.a) {
        const std::string &text = c.get("text")->s;
        const std::vector<pulsar_text_span> spans = json_spans(*c.get("spans"));
        const std::vector<int> want = json_ids(*c.get("ids"));
        const std::vector<int> got = encode(t, text, &spans);
        g_checks++;
        if (got != want) {
            bad++;
            fail("T2 %s", first_diff(got, want).c_str());
        }
        guarded += encode(t, text) != got;
    }
    printf("T2 span guard: %zu cases (%d where the guard changes the ids), %d mismatched\n", v.a.size(), guarded,
           bad);
}

static void t3_sweep(const bpe_tokenizer *t, const std::string &dir,
                     std::string (*context)(const std::string &c)) {
    std::string tsv;
    if (!read_file(dir + "/tok_sweep.tsv", &tsv)) { fail("T3: no tok_sweep.tsv"); return; }
    int chunks = 0, bad = 0;
    long total = 0;
    for (size_t pos = 0; pos < tsv.size();) {
        const size_t nl = tsv.find('\n', pos);
        const std::string line = tsv.substr(pos, nl - pos);
        pos = nl + 1;
        int chunk = 0, n = 0;
        char sha[64];
        if (sscanf(line.c_str(), "%d\t%d\t%63s", &chunk, &n, sha) != 3) continue;
        std::string text;
        for (uint32_t cp = (uint32_t)chunk * 4096; cp < (uint32_t)chunk * 4096 + 4096; cp++) {
            if (cp >= 0xd800 && cp <= 0xdfff) continue;
            std::string c;
            if (cp < 0x80) c.push_back((char)cp);
            else if (cp < 0x800) { c.push_back((char)(0xc0 | (cp >> 6))); c.push_back((char)(0x80 | (cp & 0x3f))); }
            else if (cp < 0x10000) {
                c.push_back((char)(0xe0 | (cp >> 12))); c.push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
                c.push_back((char)(0x80 | (cp & 0x3f)));
            } else {
                c.push_back((char)(0xf0 | (cp >> 18))); c.push_back((char)(0x80 | ((cp >> 12) & 0x3f)));
                c.push_back((char)(0x80 | ((cp >> 6) & 0x3f))); c.push_back((char)(0x80 | (cp & 0x3f)));
            }
            text += context(c);
        }
        const std::vector<int> ids = encode(t, text);
        total += (long)ids.size();
        chunks++;
        g_checks++;
        if ((int)ids.size() != n || sha1_hex(ids_text(ids)) != sha) {
            bad++;
            fail("T3 chunk %d (U+%04X..): %zu ids vs HF %d", chunk, chunk * 4096, ids.size(), n);
        }
    }
    printf("T3 codepoint sweep: %d chunks, %ld tokens, %d mismatched\n", chunks, total, bad);
}

static void t4_calib(const bpe_tokenizer *t, const std::string &dir, const std::string &calib) {
    std::string tsv, rows;
    if (!read_file(dir + "/tok_calib.tsv", &tsv) || !read_file(calib, &rows)) {
        fail("T4: cannot read tok_calib.tsv or %s", calib.c_str());
        return;
    }
    std::vector<std::pair<int, std::string>> want;
    for (size_t pos = 0; pos < tsv.size();) {
        const size_t nl = tsv.find('\n', pos);
        int idx, n;
        char sha[64];
        if (sscanf(tsv.substr(pos, nl - pos).c_str(), "%d\t%d\t%63s", &idx, &n, sha) == 3) want.push_back({n, sha});
        pos = nl + 1;
    }
    int row = 0, bad = 0;
    long total = 0;
    for (size_t pos = 0; pos < rows.size();) {
        const size_t nl = rows.find('\n', pos);
        const std::string line = rows.substr(pos, nl - pos);
        pos = nl == std::string::npos ? rows.size() : nl + 1;
        if (line.empty()) continue;
        pyjson_value r;
        char err[200];
        if (!pyjson_parse(line.data(), line.size(), &r, err, sizeof err)) { fail("T4 row %d: %s", row, err); row++; continue; }
        const std::vector<int> ids = encode(t, r.get("text")->s);
        total += (long)ids.size();
        g_checks++;
        if (row >= (int)want.size() || (int)ids.size() != want[(size_t)row].first ||
            sha1_hex(ids_text(ids)) != want[(size_t)row].second) {
            bad++;
            fail("T4 row %d: %zu ids", row, ids.size());
        }
        row++;
    }
    CHECK(row == (int)want.size(), "T4: %d rows vs %zu golden", row, want.size());
    printf("T4 calibration corpus: %d rows, %ld tokens, %d mismatched\n", row, total, bad);
}
