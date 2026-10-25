/* nemotron_parse_gate.cpp -- L321: Nemotron-3.5 Parse on a container, graded against the HF fp32 reference.  Needs
 * the GPU.
 *
 *   nemotron_parse_gate MODEL REFDIR [N [OUTDIR]]
 *
 * REFDIR is retriever/parse/parse_ref.py's output: per page <stem>.png (the input), <stem>.ids.npy (HF's greedy ids,
 * the prompt first) and <stem>.raw.txt (HF's batch_decode of them).  For the first N pages (default all), sorted:
 *   D1  the decode table: decode(HF's ids) == HF's text, byte for byte (the family's tokenizer contract);
 *   P1  pulsar_engine_parse_page on the PNG: the prompt is HF's, and the generated ids are compared with HF's --
 *       identical, or the first position they differ at -- and the text with HF's text;
 *   P2  a parse refuses a non-image and a prompt that is not the model's tokens, by name; a session is refused.
 * Prints per page the image, the prefix, the tokens and the times (vision / prefill / decode); OUTDIR gets each page's
 * text as <stem>.raw.txt (for retriever/parse/grade_markdown.py).  PULSAR_PARSE_DUMP=DIR
 * writes the first page's stages for retriever/parse/grade_stages.py.  Exit 0 = D1 and P2 pass and every page parses;
 * the agreement is graded by its numbers (rule 7). */
#include "families/nemotron_parse/nemotron_parse.h"

#include <dirent.h>
#include <stdarg.h>

#include <algorithm>
#include <string>
#include <vector>

static int n_fail;

static void check(bool ok, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "  %s ", ok ? "ok  " : "FAIL");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    if (!ok) n_fail++;
}

static bool slurp(const std::string &p, std::string *out) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) return false;
    char b[1 << 16];
    size_t n;
    out->clear();
    while ((n = fread(b, 1, sizeof b, f)) > 0) out->append(b, n);
    fclose(f);
    return true;
}

/* a 1-D int32 .npy (numpy's own header) */
static bool npy_i32(const std::string &p, std::vector<int> *v) {
    std::string s;
    if (!slurp(p, &s) || s.size() < 10 || s.compare(0, 6, "\x93NUMPY") != 0) return false;
    const size_t hl = (unsigned char)s[8] | ((unsigned char)s[9] << 8), off = 10 + hl;
    if (s.find("'<i4'") == std::string::npos || off > s.size()) return false;
    v->resize((s.size() - off) / 4);
    memcpy(v->data(), s.data() + off, v->size() * 4);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: nemotron_parse_gate MODEL REFDIR [N [OUTDIR]]\n");
        return 2;
    }
    const std::string ref = argv[2];
    std::vector<std::string> stems;
    if (DIR *d = opendir(ref.c_str())) {
        while (dirent *de = readdir(d)) {
            const std::string n = de->d_name;
            if (n.size() > 4 && n.compare(n.size() - 4, 4, ".png") == 0) stems.push_back(n.substr(0, n.size() - 4));
        }
        closedir(d);
    }
    std::sort(stems.begin(), stems.end());
    if (argc > 3 && atoi(argv[3]) > 0) stems.resize(std::min(stems.size(), (size_t)atoi(argv[3])));
    const char *outdir = argc > 4 ? argv[4] : NULL;
    if (stems.empty()) {
        fprintf(stderr, "nemotron_parse_gate: no <stem>.png in %s\n", ref.c_str());
        return 2;
    }
    pulsar_engine_options opt = {};
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0 || !e) {
        fprintf(stderr, "nemotron_parse_gate: the engine did not open %s\n", argv[1]);
        return 1;
    }
    const pulsar_nemotron_parse_model *w = (const pulsar_nemotron_parse_model *)e->family_state;
    check(pulsar_engine_is_parser(e) && e->family->id == PULSAR_FAMILY_ID_NEMOTRON_PARSE, "a parser (%s)",
          pulsar_engine_family_name(e));

    fprintf(stderr, "D1 the decode table\n");
    size_t d1 = 0;
    std::vector<std::vector<int>> ref_ids(stems.size());
    std::vector<std::string> ref_txt(stems.size());
    for (size_t i = 0; i < stems.size(); i++) {
        if (!npy_i32(ref + "/" + stems[i] + ".ids.npy", &ref_ids[i]) || !slurp(ref + "/" + stems[i] + ".raw.txt", &ref_txt[i])) {
            fprintf(stderr, "nemotron_parse_gate: %s: no ids / text\n", stems[i].c_str());
            return 2;
        }
        d1 += pulsar_nemotron_parse_decode(w->vocab, ref_ids[i].data(), ref_ids[i].size()) == ref_txt[i];
    }
    check(d1 == stems.size(), "decode(HF's ids) == HF's text on %zu / %zu pages", d1, stems.size());

    fprintf(stderr, "P1 the pages\n");
    size_t same = 0, same_text = 0, parsed = 0;
    double agree = 0, toks = 0, ms = 0, dec_ms = 0;
    for (size_t i = 0; i < stems.size(); i++) {
        std::string png;
        slurp(ref + "/" + stems[i] + ".png", &png);
        const std::vector<int> &R = ref_ids[i];
        const size_t np = w->prompt_ids.size();
        if (i > 0) unsetenv("PULSAR_PARSE_DUMP");   /* the first page's stages only */
        pulsar_tokens ids = {};
        char *text = NULL, err[512] = "";
        pulsar_parse_stats st = {};
        const int rc = pulsar_engine_parse_page(e, (const uint8_t *)png.data(), png.size(), NULL, &ids, &text, &st, err,
                                                sizeof err);
        if (rc != 0) {
            check(false, "%s: %s", stems[i].c_str(), err);
            continue;
        }
        parsed++;
        size_t k = 0;
        while (k < (size_t)ids.len && k < R.size() && ids.v[k] == R[k]) k++;
        const bool eq = k == R.size() && k == (size_t)ids.len;
        same += eq;
        same_text += text && ref_txt[i] == text;
        const size_t gen_ref = R.size() > np ? R.size() - np : 0;
        agree += gen_ref ? (double)(k > np ? k - np : 0) / (double)gen_ref : 1.0;
        toks += st.generated_tokens;
        ms += st.vision_ms + st.prefill_ms + st.decode_ms;
        dec_ms += st.decode_ms;
        fprintf(stderr, "  %-44s %4ux%4u prefix %4u  %5u tok (HF %5zu)  %s  vision %5.0f prefill %5.0f decode %6.0f ms "
                        "(%4.0f tok/s)\n", stems[i].c_str(), st.image_w, st.image_h, st.prefix_rows, st.generated_tokens,
                gen_ref, eq ? "IDENTICAL       " : (std::string("diverges at ") + std::to_string(k > np ? k - np : 0)).c_str(),
                st.vision_ms, st.prefill_ms, st.decode_ms, st.generated_tokens / (st.decode_ms / 1e3));
        if (outdir && text) {
            if (FILE *f = fopen((std::string(outdir) + "/" + stems[i] + ".raw.txt").c_str(), "wb")) {
                fwrite(text, 1, strlen(text), f);
                fclose(f);
            }
        }
        free(text);
        pulsar_tokens_free(&ids);
    }
    check(parsed == stems.size(), "every page parsed (%zu / %zu)", parsed, stems.size());
    fprintf(stderr, "  info %zu / %zu pages token-identical to HF fp32 (%zu text-identical); mean agreeing prefix %.1f%% "
                    "of HF's generated tokens; %.1f s / page, decode %.0f tok/s\n", same, stems.size(), same_text,
            100.0 * agree / (double)stems.size(), ms / 1e3 / (double)stems.size(), toks / (dec_ms / 1e3));

    fprintf(stderr, "P2 refusals\n");
    {
        pulsar_tokens ids = {};
        char *text = NULL, err[512] = "";
        const uint8_t junk[16] = {1, 2, 3};
        check(pulsar_engine_parse_page(e, junk, sizeof junk, NULL, &ids, &text, NULL, err, sizeof err) != 0 && err[0],
              "a non-image: %s", err);
        std::string png;
        slurp(ref + "/" + stems[0] + ".png", &png);
        pulsar_parse_options po = {};
        po.prompt = "<s>hello";
        err[0] = 0;
        check(pulsar_engine_parse_page(e, (const uint8_t *)png.data(), png.size(), &po, &ids, &text, NULL, err,
                                       sizeof err) != 0 && err[0], "a prompt that is not the model's tokens: %s", err);
        pulsar_session *s = NULL;
        check(pulsar_session_create(&s, e, 4096) != 0 && !s, "a session is refused");
    }
    pulsar_engine_close(e);
    fprintf(stderr, "nemotron_parse_gate: %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}
