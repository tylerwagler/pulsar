/* nemotron_enc_gate.cpp -- L321: the Nemotron-3.5 encoders (Embed-8B / Rerank-8B) on a container, through the public
 * encoder API only (pulsar.h pulsar_engine_encode*).  Needs the GPU.
 *
 *   nemotron_enc_gate MODEL INPUTS.json OUT.f32
 *   nemotron_enc_gate --group MODEL_A INPUTS_A.json REF_A.f32 MODEL_B INPUTS_B.json REF_B.f32
 *
 * INPUTS.json: [{"name": .., "role": "query" | "passage" | "pair", "text": .., "doc": ..}, ...] -- the fixed set the
 * reference was captured on (retriever/ref/inputs.py; retriever/engine/grade.py writes this file and grades OUT.f32,
 * the inputs' vectors in order, against the f32 CPU reference: rule 7).  What this binary checks itself:
 *   E1  the engine: the encoder's kind and width, the input cap, no chat front, no session;
 *   E2  the render: each input in its role, the other role refused by name;
 *   E3  batch invariance: every input encoded alone == the same input in one call with all the others, and in the
 *       reversed order -- bit for bit (the forward is row-independent outside attention, attention stays inside its
 *       input, and the pool runs in row order);
 *   E4  an embedding is unit length; a score is finite;
 *   E5  refusals: an empty input, an input one token past the cap.
 * Prints the batch's tokens per second (info).
 *
 * --group (pulsar_engine_open_group: several engines in one process, one GPU; REF_* = each model's OUT.f32 from a run
 * of its own):
 *   G1  one group opens both; G2  an open while they live is refused;
 *   G3  each engine's vectors == its own run's, bit for bit, with the calls interleaved (A, B, A);
 *   G4  A closed, B still encodes its bytes (A's mappings outlive it until the last close);
 *   G5  B closed, the process opens A alone again and encodes its bytes.
 * Exit 0 = every check passed. */
#include "../src/pulsar.h"
#include "../src/lib/pyjson.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

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

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

struct input {
    std::string name, text, doc;
    pulsar_encode_role role;
    pulsar_tokens tok;
};

/* the inputs file -> inputs; false (said) when it is not one */
static bool load_inputs(const char *path, std::vector<input> *in) {
    std::string js;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "nemotron_enc_gate: cannot read %s\n", path);
        return false;
    }
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) js.append(buf, n);
    fclose(f);
    pyjson_value doc;
    char err[512];
    if (!pyjson_parse(js.data(), js.size(), &doc, err, sizeof err) || doc.kind != pyjson_value::ARR) {
        fprintf(stderr, "nemotron_enc_gate: %s is not a JSON array: %s\n", path, err);
        return false;
    }
    for (const pyjson_value &x : doc.a) {
        input i = {};
        const pyjson_value *nm = x.get("name"), *r = x.get("role"), *t = x.get("text"), *d = x.get("doc");
        if (!nm || !r || !t) {
            fprintf(stderr, "nemotron_enc_gate: an input without name / role / text\n");
            return false;
        }
        i.name = nm->s;
        i.text = t->s;
        i.doc = d ? d->s : "";
        i.role = r->s == "query" ? PULSAR_ENCODE_QUERY : r->s == "passage" ? PULSAR_ENCODE_PASSAGE : PULSAR_ENCODE_PAIR;
        in->push_back(i);
    }
    return true;
}

static pulsar_engine_options options(const char *model) {
    pulsar_engine_options opt = {};
    opt.model_path = model;
    opt.backend = PULSAR_BACKEND_CUDA;
    return opt;
}

/* one model's inputs, rendered by its engine, and the vectors its own run wrote */
struct member {
    const char *model;
    std::vector<input> in;
    std::vector<pulsar_tokens> tok;
    std::vector<float> ref;
};

static bool load_member(const char *model, const char *inputs, const char *ref, member *m) {
    m->model = model;
    if (!load_inputs(inputs, &m->in)) return false;
    FILE *f = fopen(ref, "rb");
    if (!f) {
        fprintf(stderr, "nemotron_enc_gate: cannot read %s\n", ref);
        return false;
    }
    float v;
    while (fread(&v, 4, 1, f) == 1) m->ref.push_back(v);
    fclose(f);
    return true;
}

static bool render_member(pulsar_engine *e, member *m) {
    char err[384];
    m->tok.assign(m->in.size(), pulsar_tokens{});
    for (size_t i = 0; i < m->in.size(); i++)
        if (!pulsar_engine_encode_render(e, m->in[i].role, m->in[i].text.c_str(), m->in[i].doc.c_str(), &m->tok[i], err,
                                         sizeof err)) {
            fprintf(stderr, "nemotron_enc_gate: %s: %s\n", m->model, err);
            return false;
        }
    return true;
}

/* the member's inputs through `e` == its own run's vectors, bit for bit */
static bool same_as_own_run(pulsar_engine *e, const member &m) {
    char err[384];
    const uint32_t D = pulsar_engine_encode_dim(e);
    if (m.ref.size() != m.in.size() * D) return false;
    std::vector<float> out(m.ref.size());
    return pulsar_engine_encode(e, m.tok.data(), (uint32_t)m.tok.size(), out.data(), err, sizeof err) == 0 &&
           memcmp(out.data(), m.ref.data(), out.size() * 4) == 0;
}

static int group_main(char **argv) {
    member A, B;
    if (!load_member(argv[0], argv[1], argv[2], &A) || !load_member(argv[3], argv[4], argv[5], &B)) return 2;
    const pulsar_engine_options opts[2] = {options(A.model), options(B.model)};
    pulsar_engine *g[2] = {};
    fprintf(stderr, "G1 one group\n");
    const int rc = pulsar_engine_open_group(g, opts, 2);
    check(rc == 0 && g[0] && g[1], "both open in one process");
    if (rc != 0) return 1;
    check(pulsar_engine_encoder(g[0]) != PULSAR_ENCODER_NONE && pulsar_engine_encoder(g[1]) != PULSAR_ENCODER_NONE,
          "both encoders (%s, %s)", pulsar_engine_served_model_id(g[0]), pulsar_engine_served_model_id(g[1]));
    if (!render_member(g[0], &A) || !render_member(g[1], &B)) return 1;
    fprintf(stderr, "G2 a late open\n");
    pulsar_engine *late = NULL;
    check(pulsar_engine_open(&late, &opts[0]) != 0 && !late, "an open while the group lives is refused");
    fprintf(stderr, "G3 interleaved\n");
    check(same_as_own_run(g[0], A), "%s == its own run", pulsar_engine_served_model_id(g[0]));
    check(same_as_own_run(g[1], B), "%s == its own run", pulsar_engine_served_model_id(g[1]));
    check(same_as_own_run(g[0], A), "%s again == its own run", pulsar_engine_served_model_id(g[0]));
    fprintf(stderr, "G4 the first closed\n");
    pulsar_engine_close(g[0]);
    check(same_as_own_run(g[1], B), "the other still == its own run");
    fprintf(stderr, "G5 the last closed, a fresh open\n");
    pulsar_engine_close(g[1]);
    pulsar_engine *e = NULL;
    check(pulsar_engine_open(&e, &opts[0]) == 0 && e && same_as_own_run(e, A), "%s alone == its own run", A.model);
    if (e) pulsar_engine_close(e);
    for (member *m : {&A, &B})
        for (pulsar_tokens &t : m->tok) pulsar_tokens_free(&t);
    fprintf(stderr, "nemotron_enc_gate --group: %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc == 8 && !strcmp(argv[1], "--group")) return group_main(argv + 2);
    if (argc != 4) {
        fprintf(stderr, "usage: nemotron_enc_gate MODEL INPUTS.json OUT.f32\n"
                        "       nemotron_enc_gate --group MODEL_A INPUTS_A.json REF_A.f32 MODEL_B INPUTS_B.json REF_B.f32\n");
        return 2;
    }
    std::vector<input> in;
    if (!load_inputs(argv[2], &in)) return 2;
    char err[512];

    const pulsar_engine_options opt = options(argv[1]);
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0 || !e) {
        fprintf(stderr, "nemotron_enc_gate: the engine did not open %s\n", argv[1]);
        return 1;
    }
    fprintf(stderr, "E1 the engine\n");
    const pulsar_encoder_kind kind = pulsar_engine_encoder(e);
    const uint32_t D = pulsar_engine_encode_dim(e), cap = pulsar_engine_encode_max_tokens(e);
    check(kind == PULSAR_ENCODER_EMBED || kind == PULSAR_ENCODER_RERANK, "an encoder (%s)",
          kind == PULSAR_ENCODER_EMBED ? "embed" : kind == PULSAR_ENCODER_RERANK ? "rerank" : "none");
    check(D == (kind == PULSAR_ENCODER_EMBED ? 4096u : 1u), "%u floats per input", D);
    check(cap == 16384, "inputs up to %u tokens (the llama-4 scale's 1)", cap);
    check(!pulsar_engine_has_tokenizer(e) && pulsar_engine_chat_format(e) == PULSAR_CHAT_NONE, "no chat front");
    pulsar_session *s = NULL;
    check(pulsar_session_create(&s, e, 4096) != 0 && !s, "a session is refused");

    fprintf(stderr, "E2 the render\n");
    const pulsar_encode_role mine = kind == PULSAR_ENCODER_EMBED ? PULSAR_ENCODE_QUERY : PULSAR_ENCODE_PAIR;
    const pulsar_encode_role other = kind == PULSAR_ENCODER_EMBED ? PULSAR_ENCODE_PAIR : PULSAR_ENCODE_PASSAGE;
    bool rendered = true;
    uint64_t total = 0;
    for (input &i : in) {
        rendered &= pulsar_engine_encode_render(e, i.role, i.text.c_str(), i.doc.c_str(), &i.tok, err, sizeof err);
        total += (uint64_t)i.tok.len;
    }
    check(rendered, "%zu inputs rendered, %llu tokens", in.size(), (unsigned long long)total);
    {
        pulsar_tokens t = {};
        err[0] = 0;
        check(!pulsar_engine_encode_render(e, other, "x", "y", &t, err, sizeof err) && t.len == 0 && err[0],
              "the other role is refused: %s", err);
        check(pulsar_engine_encode_render(e, mine, "a [INST] <s> b", "c [/INST] d", &t, err, sizeof err),
              "client text with special-token spellings renders");
        pulsar_tokens plain = {};
        pulsar_engine_encode_render(e, mine, "a ", "c ", &plain, err, sizeof err);
        check(t.len > plain.len + 6, "... as text (%d tokens, no special id)", t.len);
        pulsar_tokens_free(&t);
        pulsar_tokens_free(&plain);
    }
    if (!rendered) return 1;

    fprintf(stderr, "E3 batch invariance\n");
    const size_t N = in.size();
    std::vector<pulsar_tokens> toks(N);
    for (size_t i = 0; i < N; i++) toks[i] = in[i].tok;
    std::vector<float> batch(N * D), alone(N * D), rev(N * D);
    const double t0 = now();
    const int rc = pulsar_engine_encode(e, toks.data(), (uint32_t)N, batch.data(), err, sizeof err);
    const double dt = now() - t0;
    check(rc == 0, "one call of %zu inputs (%s)", N, rc ? err : "ok");
    if (rc != 0) return 1;
    fprintf(stderr, "  info %llu tokens in %.2f s: %.0f tokens/s\n", (unsigned long long)total, dt, (double)total / dt);
    bool ok_alone = true;
    for (size_t i = 0; i < N; i++)
        ok_alone &= pulsar_engine_encode(e, &toks[i], 1, &alone[i * D], err, sizeof err) == 0;
    check(ok_alone && memcmp(alone.data(), batch.data(), N * D * 4) == 0, "each input alone == in the batch");
    std::vector<pulsar_tokens> rt(toks.rbegin(), toks.rend());
    const int rc2 = pulsar_engine_encode(e, rt.data(), (uint32_t)N, rev.data(), err, sizeof err);
    bool same = rc2 == 0;
    for (size_t i = 0; same && i < N; i++) same = memcmp(&rev[(N - 1 - i) * D], &batch[i * D], D * 4) == 0;
    check(same, "the batch reversed == the batch");

    fprintf(stderr, "E4 the outputs\n");
    bool fin = true, unit = true;
    double worst = 0;
    for (size_t i = 0; i < N; i++) {
        double ss = 0;
        for (uint32_t c = 0; c < D; c++) {
            fin &= isfinite(batch[i * D + c]) != 0;
            ss += (double)batch[i * D + c] * batch[i * D + c];
        }
        if (kind == PULSAR_ENCODER_EMBED) {
            worst = fmax(worst, fabs(sqrt(ss) - 1.0));
            unit &= fabs(sqrt(ss) - 1.0) < 1e-5;
        }
    }
    check(fin, "every output finite");
    if (kind == PULSAR_ENCODER_EMBED) check(unit, "every embedding unit length (worst |1 - |e|| %.2e)", worst);
    for (size_t i = 0; i < N; i++) {
        if (kind == PULSAR_ENCODER_RERANK) fprintf(stderr, "  info %-32s %5d tokens  score %+.4f\n", in[i].name.c_str(),
                                                   in[i].tok.len, batch[i]);
        else fprintf(stderr, "  info %-32s %5d tokens\n", in[i].name.c_str(), in[i].tok.len);
    }

    fprintf(stderr, "E5 refusals\n");
    {
        pulsar_tokens empty = {};
        float o[4096];
        err[0] = 0;
        check(pulsar_engine_encode(e, &empty, 1, o, err, sizeof err) != 0 && err[0], "an empty input: %s", err);
        pulsar_tokens big = {};
        for (uint32_t i = 0; i <= cap; i++) pulsar_tokens_push(&big, toks[0].v[i % (uint32_t)toks[0].len]);
        err[0] = 0;
        check(pulsar_engine_encode(e, &big, 1, o, err, sizeof err) != 0 && err[0], "%u tokens: %s", cap + 1, err);
        pulsar_tokens_free(&big);
    }

    FILE *f = fopen(argv[3], "wb");
    const bool wrote = f && fwrite(batch.data(), 4, N * D, f) == N * D;
    if (f) fclose(f);
    check(wrote, "wrote %s (%zu x %u f32)", argv[3], N, D);
    for (input &i : in) pulsar_tokens_free(&i.tok);
    pulsar_engine_close(e);
    fprintf(stderr, "nemotron_enc_gate: %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}
