/* L284 #3: a family pool's bank physical residency, through the public API -- what the server's 2b guard does to
 * an idle bank (server spill_bank / bank_restore_spilled), graded on the real model.
 *
 *   PULSAR_MSEQ_BANKS=4 ./tests/bank_residency_gate <model> [G]
 *
 * Bank 1 and bank 2 prefill the same G tokens of the story prompt (G a multiple of 128, so a checkpoint stands at
 * the prompt's end).  Bank 1 is persisted as a segment [0, G), bank 0 installed, then:
 *   R1  free_physical(live bank 0) refuses; free_physical(1) releases bank 1: is_evicted, its touched KV 0, the
 *       session's touched KV drops by exactly what bank 1 held, and MemAvailable rises by at least 3/4 of it
 *       within 60 s (GB10 hands a cudaFree'd managed range back to the kernel asynchronously, at ~10 MiB/s)
 *   R2  alloc_physical(1) re-backs it, twice (idempotent): not evicted; the segment loads it back to G and its
 *       touched KV is what it was
 *   R3  bank 1 (freed and restored) and bank 2 (never freed) each sync the same extension Q of the prompt and
 *       decode 16 greedy tokens: the logits after the sync and after every step are byte-identical
 * DeepSeek's pool has its own gate (bank_evict_restore_gate); this one runs on any family that declares banks and
 * segments. */
#include "pulsar.h"
#include "gate_entry.h"
#include "gate_util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) GATE_CHECK(n_fail, c, __VA_ARGS__)

static long long mem_available_kb(void) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return -1;
    char line[256];
    long long kb = -1;
    while (fgets(line, sizeof(line), fp))
        if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
    fclose(fp);
    return kb;
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    const long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (buf && fread(buf, 1, (size_t)n, fp) != (size_t)n) { free(buf); buf = NULL; }
    if (buf) buf[n] = '\0';
    fclose(fp);
    return buf;
}

static std::vector<float> logits_of(pulsar_session *s, int W) {
    std::vector<float> v((size_t)W);
    if (pulsar_session_copy_logits(s, v.data(), W) != W) v.clear();
    return v;
}

static int argmax(const std::vector<float> &v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); i++) if (v[i] > v[(size_t)best]) best = (int)i;
    return best;
}

/* The installed bank syncs Q and decodes `steps` greedy tokens: the logits after the sync and after each step. */
static std::vector<std::vector<float>> continue_greedy(pulsar_session *s, const pulsar_tokens *Q, int steps, int W,
                                                       std::vector<int> *toks) {
    std::vector<std::vector<float>> out;
    char err[256] = "";
    if (pulsar_session_sync(s, Q, err, sizeof(err)) != 0) {
        fprintf(stderr, "bank-residency: sync(Q): %s\n", err);
        return out;
    }
    out.push_back(logits_of(s, W));
    for (int i = 0; i < steps; i++) {
        const int t = argmax(out.back());
        toks->push_back(t);
        if (pulsar_session_eval(s, t, err, sizeof(err)) != 0) {
            fprintf(stderr, "bank-residency: eval: %s\n", err);
            out.clear();
            return out;
        }
        out.push_back(logits_of(s, W));
    }
    return out;
}

int GATE_ENTRY(int argc, char **argv) {
    n_fail = 0;
    if (argc < 2) { fprintf(stderr, "usage: %s <model> [G]\n", argv[0]); return 2; }
    int G = argc > 2 ? atoi(argv[2]) : 16384;
    G = G / 128 * 128;
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "bank-residency: %s did not open\n", argv[1]); return 2; }
    const int W = pulsar_engine_logits_width(e);

    char *text = read_file("tests/long_context_story_prompt.txt");
    pulsar_tokens all = {0};
    if (text) pulsar_tokenize_text(e, text, &all);
    free(text);
    if (all.len < G + 64) G = (all.len - 64) / 128 * 128;
    if (G < 128) {
        fprintf(stderr, "bank-residency: tests/long_context_story_prompt.txt gave %d tokens\n", all.len);
        pulsar_tokens_free(&all);
        gate_engine_close(e);
        return 2;
    }
    pulsar_tokens P = all, Q = all;
    P.len = G;
    Q.len = G + 37;   /* a returning conversation: its history and a new turn */
    printf("bank-residency: G %d, Q %d tokens, 4 banks\n", G, Q.len);

    pulsar_engine_set_bank_pool(4);
    pulsar_session *s = NULL;
    char err[256] = "";
    if (pulsar_session_create(&s, e, G + 4096) != 0 || pulsar_session_bank_count(s) != 4) {
        CHECK(false, "a 4-bank session at ctx %d", G + 4096);
        if (s) pulsar_session_free(s);
        pulsar_tokens_free(&all);
        gate_engine_close(e);
        return 1;
    }
    bool ok = true;
    for (uint32_t b : {1u, 2u}) {
        ok = ok && pulsar_session_bank_state_restore(s, b) && pulsar_session_sync(s, &P, err, sizeof(err)) == 0;
        if (ok) pulsar_session_bank_state_save(s, b);
    }
    CHECK(ok, "banks 1 and 2 prefilled G: %s", err);

    /* the spill: bank 1 installed, its chain persisted, its carry saved, bank 0 installed */
    FILE *seg = tmpfile();
    const uint64_t seg_bytes = ok ? pulsar_session_segment_bytes(s, 0, G) : 0;
    ok = ok && seg && pulsar_session_bank_state_restore(s, 1) && pulsar_session_checkpoint_best(s, G) == G &&
         pulsar_session_save_segment(s, seg, 0, G, NULL, err, sizeof(err)) == 0;
    if (ok) pulsar_session_bank_state_save(s, 1);
    ok = ok && pulsar_session_bank_state_restore(s, 0);
    CHECK(ok, "bank 1 persisted as [0, %d) (%.1f MiB): %s", G, seg_bytes / 1048576.0, err);

    /* R1 */
    const uint64_t t1 = pulsar_session_bank_touched_kv_bytes(s, 1), all0 = pulsar_session_touched_kv_bytes(s);
    CHECK(!pulsar_session_bank_free_physical(s, 0) && !pulsar_session_bank_is_evicted(s, 0),
          "R1 free_physical refuses the live bank 0 and frees nothing");
    const long long ma0 = mem_available_kb();
    const bool freed = ok && pulsar_session_bank_free_physical(s, 1);
    /* GB10 returns a cudaFree'd managed range to the kernel ASYNCHRONOUSLY and slowly (measured 2026-10-07: +0 at
     * once, +50 MiB at 5 s, +223 MiB of 227.5 at 30 s): watch MemAvailable for up to 60 s */
    long long ma1 = mem_available_kb();
    int waited_ms = 0;
    for (; waited_ms < 60000 && ma0 >= 0 && (double)(ma1 - ma0) * 1024.0 < 0.75 * (double)t1; waited_ms += 250) {
        usleep(250 * 1000);
        const long long m = mem_available_kb();
        if (m > ma1) ma1 = m;
    }
    const uint64_t t1_after = pulsar_session_bank_touched_kv_bytes(s, 1), all1 = pulsar_session_touched_kv_bytes(s);
    CHECK(freed && pulsar_session_bank_is_evicted(s, 1) && t1_after == 0 && all0 - all1 == t1 && t1 > 0,
          "R1 bank 1 freed: evicted, touched %.1f -> %.1f MiB, session %.1f -> %.1f MiB", t1 / 1048576.0,
          t1_after / 1048576.0, all0 / 1048576.0, all1 / 1048576.0);
    const double rise_mib = (double)(ma1 - ma0) / 1024.0;
    CHECK(ma0 >= 0 && rise_mib >= 0.75 * (double)t1 / 1048576.0,
          "R1 MemAvailable rose %.1f MiB within %.2f s (bank 1's touched KV %.1f MiB)", rise_mib, waited_ms / 1000.0,
          t1 / 1048576.0);
    CHECK(!pulsar_session_bank_is_evicted(s, 2), "R1 bank 2 untouched by bank 1's eviction");

    /* R2: the restore -- re-back, install, load the chain */
    const bool backed = freed && pulsar_session_bank_alloc_physical(s, 1) && pulsar_session_bank_alloc_physical(s, 1);
    CHECK(backed && !pulsar_session_bank_is_evicted(s, 1), "R2 alloc_physical re-backs bank 1 (and again: idempotent)");
    int g = 0;
    if (seg) rewind(seg);
    const bool loaded = backed && pulsar_session_bank_state_restore(s, 1) &&
                        pulsar_session_load_segment(s, seg, seg_bytes, true, &g, NULL, err, sizeof(err)) == 0;
    CHECK(loaded && g == G && pulsar_session_bank_touched_kv_bytes(s, 1) == t1,
          "R2 the chain loaded bank 1 back to %d (touched %.1f MiB): %s", g,
          pulsar_session_bank_touched_kv_bytes(s, 1) / 1048576.0, err);

    /* R3 */
    std::vector<int> toks1, toks2;
    const std::vector<std::vector<float>> got = loaded ? continue_greedy(s, &Q, 16, W, &toks1)
                                                       : std::vector<std::vector<float>>();
    pulsar_session_bank_state_save(s, 1);
    const std::vector<std::vector<float>> ref = pulsar_session_bank_state_restore(s, 2)
                                                    ? continue_greedy(s, &Q, 16, W, &toks2)
                                                    : std::vector<std::vector<float>>();
    bool same = got.size() == 17 && ref.size() == 17;
    double worst = 0;
    int first_diff = -1;
    for (size_t i = 0; same && i < got.size(); i++) {
        if (got[i].size() != (size_t)W || ref[i].size() != (size_t)W) { same = false; break; }
        if (memcmp(got[i].data(), ref[i].data(), (size_t)W * sizeof(float)) != 0 && first_diff < 0) first_diff = (int)i;
        for (int k = 0; k < W; k++) worst = fmax(worst, fabs((double)got[i][(size_t)k] - (double)ref[i][(size_t)k]));
    }
    CHECK(same && first_diff < 0 && toks1 == toks2,
          "R3 freed+restored bank 1 == never-freed bank 2: sync(Q) + 16 greedy steps byte-identical "
          "(first diff at step %d, max |diff| %.3g)", first_diff, worst);

    if (seg) fclose(seg);
    pulsar_session_free(s);
    pulsar_tokens_free(&all);
    gate_engine_close(e);
    printf(n_fail ? "BANK-RESIDENCY GATE FAIL (%d)\n" : "BANK-RESIDENCY GATE PASS\n", n_fail);
    return n_fail != 0;
}
