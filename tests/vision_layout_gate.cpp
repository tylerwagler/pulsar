/* L216 vision-layout gate: grade src/engine/vision.cpp against the checkpoint's
 * own inference/image_processor.py.
 *
 * The goldens in tests/test-vectors/vision-layout-goldens.txt were produced by
 * tests/vision_layout_goldens.py RUNNING the reference module -- not by a second
 * transcription of its formulas -- so a match here means the C++ port agrees
 * with the authority on the parts that are easy to get subtly wrong: the 2-row
 * N-layout interleave, the compressor pad that depends on the block's position
 * in the prompt, `num_tokens`' operator precedence, and the shrink loop.
 *
 *   ./tests/vision_layout_gate [goldens-file] */
#include "pulsar_engine_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TYPES 4096

static int n_grid, n_solve, n_safe, n_block, n_fail;

static void bad(const char *kind, int line, const char *fmt, ...) {
    va_list ap;
    n_fail++;
    fprintf(stderr, "  FAIL line %d (%s): ", line, kind);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

/* The reference's load_image() rounds the requested size UP to a patch multiple
 * before calling safe_resize; the goldens carry h/w, so the gate does that step
 * here exactly as the reference does. */
static int ceil_patch(int v, int p) { return (v + p - 1) / p * p; }

static void check_grid(const char *p, int lineno) {
    int bh, bw, pat, rat, nh, nw, n;
    if (sscanf(p, "grid %d %d %d %d %d %d %d", &bh, &bw, &pat, &rat, &nh, &nw, &n) != 7) {
        bad("grid", lineno, "unparseable");
        return;
    }
    pulsar_vision_grid g = vision_grid_tokens(bh, bw, pat, rat);
    n_grid++;
    if (g.n_llm_h != nh || g.n_llm_w != nw || g.num_tokens != n)
        bad("grid", lineno, "bh=%d bw=%d got %d/%d/%d want %d/%d/%d",
            bh, bw, g.n_llm_h, g.n_llm_w, g.num_tokens, nh, nw, n);
}

static void check_solve(const char *p, int lineno) {
    int h, w, pat, rat, maxn, nh, nw, bh, bw, n;
    if (sscanf(p, "solve %d %d %d %d %d %d %d %d %d %d",
               &h, &w, &pat, &rat, &maxn, &nh, &nw, &bh, &bw, &n) != 10) {
        bad("solve", lineno, "unparseable");
        return;
    }
    pulsar_vision_resize r = vision_solve_resize_ratio(h, w, pat, rat, maxn);
    n_solve++;
    if (r.n_llm_h != nh || r.n_llm_w != nw || r.best_height != bh ||
        r.best_width != bw || r.num_tokens != n)
        bad("solve", lineno, "h=%d w=%d maxn=%d got %d/%d/%d/%d/%d want %d/%d/%d/%d/%d",
            h, w, maxn, r.n_llm_h, r.n_llm_w, r.best_height, r.best_width, r.num_tokens,
            nh, nw, bh, bw, n);
}

static void check_safe(const char *p, int lineno) {
    int h, w, pat, rat, maxn, nh, nw, bh, bw;
    if (sscanf(p, "safe %d %d %d %d %d %d %d %d %d",
               &h, &w, &pat, &rat, &maxn, &nh, &nw, &bh, &bw) != 9) {
        bad("safe", lineno, "unparseable");
        return;
    }
    pulsar_vision_resize r;
    int ok = vision_safe_resize(h, w, ceil_patch(h, pat), ceil_patch(w, pat),
                                pat, rat, maxn, &r);
    n_safe++;
    if (!ok || r.n_llm_h != nh || r.n_llm_w != nw ||
        r.best_height != bh || r.best_width != bw)
        bad("safe", lineno, "h=%d w=%d maxn=%d got ok=%d %d/%d/%d/%d want %d/%d/%d/%d",
            h, w, maxn, ok, r.n_llm_h, r.n_llm_w, r.best_height, r.best_width,
            nh, nw, bh, bw);
}

static void check_block(char *p, int lineno) {
    int nh, nw, sp;
    if (sscanf(p, "block %d %d %d", &nh, &nw, &sp) != 3) {
        bad("block", lineno, "unparseable");
        return;
    }
    int want_types[MAX_TYPES], want_perm[MAX_TYPES];
    int nt = 0, np = 0, hdr = 0, seen_sep = 0, overflow = 0;
    char *save = NULL;
    for (char *tok = strtok_r(p, " \t\n", &save); tok; tok = strtok_r(NULL, " \t\n", &save)) {
        if (!strcmp(tok, "block")) continue;
        if (!strcmp(tok, ";")) { seen_sep = 1; continue; }
        if (hdr < 3) { hdr++; continue; }        /* nh nw sp, already read above */
        int v = atoi(tok);
        if (!seen_sep) {
            if (nt >= MAX_TYPES) { overflow = 1; break; }
            want_types[nt++] = v;
        } else {
            if (np >= MAX_TYPES) { overflow = 1; break; }
            want_perm[np++] = v;
        }
    }
    if (overflow) {
        bad("block", lineno, "golden line longer than %d entries", MAX_TYPES);
        return;
    }

    int got_types[MAX_TYPES], got_perm[MAX_TYPES];
    int got_n = vision_build_image_block(nh, nw, sp, got_types, MAX_TYPES, got_perm, MAX_TYPES);
    n_block++;
    if (got_n != nt) {
        bad("block", lineno, "nh=%d nw=%d sp=%d types count got %d want %d", nh, nw, sp, got_n, nt);
        return;
    }
    for (int i = 0; i < nt; i++) {
        if (got_types[i] != want_types[i]) {
            bad("block", lineno, "nh=%d nw=%d sp=%d types[%d] got %d want %d",
                nh, nw, sp, i, got_types[i], want_types[i]);
            return;
        }
    }
    if (np != nh * nw) {
        bad("block", lineno, "nh=%d nw=%d perm count got %d want %d", nh, nw, np, nh * nw);
        return;
    }
    for (int i = 0; i < np; i++) {
        if (got_perm[i] != want_perm[i]) {
            bad("block", lineno, "nh=%d nw=%d sp=%d perm[%d] got %d want %d",
                nh, nw, sp, i, got_perm[i], want_perm[i]);
            return;
        }
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "tests/test-vectors/vision-layout-goldens.txt";
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "vision-layout gate: cannot open %s\n", path);
        return 2;
    }
    static char line[1 << 20];
    int lineno = 0;
    while (fgets(line, sizeof line, f)) {
        lineno++;
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        if      (!strncmp(p, "grid ",  5)) check_grid(p, lineno);
        else if (!strncmp(p, "solve ", 6)) check_solve(p, lineno);
        else if (!strncmp(p, "safe ",  5)) check_safe(p, lineno);
        else if (!strncmp(p, "block ", 6)) check_block(p, lineno);
        else bad("unknown", lineno, "unrecognised golden kind: %.20s", p);
    }
    fclose(f);
    int cases = n_grid + n_solve + n_safe + n_block;
    printf("VISION-LAYOUT GATE: %s (%d cases: %d grid, %d solve, %d safe, %d block; %d mismatches)\n",
           n_fail ? "FAIL" : "PASS", cases, n_grid, n_solve, n_safe, n_block, n_fail);
    return n_fail ? 1 : 0;
}
