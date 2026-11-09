/* L278 (contract 6): the chat-decode smoke for every family -- one greedy chat answer through the engine's public
 * text API (the family's chat renderer, sync, argmax decode, the family's stop set, detokenize), asserting what no
 * us-vs-us comparison can fake.  tests/chat_smoke_gate.py does the same through the CLI binary for the primary
 * model (it exists for the 2026-08-22 BOS salad: every gate green while serving garbage); this one is a runner
 * gate, so it runs on every hosted model on the engine the runner already holds -- no extra load.
 *
 *   ./tests/chat_decode_smoke MODEL
 *
 * Asserts, family-neutrally:
 *   1. the answer contains "12";
 *   2. the turn ENDS at one of the family's stop tokens (pulsar_token_is_stop) within the budget;
 *   3. the continuation never emits the rendered prompt's first token -- the frame opener (DeepSeek's BOS,
 *      Qwen's <|im_start|>), which a sane answer cannot contain and a degenerate decode emits first;
 *   4. the continuation is not one token repeated. */
#include "pulsar.h"
#include "gate_entry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#define SMOKE_MAX_NEW 64

int GATE_ENTRY(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "chat smoke: %s did not open\n", argv[1]); return 2; }
    pulsar_engine_set_bank_pool(1);
    pulsar_session *s = NULL;
    pulsar_tokens prompt = {0};
    int fails = 0;
    if (pulsar_session_create(&s, e, 4096) != 0) {
        fprintf(stderr, "chat smoke: session failed\n");
        gate_engine_close(e);
        return 1;
    }
    /* the family's own one-turn render (DeepSeek's template from markers; Qwen's rendered whole) */
    pulsar_encode_chat_prompt(e, NULL, "What is 7+5? Answer with just the number.", PULSAR_THINK_NONE, &prompt);
    char err[256] = "";
    std::vector<int> ids;
    std::string text;
    bool stopped = false;
    if (prompt.len < 1 || pulsar_session_sync(s, &prompt, err, sizeof err) != 0) {
        printf("  FAIL  sync of the %d-token chat prompt: %s\n", prompt.len, err);
        fails++;
    } else {
        for (int i = 0; i < SMOKE_MAX_NEW; i++) {
            const int tok = pulsar_session_argmax(s);
            if (pulsar_token_is_stop(e, tok)) { stopped = true; break; }
            ids.push_back(tok);
            size_t n = 0;
            char *piece = pulsar_token_text(e, tok, &n);
            if (piece) { text.append(piece, n); free(piece); }
            if (pulsar_session_eval(s, tok, err, sizeof err) != 0) {
                printf("  FAIL  decode step %d: %s\n", i, err);
                fails++;
                break;
            }
        }
    }
    printf("  %s: %d-token prompt, %zu generated%s: \"%s\"\n", pulsar_engine_family_name(e), prompt.len, ids.size(),
           stopped ? " then a stop token" : "", text.c_str());
    if (text.find("12") == std::string::npos) { printf("  FAIL  the answer has no \"12\"\n"); fails++; }
    if (!stopped) { printf("  FAIL  no stop token within %d tokens\n", SMOKE_MAX_NEW); fails++; }
    int frame = 0;
    for (int id : ids) frame += prompt.len > 0 && id == prompt.v[0];
    if (frame) { printf("  FAIL  the frame opener (id %d) appears %d time(s) in the answer\n", prompt.v[0], frame); fails++; }
    std::vector<int> distinct(ids);
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    if (ids.size() >= 8 && distinct.size() < std::max<size_t>(2, ids.size() / 8)) {
        printf("  FAIL  the answer is one token repeated (%zu distinct of %zu)\n", distinct.size(), ids.size());
        fails++;
    }
    pulsar_tokens_free(&prompt);
    pulsar_session_free(s);
    gate_engine_close(e);
    printf(fails ? "CHAT DECODE SMOKE FAIL (%d)\n" : "CHAT DECODE SMOKE PASS\n", fails);
    return fails != 0;
}
