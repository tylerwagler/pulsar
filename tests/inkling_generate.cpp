/* L294 / L295: greedy generation on an Inkling container -- the coherence check of the forward, through the public
 * API: each prompt rendered as one user turn by F5's renderer (src/lib/inkling_chat.h: vLLM's render_inkling_messages,
 * byte for byte, the client spans kept), tokenized by the family's tokenizer table, synced, then the argmax decoded
 * one row a step until a stop id or the token budget.  The chat front's turn-by-turn render and the server are E6's;
 * this is the CLI-level path.
 *
 *   ./tests/inkling_generate <container> <max_new_tokens> [effort_name] <prompt> [<prompt> ...]
 *
 * Prints, per prompt, the rendered token count, the prefill and decode speed, and the generated text verbatim. */
#include "pulsar.h"
#include "gate_entry.h"
#include "gate_util.h"
#include "lib/inkling_chat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

int GATE_ENTRY(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <container> <max_new_tokens> [effort_name] <prompt> [...]\n", argv[0]);
        return 2;
    }
    const int max_new = atoi(argv[2]);
    int first = 3, effort = inkling_effort_default();
    if (argc > 4 && inkling_effort_named(argv[3]) >= 0) {
        effort = inkling_effort_named(argv[3]);
        first = 4;
    }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (gate_engine_open(&e, &opt) != 0) { fprintf(stderr, "inkling-generate: %s did not open\n", argv[1]); return 2; }
    int rc = 0;
    for (int a = first; a < argc && rc == 0; a++) {
        inkling_msg_in m{};
        m.role = "user";
        m.content = argv[a];
        inkling_render_in in{};
        in.msgs = &m;
        in.n_msgs = 1;
        in.effort = effort;
        in.add_generation_prompt = true;
        inkling_render_out out;
        char err[256] = "";
        if (!inkling_chat_render(in, &out, err, sizeof(err))) { fprintf(stderr, "render: %s\n", err); rc = 1; break; }
        pulsar_tokens P;
        memset(&P, 0, sizeof(P));
        pulsar_tokenize_rendered_chat_spans(e, out.text.c_str(), out.spans.data(), (uint32_t)out.spans.size(), &P);
        pulsar_engine_set_bank_pool(1);
        pulsar_session *s = NULL;
        if (pulsar_session_create(&s, e, P.len + max_new + 16) != 0 || !s) { rc = 1; pulsar_tokens_free(&P); break; }
        const double t0 = gate_now_s();
        if (pulsar_session_sync(s, &P, err, sizeof(err)) != 0) {
            fprintf(stderr, "sync: %s\n", err);
            rc = 1;
        }
        const double t1 = gate_now_s();
        std::string text;
        int n = 0;
        bool stopped = false;
        while (rc == 0 && n < max_new) {
            const int tok = pulsar_session_argmax(s);
            if (tok < 0) { rc = 1; break; }
            if (pulsar_token_is_stop(e, tok)) { stopped = true; break; }
            size_t len = 0;
            char *piece = pulsar_token_text(e, tok, &len);
            text.append(piece, len);
            free(piece);
            n++;
            if (pulsar_session_eval(s, tok, err, sizeof(err)) != 0) { fprintf(stderr, "eval: %s\n", err); rc = 1; }
        }
        const double t2 = gate_now_s();
        printf("=== prompt %d: %d rendered tokens, prefill %.2f s (%.0f tok/s); %d new tokens in %.2f s (%.1f tok/s)%s\n",
               a - first + 1, P.len, t1 - t0, P.len / (t1 - t0 > 0 ? t1 - t0 : 1e-9), n, t2 - t1,
               n / (t2 - t1 > 0 ? t2 - t1 : 1e-9), stopped ? ", stopped at the stop id" : ", at the budget");
        printf("--- user: %s\n--- model (raw, specials kept):\n%s\n", argv[a], text.c_str());
        fflush(stdout);
        pulsar_tokens_free(&P);
        pulsar_session_free(s);
    }
    gate_engine_close(e);
    return rc;
}
