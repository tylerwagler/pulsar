/* L251: one chat turn through the Qwen lane, text in and text out -- the S5 renderer, the engine's
 * tokenizer entries (dispatching to the Qwen tokenizer), the session's prefill and decode, the stop
 * set, and the output parser, with nothing from Python in the loop.
 *
 *   ./tests/qwen_chat_smoke <container> "<user message>" [max_new] [effort: none|low|medium|xhigh]
 *
 * Greedy.  Prints the rendered prompt's size, the reasoning and the answer as the parser split
 * them, the stop reason, and the prefill / decode rates.  Not part of the battery: it needs the
 * real container (with the checkpoint's tokenizer.json + generation_config.json beside it). */
#include "pulsar.h"
#include "lib/qwen_chat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>
#include <vector>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <container> \"<user message>\" [max_new] [effort]\n", argv[0]);
        return 2;
    }
    const int max_new = argc > 3 ? atoi(argv[3]) : 256;
    qwen_effort effort = qwen_effort_default();
    char err[512] = "";
    if (argc > 4 && !qwen_effort_resolve(argv[4], strcmp(argv[4], "none") == 0 ? 0 : 1, &effort, err, sizeof(err))) {
        fprintf(stderr, "qwen-chat-smoke: effort: %s\n", err);
        return 2;
    }

    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "qwen-chat-smoke: %s did not open\n", argv[1]); return 2; }
    if (!pulsar_engine_has_tokenizer(e)) { fprintf(stderr, "qwen-chat-smoke: the engine has no tokenizer\n"); return 2; }
    const int W = pulsar_engine_logits_width(e);

    /* render: one user message, the generation prompt, the chosen effort */
    const qwen_msg_in msg = {"user", argv[2], NULL, NULL, 0};
    qwen_render_in in;
    in.msgs = &msg;
    in.n_msgs = 1;
    in.tools_json = NULL;
    in.effort = effort;
    in.add_generation_prompt = true;
    qwen_render_out r;
    if (!qwen_chat_render(in, &r, err, sizeof(err))) { fprintf(stderr, "qwen-chat-smoke: render refused: %s\n", err); return 1; }

    pulsar_tokens prompt = {0};
    pulsar_tokenize_rendered_chat_spans(e, r.text.c_str(), r.spans.data(), (uint32_t)r.spans.size(), &prompt);
    printf("qwen-chat-smoke: effort %s, rendered %zu bytes -> %d tokens\n", qwen_effort_name(effort), r.text.size(), prompt.len);
    if (prompt.len <= 0) { fprintf(stderr, "qwen-chat-smoke: the tokenizer produced nothing\n"); return 1; }

    pulsar_session *sess = NULL;
    if (pulsar_session_create(&sess, e, prompt.len + max_new + 8) != 0) { fprintf(stderr, "qwen-chat-smoke: session refused\n"); return 2; }
    qwen_output_parser parser;
    /* the generation prompt opens a think block unless thinking is off */
    if (!parser.init(effort != QWEN_EFFORT_NONE, NULL, err, sizeof(err))) { fprintf(stderr, "qwen-chat-smoke: parser: %s\n", err); return 1; }
    std::vector<qwen_out_event> ev;
    std::vector<float> row((size_t)W);

    const double t0 = now_s();
    if (pulsar_session_sync(sess, &prompt, err, sizeof(err)) != 0) { fprintf(stderr, "qwen-chat-smoke: sync: %s\n", err); return 1; }
    const double t_prefill = now_s() - t0;
    const char *stop_reason = "length";
    int n_new = 0;
    const double t1 = now_s();
    for (int i = 0; i < max_new; i++) {
        if (i > 0 && pulsar_session_eval(sess, prompt.v[prompt.len - 1], err, sizeof(err)) != 0) {
            fprintf(stderr, "qwen-chat-smoke: eval: %s\n", err);
            return 1;
        }
        if (pulsar_session_copy_logits(sess, row.data(), W) != W) { fprintf(stderr, "qwen-chat-smoke: copy_logits\n"); return 1; }
        int am = 0;
        for (int j = 1; j < W; j++) if (row[j] > row[am]) am = j;
        pulsar_tokens_push(&prompt, am);
        n_new++;
        if (pulsar_token_is_stop(e, am)) { stop_reason = "stop token"; break; }
        size_t n = 0;
        char *txt = pulsar_token_text(e, am, &n);
        parser.feed(txt, n, &ev);
        free(txt);
    }
    const double t_decode = now_s() - t1;
    parser.finish(&ev);

    printf("---- reasoning (%zu bytes) ----\n%s\n", parser.reasoning().size(), parser.reasoning().c_str());
    printf("---- answer (%zu bytes) ----\n%s\n", parser.content().size(), parser.content().c_str());
    printf("---- %d tokens, ended by %s; parser errors %d ----\n", n_new, stop_reason, parser.errors());
    printf("qwen-chat-smoke: SPEED prefill %d tokens in %.3f s | decode %d tokens in %.3f s = %.2f tok/s\n",
           prompt.len - n_new, t_prefill, n_new - 1, t_decode, n_new > 1 ? (n_new - 1) / t_decode : 0.0);
    pulsar_tokens_free(&prompt);
    pulsar_session_free(sess);
    pulsar_engine_close(e);
    return 0;
}
