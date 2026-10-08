/* L268: one image question through the engine, end to end -- the family's chat render with its image placeholder,
 * the core's expansion (pulsar_expand_image_placeholders), the image sync (the tower, the chunk merge and, for Qwen,
 * the multi-axis rope), greedy decode to the family's stop -- graded on content: the answer must contain EXPECT.
 *
 *   ./tests/image_chat_smoke MODEL IMAGE "question" EXPECT [max_new]
 *
 * The prompt is the family's rendered template text (one user turn, thinking off) with the image placeholder
 * before the question; the server's renderers (src/server) produce the same bytes for a request with one image. */
#include "pulsar.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

static bool read_file(const char *path, std::vector<uint8_t> *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    uint8_t buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out->insert(out->end(), buf, buf + n);
    fclose(f);
    return !out->empty();
}

static std::string lower(std::string s) {
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s MODEL IMAGE \"question\" EXPECT [max_new]\n", argv[0]); return 2; }
    const int max_new = argc > 5 ? atoi(argv[5]) : 48;
    std::vector<uint8_t> bytes;
    if (!read_file(argv[2], &bytes)) { fprintf(stderr, "image smoke: cannot read %s\n", argv[2]); return 2; }
    pulsar_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    pulsar_engine *e = NULL;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "image smoke: %s did not open\n", argv[1]); return 2; }
    std::string text;
    if (pulsar_engine_chat_format(e) == PULSAR_CHAT_QWEN) {
        text = std::string("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>") + argv[3] +
               "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    } else {
        fprintf(stderr, "image smoke: the %s family's template is not written here (DeepSeek's image path has its "
                        "own gates)\n", pulsar_engine_family_name(e));
        pulsar_engine_close(e);
        return 2;
    }
    pulsar_tokens raw = {0}, prompt = {0};
    pulsar_tokenize_rendered_chat(e, text.c_str(), &raw);
    pulsar_image_ref img = { bytes.data(), bytes.size(), -1 };
    char err[512] = "";
    int rc = 1;
    pulsar_session *s = NULL;
    if (!pulsar_expand_image_placeholders(e, &raw, 0, &img, 1, &prompt, err, sizeof err)) {
        printf("IMAGE CHAT SMOKE FAIL: the expansion refused: %s\n", err);
    } else if (pulsar_session_create(&s, e, prompt.len + max_new + 64) != 0) {
        printf("IMAGE CHAT SMOKE FAIL: no session\n");
    } else if (pulsar_session_sync_mm(s, &prompt, &img, 1, err, sizeof err) != 0) {
        printf("IMAGE CHAT SMOKE FAIL: the image sync refused: %s\n", err);
    } else {
        std::string answer;
        bool stopped = false;
        for (int i = 0; i < max_new; i++) {
            const int tok = pulsar_session_argmax(s);
            if (pulsar_token_is_stop(e, tok)) { stopped = true; break; }
            size_t n = 0;
            char *piece = pulsar_token_text(e, tok, &n);
            if (piece) { answer.append(piece, n); free(piece); }
            if (pulsar_session_eval(s, tok, err, sizeof err) != 0) { printf("decode: %s\n", err); break; }
        }
        const bool hit = lower(answer).find(lower(argv[4])) != std::string::npos;
        printf("  %s: %d-token prompt (image block at %d), answer%s: \"%s\"\n", pulsar_engine_family_name(e),
               prompt.len, img.start_pos, stopped ? " (stopped)" : " (budget)", answer.c_str());
        printf(hit ? "IMAGE CHAT SMOKE PASS\n" : "IMAGE CHAT SMOKE FAIL: the answer has no \"%s\"\n", argv[4]);
        rc = hit ? 0 : 1;
    }
    if (s) pulsar_session_free(s);
    pulsar_tokens_free(&raw);
    pulsar_tokens_free(&prompt);
    pulsar_engine_close(e);
    return rc;
}
