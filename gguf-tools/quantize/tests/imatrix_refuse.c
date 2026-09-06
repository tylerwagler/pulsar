/* imatrix_refuse -- the quantizer has ONE column-weighting path for a target
 * that requires an imatrix (IQ2_XXS): the entry the collector recorded for
 * that tensor.  A tensor without one must be refused, naming the tensor, not
 * quantized with a synthetic weight-energy vector (the arm L192 deleted).
 *
 * f32_to_type() exits the process on refusal, so the refusal runs in a forked
 * child whose stderr is captured; the parent asserts the exit status and the
 * message.  The positive control (same tensor, flat imatrix) proves the call
 * is the live one and that the refusal is the ONLY thing the missing entry
 * changes.  CPU-only; no model. */
#include "dsq_internal.h"

#include <sys/wait.h>
#include <unistd.h>

#define NCOLS 256
#define NROWS 2
static const char *TENSOR = "blk.7.ffn_gate_exps.weight";

static void fill(float *x) {
    uint64_t s = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < NCOLS * NROWS; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        x[i] = ((float)(s >> 40) / 8388608.0f - 1.0f) * 0.05f;
    }
}

int main(void) {
    float x[NCOLS * NROWS];
    fill(x);
    ds4q_quantize_init(DS4Q_TYPE_IQ2_XXS);
    int failures = 0;

    /* Positive control: with an imatrix the tensor quantizes to the row size. */
    float flat[NCOLS];
    for (int i = 0; i < NCOLS; i++) flat[i] = 1.0f;
    byte_buf q = f32_to_type(x, NCOLS * NROWS, DS4Q_TYPE_IQ2_XXS, NCOLS, flat, TENSOR);
    size_t want = (size_t)NROWS * ds4q_row_size(DS4Q_TYPE_IQ2_XXS, NCOLS);
    if (q.size != want) {
        printf("FAIL: with imatrix, iq2_xxs wrote %zu bytes, want %zu\n", q.size, want);
        failures++;
    }
    free(q.data);

    /* Refusal: same tensor, no imatrix entry -> exit 1 naming type and tensor. */
    int pipefd[2];
    if (pipe(pipefd) != 0) { perror("pipe"); return 2; }
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 2; }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        byte_buf r = f32_to_type(x, NCOLS * NROWS, DS4Q_TYPE_IQ2_XXS, NCOLS, NULL, TENSOR);
        /* Reaching here means the quantizer synthesized a weighting. */
        free(r.data);
        _exit(0);
    }
    close(pipefd[1]);
    char msg[1024] = {0};
    size_t got = 0;
    for (;;) {
        ssize_t n = read(pipefd[0], msg + got, sizeof(msg) - 1 - got);
        if (n <= 0) break;
        got += (size_t)n;
        if (got >= sizeof(msg) - 1) break;
    }
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 1) {
        printf("FAIL: iq2_xxs without an imatrix entry did not exit 1 (status 0x%x); stderr: %s\n",
               status, msg);
        failures++;
    }
    char want_msg[512];
    snprintf(want_msg, sizeof(want_msg), "iq2_xxs requires an imatrix entry for %s; pass --imatrix", TENSOR);
    if (!strstr(msg, want_msg)) {
        printf("FAIL: refusal did not name the type and tensor; want '%s', stderr: %s\n", want_msg, msg);
        failures++;
    }

    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("imatrix refusal: OK (%s)\n", TENSOR);
    return 0;
}
