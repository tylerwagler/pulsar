/* dsq_requant_tensor: re-quantise ONE bf16 tensor out of an existing GGUF into a
 * raw blob of another type, through the quantizer's own f32_to_type() -- the
 * same codec a full rebuild would run, so the blob is what deepseek4-quantize
 * would have emitted for that tensor.  gguf_replace_tensor.py splices the blob
 * into a copy of the artifact (directory reflow) and proves every other tensor
 * byte-identical.
 *
 * L213 step 2 uses it to produce the drafter markov_w2 variants (type 45 int8 +
 * row scale, type 38 MXFP8) from the shipped bf16 k-major artifact without a
 * 92 GB re-quantise.  ne0 is the fastest dim (the row length f32_to_type is
 * told); for markov_w2 k-major that is vocab.
 *
 *   dsq_requant_tensor SRC.gguf TENSOR_NAME TYPE_CODE NE0 NE1 OUT.blob
 */
#include "dsq_internal.h"   /* first: it sets _POSIX_C_SOURCE before the system headers */
#include "quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s SRC.gguf TENSOR_NAME TYPE_CODE NE0 NE1 OUT.blob\n", argv[0]);
        return 2;
    }
    const char *src = argv[1], *name = argv[2], *out_path = argv[6];
    const ds4q_type target = (ds4q_type)atoi(argv[3]);
    const int64_t ne0 = atoll(argv[4]), ne1 = atoll(argv[5]);
    if (ne0 <= 0 || ne1 <= 0) die("bad dims");
    const int64_t n = ne0 * ne1;

    gguf_file g = load_gguf_metadata(src);
    byte_buf raw = read_gguf_tensor_data(&g, src, name);
    if (raw.size != (size_t)n * 2u) {
        fprintf(stderr, "error: %s holds %zu bytes, expected %lld bf16 elements (%zu bytes)\n",
                name, raw.size, (long long)n, (size_t)n * 2u);
        return 3;
    }
    float *f = (float *)xmalloc((size_t)n * sizeof(float));
    const uint16_t *b = (const uint16_t *)raw.data;
    for (int64_t k = 0; k < n; k++) f[k] = ds4q_bf16_to_f32(b[k]);

    byte_buf out = f32_to_type(f, n, target, ne0, NULL, name);
    FILE *fo = fopen(out_path, "wb");
    if (!fo || fwrite(out.data, 1, out.size, fo) != out.size || fclose(fo) != 0) die("write failed");
    printf("%s: %lld bf16 elements (ne0=%lld ne1=%lld) -> type %d, %zu bytes -> %s\n",
           name, (long long)n, (long long)ne0, (long long)ne1, (int)target, out.size, out_path);
    return 0;
}
