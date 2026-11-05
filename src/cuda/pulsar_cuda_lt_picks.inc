/* L336: the committed cuBLASLt kernel choices for pulsar_lt_fixed (pulsar_cuda_lt_fixed.h, pulsar_lt_pick), one per
 * shape: written by tests/lt_picks, which times cuBLASLt's candidates at the class's canonical rows and keeps the
 * fastest of those whose row bytes do not depend on the row count, when it beats the first suggestion by >= 3 %.
 * Replacing a row changes that shape's bits: commit it with the reference grade.  Fields: cc, cuBLASLt version,
 * el (0 fp16, 1 bf16), a (0 KN, 1 NK), cls (0 prompt, 1 decode), K, N, algo, tile, stages, splitK, reduction,
 * swizzle, custom, inner, cluster. */
/* measured by tests/lt_picks on NVIDIA GB10 (sm_121), cuBLASLt 130600 */
{121, 130600, 0, 0, 0, 2560, 10240, 67, 409, 35, 1, 0, 0, 30, 0, 0},   /* fp16 KN K=2560 N=10240 prompt: 3.003 -> 2.501 ms (-17 %) */
{121, 130600, 0, 0, 0, 2560, 6144, 67, 409, 35, 1, 0, 0, 30, 0, 0},   /* fp16 KN K=2560 N=6144 prompt: 1.876 -> 1.509 ms (-20 %) */
{121, 130600, 0, 0, 0, 6144, 2560, 21, 20, 15, 1, 0, 0, 0, 0, 0},   /* fp16 KN K=6144 N=2560 prompt: 1.951 -> 1.600 ms (-18 %) */
/* fp16 KN K=2560 N=640 prompt: the first suggestion kept (best of the kept candidates 0.0 % faster) */
{121, 130600, 0, 0, 0, 640, 2560, 67, 409, 35, 1, 0, 0, 68, 0, 0},   /* fp16 KN K=640 N=2560 prompt: 0.300 -> 0.265 ms (-12 %) */
/* bf16 NK K=2560 N=512 prompt: the first suggestion kept (best of the kept candidates 2.1 % faster) */
/* fp16 KN K=2560 N=2560 prompt: the first suggestion kept (best of the kept candidates 1.0 % faster) */
{121, 130600, 0, 0, 0, 2560, 12288, 67, 409, 35, 1, 0, 0, 30, 0, 0},   /* fp16 KN K=2560 N=12288 prompt: 3.520 -> 2.972 ms (-16 %) */
/* fp16 KN K=2560 N=512 prompt: the first suggestion kept (best of the kept candidates 0.0 % faster) */
