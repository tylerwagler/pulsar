/* L336: the committed cuBLASLt kernel choices for pulsar_lt_fixed (pulsar_cuda_lt_fixed.h, pulsar_lt_pick), one per
 * shape: written by tests/lt_picks, which times cuBLASLt's candidates at the class's canonical rows and keeps the
 * fastest of those whose row bytes do not depend on the row count, when it beats the first suggestion by >= 3 %.
 * Replacing a row changes that shape's bits: commit it with the reference grade.  Fields: cc, cuBLASLt version,
 * el (0 fp16, 1 bf16), a (0 KN, 1 NK), cls (0 prompt, 1 decode), K, N, algo, tile, stages, splitK, reduction,
 * swizzle, custom, inner, cluster. */
/* measured by tests/lt_picks on NVIDIA GB10 (sm_121), cuBLASLt 131000 (CUDA 13.5, 2026-11-05): the same kernels as the
 * table graded on cuBLASLt 130600.  6144 -> 2560 keeps algo 21 tile 20, graded, over algo 67 tile 412's 1.614 ms: under 1 %
 * apart, not worth a change of bits. */
{121, 131000, 0, 0, 0, 2560, 10240, 67, 409, 35, 1, 0, 0, 30, 0, 0},   /* fp16 KN K=2560 N=10240 prompt: 2.963 -> 2.541 ms (-14 %) */
{121, 131000, 0, 0, 0, 2560, 6144, 67, 409, 35, 1, 0, 0, 30, 0, 0},   /* fp16 KN K=2560 N=6144 prompt: 1.826 -> 1.537 ms (-16 %) */
{121, 131000, 0, 0, 0, 6144, 2560, 21, 20, 15, 1, 0, 0, 0, 0, 0},   /* fp16 KN K=6144 N=2560 prompt: 1.917 -> 1.628 ms (-15 %) */
/* fp16 KN K=2560 N=640 prompt: the first suggestion kept (best of the kept candidates 0.0 % faster) */
{121, 131000, 0, 0, 0, 640, 2560, 67, 409, 35, 1, 0, 0, 68, 0, 0},   /* fp16 KN K=640 N=2560 prompt: 0.308 -> 0.268 ms (-13 %) */
/* bf16 NK K=2560 N=512 prompt: the first suggestion kept (best of the kept candidates 0.2 % faster) */
/* fp16 KN K=2560 N=2560 prompt: the first suggestion kept (best of the kept candidates 0.0 % faster) */
{121, 131000, 0, 0, 0, 2560, 12288, 67, 409, 35, 1, 0, 0, 30, 0, 0},   /* fp16 KN K=2560 N=12288 prompt: 3.428 -> 3.005 ms (-12 %) */
/* fp16 KN K=2560 N=512 prompt: the first suggestion kept (best of the kept candidates 0.0 % faster) */
