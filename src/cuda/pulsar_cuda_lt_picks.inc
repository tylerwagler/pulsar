/* L336: the committed cuBLASLt kernel choices for pulsar_lt_fixed (pulsar_cuda_lt_fixed.h, pulsar_lt_pick), one per
 * shape: written by tests/lt_picks, which times cuBLASLt's candidates at the class's canonical rows and keeps the
 * fastest of those whose row bytes do not depend on the row count, when it beats the first suggestion by >= 3 %.
 * Replacing a row changes that shape's bits: commit it with the reference grade.  Fields: cc, cuBLASLt version,
 * el (0 fp16, 1 bf16), a (0 KN, 1 NK), cls (0 prompt, 1 decode), K, N, algo, tile, stages, splitK, reduction,
 * swizzle, custom, inner, cluster. */
