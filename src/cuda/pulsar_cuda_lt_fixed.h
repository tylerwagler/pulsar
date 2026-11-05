/** @file
 *  The fixed-algorithm fp16 / bf16 cuBLASLt GEMM contract (row-count independent bytes). */
/* The fixed-algorithm fp16 GEMM (L321; L266's contract, out of qwen_exl3_dense_prefill.cu): D = f32(B . A) on the
 * fp16 tensor cores (sm_75+), cuBLASLt, f32 accumulate and output, with ONE algorithm per (A's layout, K, N, row
 * class) -- chosen once at the class's canonical row count with the reduction scheme pinned to NONE (no split-K), and
 * checked to support every other row count it is asked for (an unsupported one refuses).  So a row's bytes never
 * depend on how many rows share the call: a prompt cut anywhere is the prompt prefilled whole, a decode row alone is
 * that row batched.  cublasGemmEx(CUBLAS_GEMM_DEFAULT) picks a kernel (and split-K) by M; that is what this replaces.
 *
 * Column-major, as cuBLASLt sees it:  D^T (n x M, ld ldd) = A' (n x K) . X^T (K x M, ld K), where A' is
 *   PULSAR_LT_A_KN   the weight stored [K][N] row-major (ld N)        -- an EXL3 trellis decoded to W_q
 *   PULSAR_LT_A_NK   the weight stored [N][K] row-major (ld K), transposed by the op -- a Linear as stored [out][in]
 * and X is M fp16 rows of K.  Row classes: PROMPT (canonical PULSAR_PREFILL_CHUNK_DEFAULT rows) and DECODE (canonical
 * PULSAR_GPU_MNEUTRAL_ROWS_MAX rows: a GEMV-shaped algorithm, so a decode row does not run a 4096-row tile). */
#ifndef PULSAR_CUDA_LT_FIXED_H
#define PULSAR_CUDA_LT_FIXED_H

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

/// How A (the weight) is stored: PULSAR_LT_A_KN = [K][N] row-major (ld N, an EXL3 trellis decoded to W_q);
/// PULSAR_LT_A_NK = [N][K] row-major (ld K, a Linear as stored [out][in], transposed by the op).
enum pulsar_lt_a { PULSAR_LT_A_KN = 0, PULSAR_LT_A_NK = 1 };
/// Row class whose canonical row count picks the algorithm: PULSAR_LT_PROMPT (PULSAR_PREFILL_CHUNK_DEFAULT rows) or
/// PULSAR_LT_DECODE (PULSAR_GPU_MNEUTRAL_ROWS_MAX rows).
enum pulsar_lt_rows { PULSAR_LT_PROMPT = 0, PULSAR_LT_DECODE = 1 };

/** L336: one recorded kernel choice -- the committed table (pulsar_cuda_lt_picks.inc, written by tests/lt_picks) maps a
 *  shape to the cuBLASLt algorithm that is fastest there AND keeps a row's bytes independent of the row count, so the
 *  pick never depends on a timing taken at startup: bits change only when a new table is committed (and graded).  A
 *  shape with no entry keeps cuBLASLt's first suggestion, the behaviour before the table.  Keyed by the device's
 *  compute capability and cuBLASLt's version, since both change which kernels exist and what they compute. */
typedef struct {
    int cc;            ///< compute capability, major * 10 + minor (121 = GB10)
    int lt_version;    ///< cublasLtGetVersion() the entry was measured with
    int el;            ///< element type of A and X: 0 fp16, 1 bf16
    int a;             ///< A's layout (pulsar_lt_a)
    int cls;           ///< row class (pulsar_lt_rows)
    int K, N;          ///< reduction depth, A's full output width
    /** @var K
     *  Reduction depth.
     *  @var N
     *  A's full output width. */
    int algo_id;       ///< CUBLASLT_ALGO_CONFIG_ID
    int tile;          ///< CUBLASLT_ALGO_CONFIG_TILE_ID
    int stages;        ///< CUBLASLT_ALGO_CONFIG_STAGES_ID
    int splitk;        ///< CUBLASLT_ALGO_CONFIG_SPLITK_NUM (1: none, the contract)
    int reduction;     ///< CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME (NONE, the contract)
    int swizzle;       ///< CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING
    int custom;        ///< CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION
    int inner;         ///< CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID
    int cluster;       ///< CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID
} pulsar_lt_pick;

/** L336: the recorded pick for a shape on this device and cuBLASLt, or NULL when the table has none.
 *  @param el  0 fp16, 1 bf16
 *  @param a   pulsar_lt_a
 *  @param cls pulsar_lt_rows
 *  @param K   reduction depth
 *  @param N   A's full output width
 *  @return the table's entry, or NULL */
const pulsar_lt_pick *pulsar_lt_pick_find(int el, int a, int cls, int K, int N);

/** L336: fill a pick record from a cuBLASLt algorithm's config (tests/lt_picks writes the table with it).
 *  @param algo the algorithm (cublasLtMatmulAlgo_t, passed as its address)
 *  @param out  receives the config fields; the key fields are left as they were */
void pulsar_lt_pick_from_algo(const void *algo, pulsar_lt_pick *out);

/** L336: rebuild a recorded pick's algorithm (cublasLtMatmulAlgoInit + its config) for the fixed GEMM's types.
 *  @param handle the cublasLtHandle_t (as its value)
 *  @param p      the pick
 *  @param algo   receives the cublasLtMatmulAlgo_t (its address)
 *  @return true when cuBLASLt accepted the id and every config field */
bool pulsar_lt_pick_algo(void *handle, const pulsar_lt_pick *p, void *algo);

/** The cuBLASLt workspace every fixed GEMM may use (the module owns it). */
#define PULSAR_LT_FIXED_WORKSPACE (32u << 20)

/** D[M][n] (f32, rows ldd apart) = X[M][K] (fp16, rows K apart) times columns [0, n) of A (fp16; a's layout; the
 *  shape's N = its full output width, n <= N, n_off the first column) -- the shape's one algorithm for `rows`.
 *  @return 0, -1 refused (named on stderr: no algorithm, an unsupported call), -3 the GEMM failed. */
/// @param a_layout how A is stored (pulsar_lt_a)
/// @param rows     the row class whose algorithm runs
/// @param a        device fp16 weight A
/// @param K        reduction depth
/// @param N        A's full output width
/// @param n_off    first output column of A to compute
/// @param n        output columns (n_off + n <= N)
/// @param x        device fp16 activations [M][K]
/// @param M        activation rows
/// @param d        device f32 output [M][ldd]
/// @param ldd      d row pitch in floats (>= n)
/// @param stream   stream to launch on
/// @param tag      caller's name for refusal messages
int pulsar_lt_fixed_gemm(pulsar_lt_a a_layout, pulsar_lt_rows rows, const void *a, int K, int N, int n_off, int n,
                         const void *x, int M, float *d, int ldd, cudaStream_t stream, const char *tag);

/** L336: pulsar_lt_fixed_gemm with bf16 A and X (the bf16 tensor cores: sm_80+; on sm_75 cuBLASLt has no such kernel
 *  and the call refuses).  Same contract: one algorithm per (A's layout, K, N, row class), no split-K.
 *  @return 0, -1 refused (named on stderr), -3 the GEMM failed. */
/// @param a_layout how A is stored (pulsar_lt_a)
/// @param rows     the row class whose algorithm runs
/// @param a        device bf16 weight A
/// @param K        reduction depth
/// @param N        A's full output width
/// @param n_off    first output column of A to compute
/// @param n        output columns (n_off + n <= N)
/// @param x        device bf16 activations [M][K]
/// @param M        activation rows
/// @param d        device f32 output [M][ldd]
/// @param ldd      d row pitch in floats (>= n)
/// @param stream   stream to launch on
/// @param tag      caller's name for refusal messages
int pulsar_lt_fixed_gemm_bf16(pulsar_lt_a a_layout, pulsar_lt_rows rows, const void *a, int K, int N, int n_off,
                              int n, const void *x, int M, float *d, int ldd, cudaStream_t stream, const char *tag);

#endif
