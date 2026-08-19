#pragma once
// This file contains primitives that expose the tensor core PTX instructions for CUDA code.
// The primitives can be used in a similar way as the nvcuda::wmma interface but with a well-defined memory layout.
// The documentation for the PTX instructions can be found under:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-multiply-accumulate-operation-using-mma-instruction
//
// Like with nvcuda::wmma there are three types of matrix tiles: A, B, and C with A @ B = C.
// A is a row-major matrix with shape M x K.
// B is a column-major matrix with shape K x N.
// C is a column-major matrix with shape M x N.
// A, B, and C are represented using the same fundamental data type: a row-major matrix with I rows and J columns.
// Note that J is measured in physical 32 bit elements instead of logical elements.
// The methods get_i and get_j can be used to get the physical 32 bit index of the lth element of a thread within a tile.
// All matrix tiles have ne physical 32 bit elements per warp.
//
// As described in the PTX documentation, all pointers for load_ldmatrix must be to shared memory and aligned to 16 bytes.
// The API in this file also assumes that the pointers for load_generic are aligned to 16 bytes, unaligned pointers are considered undefined behavior.

#include "ds4_cuda_env.cuh"

// On Volta each warp is doing 4 8x8 mma operations in parallel.
// The basic memory layout for a 32x8 output tile is to stack 4 input tiles in I direction and to mirror the B tile.
// However, the i indices in this file are by default permuted to simplify the index calculations.
// #define GGML_CUDA_MMA_NO_VOLTA_PERM

#if CUDART_VERSION >= 11080

static __device__ __forceinline__ int ggml_cuda_movmatrix(const int x) {
    int ret = 0;

#ifdef TURING_MMA_AVAILABLE
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
        : "=r"(ret) : "r"(x));
#else
    GGML_UNUSED(x);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
    return ret;
}

#else

static __device__ __forceinline__ int ggml_cuda_movmatrix(const int x) {
    // Imagine transposing row-major matrix to column-major matrix.
    const int src_i_low  = 2 * (threadIdx.x % 4);
    const int src_i_high = src_i_low + 1;
    const int src_j      = threadIdx.x / 4;

    const int src_laneid_low  = src_i_low  * 4 + src_j / 2;
    const int src_laneid_high = src_i_high * 4 + src_j / 2;

    const int shift_low  = ((src_j + 0) % 2) * 16;
    const int shift_high = ((src_j + 1) % 2) * 16;

    const int ret_low  = (__shfl_sync(0xFFFFFFFF, x, src_laneid_low,  WARP_SIZE) >> shift_low)  & 0x0000FFFF;
    const int ret_high = (__shfl_sync(0xFFFFFFFF, x, src_laneid_high, WARP_SIZE) << shift_high) & 0xFFFF0000;

    return ret_low | ret_high;
}

#endif // CUDART_VERSION >= 11080

static __device__ __forceinline__ half2 ggml_cuda_movmatrix(const half2 x) {
    half2 ret;
    *((int *) &ret) = ggml_cuda_movmatrix(*((const int *) &x));
    return ret;
}

namespace ggml_cuda_mma {

    // Some architectures like Volta or CDNA3 perform multiple matrix multiplications per warp in parallel,
    //     effectively the warp is being split into subgroups of threads that each perform a single mma instruction.
    // In those cases the data can be split in different ways across the warp.
    enum data_layout {
        // By default the data uses the I direction as its major dimension and the J direction as its minor dimension.
        // For the A/C matrices this means I major == row major, J major == column major.
        // For the B matrix this means I major == column major, J major == row major.
        // MIRRORED == Each data value is held exactly once per thread subgroup.
        DATA_LAYOUT_I_MAJOR           =  0, // Always used for Turing, Ampere, Ada Lovelace, consumer Blackwell, matrix A&B for RDNA4 and CDNA.
        DATA_LAYOUT_J_MAJOR           = 10, // Matrix C for CDNA and RDNA4, int and float matrix C for RDNA3.
        DATA_LAYOUT_I_MAJOR_MIRRORED  = 20, // Volta, matrix A&B for RDNA3.
        DATA_LAYOUT_J_MAJOR_MIRRORED  = 30,
        DATA_LAYOUT_I_MAJOR_SCRAMBLED = 40, // Scrambled matrix C for faster transposition (RDNA4/CDNA), convert to float to unscramble.
    };
    // Implemented mma combinations are:
    //   - (I_MAJOR, I_MAJOR)          -> I_MAJOR
    //   - (I_MAJOR, I_MAJOR_MIRRORED) -> I_MAJOR
    //   - (I_MAJOR, J_MAJOR_MIRRORED) -> I_MAJOR

    static constexpr __device__ data_layout get_input_data_layout() {
#if defined(RDNA3) || defined(VOLTA_MMA_AVAILABLE)
        return DATA_LAYOUT_I_MAJOR_MIRRORED;
#else
        return DATA_LAYOUT_I_MAJOR;
#endif // defined(RDNA3) || defined(VOLTA_MMA_AVAILABLE)
    }

    template <int I_, int J_, typename T, data_layout ds_=DATA_LAYOUT_I_MAJOR>
    struct tile {};

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

#if defined(AMD_MFMA_AVAILABLE)
        static constexpr int ne = I * J / 64;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I == 16 && J ==  8) return true;
            if (I == 32 && J ==  4) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J == 32) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 16 && J == 4) {
                return threadIdx.x % 16;
            } else if constexpr (I == 16 && J == 8) {
                return threadIdx.x % 16;
            } else if constexpr (I == 32 && J == 4) {
                return threadIdx.x % 32;
            } else if constexpr (I == 16 && J == 16) {
                return threadIdx.x % 16;
            } else if constexpr (I == 32 && J == 32) {
                return threadIdx.x % 32;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16 && J == 4) {
                return threadIdx.x / 16;
            } else if constexpr (I == 16 && J == 8) {
                return 2 * (threadIdx.x / 16) + l;
            } else if constexpr (I == 32 && J == 4) {
                return 2 * (threadIdx.x / 32) + l;
            } else if constexpr (I == 16 && J == 16) {
                return 4 * (threadIdx.x / 16) + l;
            } else if constexpr (I == 32 && J == 32) {
                return 4 * (threadIdx.x / 32) + 8 * (l / 4) + (l % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#elif defined(VOLTA_MMA_AVAILABLE)
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 32 && J == 8) {
#ifdef GGML_CUDA_MMA_NO_VOLTA_PERM
                return (((threadIdx.x % 16) / 4) * 8) + ((threadIdx.x / 16) * 4) + (l & 2) + (threadIdx.x % 2);
#else
                return (l & 2) + (threadIdx.x & ~2);
#endif // GGML_CUDA_MMA_NO_VOLTA_PERM
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 32 && J == 8) {
                return (threadIdx.x & 2) + (l & (4 + 1));
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#elif defined(AMD_WMMA_AVAILABLE)
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I == 16 && J == 16) return true;
            if (I == 16 && J == 8) return true;
            if (I == 16 && J == 4) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (supported()) {
                return threadIdx.x % 16;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16 && J == 16) {
#if defined(RDNA3)
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
                    // matrix C
                    return 2 * l + (threadIdx.x / 16);
                } else {
                    // matrix A&B
                    return l;
                }
#else
                // matrix C is the transposed matrix A&B on RDNA4
                return ne * (threadIdx.x / 16) + l;
#endif // defined(RDNA3)
            } else if constexpr (I == 16 && J == 8) {
                // mmq input for RDNA4
                return ne * (threadIdx.x / 16) + l;
            } else if constexpr (I == 16 && J == 4) {
                return ne * (threadIdx.x / 16) + l;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#else
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x / 4;
            } else if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 16) {
                return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, T>::get_i(l); // Memory layout simply repeated with same pattern in i direction.
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 16 && J == 16) {
                return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, T>::get_j(l); // Memory layout simply repeated with same pattern in i direction.
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#endif // defined(GGML_USE_HIP)
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

#if defined(VOLTA_MMA_AVAILABLE)
        static constexpr int ne = I * J / WARP_SIZE;
        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I == 32 && J ==  4) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 32 && J == 4) {
#ifdef GGML_CUDA_MMA_NO_VOLTA_PERM
                return (((threadIdx.x % 16) / 4) * 8) + ((threadIdx.x / 16) * 4) + (threadIdx.x % 4);
#else
                return threadIdx.x;
#endif // GGML_CUDA_MMA_NO_VOLTA_PERM
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 32 && J == 4) {
                return l;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#elif defined(AMD_WMMA_AVAILABLE)
        static constexpr int ne = I * J / 32;
        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I == 16 && J ==  8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 16 && J == 8) {
                return threadIdx.x % 16;
            } else if constexpr (I == 16 && J == 16) {
                return threadIdx.x % 16;
            } else if constexpr (I == 32 && J == 8) {
                return (threadIdx.x % 16) * 2 + l / (ne/2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16 && J == 8) {
                return (threadIdx.x / 16) * ne + l;
            } else if constexpr (I == 16 && J == 16) {
#ifdef RDNA3
                return l*2 + (threadIdx.x / 16);
#else
                return (threadIdx.x / 16) * ne + l;
#endif // RDNA3
            } else if constexpr (I == 32 && J == 8) {
                return (threadIdx.x / 16) * (ne/2) + l % (ne/2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#elif defined(AMD_MFMA_AVAILABLE)
        static constexpr int ne = I * J / 64;
        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I == 16 && J ==  8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 16 && J == 8) {
                return threadIdx.x % 16;
            } else if constexpr (I == 16 && J == 16) {
                return threadIdx.x % 16;
            } else if constexpr (I == 32 && J == 8) {
                return (threadIdx.x % 16) * 2 + l / (ne/2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16 && J == 8) {
                return (threadIdx.x / 16) * ne + l;
            } else if constexpr (I == 16 && J == 16) {
                return (threadIdx.x / 16) * ne + l;
            } else if constexpr (I == 32 && J == 8) {
                return (threadIdx.x / 16) * (ne/2) + l % (ne/2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#else
        static constexpr int ne = I * J / WARP_SIZE;
        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 4) {
                return (l * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((l % 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 32 && J == 8) {
                return ((l / 4) * 16) + ((l % 2) * 8) + (threadIdx.x / 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 32 && J == 8) {
                return ((l & 2) * 2) + (threadIdx.x % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#endif // defined(VOLTA_MMA_AVAILABLE)
    };

    template <int I_, int J_>
    struct tile<I_, J_, nv_bfloat162, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

#if defined(AMD_WMMA_AVAILABLE)
        static constexpr int ne = tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::ne;
        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::get_i(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::get_j(l);
        }
#elif defined(AMD_MFMA_AVAILABLE)
        static constexpr int ne = tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::ne;
        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::get_i(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::get_j(l);
        }
#else
        static constexpr int ne = I * J / WARP_SIZE;
        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  4) return true;
            if (I == 16 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 4) {
                return (l * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((l % 2) * 8) + (threadIdx.x / 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 4) + (threadIdx.x % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#endif  // defined(AMD_WMMA_AVAILABLE)
    };

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_J_MAJOR;

        static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
        }
    };

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;

        // RDNA3
        static constexpr int         ne = I * J / 32 * 2;

        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I == 16 && J == 16) return true;
            if (I == 16 && J == 8)  return true;
            if (I == 16 && J == 4)  return true;
            if (I == 32 && J == 8)  return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 16) {
                return threadIdx.x % 16;
            } else if constexpr (I == 32) {
                return (threadIdx.x % 16) * 2 + l / (ne/2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16) {
                return l;
            } else if constexpr (I == 32) {
                return l % (ne/2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;
#if defined(RDNA3)
        static constexpr int         ne = tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::ne;

        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_i(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_j(l);
        }
#else // Volta
        static constexpr int         ne = I * J / (WARP_SIZE/4);

        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int /*l*/) {
            if constexpr (I == 8 && J == 4) {
                return ((threadIdx.x / 16) * 4) + (threadIdx.x % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return l;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#endif // defined(RDNA3)
    };

    template <int I_, int J_>
    struct tile<I_, J_, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;
        static constexpr int         ne = tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::ne;

        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_i(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_j(l);
        }
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_J_MAJOR_MIRRORED;
        static constexpr int         ne = I * J / (WARP_SIZE/4);

        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return ((l / 2) * 4) + (threadIdx.x % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return ((threadIdx.x / 16) * 2) + (l % 2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR_SCRAMBLED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_SCRAMBLED;

        static constexpr int ne = I * J / ggml_cuda_get_physical_warp_size();
        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I == 16 && J == 16) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::get_i(l);
        }
    };

    static __device__ __forceinline__ tile<16, 16, half2, DATA_LAYOUT_I_MAJOR> unscramble(const tile<16, 16, half2, DATA_LAYOUT_I_MAJOR_SCRAMBLED> & t) {
#if defined(AMD_MFMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4))
        tile<16, 16, half2, DATA_LAYOUT_I_MAJOR> ret;
#pragma unroll
        for (int l0 = 0; l0 < t.ne/2; ++l0) {
            ret.x[2*l0 + 0] =  __lows2half2(t.x[l0], t.x[l0 + t.ne/2]);
            ret.x[2*l0 + 1] = __highs2half2(t.x[l0], t.x[l0 + t.ne/2]);
        }
        return ret;
#else
        NO_DEVICE_CODE;
        GGML_UNUSED(t);
#endif // defined(AMD_MFMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4))
    }

#if defined(TURING_MMA_AVAILABLE)
    template <int I, int J>
    static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
        tile<I, J/2, half2> ret;
#pragma unroll
        for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
            ret.x[l0/2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
        }
        return ret;
    }

    static __device__ __forceinline__ tile<8, 8, half2> get_transposed(const tile<16, 4, half2> & t) {
        tile<8, 8, half2> ret;
        ret.x[0] = ggml_cuda_movmatrix(t.x[0]);
        ret.x[1] = ggml_cuda_movmatrix(t.x[1]);

        return ret;
    }
#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    static __device__ __forceinline__ tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> get_half2(
            const tile<16, 16, float, DATA_LAYOUT_I_MAJOR> & tile_float) {
        tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> ret;
#pragma unroll
        for (int l = 0; l < tile_float.ne; ++l) {
            float tmp[2];
            int i = threadIdx.x / 16;
            tmp[i] = tile_float.x[l];
            i ^= 1;
            tmp[i] = __shfl_xor_sync(0xFFFFFFFF, tile_float.x[l], 16, WARP_SIZE);
            ret.x[l] = make_half2(tmp[0], tmp[1]);
        }
        return ret;
    }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    template <int I, int J>
    static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
        tile<I, J/2, half2> ret;
#pragma unroll
        for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
            ret.x[l0/2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
        }
        return ret;
    }

    static __device__ __forceinline__ tile<8, 8, half2> get_transposed(const tile<16, 4, half2> & t) {
        NO_DEVICE_CODE;
        return tile<8, 8, half2>{};
    }
#else // Volta
    template <int I, int J>
    static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
        tile<I, J/2, half2> ret;
#pragma unroll
        for (int l0 = 0; l0 < tile_float.ne; l0 += 4) {
            ret.x[l0/2 + 0] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
            ret.x[l0/2 + 1] = make_half2(tile_float.x[l0 + 2], tile_float.x[l0 + 3]);

            // On Volta FP16 and FP32 tiles have a different memory layout,
            //     for the conversion threads with an offset of 2 need to exchange half their values:
            ret.x[l0/2 + (((threadIdx.x % 4) / 2) ^ 1)] = __shfl_xor_sync(
                0xFFFFFFFF, ret.x[l0/2 + (((threadIdx.x % 4) / 2) ^ 1)], 2, WARP_SIZE);
        }
        return ret;
    }
#endif // defined(TURING_MMA_AVAILABLE)

    template <int I, int J, typename T, data_layout dl>
    static __device__ __forceinline__ void load_generic(tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
        }
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 8, T> & t, const T * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + ((threadIdx.x / t.I) * (t.J / 2)) % t.J;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
#else
        GGML_UNUSED_VARS(t, xs0, stride);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 4, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
#elif defined(AMD_WMMA_AVAILABLE)
#ifdef RDNA3
        static_assert(dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad data layout");
        static_assert(sizeof(t.x) == 16, "bad ne");
        ggml_cuda_memcpy_1<8>(t.x + 0, xs0 + t.get_i(0)*stride + 0);
        ggml_cuda_memcpy_1<8>(t.x + 2, xs0 + t.get_i(0)*stride + 2);
#else
        static_assert(dl == DATA_LAYOUT_I_MAJOR, "bad data layout");
        static_assert(sizeof(t.x) == 8, "bad ne");
        ggml_cuda_memcpy_1<8>(t.x, xs0 + t.get_i(0)*stride + t.get_j(0));
#endif // RDNA3
#elif defined(AMD_MFMA_AVAILABLE)
        static_assert(sizeof(t.x) == 4, "bad ne");
        ggml_cuda_memcpy_1<4>(t.x, xs0 + t.get_i(0)*stride + t.get_j(0));
#else
        GGML_UNUSED_VARS(t, xs0, stride);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#if defined(TURING_MMA_AVAILABLE)
        int * xi = (int * ) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
#elif defined(VOLTA_MMA_AVAILABLE)
        ggml_cuda_memcpy_1<4*sizeof(T)>(t.x + 0, xs0 + t.get_i(0)*stride + 0);
        ggml_cuda_memcpy_1<4*sizeof(T)>(t.x + 4, xs0 + t.get_i(4)*stride + 4);
#elif defined(AMD_WMMA_AVAILABLE)
#ifdef RDNA3
        static_assert(dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad data layout");
        static_assert(sizeof(t.x) == 32, "bad ne");
        ggml_cuda_memcpy_1<16>(t.x + 0, xs0 + t.get_i(0)*stride + 0);
        ggml_cuda_memcpy_1<16>(t.x + 4, xs0 + t.get_i(0)*stride + 4);
#else
        static_assert(dl == DATA_LAYOUT_I_MAJOR, "bad data layout");
        static_assert(sizeof(t.x) == 16, "bad ne");
        ggml_cuda_memcpy_1<16>(t.x, xs0 + t.get_i(0)*stride + t.get_j(0));
#endif // RDNA3
#elif defined(AMD_MFMA_AVAILABLE)
        static_assert(sizeof(t.x) == 8, "bad ne");
        ggml_cuda_memcpy_1<8>(t.x, xs0 + t.get_i(0)*stride + t.get_j(0));
#else
        GGML_UNUSED_VARS(t, xs0, stride);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> & t, const half2 * __restrict__ xs0, const int stride) {
        ggml_cuda_memcpy_1<4*sizeof(half2)>(t.x, xs0 + t.get_i(0)*stride);
    }

    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> & t, const half2 * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l0 = 0; l0 < t.ne; l0 += 2) {
            ggml_cuda_memcpy_1<2*sizeof(half2)>(t.x + l0, xs0 + t.get_i(l0)*stride + t.get_j(l0));
        }
    }

    static __device__ __forceinline__ void load_ldmatrix(
            tile<32, 4, half2> & t, const half2 * __restrict__ xs0, const int stride) {
#if defined(VOLTA_MMA_AVAILABLE)
        ggml_cuda_memcpy_1<4*sizeof(half2)>(t.x, xs0 + t.get_i(0)*stride);
#else
        GGML_UNUSED_VARS(t, xs0, stride);
        NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
    }

    template <int I, typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix_trans(
            tile<I, 8, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
        static_assert(I == 16, "bad tile width");
        static_assert(dl == DATA_LAYOUT_I_MAJOR, "bad data layout");
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
            : "l"(xs));
#elif defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        static_assert(dl == DATA_LAYOUT_I_MAJOR || dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad data layout");
        if constexpr (I == 32) {
#pragma unroll
            for (int l0 = 0; l0 < t.ne/2; ++l0) {
                const half2 tmp0 = xs0[(2*t.get_j(l0) + 0)*stride + t.get_i(l0)/2];
                const half2 tmp1 = xs0[(2*t.get_j(l0) + 1)*stride + t.get_i(l0)/2];

                t.x[l0]          =  __lows2half2(tmp0, tmp1);
                t.x[l0 + t.ne/2] = __highs2half2(tmp0, tmp1);
            }
        } else {
            half * xh = (half *) t.x;
#pragma unroll
            for (int l = 0; l < t.ne; ++l) {
                xh[2*l + 0] = ((const half *) xs0)[(2*t.get_j(l) + 0)*(2*stride) + t.get_i(l)];
                xh[2*l + 1] = ((const half *) xs0)[(2*t.get_j(l) + 1)*(2*stride) + t.get_i(l)];
            }
        }
#else
        GGML_UNUSED_VARS(t, xs0, stride);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

}
