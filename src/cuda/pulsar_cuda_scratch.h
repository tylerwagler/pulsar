/* Device scratch: the reservations, and the bump arena that slices them.
 *
 * Split out of pulsar_cuda_internal.h so the vendored-adjacent MMQ sources can
 * use the arena without pulling in cub, cuBLASLt and the rest of the engine's
 * internal surface.  Self-contained on purpose: stdint only.
 */
#ifndef PULSAR_CUDA_SCRATCH_H
#define PULSAR_CUDA_SCRATCH_H

#include <stdint.h>

/* SLOTS -- and this is a correctness mechanism, not tidiness.
 *
 * A reservation returns the SAME base pointer whenever the existing block is
 * already large enough, and cudaFree/cudaMalloc's it when it is not.  So two
 * arenas over one slot do not merely share a region, they hand out the same
 * bytes -- or the second one frees the block the first is still holding.
 *
 * That is reachable, not theoretical: routed_moe_launch_mixed40 begins an arena
 * (x_gathered, w_gathered, proj_scratch, ...) and then calls into the MMQ expert
 * GEMM with every one of those slices still live.  The MMQ path needs scratch of
 * its own for exactly the same call.  It used to get it from the vendored ggml
 * pool, which was a physically separate allocator; slots are what replaces that
 * separation once the pool is gone.
 *
 * THE RULE: one live arena per slot.  A caller that holds an arena on a slot may
 * not call anything that begins another arena on that same slot.  Nesting ACROSS
 * slots is fine and is the whole point.  There is no end() -- an arena is valid
 * until the next reservation on its own slot -- so this rule is not enforced by
 * the type system and has to be kept by construction.  Two slots, two owners:
 * keep it that way.
 */
enum {
    CUDA_SCRATCH_MAIN = 0,   /* matmul, attention, moe, indexer */
    CUDA_SCRATCH_MMQ  = 1,   /* the MMQ expert GEMM, called from under MAIN */
    CUDA_SCRATCH_SLOTS = 2
};

/* Single-buffer reservation on CUDA_SCRATCH_MAIN.  For more than one buffer,
 * use the arena below rather than slicing this by hand. */
void *cuda_tmp_alloc(uint64_t bytes, const char *what);

/* Bump arena over one scratch reservation.
 *
 * Use this instead of cuda_tmp_alloc whenever a launch needs MORE THAN ONE
 * buffer.  cuda_tmp_alloc returns one pointer, so multi-buffer callers were
 * slicing it by hand -- eight offsets in routed_moe_cutlass -- and offset
 * arithmetic at the call site is exactly the kind that compiles clean while two
 * buffers overlap.
 *
 * Contract: begin() reserves and resets; take() bump-allocates aligned slices
 * and returns NULL if the reservation would be exceeded, LATCHING the failure
 * so every later take fails too.  One NULL check after the last take is
 * therefore sufficient -- a partial success cannot hand back aliased memory.
 *
 * Lifetime is the slot's scratch buffer: valid until the next reservation on
 * that slot. */
typedef struct {
    uint8_t    *base;
    uint64_t    cap;
    uint64_t    used;
    const char *what;
    int         failed;
} cuda_arena;

/* begin() reserves on CUDA_SCRATCH_MAIN; begin_slot() names the slot. */
int      cuda_arena_begin(cuda_arena *a, uint64_t bytes, const char *what);
int      cuda_arena_begin_slot(cuda_arena *a, int slot, uint64_t bytes, const char *what);
void    *cuda_arena_take(cuda_arena *a, uint64_t bytes, uint64_t align);
uint64_t cuda_arena_used(const cuda_arena *a);

#endif /* PULSAR_CUDA_SCRATCH_H */
