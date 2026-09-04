#!/usr/bin/env python3
"""Does SparkInfer/b12x's sparse-MLA extend kernel beat OUR attention at OUR shape?

Context.  FlashInfer's sm120 sparse-MLA prefill kernel does not build for GB10 --
prefill_kernel.cuh uses setmaxnreg unconditionally and sm_120/121 lack it.  b12x
is a CuTe-DSL kernel library built specifically for SM120/SM121, so it is the one
external sparse-MLA path that can actually run here.

Reference to beat, from tests/attn_indexed_bench.cu (calibrated to the engine
ramp within 3.9%), at the plateau (causal depth >= 2048):

    attention_indexed_mixed_heads8_online_kernel<8,16>   6.917 ms / 512 tokens
    (the f32 indexed kernel of the time; deleted in L166 -- the fp16
    tensor-core kernel in pulsar_cuda_attn_f16.cu is the live comparison)

Our shape, verified against the gguf and the engine trace:
    num_q_heads 64, head_dim 512 (DSV4: d_qk == d_v == 512), topk 512,
    512 tokens per launch, page_size 64.

Honest caveats, because they decide how to read the number:
  * Our kernel additionally folds a 128-token raw window into the SAME online
    softmax.  This bench measures the sparse part alone, so it is an UPPER BOUND
    on what a port could win.
  * run_extend() exposes no attn_sink (only run_decode does), so per-head sinks
    are not modelled here either -- again favouring b12x.
  * b12x is explicitly "not intended for production" per its own README.

If it does not clearly beat 6.917 ms while doing strictly less work, the external
route for attention is closed and we stop.
"""
import sys
import time

import torch

import b12x.attention.sparse_mla as S

OURS_MS = 6.917

NUM_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
NUM_HEADS = 64
HEAD_DIM = 512      # DSV4: d_qk == d_v
V_HEAD_DIM = 512
TOPK = 512
PAGE = 64


def main() -> int:
    dev = torch.device("cuda:0")
    print(f"torch {torch.__version__}  cc={torch.cuda.get_device_capability(0)}")
    print(f"b12x sparse_mla.is_supported() = {S.is_supported(dev)}")
    if not S.is_supported(dev):
        print("\n  UNSUPPORTED on this device -- that is the answer.")
        return 2

    print(f"shape tokens={NUM_TOKENS} heads={NUM_HEADS} head_dim={HEAD_DIM} "
          f"topk={TOPK} page={PAGE}")

    caps = S.Caps(
        device=dev,
        num_q_heads=NUM_HEADS,
        max_q_rows=NUM_TOKENS,
        max_width=TOPK,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        head_dim=HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        mode="extend",
        page_size=PAGE,
        max_kv_rows=NUM_TOKENS * 8,
    )
    plan = S.plan(caps)
    spec = plan.scratch_specs()[0]
    print(f"plan built OK; scratch {spec.shape[0]/1048576:.1f} MiB {spec.dtype}")
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

    n_kv_rows = NUM_TOKENS * 8
    n_pages = (n_kv_rows + PAGE - 1) // PAGE

    q = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    kv_cache = torch.randn(n_pages, PAGE, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    # indexer output: topk selected token offsets per query row
    sel = torch.randint(0, n_kv_rows, (NUM_TOKENS, TOPK), device=dev, dtype=torch.int32)
    sel, _ = torch.sort(sel, dim=-1)
    # extend mode validates these per Q ROW, not per batch
    cache_seqlens = torch.full((NUM_TOKENS,), n_kv_rows, device=dev, dtype=torch.int32)
    nsa_seqlens = torch.full((NUM_TOKENS,), TOPK, device=dev, dtype=torch.int32)

    binding = plan.bind(
        scratch=scratch,
        q=q,
        selected_indices=sel,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_seqlens,
    )

    def run():
        return S.run_extend(
            kv_cache=kv_cache,
            binding=binding,
            sm_scale=1.0 / 32.0,
            v_head_dim=V_HEAD_DIM,
            identity_page_table=True,
        )

    try:
        out = run()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"\n  run_extend FAILED: {type(e).__name__}: {e}")
        print("  Report this as a shape/support limit, not as a performance result.")
        return 3
    print(f"ran OK, output {tuple(out.shape) if hasattr(out,'shape') else type(out)}")

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3)

    print(f"\n  b12x sparse_mla  {best:8.3f} ms/launch (sparse only, no window, no sinks)")
    print(f"  ours             {OURS_MS:8.3f} ms/launch (sparse + 128-tok window fused)")
    print(f"  ratio            {OURS_MS/best:8.3f}x "
          f"{'(b12x faster)' if best < OURS_MS else '(OURS faster)'}")
    print("\n  b12x does strictly LESS work here -- treat as an upper bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
