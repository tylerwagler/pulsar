#!/usr/bin/env python3
"""b12x compressed_mla vs our attention kernel, at our shape.

sparse_mla was the wrong entry: it enforces the GLM_NSA/DSV3.2 contract
(q_head_dim=576) and rejects our 512.  compressed_mla is the DSV4 one --
"Head dim is fixed to DSV4's 512 + 64", recipes=("dsv4",) -- and its run()
signature mirrors OUR kernel's structure exactly:

    swa_k_cache / swa_indices          <- our 128-token raw window
    indexed_k_cache / indexed_indices  <- our compressed KV + indexer top-k
    attn_sink                          <- our per-head sinks
    head_dim 512, v_head_dim 512       <- DSV4, matches ours

i.e. sliding-window + indexed-sparse + sinks fused in a single pass, which is
what the engine's indexed attention does (attn_f16_kernel since the f32
attention_indexed_mixed_heads8_online_kernel was deleted in L166).

Reference to beat (tests/attn_indexed_bench.cu, calibrated to the engine ramp
within 3.9%, plateau at causal depth >= 2048):

    ours   6.917 ms / 512 tokens

Open question this answers: the docstring says "Single-pass DECODE on SM121",
so it may refuse our 512 q rows.  If it only serves decode, it cannot replace
our prefill kernel regardless of speed.
"""
import sys
import time

import torch

import b12x.attention.compressed_mla as C

OURS_MS = 6.917

NUM_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
NUM_HEADS = 64
HEAD_DIM = 512
V_HEAD_DIM = 512
TOPK = 512
WINDOW = 128
PAGE = 64
SWA_PAGE = 256


def main() -> int:
    dev = torch.device("cuda:0")
    print(f"torch {torch.__version__}  cc={torch.cuda.get_device_capability(0)}")
    print(f"compressed_mla.is_supported() = {C.is_supported(dev)}")
    if not C.is_supported(dev):
        print("\n  UNSUPPORTED -- that is the answer.")
        return 2
    print(f"shape tokens={NUM_TOKENS} heads={NUM_HEADS} head_dim={HEAD_DIM} "
          f"topk={TOPK} window={WINDOW}")

    n_kv_rows = NUM_TOKENS * 8
    caps = C.Caps(
        device=dev,
        num_q_heads=NUM_HEADS,
        max_q_rows=NUM_TOKENS,
        max_width=TOPK + WINDOW,   # total width = indexed + window, not topk alone
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        head_dim=HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        page_size=PAGE,
        max_kv_rows=n_kv_rows,
    )
    try:
        plan = C.plan(caps)
    except Exception as e:  # noqa: BLE001
        print(f"\n  plan() REFUSED: {type(e).__name__}: {e}")
        return 3
    spec = plan.scratch_specs()[0]
    print(f"plan OK; scratch {spec.shape[0]/1048576:.1f} MiB {spec.dtype}")
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

    n_pages = (n_kv_rows + PAGE - 1) // PAGE
    n_swa_pages = (n_kv_rows + SWA_PAGE - 1) // SWA_PAGE

    q = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    # KV must be the packed FP8 byte record (uint8 storage), not bf16 -- this is
    # the same 584 B/token DSV4 layout FlashInfer defines as _BPT_DSV4.
    BPT = 584
    # 2-D [pages, page_bytes] -- the page is a flat byte record, not [tok, bytes]
    idx_kv = torch.randint(0, 255, (n_pages, PAGE * BPT), device=dev, dtype=torch.uint8)
    swa_kv = torch.randint(0, 255, (n_swa_pages, SWA_PAGE * BPT), device=dev, dtype=torch.uint8)
    idx_ind = torch.randint(0, n_kv_rows, (NUM_TOKENS, TOPK), device=dev, dtype=torch.int32)
    idx_ind, _ = torch.sort(idx_ind, dim=-1)
    swa_ind = torch.randint(0, n_kv_rows, (NUM_TOKENS, WINDOW), device=dev, dtype=torch.int32)
    swa_ind, _ = torch.sort(swa_ind, dim=-1)
    idx_len = torch.full((NUM_TOKENS,), TOPK, device=dev, dtype=torch.int32)
    swa_len = torch.full((NUM_TOKENS,), WINDOW, device=dev, dtype=torch.int32)
    sinks = torch.randn(NUM_HEADS, device=dev, dtype=torch.float32)

    binding = plan.bind(
        scratch=scratch, q=q,
        swa_indices=swa_ind, swa_lengths=swa_len,
        indexed_indices=idx_ind, indexed_lengths=idx_len,
    )

    def run():
        # the binding OWNS q and all index/length tensors; run() takes only the
        # caches and scalars, or it raises.
        return C.run(
            swa_k_cache=swa_kv,
            binding=binding,
            sm_scale=1.0 / 32.0,
            swa_page_size=SWA_PAGE,
            indexed_k_cache=idx_kv,
            indexed_page_size=PAGE,
            attn_sink=sinks,
            expected_num_q_heads=NUM_HEADS,
        )

    try:
        out = run()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"\n  run() FAILED: {type(e).__name__}: {e}")
        print("  Report as a shape/support limit, not a performance result.")
        return 4
    print(f"ran OK, output {tuple(out.shape)}")

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3)

    print(f"\n  b12x compressed_mla  {best:8.3f} ms/launch  (window + sparse + sinks)")
    print(f"  ours                 {OURS_MS:8.3f} ms/launch  (same structure)")
    print(f"  ratio                {OURS_MS/best:8.3f}x "
          f"{'(b12x faster)' if best < OURS_MS else '(OURS faster)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
