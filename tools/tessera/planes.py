"""planes.py -- the kernel-ready planes of a Tessera expert stack, by Tessera's own load-time prep (L255).

The one place pulsar's tools turn encoded unit blobs into the arrays src/cuda/mmq/pulsar_tessera.cu reads:
tests/tessera_kernel_gate's fixture (kernel_fixture.py) and the container builder (overlay_layer.py) both come
through here, so the engine is fed exactly the planes the byte gate proved.  Value family (BF16 grid, folded).

Made with Tessera by Robert Tand - https://github.com/RobTand/tessera
"""
import torch

#: plane order and names -- family_qwen.h PULSAR_QWEN_TESS_* (the engine binds mlp.experts.tessera.<proj>.<plane>)
PLANES = ("words", "table", "init", "has_init", "wscale", "runs", "bdesc", "geom")
PROJS = ("gate", "up", "down")


def fused_stack(blobs, device):
    """Tessera's FusedRoutedWindowMoE over the blobs' gate/up/down lists (one blob per expert, expert order)."""
    from tessera import routed_fused as rf
    from tessera import window_gemm_grouped as wgg
    from tessera.compact_prep import parse_compact_wire, prepare_window_compact

    unit = lambda b: prepare_window_compact(parse_compact_wire(b), device=device, family="value")
    bundles = {k: wgg.prepare_grouped_window_gemm([unit(b) for b in blobs[k]], block_m=32, block_n=64, block_k=64,
                                                  arithmetic="folded") for k in PROJS}
    return rf.FusedRoutedWindowMoE.from_bundles(bundles["gate"], bundles["up"], bundles["down"])


def stack_planes(fused):
    """{proj: {plane: CPU tensor}} for the three projections of a fused stack."""
    out = {}
    for k in PROJS:
        b = getattr(fused, k)
        tile, slot = ((fused.tile_words_down, fused.slot_words_down) if k == "down"
                      else (fused.tile_words_gate_up, fused.slot_words_gate_up))
        out[k] = {
            "words": getattr(fused, f"words_{k}"),
            "table": getattr(fused, f"table_{k}").view(torch.bfloat16),
            "init": b.init_all,
            "has_init": b.has_init,
            "wscale": b.scale_all,
            "runs": getattr(fused, f"runs_{k}"),
            "bdesc": getattr(fused, f"bdesc_{k}"),
            "geom": torch.tensor([int(tile), int(slot)], dtype=torch.int32),
        }
        out[k] = {p: t.detach().contiguous().cpu() for p, t in out[k].items()}
        for p in PLANES:
            want = torch.bfloat16 if p == "table" else (torch.float32 if p == "wscale" else torch.int32)
            if out[k][p].dtype != want:
                raise SystemExit(f"{k}.{p}: Tessera prepared {out[k][p].dtype}, the engine reads {want}")
    return out
