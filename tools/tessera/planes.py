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
    # Tessera 37742e0f (the routed class cutover): from_bundles takes the expert classes -- ordered contiguous
    # partitions of the experts, each with its q256 profile per projection group (w13 = gate, up; w2 = down).
    # One class over every expert here; its q256 is the stored schedule's own rung (what _profile_schedule
    # re-derives through bresenham_rate_schedule and compares to runs_all), so a uniform rate-4 stack is 1024.
    E = int(bundles["down"].experts)
    q = {k: _stack_q256(bundles[k]) for k in PROJS}
    classes = [{"start": 0, "end": E, "q256": {"w13": [q["gate"], q["up"]], "w2": [q["down"]]}}]
    return rf.FusedRoutedWindowMoE.from_bundles(bundles["gate"], bundles["up"], bundles["down"],
                                                expert_classes=classes)


def _stack_q256(bundle):
    """The rung (q256) of a grouped bundle's stored schedule: 256 x (body bits per column) / columns, from the
    first expert's run table [(rate, _, count, _), ...]; every expert of a stack shares one schedule."""
    runs = bundle.runs_all.view(int(bundle.experts), -1, 4)[0].tolist()
    bits = sum(int(r[0]) * int(r[2]) for r in runs)
    cols = int(bundle.cols)
    q, rem = divmod(256 * bits, cols)
    if rem:
        raise SystemExit(f"stack schedule {runs} over {cols} columns is not a q256 rung")
    return q


def stack_planes(fused):
    """{proj: {plane: CPU tensor}} for the three projections of a fused stack."""
    out = {}
    # 37742e0f: the kernel-ready planes live on the stack's _WindowClass (one here: every expert in one class);
    # the class's gate / up / down are the bundles' native views, which keep init_all / has_init / scale_all
    if len(fused.classes) != 1:
        raise SystemExit(f"the engine reads ONE expert class, Tessera built {len(fused.classes)}")
    c = fused.classes[0]
    for k in PROJS:
        b = getattr(c, k)
        tile, slot = ((c.tile_words_down, c.slot_words_down) if k == "down"
                      else (c.tile_words_gate_up, c.slot_words_gate_up))
        out[k] = {
            "words": getattr(c, f"words_{k}"),
            "table": getattr(c, f"table_{k}").view(torch.bfloat16),
            "init": b.init_all,
            "has_init": b.has_init,
            "wscale": b.scale_all,
            "runs": getattr(c, f"runs_{k}"),
            "bdesc": getattr(c, f"bdesc_{k}"),
            "geom": torch.tensor([int(tile), int(slot)], dtype=torch.int32),
        }
        out[k] = {p: t.detach().contiguous().cpu() for p, t in out[k].items()}
        for p in PLANES:
            want = torch.bfloat16 if p == "table" else (torch.float32 if p == "wscale" else torch.int32)
            if out[k][p].dtype != want:
                raise SystemExit(f"{k}.{p}: Tessera prepared {out[k][p].dtype}, the engine reads {want}")
    return out
