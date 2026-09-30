#!/usr/bin/env python3
"""kernel_fixture.py -- the byte oracle for tests/tessera_kernel_gate (L255).

Runs Tessera's OWN fused window kernels (its torch extension, built from Tessera's source by Tessera's loader)
on Qwen3.8-Flash-Next layer-12 weights, and writes every kernel input plus the output bytes to one fixture
file.  The gate then runs pulsar's launcher (src/cuda/mmq/pulsar_tessera.cu, the same device code vendored)
on the same inputs and requires the same bytes.  Value family (BF16 grid, folded arithmetic) only -- the one
pulsar vendors.

    python3 kernel_fixture.py --blobs BLOBS_BF16.pt --act /mnt/models/qwen38fn-act-L12 --out tessera_kernel.fx

BLOBS is a torch.save dict {"experts": [ids], "gate"/"up"/"down": [unit blob bytes per expert],
"qkv"/"oproj": [one blob]} of q256 1024 BF16-grid units (``tessera.export.encode_linears``).  Needs Tessera on
PYTHONPATH and a CUDA device (Tessera builds its extension on first use).

Fixture format (TSFX1): 8-byte magic ``TSFX1\\0\\0\\0``, u32 record count, then per record: u16 name length,
name (utf-8), u8 dtype (0 u8, 1 i16, 2 i32, 3 f32, 4 bf16), u8 ndim, i64 dims[ndim], u64 nbytes, zero padding
to a 16-byte boundary, the data, zero padding to a 16-byte boundary.  Little-endian throughout.

Made with Tessera by Robert Tand - https://github.com/RobTand/tessera
"""
import argparse
import struct

import torch

DTYPES = {torch.uint8: 0, torch.int16: 1, torch.int32: 2, torch.float32: 3, torch.bfloat16: 4}


class Fixture:
    def __init__(self):
        self.recs = []

    def add(self, name, t):
        t = t.detach().contiguous().cpu()
        if t.dtype not in DTYPES:
            raise SystemExit(f"{name}: dtype {t.dtype} has no fixture code")
        self.recs.append((name, t))

    def scalars(self, name, values):
        self.add(name, torch.tensor([int(v) for v in values], dtype=torch.int32))

    def write(self, path):
        with open(path, "wb") as f:
            f.write(b"TSFX1\0\0\0" + struct.pack("<I", len(self.recs)))
            for name, t in self.recs:
                nb = name.encode()
                data = t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bfloat16 \
                    else t.view(torch.int16).numpy().tobytes()
                f.write(struct.pack("<H", len(nb)) + nb + struct.pack("<BB", DTYPES[t.dtype], t.dim()))
                f.write(struct.pack(f"<{t.dim()}q", *t.shape) + struct.pack("<Q", len(data)))
                f.write(b"\0" * (-f.tell() % 16))
                f.write(data)
                f.write(b"\0" * (-f.tell() % 16))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blobs", required=True)
    ap.add_argument("--act", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=255)
    a = ap.parse_args()

    from tessera import routed_fused as rf
    from tessera import window_gemm as wg
    from tessera import window_gemm_grouped as wgg
    from tessera.compact_prep import parse_compact_wire, prepare_window_compact

    dev = torch.device("cuda")
    blobs = torch.load(a.blobs)
    fx = Fixture()
    unit = lambda b: prepare_window_compact(parse_compact_wire(b), device=dev, family="value")

    # ---- dense: one role per Linear, x = the layer's real inputs
    for role, act_file, ms in (("qkv", "gdn_in.pt", (1, 8, 64, 4096)), ("oproj", "oproj_in.pt", (1, 8, 4096))):
        bundle = wg.prepare_window_gemm(unit(blobs[role][0]), block_m=64, block_n=64, block_k=64,
                                        arithmetic="folded")
        why = rf.fused_dense_window_supported(bundle)
        if why is not None:
            raise SystemExit(f"dense {role}: Tessera refuses the fused dense identity: {why}")
        r = rf.prepare_dense_role(bundle)
        p = f"dense.{role}."
        fx.add(p + "words", r.words)
        fx.add(p + "table", r.table16)
        fx.add(p + "init", r.init)
        fx.add(p + "has_init", r.has_init)
        fx.add(p + "wscale", r.wscale)
        fx.add(p + "runs", r.runs)
        fx.add(p + "bdesc", r.bdesc)
        fx.scalars(p + "geom", (r.rows, r.cols, r.tile_words, r.slot_words))
        xin = torch.load(f"{a.act}/{act_file}", mmap=True)
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        for m in ms:
            x = xin[:m].to(dev).to(torch.bfloat16).contiguous()
            out = torch.empty((m, r.rows), dtype=torch.bfloat16, device=dev)
            counter = torch.zeros(1, dtype=torch.int32, device=dev)
            rf.dense_forward(r, x, None, out, counter)
            torch.cuda.synchronize()
            c = f"case.dense.{role}.M{m}."
            fx.add(c + "x", x)
            fx.add(c + "y", out)
            fx.scalars(c + "ksplit", (rf.dense_k_split(m, r.rows, r.cols, sms, tile_words=r.tile_words), sms))
            print(f"dense {role} M={m}: k_split {rf.dense_k_split(m, r.rows, r.cols, sms, tile_words=r.tile_words)}",
                  flush=True)

    # ---- routed: the encoded experts as one stack, distinct top-10 routing per token
    bundles = {k: wgg.prepare_grouped_window_gemm([unit(b) for b in blobs[k]], block_m=32, block_n=64, block_k=64,
                                                  arithmetic="folded") for k in ("gate", "up", "down")}
    fused = rf.FusedRoutedWindowMoE.from_bundles(bundles["gate"], bundles["up"], bundles["down"])
    for k in ("gate", "up", "down"):
        b = getattr(fused, k)
        p = f"moe.{k}."
        fx.add(p + "words", getattr(fused, f"words_{k}"))
        fx.add(p + "table", getattr(fused, f"table_{k}"))
        fx.add(p + "init", b.init_all)
        fx.add(p + "has_init", b.has_init)
        fx.add(p + "wscale", b.scale_all)
        fx.add(p + "runs", getattr(fused, f"runs_{k}"))
        fx.add(p + "bdesc", getattr(fused, f"bdesc_{k}"))
    E = len(blobs["gate"])
    fx.scalars("moe.geom", (E, fused.gate.rows, fused.gate.cols, fused.tile_words_gate_up, fused.slot_words_gate_up,
                            fused.tile_words_down, fused.slot_words_down))
    xin = torch.load(f"{a.act}/moe_in.pt", mmap=True)
    g = torch.Generator().manual_seed(a.seed)
    for t in (1, 8, 64, 512, 4096):
        x = xin[:t].to(dev).to(torch.bfloat16).contiguous()
        logits = torch.randn(t, E, generator=g)
        top = torch.topk(logits, 10, dim=-1)
        ids = top.indices.to(torch.int32).to(dev).contiguous()
        w = torch.softmax(top.values, dim=-1).to(torch.float32).to(dev).contiguous()
        y = fused(x, ids, w)
        torch.cuda.synchronize()
        c = f"case.moe.T{t}."
        fx.add(c + "x", x)
        fx.add(c + "ids", ids)
        fx.add(c + "w", w)
        fx.add(c + "y", y)
        print(f"moe T={t}: out {tuple(y.shape)}", flush=True)
    fx.write(a.out)
    print(f"wrote {a.out}: {len(fx.recs)} records", flush=True)


if __name__ == "__main__":
    main()
