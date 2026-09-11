#!/usr/bin/env python3
"""prismaquant activation cache -> the imatrix `.dat` our quantizer reads (dsq_codecs.c:imatrix_load).

Why: prisma is the calibration authority for the Vision-Exp build (L216, 2026-09-10 correction).  Its streaming probe leaves one `.pt`
per Linear in --activation-cache-dir: {"inputs": X[rows, in_features], "name": live module name}, the rows being the calibration
tokens that Linear actually saw (for a routed expert: the tokens routed to it, capped by --activation-rows-limit).  Upstream's
`export_gguf.build_imatrix_from_act_cache` derives llama.cpp's per-input-column importance, mean(x^2) over rows, in memory only;
this writes it in the binary layout the C quantizer parses, keyed by the artifact's tensor names.

Layout (little-endian): i32 n_entries; per entry i32 name_len, name, i32 ncall, i32 nval, f32[nval]; trailer i32 chunks, i32 dataset_len,
dataset.  ncall = 1 and the values are already means (the loader divides by ncall when > 0).  One entry per (layer, projection)
`blk.N.ffn_{gate,up,down}_exps.weight` with n_experts x in_features values in expert order -- `imatrix_find` slices per expert when
n_values == ncols * n_experts (that is how the 2026-07 `...-ds4-1p5m.dat` is laid out: 129 entries, 1048576 / 524288 values).

Experts absent from the cache (never routed in the calibration) get the LAYER MEAN over the experts that were, and are counted in the
sidecar; a layer with no expert at all is an error.  Dense projections are not written: the engine serves them MXFP8 (no codeword search).

usage: act_cache_to_imatrix_dat.py --act DIR --out FILE.dat [--n-experts 256] [--layers 43] [--dataset LABEL] [--chunks N]
"""
import argparse, glob, json, os, re, struct, sys, time

import numpy as np
import torch

PROJ = {"gate_proj": "ffn_gate_exps", "up_proj": "ffn_up_exps", "down_proj": "ffn_down_exps"}
RX = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--act", required=True, help="prisma --activation-cache-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-experts", type=int, default=256)
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--dataset", default="", help="label recorded in the trailer (the calibration corpus)")
    ap.add_argument("--chunks", type=int, default=0, help="calibration sample count recorded in the trailer")
    ap.add_argument("--sidecar", default=None, help="coverage JSON (default: OUT.json)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.act, "*.pt")))
    if not files:
        sys.exit(f"no .pt files under {a.act}")
    # (layer, proj) -> {expert: importance[in]}
    imp = {}
    rows_seen = {}
    skipped = 0
    t0 = time.time()
    for i, p in enumerate(files):
        stem = os.path.basename(p)[:-3]
        name = stem.replace("__", ".")
        m = RX.match(name)
        if not m:
            skipped += 1
            continue
        layer, xid, proj = int(m.group(1)), int(m.group(2)), m.group(3)
        blob = torch.load(p, map_location="cpu", weights_only=False)
        x = blob.get("inputs") if isinstance(blob, dict) else None
        if x is None or x.ndim != 2 or x.shape[0] == 0:
            skipped += 1
            continue
        v = x.float().pow(2).mean(dim=0).numpy().astype(np.float32)
        imp.setdefault((layer, proj), {})[xid] = v
        rows_seen[(layer, proj, xid)] = int(x.shape[0])
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(files)} files, {time.time()-t0:.0f}s", flush=True)

    entries = []
    coverage = {}
    for layer in range(a.layers):
        for proj, suffix in PROJ.items():
            per = imp.get((layer, proj))
            if not per:
                sys.exit(f"layer {layer} {proj}: no expert activations at all in {a.act}")
            width = len(next(iter(per.values())))
            for xid, v in per.items():
                if len(v) != width:
                    sys.exit(f"layer {layer} {proj} expert {xid}: width {len(v)} != {width}")
            layer_mean = np.mean(np.stack(list(per.values())), axis=0).astype(np.float32)
            values = np.empty((a.n_experts, width), dtype=np.float32)
            missing = 0
            for xid in range(a.n_experts):
                if xid in per:
                    values[xid] = per[xid]
                else:
                    values[xid] = layer_mean
                    missing += 1
            if not np.all(np.isfinite(values)):
                sys.exit(f"layer {layer} {proj}: non-finite importance")
            entries.append((f"blk.{layer}.{suffix}.weight", values.reshape(-1)))
            coverage[f"blk.{layer}.{suffix}"] = {"experts_seen": a.n_experts - missing, "filled_with_layer_mean": missing,
                                                 "rows_min": min(rows_seen[(layer, proj, x)] for x in per), "rows_max": max(rows_seen[(layer, proj, x)] for x in per)}

    with open(a.out, "wb") as f:
        f.write(struct.pack("<i", len(entries)))
        for name, values in entries:
            nb = name.encode()
            f.write(struct.pack("<i", len(nb))); f.write(nb)
            f.write(struct.pack("<ii", 1, len(values)))
            f.write(values.astype("<f4").tobytes())
        ds = a.dataset.encode()
        f.write(struct.pack("<ii", a.chunks, len(ds))); f.write(ds)
    total_missing = sum(c["filled_with_layer_mean"] for c in coverage.values())
    side = {"act_dir": os.path.abspath(a.act), "files": len(files), "expert_files_used": len(rows_seen), "non_expert_files_skipped": skipped,
            "entries": len(entries), "n_experts": a.n_experts, "layers": a.layers, "experts_filled_with_layer_mean": total_missing,
            "dataset": a.dataset, "chunks": a.chunks, "coverage": coverage, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(side, open(a.sidecar or (a.out + ".json"), "w"), indent=1)
    print(f"wrote {a.out}: {len(entries)} entries from {len(rows_seen)} expert activation files; "
          f"{total_missing} of {a.layers * 3 * a.n_experts} expert slots filled with their layer mean; {skipped} non-expert files skipped")


if __name__ == "__main__":
    main()
