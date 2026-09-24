#!/usr/bin/env python3
"""test_names.py -- names.py + policy.py against the SERVED artifact (L247).

Plain python, headers only, no GPU, no torch.  For every HF tensor of the
Vision-Exp checkpoint, map_hf + layout_for must reproduce the served
source-precision repack's container_name, gguf_name, shard file and layout
EXACTLY, and every served entry must be produced by some HF name (both
directions; expert `.scale` companions fold into their `.weight`).  Then
map_hf must know every name of the V4.1 checkpoint and of the MiaAI EXL3
checkpoint (none returns None).  Exit non-zero on any mismatch.

  SERVED  /mnt/models/DeepSeek-v4-Flash/                       48 shards, pulsar.tensors + pulsar.experts
  SOURCE  hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp  model.safetensors.index.json + shard headers
  V4.1    hub/models--deepseek-ai--DeepSeek-V4.1-Flash
  EXL3    /mnt/models/mia-v41-exl3/model-000*-of-00039.safetensors
"""
import collections
import glob
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from names import Mapped, ModelShape, map_hf, shard_file, shard_order   # noqa: E402
from policy import NATIVE, PolicyError, declared_shape, layout_for, rekey_format_map  # noqa: E402

SERVED = "/mnt/models/DeepSeek-v4-Flash/"
VEXP = "/mnt/models/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/*/"
V41 = "/mnt/models/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*/"
EXL3 = "/mnt/models/mia-v41-exl3/model-000*-of-00039.safetensors"
FORMAT_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "format-maps", "v5mx4-format-map.json")

EXPERT_ENTRY = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.\d+\.(w[123])\.weight$")


def header(path):
    with open(path, "rb") as fh:
        n, = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def one(pattern):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise SystemExit(f"{pattern}: expected exactly one match, got {hits}")
    return hits[0]


def hf_checkpoint(snap):
    """{hf name: (dtype, shape)} from the index + every shard header, and the config."""
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    hdr = {}
    for s in sorted(set(idx.values())):
        for k, v in header(os.path.join(snap, s)).items():
            if k != "__metadata__":
                hdr[k] = (v["dtype"], v["shape"])
    missing = [n for n in idx if n not in hdr]
    if missing:
        raise SystemExit(f"{snap}: {len(missing)} index names have no header entry, e.g. {missing[:3]}")
    return {n: hdr[n] for n in idx}, json.load(open(os.path.join(snap, "config.json")))


def served_artifact(d):
    """tensors: {container: (shard file, gguf_name, layout, dtype, dims_ne)};
    experts: {container: (shard file, layout, gguf_name, part)} per U8 expert entry."""
    tensors, experts = {}, {}
    files = sorted(glob.glob(os.path.join(d, "model-*.safetensors")))
    for f in files:
        base = os.path.basename(f)
        h = header(f)
        md = h.pop("__metadata__")
        pt = json.loads(md["pulsar.tensors"])
        pe = json.loads(md["pulsar.experts"])
        for k, v in pt.items():
            tensors[k] = (base, v["gguf_name"], v["layout"], h[k]["dtype"], list(v["dims_ne"]))
        stacks = {(e["gguf_name"].split(".")[0], int(e["gguf_name"].split(".")[1]), e["part"]): e for e in pe}
        for k in h:
            if k in pt:
                continue
            m = EXPERT_ENTRY.match(k)
            if not m:
                raise SystemExit(f"{base}: {k} is neither declared in pulsar.tensors nor an expert entry")
            ns, idx, part = m.group(1), int(m.group(2)), m.group(3)
            st = stacks[("blk" if ns == "layers" else "dspark", idx, part)]
            experts[k] = (base, st["layout"], st["gguf_name"], part)
    return tensors, experts, len(files)


def served_layout_equal(ours, served_layout, served_dtype):
    if served_layout == "native":
        return NATIVE.get(served_dtype) == ours
    return served_layout == ours


def main():
    failures = []

    def fail(msg):
        failures.append(msg)
        if len(failures) <= 25:
            print("MISMATCH", msg)

    # ------------------------------------------------------------------ Vision-Exp vs served
    snap = one(VEXP)
    src, cfg = hf_checkpoint(snap)
    shape = ModelShape.from_config(cfg)
    print(f"source {snap}: {len(src)} tensors; shape {shape}")
    tensors, experts, n_files = served_artifact(SERVED)
    print(f"served {SERVED}: {n_files} shards, {len(tensors)} declared tensors, {len(experts)} expert entries")
    if n_files != len(shard_order(shape)):
        fail(f"shard count: served {n_files}, shard_order says {len(shard_order(shape))}")

    fam = collections.Counter()
    produced_tensors, produced_experts, dropped = {}, {}, collections.Counter()
    companions = collections.Counter()
    for hf, (dtype, hshape) in src.items():
        m = map_hf(hf, shape)
        if m is None:
            fail(f"{hf}: map_hf -> None")
            continue
        if m.is_scale:
            companions[m.family] += 1
            continue
        if not m.emit:
            dropped[re.sub(r"\d+", "N", hf)] += 1
            if m.container_name in tensors or m.container_name in experts:
                fail(f"{hf}: emit=False but the served artifact carries {m.container_name}")
            continue
        fam[m.family] += 1
        try:
            layout = layout_for(m, dtype, hshape, {})
        except PolicyError as e:
            fail(f"{hf}: {e}")
            continue
        want_shard = shard_file(shape, m.shard)
        if m.family == "expert":
            produced_experts[m.container_name] = hf
            got = experts.get(m.container_name)
            if got is None:
                fail(f"{hf} -> {m.container_name}: not an expert entry of the served artifact")
                continue
            sfile, slayout, sgguf, spart = got
            if (sfile, slayout, sgguf, spart) != (want_shard, layout, m.gguf_name, m.part):
                fail(f"{hf}: expert ({sfile},{slayout},{sgguf},{spart}) served vs "
                     f"({want_shard},{layout},{m.gguf_name},{m.part}) ours")
        else:
            produced_tensors[m.container_name] = hf
            got = tensors.get(m.container_name)
            if got is None:
                fail(f"{hf} -> {m.container_name}: not in the served artifact")
                continue
            sfile, sgguf, slayout, sdtype, sdims = got
            dims = list(reversed(declared_shape(m, hshape)))
            if sdims != dims:
                fail(f"{hf}: dims_ne {sdims} served vs {dims} ours (source shape {hshape})")
            if sfile != want_shard:
                fail(f"{hf}: shard {sfile} served vs {want_shard} ours")
            if sgguf != m.gguf_name:
                fail(f"{hf}: gguf_name {sgguf} served vs {m.gguf_name} ours")
            if not served_layout_equal(layout, slayout, sdtype):
                fail(f"{hf}: layout {slayout}/{sdtype} served vs {layout} ours")

    # every companion must have a primary that was produced
    for hf in src:
        m = map_hf(hf, shape)
        if m is not None and m.is_scale and m.emit:
            if m.container_name not in produced_tensors and m.container_name not in produced_experts:
                fail(f"{hf}: companion of {m.container_name}, which nothing produced")

    # reverse direction
    for k in tensors:
        if k not in produced_tensors:
            fail(f"served {k} ({tensors[k][1]}) is produced by no HF name")
    for k in experts:
        if k not in produced_experts:
            fail(f"served expert entry {k} is produced by no HF name")

    print("Vision-Exp -> served: emitted per family", dict(sorted(fam.items())),
          "| companions folded", dict(sorted(companions.items())),
          "| consumed-not-emitted", dict(sorted(dropped.items())))
    print(f"  {len(produced_tensors)}/{len(tensors)} declared tensors and "
          f"{len(produced_experts)}/{len(experts)} expert entries reproduced")

    # ------------------------------------------------------------------ format map re-key
    fmap = json.load(open(FORMAT_MAP))
    try:
        rekeyed = rekey_format_map(fmap, shape)
    except PolicyError as e:
        fail(f"rekey_format_map: {e}")
        rekeyed = {}
    unresolved = [k for k in rekeyed if "*" not in k and k not in src]
    if unresolved:
        fail(f"rekey_format_map: {len(unresolved)} keys are not source tensors, e.g. {unresolved[:3]}")
    by_layout = collections.Counter(rekeyed.values())
    print(f"format map {os.path.basename(FORMAT_MAP)}: {len(fmap)} rows -> {len(rekeyed)} HF keys, layouts {dict(by_layout)}")
    # the map applied as overrides: buildable rows agree with the defaults, IQ2 rows refuse loudly
    refused, applied = 0, 0
    for pat, want in rekeyed.items():
        if "*" in pat:
            hf = pat.replace("*", "0")
        else:
            hf = pat
        m = map_hf(hf, shape)
        try:
            layout_for(m, *src[hf], {pat: want})
            applied += 1
        except PolicyError:
            refused += 1
            if want != "iq2_xxs_mmq_k":
                fail(f"format map row {pat}={want} refused although the builder produces {want}")
    print(f"  as overrides: {applied} rows applied, {refused} refused (all {by_layout.get('iq2_xxs_mmq_k', 0)} IQ2 rows: no producer in the builder)")

    # ------------------------------------------------------------------ V4.1 names
    snap41 = one(V41)
    src41, cfg41 = hf_checkpoint(snap41)
    shape41 = ModelShape.from_config(cfg41)
    print(f"V4.1 {snap41}: {len(src41)} tensors; shape {shape41}")
    fam41, none41, drop41 = collections.Counter(), [], collections.Counter()
    vexp_suffixes = {re.sub(r"^(layers|mtp)\.\d+\.", r"\1.N.", n) for n in src}
    v41_only = collections.Counter()
    for hf, (dtype, hshape) in src41.items():
        m = map_hf(hf, shape41)
        if m is None:
            none41.append(hf)
            continue
        key = re.sub(r"^(layers|mtp)\.\d+\.", r"\1.N.", hf)
        key = re.sub(r"experts\.\d+\.", "experts.E.", key)
        if key not in vexp_suffixes and re.sub(r"experts\.E\.", "experts.0.", key) not in vexp_suffixes:
            v41_only[(key, m.gguf_name if m.layer is None else re.sub(r"\.\d+\.", ".N.", m.gguf_name, count=1), m.emit)] += 1
        if not m.emit:
            drop41[re.sub(r"\d+", "N", hf)] += 1
            continue
        if m.is_scale:
            continue
        fam41[m.family] += 1
        try:
            layout_for(m, dtype, hshape, {})
        except PolicyError as e:
            fail(f"V4.1 {hf}: {e}")
    if none41:
        fail(f"V4.1: {len(none41)} names map to None, e.g. {none41[:5]}")
    print("V4.1: emitted per family", dict(sorted(fam41.items())), "| consumed-not-emitted", dict(sorted(drop41.items())))
    print("V4.1-only rows (HF pattern, gguf_name, emit) x count:")
    for (k, g, e), c in sorted(v41_only.items()):
        print(f"  {k:48s} -> {g:44s} emit={e} x{c}")

    # ------------------------------------------------------------------ EXL3 names
    exl3_names = []
    for f in sorted(glob.glob(EXL3)):
        for k, v in header(f).items():
            if k != "__metadata__":
                exl3_names.append((k, v["dtype"], v["shape"]))
    fam3, none3, rates, refused3 = collections.Counter(), [], collections.Counter(), collections.Counter()
    primaries, entries = 0, set()
    for k, dtype, hshape in exl3_names:
        m = map_hf(k, shape41)
        if m is None:
            none3.append(k)
            continue
        fam3[m.family] += 1
        if m.emit and not m.is_scale:
            primaries += 1
            entries.add(m.container_name)
            if m.family == "expert":
                try:
                    rates[(m.shard.split(".")[0], layout_for(m, dtype, hshape, {}))] += 1
                except PolicyError:
                    # MiaAI quantized the drafter's experts at 4 bits (mtp_bits: 4 -> 64
                    # words per tile), a rate the engine has no arm for: refused, not built.
                    refused3[(m.shard.split(".")[0], hshape[2])] += 1
    if none3:
        fail(f"EXL3: {len(none3)} names map to None, e.g. {none3[:5]}")
    print(f"EXL3 {os.path.dirname(EXL3)}: {len(exl3_names)} tensors mapped, per family {dict(sorted(fam3.items()))}; "
          f"{primaries} primaries -> {len(entries)} container entries; expert rates {dict(rates)}; "
          f"refused (family, words/tile) {dict(refused3)}")
    if any(fam == "layers" for fam, _ in refused3):
        fail(f"EXL3: a text-stack expert rate was refused: {dict(refused3)}")

    # ------------------------------------------------------------------ pinned expectations
    def expect(hf, sh, **want):
        m = map_hf(hf, sh)
        if m is None:
            fail(f"pinned: {hf} -> None")
            return
        for k, v in want.items():
            if getattr(m, k) != v:
                fail(f"pinned: {hf}.{k} = {getattr(m, k)!r}, want {v!r}")
    expect("mtp.0.main_proj.weight", shape, gguf_name="dspark.main_proj.weight", shard="mtp.0", family="mtp")
    expect("mtp.0.main_proj.scale", shape, container_name="mtp.0.main_proj.weight", is_scale=True)
    expect("mtp.2.markov_head.embed.weight", shape41, container_name="mtp.2.markov_head.embed.weight",
           gguf_name="dspark.2.markov_head.markov_w1.weight")
    expect("mtp.2.markov_head.head.weight", shape41, gguf_name="dspark.2.markov_head.markov_w2.weight")
    expect("layers.20.ffn.experts.5.w1.trellis", shape41, container_name="layers.20.ffn.experts.5.w1.weight",
           gguf_name="blk.20.ffn_gate_exps.weight", family="expert", expert=5, part="w1", is_scale=False)
    expect("layers.20.ffn.experts.5.w1.mul1", shape41, container_name="layers.20.ffn.experts.5.w1.weight", is_scale=True)
    expect("layers.3.attn.wo_a.slice.7.svh", shape41, container_name="layers.3.attn.wo_a.weight",
           gguf_name="blk.3.attn_output_a.weight", is_scale=True)
    expect("head.trellis", shape41, container_name="head.weight", gguf_name="output.weight", shard="top")
    expect("layers.1.engram.wkv.weight", shape41, gguf_name="layers.1.engram.wkv.weight", shard="layers.1", emit=True)
    expect("layers.1.engram.embed.weight", shape41, emit=False)
    expect("layers.0.ffn.gate.bias", shape, emit=False)
    expect("layers.3.ffn.gate.bias", shape, emit=True, gguf_name="blk.3.exp_probs_b.bias")
    expect("mtp.1.ffn.gate.bias_vl", shape, emit=False)
    expect("hc_head_fn", shape, container_name="hc_head_fn", gguf_name="output_hc_fn.weight", shard="top")
    for bad in ("layers.0.ffn.experts.5.w1.mcg", "layers.43.attn_norm.weight", "mtp.3.norm.weight",
                "lm_head.weight", "layers.0.attn.wq_a", "model.embed_tokens.weight", "layers.0.norm.weight"):
        if map_hf(bad, shape) is not None:
            fail(f"pinned: {bad} should map to None")
    m = map_hf("layers.0.ffn.gate.tid2eid", shape)
    if layout_for(m, "I64", [129280, 6], {}) != "i32":
        fail("pinned: tid2eid I64 -> i32")
    for hf, dtype, hshape in (("layers.0.attn_norm.weight", "F16", [4096]), ("layers.0.attn.wkv.weight", "F8_E4M3", [4096]),
                              ("layers.0.hc_attn_base", "I64", [24])):
        try:
            layout_for(map_hf(hf, shape), dtype, hshape, {})
            fail(f"pinned: {hf} {dtype} {hshape} should refuse")
        except PolicyError:
            pass
    try:
        layout_for(map_hf("layers.0.attn.wkv.weight", shape), "F8_E4M3", [512, 4096], {"layers.*.attn.wkv.weight": "bf16"})
        fail("pinned: an override contradicting the source dtype should refuse")
    except PolicyError:
        pass

    total = sum(fam.values()) + sum(companions.values()) + sum(dropped.values())
    print(f"\nRESULT: {total} Vision-Exp names walked ({sum(fam.values())} entries, {sum(companions.values())} companions, "
          f"{sum(dropped.values())} consumed-not-emitted), {len(src41)} V4.1 names, {len(exl3_names)} EXL3 names, "
          f"{len(failures)} mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
