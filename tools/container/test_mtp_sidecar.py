#!/usr/bin/env python3
"""test_mtp_sidecar.py -- grade an emitted MTP sidecar (mtp_sidecar.py) against its two sources and the loader.

    python3 test_mtp_sidecar.py --sidecar OUT_DIR --hf BF16_SNAPSHOT --exl3 TURBODERP_EXL3_SNAPSHOT --base CONTAINER_DIR

  0. rates   the emitter's layout -> k2 table equals the engine's (exl3_trellis.h exl3_type_k2 + model.cpp's
             layout names), and every EXL3 tensor's rate is one its arm reads (exl3_arm_has_rate: dense Linears
             DENSE, routed gate / up PAIR, routed down DOWN).
  1. EXL3    every EXL3 tensor (dense + each of the 3 x 512 expert projections): the sidecar's [trellis | suh | svh]
             slice and turboderp's own .trellis / .suh / .svh are DECODED the same way -- the mul1 codebook tile
             decode of exl3_trellis.h, mirrored here in numpy -- and the fp16 W_hat bits, suh and svh compared
             bit for bit (plus the raw bytes and the .mul1 marker).  Then a full reconstruction
             W = diag(suh) H128 W_hat H128 diag(svh) of every dense tensor and a sample of experts is compared bit
             for bit on both sides and graded against the BF16 source (fidelity, and the gate / up split order).
  2. native  every bf16 tensor equals the BF16 source byte for byte; every mxfp8_lt tensor equals the emitter's
             producer re-run on the BF16 source byte for byte and decodes within E4M3 rounding of it; the same
             producer reproduces a base-container main-layer mxfp8_lt tensor byte for byte (it IS the container's
             codec).
  3. names   the loader's two passes (st_add_declared / st_add_expert_stacks) replayed over the combined directory:
             every declaration resolves, byte counts equal the layout model, experts are contiguous, one shard
             carries pulsar.kv (the base primary), no gguf_name repeats; the Qwen binder's names and ne dims for
             the MTP block; names.py (the DeepSeek table) is shown not to apply.  Prints the bound key list.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_sidecar as MS  # noqa: E402  (first: it pins the emitter's modules on sys.path)

import numpy as np  # noqa: E402

REPO = MS.REPO
FAIL = []


def check(ok, msg):
    if not ok:
        FAIL.append(msg)
        print("FAIL", msg, flush=True)
    return ok


# ---------------------------------------------------------------------------
# the EXL3 decode: a numpy mirror of src/engine/exl3_trellis.h
# ---------------------------------------------------------------------------
MUL1 = 0x83DCD12D


def _f16(bits):
    return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])


def mul1_table() -> np.ndarray:
    """exl3_mul1_decode for all 65536 states -> fp16 bits.  s * 0x1eee and the sum with 0xc931 are exact in fp32
    (the header's claim, asserted here by computing both in fp32 and in fp64), so the one rounding is to fp16."""
    x = np.arange(65536, dtype=np.uint64)
    y = (x * np.uint64(MUL1)) & np.uint64(0xFFFFFFFF)
    s = (0x6400 + (y & 0xFF) + ((y >> 8) & 0xFF) + ((y >> 16) & 0xFF) + (y >> 24)).astype(np.uint16)
    h = s.view(np.float16)
    v32 = h.astype(np.float32) * np.float32(_f16(0x1EEE)) + np.float32(_f16(0xC931))
    v64 = h.astype(np.float64) * _f16(0x1EEE) + _f16(0xC931)
    assert np.array_equal(v32.astype(np.float64), v64), "mul1: fp32 product/sum not exact"
    return v64.astype(np.float16).view(np.uint16)


TABLE = mul1_table()


def k2_of_words(words):
    """exl3_k2_from_words."""
    if words <= 0 or words % 8:
        return 0
    k2 = 2 * (words // 16) if words % 16 == 0 else 2 * (words // 16) + 1
    bits = k2 >> 1
    valid = (1 <= bits <= 3) if (k2 & 1) else (1 <= bits <= 8)
    return k2 if valid else 0


def _positions(k2):
    p = np.arange(256)
    bits = k2 >> 1
    if k2 & 1:
        b2 = 2 * bits + 1
        end = (p >> 1) * b2 + np.where(p & 1, b2, bits)
    else:
        end = (p + 1) * bits
    t, i = p >> 3, p & 7
    row = (t % 4) * 2 + (i & 1) + 8 * ((i >> 1) & 1)
    col = t // 4 + 8 * (i >> 2)
    return end, row, col


def decode_w_hat(trellis_bytes: bytes, k: int, n: int, words: int) -> np.ndarray:
    """[k/16, n/16, words] int16 trellis -> fp16 bits W_hat [k, n] (row along in, col along out)."""
    k2 = k2_of_words(words)
    assert k2, words
    kt, nt = k // 16, n // 16
    w = np.frombuffer(trellis_bytes, dtype="<u2").reshape(kt * nt, words)
    w32 = w[:, 0::2].astype(np.uint32) | (w[:, 1::2].astype(np.uint32) << 16)   # undo the packer's half swap
    stream = w32.astype(">u4").view(np.uint8).reshape(kt * nt, 2 * words)       # stream bit s = MSB-first bit s
    # the ring: a state whose window starts before bit 0 wraps to the stream's tail -- prepend the last 16 bits
    pad = np.concatenate([stream[:, -2:], stream, np.zeros((kt * nt, 1), np.uint8)], axis=1).astype(np.uint32)
    end, row, col = _positions(k2)
    o = end                         # window [end-16, end) in stream bits == [end, end+16) in the padded stream
    byte, sh = o >> 3, o & 7
    v24 = (pad[:, byte] << 16) | (pad[:, byte + 1] << 8) | pad[:, byte + 2]
    state = (v24 >> (8 - sh)) & 0xFFFF
    tiles = np.empty((kt * nt, 16, 16), np.uint16)
    tiles[:, row, col] = TABLE[state]
    return tiles.reshape(kt, nt, 16, 16).transpose(0, 2, 1, 3).reshape(k, n)


def _had128():
    h = np.array([[1.0]])
    while h.shape[0] < 128:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(128.0)


H128 = _had128()


def reconstruct(w_hat_bits, suh_bits, svh_bits):
    """W (in, out) = diag(suh) H128 W_hat H128 diag(svh), fp64."""
    k, n = w_hat_bits.shape
    a = w_hat_bits.view(np.float16).astype(np.float64)
    a = np.einsum("ij,bjn->bin", H128, a.reshape(k // 128, 128, n)).reshape(k, n)
    a = (a.reshape(k, n // 128, 128) @ H128).reshape(k, n)
    suh = suh_bits.view(np.float16).astype(np.float64)
    svh = svh_bits.view(np.float16).astype(np.float64)
    return suh[:, None] * a * svh[None, :]


def bf16_f64(raw, shape):
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32).astype(np.float64).reshape(shape)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
class ShardDir:
    """name -> (path, abs offset, nbytes, dtype, shape) over every *.safetensors in a directory."""

    def __init__(self, d, only=None):
        self.entries, self.meta = {}, {}
        for f in sorted(os.listdir(d)):
            if not f.endswith(".safetensors") or f.startswith("._") or (only and f not in only):
                continue
            p = os.path.join(d, f)
            hdr, md, base = MS.read_header(p)
            self.meta[f] = (md, hdr, base, os.path.getsize(p), p)
            for k, v in hdr.items():
                o0, o1 = v["data_offsets"]
                self.entries[k] = (p, base + o0, o1 - o0, v["dtype"], v["shape"])

    def raw(self, name, lo=0, n=None):
        p, off, nb, _d, _s = self.entries[name]
        n = nb - lo if n is None else n
        with open(p, "rb") as f:
            f.seek(off + lo)
            b = f.read(n)
        assert len(b) == n
        return b


# ---------------------------------------------------------------------------
# 0. rates
# ---------------------------------------------------------------------------
def engine_rate_tables():
    hdr = open(os.path.join(REPO, "src/engine/exl3_trellis.h")).read()
    k2_of_id = dict(re.findall(r"case PULSAR_TENSOR_(EXL3M_\w+):\s*return (\d+);", hdr))
    mc = open(os.path.join(REPO, "src/engine/model.cpp")).read()
    names = dict(re.findall(r'\{"(exl3m_\w+)",\s*PULSAR_TENSOR_(EXL3M_\w+)\}', mc))
    layout_k2 = {lay: int(k2_of_id[i]) for lay, i in names.items()}
    arms = {}
    for arm, expr in re.findall(r"case EXL3_ARM_(\w+):\s*return ([^;]+);", hdr):
        arms[arm] = {int(x) for x in re.findall(r"k2 == (\d+)", expr)}
    return layout_k2, arms


def test_rates(em, sc):
    layout_k2, arms = engine_rate_tables()
    check(layout_k2 == em.producers._EXL3_K2, f"emitter _EXL3_K2 {em.producers._EXL3_K2} != engine {layout_k2}")
    md = sc.meta[MS.SIDECAR][0]
    rows = []
    for name, t in json.loads(md["pulsar.tensors"]).items():
        if t["layout"].startswith("exl3m_"):
            rows.append((name, t["layout"], "DENSE"))
    for fam in json.loads(md["pulsar.experts"]):
        rows.append((fam["gguf_name"], fam["layout"], "DOWN" if fam["part"] == "down_proj" else "PAIR"))
    for name, lay, arm in rows:
        k2 = layout_k2[lay]
        check(k2 in arms[arm], f"{name}: {lay} (k2 {k2}) is not a rate the {arm} arm reads {sorted(arms[arm])}")
    print(f"[0] rates: engine table {layout_k2}; {len(rows)} EXL3 tensors/families admitted by their arms "
          f"(arms {dict((a, sorted(v)) for a, v in arms.items())})", flush=True)


# ---------------------------------------------------------------------------
# 1. EXL3
# ---------------------------------------------------------------------------
def exl3_items(sc):
    """(sidecar entry name, source key, k, n, layout) for every EXL3 slice in the sidecar."""
    md = sc.meta[MS.SIDECAR][0]
    out = []
    for name, t in json.loads(md["pulsar.tensors"]).items():
        if t["layout"].startswith("exl3m_"):
            k, n = t["dims_ne"]
            out.append((name, name[:-len(".weight")], k, n, t["layout"]))
    for fam in json.loads(md["pulsar.experts"]):
        k, n = fam["dims_per_expert_ne"]
        for e in range(fam["n_experts"]):
            nm = fam["entry_name"].replace("{e}", str(e))
            out.append((nm, nm[:-len(".weight")], k, n, fam["layout"]))
    return out


def test_exl3(em, sc, tb, hf, n_sample):
    words_of = {lay: 16 * (k2 >> 1) + (8 if k2 & 1 else 0) for lay, k2 in em.producers._EXL3_K2.items()}
    items = exl3_items(sc)
    n_ok = 0
    sampled = set()
    for name, key, k, n, lay in items:
        words = words_of[lay]
        tbytes = (k // 16) * (n // 16) * words * 2
        blob = sc.raw(name)
        if not check(len(blob) == tbytes + 2 * (k + n), f"{name}: {len(blob)} B, layout says {tbytes + 2 * (k + n)}"):
            continue
        t_s, suh_s, svh_s = blob[:tbytes], blob[tbytes:tbytes + 2 * k], blob[tbytes + 2 * k:]
        tsh = tb.entries[key + ".trellis"][4]
        check(tsh == [k // 16, n // 16, words], f"{key}.trellis shape {tsh}, expected [{k // 16}, {n // 16}, {words}]")
        t_o, suh_o, svh_o = tb.raw(key + ".trellis"), tb.raw(key + ".suh"), tb.raw(key + ".svh")
        check(struct.unpack("<I", tb.raw(key + ".mul1"))[0] == MUL1, f"{key}.mul1 is not the mul1 codebook")
        same_raw = t_s == t_o and suh_s == suh_o and svh_s == svh_o
        wh_s = decode_w_hat(t_s, k, n, words)
        wh_o = decode_w_hat(t_o, k, n, words)
        su_s, su_o = np.frombuffer(suh_s, "<u2"), np.frombuffer(suh_o, "<u2")
        sv_s, sv_o = np.frombuffer(svh_s, "<u2"), np.frombuffer(svh_o, "<u2")
        same_dec = np.array_equal(wh_s, wh_o) and np.array_equal(su_s, su_o) and np.array_equal(sv_s, sv_o)
        if check(same_raw and same_dec, f"{name}: raw equal {same_raw}, decoded equal {same_dec}"):
            n_ok += 1
        m = re.match(r"^(mtp\.layers\.\d+\.mlp\.experts\.)(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name)
        full = m is None or int(m.group(2)) in n_sample
        if full:
            w_s, w_o = reconstruct(wh_s, su_s, sv_s), reconstruct(wh_o, su_o, sv_o)
            check(np.array_equal(w_s.view(np.uint64), w_o.view(np.uint64)), f"{name}: reconstruction differs")
            if m is None:
                src = bf16_f64(hf.raw(name), hf.shape(name))                   # [out, in]
                r = rel(w_s.T, src)
                print(f"    {name:62s} {lay}  [{k}->{n}]  rel.err vs BF16 {r:.4f}", flush=True)
            else:
                pre, e, part = m.group(1), int(m.group(2)), m.group(3)
                if part == "down_proj":
                    stack = bf16_f64(hf.raw(pre + "down_proj"), hf.shape(pre + "down_proj"))
                    r, rx = rel(w_s.T, stack[e]), None
                else:
                    stack = bf16_f64(hf.raw(pre + "gate_up_proj"), hf.shape(pre + "gate_up_proj"))
                    half = stack.shape[1] // 2
                    mine, other = (stack[e, :half], stack[e, half:]) if part == "gate_proj" else (stack[e, half:], stack[e, :half])
                    r, rx = rel(w_s.T, mine), rel(w_s.T, other)
                    check(r < 0.5 < rx, f"{name}: rel.err {r:.3f} vs its half, {rx:.3f} vs the other -- gate/up order?")
                sampled.add(name)
                print(f"    {name:62s} {lay}  rel.err vs BF16 {r:.4f}" + (f" (vs the other half {rx:.3f})" if rx else ""),
                      flush=True)
            check(r < 0.5, f"{name}: rel.err {r:.3f} vs BF16 -- orientation or decode wrong")
    print(f"[1] EXL3: {n_ok}/{len(items)} slices bit-identical to turboderp's (raw + decoded W_hat/suh/svh); "
          f"full reconstructions graded: dense + {len(sampled)} expert projections", flush=True)


# ---------------------------------------------------------------------------
# 2. native
# ---------------------------------------------------------------------------
def test_native(em, sc, hf, base_dir):
    PR = em.producers
    md = sc.meta[MS.SIDECAR][0]
    nb = nm = 0
    for name, t in sorted(json.loads(md["pulsar.tensors"]).items()):
        if t["layout"] == "bf16":
            check(sc.entries[name][3] == "BF16" and sc.entries[name][4] == hf.shape(name),
                  f"{name}: declared {sc.entries[name][3]} {sc.entries[name][4]}")
            if check(sc.raw(name) == hf.raw(name), f"{name}: bf16 bytes differ from the BF16 source"):
                nb += 1
        elif t["layout"] == "mxfp8_lt":
            inp, out = t["dims_ne"]
            got = sc.raw(name)
            same = got == PR.mxfp8_lt_from_bf16(hf.raw(name), out, inp)
            src = bf16_f64(hf.raw(name), [out, inp])
            dec = PR.mxfp8_lt_decode(got, out, inp).astype(np.float64)
            sf = PR.unswizzle_sf(np.frombuffer(got, np.uint8, offset=out * inp), out, inp // 32).astype(np.int64) - 127
            ulp = np.ldexp(1.0, np.repeat(sf, 32, axis=1) - 9)          # E4M3 subnormal step in the group's scale
            bound = np.maximum(np.abs(src) * 2.0 ** -4, ulp / 2)
            within = bool((np.abs(dec - src) <= bound).all())
            if check(same and within, f"{name}: producer re-run equal {same}, within E4M3 rounding {within}"):
                nm += 1
            print(f"    {name:62s} mxfp8_lt [{inp}->{out}] rel.err vs BF16 {rel(dec, src):.5f}", flush=True)
    # the producer is the container's: re-derive one main-layer mxfp8_lt tensor of the base
    probe = "model.language_model.layers.3.attn_hyper_connection.input_mix_weight_down.weight"
    bd = ShardDir(base_dir, only={f for f in os.listdir(base_dir) if f.endswith(".safetensors")})
    out, inp = hf.shape(probe)
    check(bd.raw(probe) == PR.mxfp8_lt_from_bf16(hf.raw(probe), out, inp),
          f"{probe}: the emitter's producer does not reproduce the base container's bytes")
    print(f"[2] native: {nb} bf16 byte-identical to BF16, {nm} mxfp8_lt byte-identical to the producer and within "
          f"E4M3 rounding; producer reproduces base {probe}", flush=True)
    return bd


# ---------------------------------------------------------------------------
# 3. names / the loader
# ---------------------------------------------------------------------------
def test_names(em, sc, bd, hf, combined):
    PR = em.producers
    # names.py is the DeepSeek table: it refuses a qwen4_exp config and maps no Qwen name
    names_mod = __import__("names")
    try:
        names_mod.ModelShape.from_config(hf.config["top_level"])
        check(False, "names.ModelShape accepted a qwen4_exp config")
    except ValueError as e:
        print(f"    names.py: ModelShape.from_config refuses qwen4_exp ({e}); the Qwen family names by HF name "
              f"(qwen.py: gguf_name == HF name / HF stack name), which is what is checked below", flush=True)
    ds = names_mod.ModelShape(n_layer=48, n_mtp=1, has_vision=True, v41=True, n_hash=0)
    mapped = [n for n in sc.entries if names_mod.map_hf(n, ds) is not None]
    check(not mapped, f"names.py maps sidecar names under the DeepSeek table: {mapped[:4]}")

    # the loader over combined/: the primary, then st_add_declared + st_add_expert_stacks per shard
    cd = ShardDir(combined)
    prim = [f for f, (md, *_r) in cd.meta.items() if "pulsar.kv" in md]
    check(prim == [bd_primary(bd)], f"shards carrying pulsar.kv: {prim}")
    bound = {}
    for f, (md, hdr, base, size, _p) in cd.meta.items():
        for hfn, t in json.loads(md.get("pulsar.tensors", "{}")).items():
            if not check(hfn in hdr, f"{f}: declares {hfn}, not in the file"):
                continue
            want = PR.bytes_for(t["layout"], t["dims_ne"], hdr[hfn]["dtype"])
            o0, o1 = hdr[hfn]["data_offsets"]
            check(want == o1 - o0, f"{f}: {hfn} declares {want} B, holds {o1 - o0}")
            check(t["gguf_name"] not in bound, f"gguf_name {t['gguf_name']} bound twice")
            bound[t["gguf_name"]] = (f, t["layout"], list(t["dims_ne"]))
        for fam in json.loads(md.get("pulsar.experts", "[]")):
            first = None
            for e in range(fam["n_experts"]):
                tn = fam["entry_name"].replace("{e}", str(e))
                if not check(tn in hdr, f"{f}: missing {tn}"):
                    break
                o0, o1 = hdr[tn]["data_offsets"]
                check(o1 - o0 == fam["expert_bytes"], f"{tn}: {o1 - o0} B != expert_bytes")
                first = o0 if e == 0 else first
                check(o0 == first + e * fam["expert_bytes"], f"{tn}: not contiguous")
            k, n = fam["dims_per_expert_ne"]
            check(PR.bytes_for(fam["layout"], [k, n]) == fam["expert_bytes"], f"{fam['gguf_name']}: expert_bytes model")
            check(fam["gguf_name"] not in bound, f"gguf_name {fam['gguf_name']} bound twice")
            bound[fam["gguf_name"]] = (f, fam["layout"], [k, n, fam["n_experts"]])

    # the Qwen binder's view of the MTP block (family_qwen.cpp qbind: HF name, ne dims)
    c = hf.config
    E, V = c["hidden_size"], c["vocab_size"]
    hc, lr, nhc = c["hidden_size"] * c["hc_count"], c["hc_lowrank"], c["hc_count"]
    q_out, kv_out = 2 * c["num_attention_heads"] * c["head_dim"], c["num_key_value_heads"] * c["head_dim"]
    o_in = c["num_attention_heads"] * c["head_dim"]
    idx_out = (c["indexer_n_heads"] + c["indexer_kv_heads"]) * c["indexer_head_dim"]
    ff, nx, ffs = c["moe_intermediate_size"], c["num_experts"], c["shared_expert_intermediate_size"]
    L = "mtp.layers.0."
    want = {
        "mtp.fc_embedding.weight": [E, E], "mtp.fc_hidden.weight": [E, E],
        "mtp.pre_fc_norm_embedding.weight": [E], "mtp.pre_fc_norm_hidden.weight": [hc],
        "mtp.hyper_connection_mixer.hc_norm.weight": [hc],
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": [hc, lr],
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": [lr, hc],
    }
    for g in ("attn", "mlp"):
        p = f"{L}{g}_hyper_connection."
        want.update({p + "hc_norm.weight": [hc], p + "input_mix_weight_down.weight": [hc, lr],
                     p + "input_mix_weight_up.weight": [lr, hc], p + "block_inject_weight.weight": [hc, nhc]})
    want.update({
        L + "self_attn.q_proj.weight": [E, q_out], L + "self_attn.k_proj.weight": [E, kv_out],
        L + "self_attn.v_proj.weight": [E, kv_out], L + "self_attn.o_proj.weight": [o_in, E],
        L + "self_attn.q_norm.weight": [c["head_dim"]], L + "self_attn.k_norm.weight": [c["head_dim"]],
        L + "self_attn.indexer.index_qk_proj.weight": [E, idx_out],
        L + "self_attn.indexer.q_layernorm.weight": [c["indexer_head_dim"]],
        L + "self_attn.indexer.k_layernorm.weight": [c["indexer_head_dim"]],
        L + "mlp.gate.weight": [E, nx],
        L + "mlp.experts.gate_proj": [E, ff, nx], L + "mlp.experts.up_proj": [E, ff, nx],
        L + "mlp.experts.down_proj": [ff, E, nx],
        L + "mlp.shared_expert.gate_proj.weight": [E, ffs], L + "mlp.shared_expert.up_proj.weight": [E, ffs],
        L + "mlp.shared_expert.down_proj.weight": [ffs, E], L + "mlp.shared_expert_gate.weight": [E, 1],
    })
    del V
    mtp_bound = {g: v for g, v in bound.items() if g.startswith("mtp.")}
    check(set(mtp_bound) == set(want), f"bound mtp names != binder names: extra {sorted(set(mtp_bound) - set(want))}, "
                                       f"missing {sorted(set(want) - set(mtp_bound))}")
    for g, dims in want.items():
        if g in mtp_bound:
            check(mtp_bound[g][2] == dims, f"{g}: ne {mtp_bound[g][2]}, binder wants {dims}")
            check(mtp_bound[g][0] == MS.SIDECAR, f"{g}: bound from {mtp_bound[g][0]}")
    n_base = len(bound) - len(mtp_bound)
    print(f"[3] names: combined dir binds {len(bound)} engine tensors ({n_base} base + {len(mtp_bound)} mtp), "
          f"primary {prim}; bound MTP key list (gguf_name, layout, ne):", flush=True)
    for g in sorted(mtp_bound):
        print(f"      {g:62s} {mtp_bound[g][1]:9s} {mtp_bound[g][2]}", flush=True)


def bd_primary(bd):
    return [f for f, (md, *_r) in bd.meta.items() if "pulsar.kv" in md][0]


def test_meta(em, sc, bd, base_dir):
    md = sc.meta[MS.SIDECAR][0]
    check("pulsar.kv" not in md, "the sidecar carries pulsar.kv")
    kv = {e["key"]: e for e in json.loads(md["pulsar.mtp.kv"])}
    check(kv["pulsar.mtp_present"]["value"] is True, "pulsar.mtp_present")
    prim = bd_primary(bd)
    check(kv["pulsar.mtp.base.primary_sha256"]["value"] == MS.sha256_file(os.path.join(base_dir, prim)),
          "base primary sha256 differs from the recorded one")
    print(f"[meta] sidecar metadata keys: {sorted(md)}", flush=True)
    for k, e in kv.items():
        print(f"      {k} {e['type']} {json.dumps(e['value'])[:100]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--exl3", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--emitter-rev", default=MS.EMITTER_REV)
    ap.add_argument("--sample-experts", default="0,1,255,511")
    a = ap.parse_args()
    em = MS.Emitter(a.emitter_rev)
    hf = em.hf_source.HFCheckpoint(a.hf)
    sc = ShardDir(a.sidecar, only={MS.SIDECAR})
    tb = ShardDir(a.exl3, only={f for f in os.listdir(a.exl3) if f.startswith("model-")})
    test_rates(em, sc)
    test_exl3(em, sc, tb, hf, {int(x) for x in a.sample_experts.split(",")})
    bd = test_native(em, sc, hf, a.base)
    test_names(em, sc, bd, hf, os.path.join(a.sidecar, "combined"))
    test_meta(em, sc, bd, a.base)
    print("RESULT:", "PASS" if not FAIL else f"FAIL ({len(FAIL)})", flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
