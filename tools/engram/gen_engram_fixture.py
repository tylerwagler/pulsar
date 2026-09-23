#!/usr/bin/env python3
"""Engram DEVICE-PATH fixture (L242): real rows, real `wkv`/`q`/`k`, the reference's math in f32.

The engine's Engram path is: hash -> 24 table rows -> the rows AS E4M3+E8M0 into the MXFP8
activation slot -> `wkv` GEMM -> the per-hc-copy gate -> a gated add into the bf16 residual copies.
tests/engram_gate runs that path on one GPU and grades it against this fixture, which holds:

  * the token ids of a fixed window and the 24 row ids per token the reference hash names
    (transcribed from engram.py / engram.cpp; the hash itself is separately gated by
    engram-hash-check, so this is a second, independent statement of the same arithmetic),
  * the 264-byte rows themselves, read from the checkpoint shard at those ids,
  * a seeded bf16 residual `h [T, 4, 5120]`,
  * the reference's `kv [T, 25600]` and `h_out [T, 4, 5120]` in f32, computed exactly as
    model.py's Engram.forward does with every operand dequantised to f32 (rows: E4M3 x E8M0;
    `wkv`: E4M3 x its 32x32 block E8M0); the gate's own quantisation of `wkv`'s activation and
    the MX GEMM's accumulation order are the deviations the gate measures.

Nothing here needs the 189 GiB row files: rows are read straight from the shard.

usage: gen_engram_fixture.py SNAPSHOT_DIR LAYOUT_JSON LAYER OUT.fix [--tokens N] [--seed S]
  SNAPSHOT_DIR  the HF snapshot (inference/config.json + model-000NN-of-00048.safetensors)
  LAYOUT_JSON   gate-baseline/l218-v41/engram-layout-v41.json (token map, primes, offsets, multipliers)
"""
import argparse, json, os, struct, sys
import numpy as np

DIM, N_SCALE, ROW_BYTES = 256, 8, 264
MAX_NGRAM, N_HEADS = 4, 8
N_COLS = (MAX_NGRAM - 1) * N_HEADS
HC_MULT, MODEL_DIM = 4, 5120
WKV_IN, WKV_OUT = N_COLS * DIM, MODEL_DIM * (HC_MULT + 1)
EPS = 1e-20            # V4.1 norm_eps
CLAMP = 1e-6

# ---- number formats ---------------------------------------------------------------------
def e4m3_lut():
    """float8_e4m3fn: sign, 4-bit exponent (bias 7), 3-bit mantissa; no inf; 0x7f/0xff = NaN."""
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 7
        if e == 0xF and m == 7:
            lut[b] = np.nan
        elif e == 0:
            lut[b] = s * (m / 8.0) * 2.0 ** (-6)
        else:
            lut[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return lut

E4M3 = e4m3_lut()

def e8m0_to_f32(b):
    b = np.asarray(b, dtype=np.int32)
    out = np.exp2((b - 127).astype(np.float32))
    return np.where(b == 255, np.nan, out).astype(np.float32)

def bf16_bits_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)

def f32_to_bf16_bits(f):
    u = np.asarray(f, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)          # round to nearest even
    return (u >> 16).astype(np.uint16)

# ---- safetensors ---------------------------------------------------------------------------
def st_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n

def st_locate(snapshot, name):
    idx = json.load(open(os.path.join(snapshot, 'model.safetensors.index.json')))['weight_map']
    shard = os.path.join(snapshot, idx[name])
    hdr, data_start = st_header(shard)
    t = hdr[name]
    return shard, t['dtype'], t['shape'], data_start + t['data_offsets'][0], data_start + t['data_offsets'][1]

def read_at(path, off, n):
    with open(path, 'rb') as f:
        f.seek(off)
        b = f.read(n)
    assert len(b) == n, (path, off, n, len(b))
    return np.frombuffer(b, dtype=np.uint8)

# ---- the hash (engram.py NgramHashState.forward; engram.cpp pulsar_engram_hash_pos) --------
def hash_cols(layout, li, ids, pos):
    tm = layout['token_map']
    mult = [int(m) for m in layout['multipliers'][li]]
    primes = [int(p) for per in layout['primes'][li] for p in per]
    offs = [int(o) for o in layout['offsets'][li]]
    pad = int(layout['pad_compressed_id'])
    cols = [0] * N_COLS
    rolling = 0
    for i in range(MAX_NGRAM):
        tok = pad if pos < i else tm[ids[pos - i]]
        prod = tok * mult[i]
        if i == 0:
            rolling = prod
            continue
        rolling ^= prod
        base = (i - 1) * N_HEADS
        for h in range(N_HEADS):
            cols[base + h] = rolling % primes[base + h] + offs[base + h]
    return cols

# ---- main ------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('snapshot'); ap.add_argument('layout'); ap.add_argument('layer', type=int); ap.add_argument('out')
    ap.add_argument('--tokens', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0x4C242)
    a = ap.parse_args()
    layout = json.load(open(a.layout))
    li = layout['layer_ids'].index(a.layer)
    n_rows = int(layout['num_embeddings'][li])

    # a fixed window of REAL ids: the hash fixture's eight plus eight more, so both the
    # pad-filled first positions and full 4-gram windows appear
    ids = [128000, 42, 7, 100, 5, 999, 31337, 88, 1234, 56789, 2, 90210, 4242, 128001, 17, 65535][:a.tokens]
    T = len(ids)
    cols = np.array([hash_cols(layout, li, ids, p) for p in range(T)], dtype=np.uint64)
    assert cols.max() < n_rows

    pre = f'layers.{a.layer}.engram.'
    shard_w, dt_w, shp_w, w0, _ = st_locate(a.snapshot, pre + 'embed.weight')
    shard_s, dt_s, shp_s, s0, _ = st_locate(a.snapshot, pre + 'embed.scale')
    assert dt_w == 'F8_E4M3' and dt_s == 'F8_E8M0' and shp_w == [n_rows, DIM] and shp_s == [n_rows, N_SCALE], (dt_w, dt_s, shp_w, shp_s)
    rows = np.zeros((T, N_COLS, ROW_BYTES), dtype=np.uint8)
    for t in range(T):
        for c in range(N_COLS):
            r = int(cols[t, c])
            rows[t, c, :DIM] = read_at(shard_w, w0 + r * DIM, DIM)
            rows[t, c, DIM:] = read_at(shard_s, s0 + r * N_SCALE, N_SCALE)
    # dequantised activation [T, 6144]: value * scale of its 32-block
    vals = E4M3[rows[:, :, :DIM].astype(np.int32)].reshape(T, N_COLS, N_SCALE, 32)
    scl = e8m0_to_f32(rows[:, :, DIM:]).reshape(T, N_COLS, N_SCALE, 1)
    x = (vals * scl).reshape(T, WKV_IN).astype(np.float32)

    # wkv [25600, 6144] E4M3 with [800, 192] E8M0 (32x32 blocks)
    shard_k, dt_k, shp_k, k0, k1 = st_locate(a.snapshot, pre + 'wkv.weight')
    shard_ks, dt_ks, shp_ks, ks0, ks1 = st_locate(a.snapshot, pre + 'wkv.scale')
    assert dt_k == 'F8_E4M3' and shp_k == [WKV_OUT, WKV_IN] and dt_ks == 'F8_E8M0' and shp_ks == [WKV_OUT // 32, WKV_IN // 32]
    wq = read_at(shard_k, k0, k1 - k0).reshape(WKV_OUT, WKV_IN)
    ws = e8m0_to_f32(read_at(shard_ks, ks0, ks1 - ks0)).reshape(WKV_OUT // 32, WKV_IN // 32)
    w = E4M3[wq.astype(np.int32)].reshape(WKV_OUT // 32, 32, WKV_IN // 32, 32) * ws[:, None, :, None]
    w = w.reshape(WKV_OUT, WKV_IN).astype(np.float32)
    kv = (x.astype(np.float64) @ w.astype(np.float64).T).astype(np.float32)   # [T, 25600], exact-ish reference

    # q/k weights [4, 5120] bf16, used only as their product
    _, dt_q, shp_q, q0, q1 = st_locate(a.snapshot, pre + 'q_weight')
    shard_q = st_locate(a.snapshot, pre + 'q_weight')[0]
    _, dt_kk, shp_kk, kk0, kk1 = st_locate(a.snapshot, pre + 'k_weight')
    shard_kk = st_locate(a.snapshot, pre + 'k_weight')[0]
    assert dt_q == 'BF16' and dt_kk == 'BF16' and shp_q == [HC_MULT, MODEL_DIM] and shp_kk == [HC_MULT, MODEL_DIM]
    qb = read_at(shard_q, q0, q1 - q0).view(np.uint16).reshape(HC_MULT, MODEL_DIM)
    kb = read_at(shard_kk, kk0, kk1 - kk0).view(np.uint16).reshape(HC_MULT, MODEL_DIM)
    weight = bf16_bits_to_f32(qb) * bf16_bits_to_f32(kb)                     # f32 [4, 5120]

    # a seeded residual in bf16, magnitudes like a real stream
    rng = np.random.default_rng(a.seed)
    h_bits = f32_to_bf16_bits(rng.standard_normal((T, HC_MULT, MODEL_DIM)).astype(np.float32) * 0.7)
    h = bf16_bits_to_f32(h_bits).reshape(T, HC_MULT, MODEL_DIM)

    # Engram.forward, f32
    key = kv[:, :HC_MULT * MODEL_DIM].reshape(T, HC_MULT, MODEL_DIM)
    value = kv[:, HC_MULT * MODEL_DIM:]                                        # [T, 5120]
    rstd = (1.0 / np.sqrt(np.mean(h * h, -1) + EPS)) * (1.0 / np.sqrt(np.mean(key * key, -1) + EPS))
    dot = np.sum(h * weight[None] * key, -1) * rstd * MODEL_DIM ** -0.5       # [T, 4]
    gate = 1.0 / (1.0 + np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(dot), CLAMP)), dot)))
    h_out = (h + gate[..., None] * value[:, None, :]).astype(np.float32)

    with open(a.out, 'wb') as f:
        f.write(b'PENGFIX1')
        f.write(struct.pack('<IIIIIIf', a.layer, T, HC_MULT, MODEL_DIM, N_COLS, ROW_BYTES, EPS))
        f.write(np.array(ids, dtype=np.int32).tobytes())
        f.write(cols.astype(np.uint64).tobytes())
        f.write(rows.tobytes())
        f.write(h_bits.astype(np.uint16).tobytes())
        f.write(weight.astype(np.float32).tobytes())
        f.write(kv.astype(np.float32).tobytes())
        f.write(dot.astype(np.float32).tobytes())
        f.write(gate.astype(np.float32).tobytes())
        f.write(h_out.astype(np.float32).tobytes())
    print(f'wrote {a.out}: layer {a.layer}, {T} tokens, rows in [{int(cols.min())}, {int(cols.max())}] of {n_rows}, '
          f'gate range [{gate.min():.4f}, {gate.max():.4f}], |kv| max {np.abs(kv).max():.3f}, {os.path.getsize(a.out)} bytes')

if __name__ == '__main__':
    main()
