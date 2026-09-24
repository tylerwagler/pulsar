#!/usr/bin/env python3
"""Grade tools/container/producers.py against the served source-precision
repack (built by the archived C codecs from the HF checkpoint).  Dev box, no
GPU; reads only the byte ranges it needs from both trees.

  * mxfp8_lt 'rederive'  -> BYTE-IDENTICAL to the served bytes (4 shapes)
  * mxfp8_lt 'verbatim'  -> decodes to the same VALUES except the design's
                            edge case; code/value difference counts reported
  * cutlass_mxfp4        -> byte-identical for every expert of one layer, 3 parts
  * fp8_e4m3_soa_k       -> byte-identical (the drafter's markov_w2)
  * native / i64_to_i32-> byte-identical
  * bytes_for            -> == every served tensor's span, all 48 shards
  * self-checks: the vectorised E4M3 encoder against a scalar port of the C,
    swizzle_sf against mx_sfoff on every shape touched

Exit status is non-zero on any byte difference in an exact producer.
"""

from __future__ import annotations

import glob
import json
import math
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import producers as P  # noqa: E402

SERVED = os.environ.get("PULSAR_SERVED", "/mnt/models/DeepSeek-v4-Flash")
HF = os.environ.get("PULSAR_HF_SNAPSHOT") or sorted(glob.glob(
    "/mnt/models/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/*/"))[0]

FAILS: list[str] = []
NOTES: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print("FAIL", msg)


def ok(msg: str) -> None:
    print("ok  ", msg)


# ---------------------------------------------------------------- readers

class Shard:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.meta = self.header.get("__metadata__", {})

    def entry(self, name: str) -> dict:
        return self.header[name]

    def read(self, name: str) -> bytes:
        o0, o1 = self.header[name]["data_offsets"]
        return self.read_range(o0, o1 - o0)

    def read_range(self, off: int, n: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(self.base + off)
            b = f.read(n)
        if len(b) != n:
            raise IOError(f"{self.path}: short read at {off}")
        return b


class Tree:
    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.index = json.load(f)["weight_map"]
        self._shards: dict[str, Shard] = {}

    def shard_of(self, name: str) -> Shard:
        return self.shard(self.index[name])

    def shard(self, file: str) -> Shard:
        if file not in self._shards:
            self._shards[file] = Shard(os.path.join(self.root, file))
        return self._shards[file]

    def entry(self, name: str) -> dict:
        return self.shard_of(name).entry(name)

    def read(self, name: str) -> bytes:
        return self.shard_of(name).read(name)


# ---------------------------------------------------------------- self-checks

def c_f32_to_e4m3(x: float) -> int:
    """Scalar port of ds4q_f32_to_e4m3 (quants_fp.c:23-50), independent of the
    vectorised encoder: the mutation reference."""
    if x != x:
        return 0x7F
    s = 0x80 if x < 0.0 else 0
    a = abs(x)
    if a >= 448.0:
        return s | 0x7E
    if a == 0.0:
        return s
    u = struct.unpack("<I", struct.pack("<f", a))[0]
    exp = ((u >> 23) & 0xFF) - 127
    sig = (1 << 23) | (u & 0x7FFFFF)
    e8 = exp + 7
    if e8 <= 0:
        code = int(np.rint(np.float32(a) * np.float32(512.0)))  # lrintf, RNE
        if code > 7:
            return s | (1 << 3)
        return s | (code & 7)
    mant3 = (sig >> 20) & 7
    rem = sig & ((1 << 20) - 1)
    half = 1 << 19
    if rem > half or (rem == half and (mant3 & 1)):
        mant3 += 1
        if mant3 == 8:
            mant3 = 0
            e8 += 1
    if e8 > 15 or (e8 == 15 and mant3 >= 7):
        return s | 0x7E
    return s | ((e8 & 0xF) << 3) | (mant3 & 7)


def check_e4m3_encoder() -> None:
    rng = np.random.default_rng(1)
    # dense coverage of the E4M3 range and the rounding boundaries
    xs = [rng.uniform(-600, 600, 20000).astype(np.float32),
          rng.uniform(-1.0, 1.0, 20000).astype(np.float32),
          (rng.uniform(-1, 1, 20000) * 2.0 ** rng.integers(-16, 3, 20000)).astype(np.float32),
          np.array([0.0, -0.0, 448.0, -448.0, 447.99, 464.0, 1e-4, 2**-9, 3 * 2**-10, 5 * 2**-10,
                    7 * 2**-10, 15 * 2**-10, 2**-6, 1.0, 1.0625, 1.1875, 255.9, 256.0, 416.0,
                    432.0, 440.0, 1e-30, -1e-30, 3.0e38, np.inf, -np.inf], dtype=np.float32)]
    # every representable E4M3 value scaled by every shift the table can see
    grid = np.ldexp(P.E4M3_VALUE[:, None], np.arange(-3, 20)[None, :]).astype(np.float32).ravel()
    xs.append(grid)
    # exact midpoints between adjacent E4M3 codes (ties -> even)
    mids = []
    for c in range(1, 0x7E):
        mids.append((float(P.E4M3_VALUE[c]) + float(P.E4M3_VALUE[c + 1])) / 2.0)
    xs.append(np.array(mids + [-m for m in mids], dtype=np.float32))
    x = np.concatenate(xs)
    got = P.f32_to_e4m3(x)
    want = np.array([c_f32_to_e4m3(float(v)) for v in x], dtype=np.uint8)
    bad = np.nonzero(got != want)[0]
    if bad.size:
        fail(f"f32_to_e4m3 disagrees with the scalar C port at {bad.size} of {x.size} inputs, "
             f"e.g. x={x[bad[0]]!r} got 0x{got[bad[0]]:02x} want 0x{want[bad[0]]:02x}")
    else:
        ok(f"f32_to_e4m3 == scalar C port on {x.size} inputs (E4M3 grid x shifts, midpoints, random)")
    try:
        P.f32_to_e4m3(np.array([np.nan], dtype=np.float32))
        fail("f32_to_e4m3 accepted NaN")
    except ValueError:
        ok("f32_to_e4m3 refuses NaN (the C dies on it)")


def check_swizzle(rows: int, kb_n: int) -> None:
    scale = np.random.default_rng(rows * 7 + kb_n).integers(0, 255, (rows, kb_n), dtype=np.uint8)
    rows_pad, kb_pad = P._rup(rows, 128), P._rup(kb_n, 4)
    ref = np.zeros(rows_pad * kb_pad, dtype=np.uint8)
    r = np.arange(rows)[:, None]
    kb = np.arange(kb_n)[None, :]
    ref[P.mx_sfoff(r, kb, kb_pad)] = scale
    got = P.swizzle_sf(scale, rows, kb_n)
    if not np.array_equal(ref, got):
        fail(f"swizzle_sf != mx_sfoff gather on [{rows}, {kb_n}]")
        return
    back = P.unswizzle_sf(got, rows, kb_n)
    if not np.array_equal(back, scale):
        fail(f"unswizzle_sf does not invert swizzle_sf on [{rows}, {kb_n}]")
        return
    ok(f"swizzle_sf == mx_sfoff on [{rows} rows, {kb_n} kb] (pad {rows_pad}x{kb_pad})")


# ---------------------------------------------------------------- decoders (for the verbatim grading)

def decode_mxfp8_lt(blob: bytes, out: int, inp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (values f64 [out, in], codes u8 [out, in], scale plane u8 [out, in/32])."""
    kb_n = inp // 32
    codes = np.frombuffer(blob, dtype=np.uint8, count=out * inp).reshape(out, inp)
    sf = np.frombuffer(blob, dtype=np.uint8, offset=out * inp)
    scale = P.unswizzle_sf(sf, out, kb_n)
    vals = P.E4M3_VALUE[codes].astype(np.float64)
    exps = np.repeat(scale.astype(np.int32) - 127, 32, axis=1)
    return np.ldexp(vals, exps), codes, scale


# ---------------------------------------------------------------- the served-vs-source runs

def run_mxfp8_lt(served: Tree, hf: Tree, name: str) -> None:
    sh = served.shard_of(name)
    decl = json.loads(sh.meta["pulsar.tensors"])[name]
    if decl["layout"] != "mxfp8_lt":
        fail(f"{name}: served layout is {decl['layout']}, not mxfp8_lt")
        return
    inp, out = decl["dims_ne"]
    we = hf.entry(name)
    se = hf.entry(name[:-len(".weight")] + ".scale")
    if we["dtype"] != "F8_E4M3" or se["dtype"] != "F8_E8M0" or we["shape"] != [out, inp]:
        fail(f"{name}: source is {we['dtype']} {we['shape']} / {se['dtype']}, expected F8_E4M3 [{out},{inp}]")
        return
    block = out // se["shape"][0]
    if se["shape"] != [out // block, inp // block]:
        fail(f"{name}: scale shape {se['shape']} does not tile [{out},{inp}] squarely")
        return
    w = hf.read(name)
    s = hf.read(name[:-len(".weight")] + ".scale")
    got_served = sh.read(name)

    t0 = time.perf_counter()
    redo = P.mxfp8_lt(w, s, out, inp, block, "rederive")
    t_re = time.perf_counter() - t0
    t0 = time.perf_counter()
    verb = P.mxfp8_lt(w, s, out, inp, block, "verbatim")
    t_vb = time.perf_counter() - t0

    n_bytes = P.bytes_for("mxfp8_lt", decl["dims_ne"])
    if len(redo) != n_bytes or len(got_served) != n_bytes:
        fail(f"{name}: rederive {len(redo)} B, served {len(got_served)} B, bytes_for {n_bytes} B")
    if redo == got_served:
        ok(f"mxfp8_lt rederive  {name} [out={out},in={inp}] block {block}: BYTE-IDENTICAL "
           f"({n_bytes / 2**20:.1f} MiB, {t_re * 1e3:.0f} ms)")
    else:
        a = np.frombuffer(redo, np.uint8)
        b = np.frombuffer(got_served, np.uint8)
        diff = np.nonzero(a != b)[0]
        where = "data" if diff[0] < out * inp else "SF plane"
        fail(f"mxfp8_lt rederive  {name}: {diff.size} bytes differ, first at {diff[0]} ({where}): "
             f"got 0x{a[diff[0]]:02x} served 0x{b[diff[0]]:02x}")
    check_swizzle(out, inp // 32)

    # verbatim grading: same values except the design's edge case
    v_val, v_codes, v_scale = decode_mxfp8_lt(verb, out, inp)
    r_val, r_codes, r_scale = decode_mxfp8_lt(got_served, out, inp)
    # the source's own decode: what verbatim must reproduce exactly
    src_codes = np.frombuffer(w, np.uint8).reshape(out, inp)
    src_scale = np.repeat(np.repeat(np.frombuffer(s, np.uint8).reshape(out // block, inp // block),
                                    block, axis=0), block // 32, axis=1)
    if not (np.array_equal(v_codes, src_codes) and np.array_equal(v_scale, src_scale)):
        fail(f"mxfp8_lt verbatim  {name}: codes/scales are not the source's")
    # the expected shift per group from the source values, independently of producers.py
    absmax = np.abs(P.E4M3_VALUE[src_codes]).reshape(out, inp // 32, 32).max(axis=2)
    s_exp = src_scale.astype(np.int64) - 127
    fl = np.where(absmax > 0, np.floor(np.log2(np.where(absmax > 0, absmax, 1.0))), 0).astype(np.int64)
    t_exp = np.clip(s_exp + fl - 7, -127, 127)
    t_exp = np.where(absmax > 0, t_exp, -127)
    shifted = t_exp != s_exp
    raised = t_exp > s_exp    # exponent raised one binade (max |code| in 0x78..0x7e): every code HALVES
    lowered = t_exp < s_exp   # exponent lowered: codes double, always exact
    plane_differs = v_scale != r_scale
    if np.array_equal(plane_differs, shifted) and np.array_equal(r_scale.astype(int) - v_scale.astype(int), t_exp - s_exp):
        ok(f"mxfp8_lt verbatim  {name}: E8M0 planes differ exactly at the {int(shifted.sum())} shifted "
           f"groups of {shifted.size} (raised by one: {int(raised.sum())}, lowered: {int(lowered.sum())}, "
           f"max lowering {int((s_exp - t_exp).max())} binades)")
    else:
        fail(f"mxfp8_lt verbatim  {name}: E8M0 planes differ at {int(plane_differs.sum())} groups, "
             f"expected shifts at {int(shifted.sum())}")
    if int((t_exp - s_exp).max()) > 1:
        fail(f"mxfp8_lt verbatim  {name}: a group's exponent was raised by {int((t_exp - s_exp).max())} binades; the codec raises by at most one")
    code_diff = v_codes != r_codes
    val_diff = v_val != r_val
    code_in_unshifted = code_diff & ~np.repeat(shifted, 32, axis=1)
    quirk = np.isin(src_codes, [0x80, 0x7F, 0xFF])
    n_quirk_diff = int((code_in_unshifted & quirk).sum())
    if int((code_in_unshifted & ~quirk).sum()):
        fail(f"mxfp8_lt verbatim  {name}: {int((code_in_unshifted & ~quirk).sum())} codes differ in "
             f"unshifted groups that are not the zero-quirk codes")
    # The design's edge case: a halved code loses its LSB to RNE when it lands on
    # the subnormal grid with an odd LSB -- the odd subnormals (0x01..0x07) AND
    # the odd codes of the minimum normal binade (0x09..0x0f), which halve into
    # subnormals.  Everything else halves exactly.
    raised32 = np.repeat(raised, 32, axis=1)
    lsb_loss = ((src_codes & 0x70) == 0) & ((src_codes & 1) == 1)
    explained = (raised32 & lsb_loss) | quirk
    unexplained = val_diff & ~explained
    src_val = np.ldexp(P.E4M3_VALUE[src_codes].astype(np.float64), np.repeat(s_exp, 32, axis=1))
    if not np.array_equal(v_val, src_val):
        fail(f"mxfp8_lt verbatim  {name}: verbatim does not decode to the source's values")
    if unexplained.any():
        i = np.argwhere(unexplained)[0]
        fail(f"mxfp8_lt verbatim  {name}: {int(unexplained.sum())} VALUE differences outside the "
             f"design's edge case, e.g. ({i[0]},{i[1]}) src 0x{src_codes[i[0], i[1]]:02x} "
             f"verbatim {v_val[i[0], i[1]]!r} served {r_val[i[0], i[1]]!r}")
    else:
        n_pred = int((raised32 & lsb_loss & ~quirk).sum())
        if n_pred != int((val_diff & ~quirk).sum()):
            fail(f"mxfp8_lt verbatim  {name}: predicted {n_pred} LSB losses, found {int((val_diff & ~quirk).sum())}")
        maxd = float(np.abs(v_val - r_val).max()) if val_diff.any() else 0.0
        ok(f"mxfp8_lt verbatim  {name}: {int(code_diff.sum())} of {code_diff.size} codes differ from "
           f"rederive ({int(code_diff.sum()) - int(val_diff.sum())} value-preserving shifts); "
           f"{int(val_diff.sum())} VALUES differ = {n_pred} odd codes < 0x10 in raised groups (RNE on the "
           f"subnormal grid) + {int((val_diff & quirk).sum())} zero-quirk codes; max |dv| {maxd:.3g}; "
           f"verbatim {t_vb * 1e3:.0f} ms")
    NOTES.append((name, out, inp, block, int(shifted.sum()), int(raised.sum()), int(code_diff.sum()),
                  int(val_diff.sum()), n_quirk_diff, int(quirk.sum()), round(t_re, 3), round(t_vb, 3)))


def run_experts(served: Tree, hf: Tree, layer: int) -> None:
    sh = served.shard(served.index[f"layers.{layer}.ffn.experts.0.w1.weight"])
    stacks = json.loads(sh.meta["pulsar.experts"])
    for st in stacks:
        part = st["part"]
        k, n = st["dims_per_expert_ne"]
        n_exp = int(st["n_experts"])
        eb = int(st["expert_bytes"])
        model = P.bytes_for("cutlass_mxfp4", [k, n])
        if model != eb:
            fail(f"{st['gguf_name']}: bytes_for cutlass_mxfp4 [{k},{n}] = {model}, declared expert_bytes {eb}")
        first = sh.entry(f"layers.{layer}.ffn.experts.0.{part}.weight")["data_offsets"][0]
        t_read = time.perf_counter()
        stack = sh.read_range(first, n_exp * eb)
        t_read = time.perf_counter() - t_read
        t_prod = 0.0
        t_src = 0.0
        n_bad = 0
        first_bad = None
        for e in range(n_exp):
            wn = f"layers.{layer}.ffn.experts.{e}.{part}.weight"
            sn = wn[:-len(".weight")] + ".scale"
            we, se = hf.entry(wn), hf.entry(sn)
            if we["dtype"] != "I8" or se["dtype"] != "F8_E8M0" or we["shape"] != [n, k // 2] or se["shape"] != [n, k // 32]:
                fail(f"{wn}: source {we['dtype']} {we['shape']} / {se['dtype']} {se['shape']} is not I8 [{n},{k // 2}] + E8M0 [{n},{k // 32}]")
                return
            t0 = time.perf_counter()
            w = hf.read(wn)
            s = hf.read(sn)
            t_src += time.perf_counter() - t0
            t0 = time.perf_counter()
            got = P.cutlass_mxfp4(w, s, n, k)
            t_prod += time.perf_counter() - t0
            if got != stack[e * eb:(e + 1) * eb]:
                n_bad += 1
                if first_bad is None:
                    a = np.frombuffer(got, np.uint8)
                    b = np.frombuffer(stack[e * eb:(e + 1) * eb], np.uint8)
                    d = np.nonzero(a != b)[0]
                    first_bad = (e, int(d[0]), int(a[d[0]]), int(b[d[0]]), int(d.size))
        if n_bad:
            fail(f"cutlass_mxfp4 {st['gguf_name']} [k={k},n={n}]: {n_bad}/{n_exp} experts differ; first expert {first_bad[0]} "
               f"at byte {first_bad[1]} ({'data' if first_bad[1] < n * k // 2 else 'SF'}) got 0x{first_bad[2]:02x} "
               f"served 0x{first_bad[3]:02x}, {first_bad[4]} bytes")
        else:
            ok(f"cutlass_mxfp4 {st['gguf_name']} [k={k},n={n}]: {n_exp}/{n_exp} experts BYTE-IDENTICAL "
               f"({n_exp * eb / 2**30:.2f} GiB; produce {t_prod * 1e3 / n_exp:.1f} ms/expert, "
               f"{t_prod:.1f} s/stack; source reads {t_src:.1f} s, served read {t_read:.1f} s)")
        check_swizzle(n, k // 32)


def run_markov_w2(served: Tree, hf: Tree) -> None:
    name = "mtp.2.markov_head.markov_w2.weight"
    sh = served.shard_of(name)
    decl = json.loads(sh.meta["pulsar.tensors"])[name]
    if decl["layout"] != "fp8_e4m3_soa_k":
        fail(f"{name}: served layout {decl['layout']}")
        return
    we = hf.entry(name)
    rows, cols = we["shape"]
    if we["dtype"] != "BF16" or decl["dims_ne"] != [rows, cols]:
        fail(f"{name}: source {we['dtype']} {we['shape']}, dims_ne {decl['dims_ne']} (k-major keeps the source order)")
        return
    w = hf.read(name)
    got_served = sh.read(name)
    t0 = time.perf_counter()
    got = P.fp8_e4m3_soa_k_from_bf16(w, rows, cols)
    dt = time.perf_counter() - t0
    if len(got) != P.bytes_for("fp8_e4m3_soa_k", decl["dims_ne"]):
        fail(f"{name}: produced {len(got)} B, bytes_for {P.bytes_for('fp8_e4m3_soa_k', decl['dims_ne'])}")
    if got == got_served:
        ok(f"fp8_e4m3_soa_k {name} bf16 [{rows},{cols}] -> k-major: BYTE-IDENTICAL ({len(got) / 2**20:.1f} MiB, {dt * 1e3:.0f} ms)")
    else:
        a = np.frombuffer(got, np.uint8)
        b = np.frombuffer(got_served, np.uint8)
        d = np.nonzero(a != b)[0]
        plane = cols * (rows // 32)
        fail(f"fp8_e4m3_soa_k {name}: {d.size} bytes differ, first at {d[0]} "
             f"({'scale plane' if d[0] < plane else 'payload'}) got 0x{a[d[0]]:02x} served 0x{b[d[0]]:02x}")


def run_native(served: Tree, hf: Tree, name: str) -> None:
    sh = served.shard_of(name)
    decl = json.loads(sh.meta["pulsar.tensors"])[name]
    se_ = sh.entry(name)
    we = hf.entry(name)
    raw = hf.read(name)
    t0 = time.perf_counter()
    if we["dtype"] == "I64" and se_["dtype"] == "I32":
        got = P.i64_to_i32(raw)
        how = "i64_to_i32"
    else:
        got = P.native(raw)
        how = "native"
    dt = time.perf_counter() - t0
    want = sh.read(name)
    span = se_["data_offsets"][1] - se_["data_offsets"][0]
    model = P.bytes_for(decl["layout"], decl["dims_ne"], dtype=se_["dtype"])
    if model != span:
        fail(f"{name}: bytes_for {model} != span {span}")
    if got == want:
        ok(f"{how} {name} {we['dtype']} {we['shape']} -> {se_['dtype']}: BYTE-IDENTICAL ({len(got)} B, {dt * 1e3:.1f} ms)")
    else:
        fail(f"{how} {name}: {len(got)} B produced vs {len(want)} B served differ")


def run_bytes_for_all(served: Tree) -> None:
    files = sorted(glob.glob(os.path.join(SERVED, "model-*.safetensors")))
    n_ok = 0
    n_exp = 0
    layouts: dict[str, int] = {}
    for f in files:
        sh = served.shard(os.path.basename(f))
        for name, d in json.loads(sh.meta.get("pulsar.tensors", "{}")).items():
            e = sh.entry(name)
            span = e["data_offsets"][1] - e["data_offsets"][0]
            try:
                model = P.bytes_for(d["layout"], d["dims_ne"], dtype=e["dtype"])
            except ValueError as ex:
                fail(f"bytes_for refused {name} {d}: {ex}")
                continue
            key = d["layout"] if d["layout"] != "native" else f"native/{e['dtype']}"
            layouts[key] = layouts.get(key, 0) + 1
            if model == span:
                n_ok += 1
            else:
                fail(f"bytes_for {name} {d['layout']} {d['dims_ne']} = {model}, span {span}")
        for st in json.loads(sh.meta.get("pulsar.experts", "[]")):
            k, n = st["dims_per_expert_ne"]
            model = P.bytes_for(st["layout"], [k, n])
            stack = P.bytes_for(st["layout"], [k, n, st["n_experts"]])
            layouts[st["layout"]] = layouts.get(st["layout"], 0) + 1
            wn0 = st["gguf_name"]
            L = int(wn0.split(".")[1])
            ns = "layers" if wn0.startswith("blk.") else "mtp"
            spans = [sh.entry(f"{ns}.{L}.ffn.experts.{e}.{st['part']}.weight")["data_offsets"] for e in range(st["n_experts"])]
            total = spans[-1][1] - spans[0][0]
            if model == st["expert_bytes"] and all(s1 - s0 == model for s0, s1 in spans) and stack == total:
                n_exp += 1
            else:
                fail(f"bytes_for {wn0} cutlass [{k},{n}] = {model}/{stack}, declared {st['expert_bytes']}, stack span {total}")
    ok(f"bytes_for == span for {n_ok} declared tensors and {n_exp} expert stacks across {len(files)} shards; "
       f"layouts {layouts}")
    # the layouts the served tree does not carry, against the engine's formulas
    cases = [("exl3m_k2", [4096, 2048], (4096 // 16) * (2048 // 16) * 32 * 2 + (4096 + 2048) * 2),
             ("exl3m_k2h", [4096, 2048], (4096 // 16) * (2048 // 16) * 40 * 2 + (4096 + 2048) * 2),
             ("exl3m_k3", [2048, 4096], (2048 // 16) * (4096 // 16) * 48 * 2 + (2048 + 4096) * 2),
             ("iq2_xxs_mmq_k", [2048, 4096], 2048 * 4096 // 256 * 66),
             ("cutlass_mxfp4", [4096, 2000], 2000 * 4096 // 2 + (2048 // 32) * 4096),
             ("mxfp8_lt", [160, 200], 200 * 160 + 256 * 8),
             ("fp8_e4m3_soa_k", [160, 200], 200 * 160 + 200 * 5)]
    for layout, dims, want in cases:
        got = P.bytes_for(layout, dims)
        if got != want:
            fail(f"bytes_for({layout}, {dims}) = {got}, want {want}")
    for layout, dims in [("exl3m_k2", [4096, 2000]), ("mxfp8_lt", [100, 128]), ("native", [4])]:
        try:
            P.bytes_for(layout, dims)
            fail(f"bytes_for({layout}, {dims}) did not refuse")
        except ValueError:
            pass
    ok("bytes_for exl3m_k2/k2h/k3, iq2, padded cutlass/mxfp8 formulas + refusals")


def check_refusals() -> None:
    w = bytes(4096 * 96)
    s = bytes((4096 // 32) * (96 // 32))
    try:
        P.mxfp8_lt(w, s, 96, 4096, 32, "rederive")
        fail("mxfp8_lt accepted out=96 (not 128-aligned)")
    except ValueError:
        pass
    w = bytes(128 * 96)
    s = bytes(4 * 3)
    try:
        P.mxfp8_lt(w, s, 128, 96, 32, "verbatim")
        fail("mxfp8_lt accepted KB=3 (not a multiple of 4)")
    except ValueError:
        pass
    try:
        P.mxfp8_lt(bytes(128 * 128), bytes(1), 128, 128, 64, "rederive")
        fail("mxfp8_lt accepted a 64x64 block")
    except ValueError:
        pass
    try:
        P.cutlass_mxfp4(bytes(8 * 24), bytes(8 * 2), 8, 48)
        fail("cutlass_mxfp4 accepted in=48")
    except ValueError:
        pass
    ok("refusals: mxfp8_lt non-128-aligned shapes and non-{32,128} blocks, cutlass_mxfp4 in%32")


def main() -> int:
    print(f"served {SERVED}\nsource {HF}")
    check_e4m3_encoder()
    check_refusals()
    for rows, kb in [(128, 4), (256, 8), (512, 128), (2048, 64), (200, 5), (130, 3)]:
        check_swizzle(rows, kb)
    served, hf = Tree(SERVED), Tree(HF)

    t0 = time.perf_counter()
    run_bytes_for_all(served)
    print(f"     (byte model over 48 headers: {time.perf_counter() - t0:.1f} s)")

    for name in ["layers.0.attn.wq_b.weight",              # [in=1024, out=32768]
                 "layers.0.ffn.shared_experts.w2.weight",  # [in=2048, out=4096]
                 "layers.0.attn.wkv.weight",               # [in=4096, out=512]
                 "mtp.2.ffn.shared_experts.w1.weight"]:    # [in=4096, out=2048], drafter shard
        run_mxfp8_lt(served, hf, name)

    run_experts(served, hf, layer=0)
    run_markov_w2(served, hf)

    for name in ["layers.0.attn_norm.weight", "layers.0.hc_attn_fn", "layers.0.ffn.gate.weight",
                 "mtp.2.confidence_head.proj.weight", "layers.0.ffn.gate.tid2eid"]:
        run_native(served, hf, name)

    print()
    print("verbatim vs rederive summary (name, out, in, block, shifted groups, raised-by-one groups, code diffs, "
          "value diffs, zero-quirk diffs, zero-quirk codes present, t_rederive s, t_verbatim s):")
    for n in NOTES:
        print("  ", n)
    if FAILS:
        print(f"\n{len(FAILS)} FAILURE(S)")
        for f in FAILS:
            print("  -", f)
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
