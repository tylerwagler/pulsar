"""weights.py -- the DFlash2 drafter's weights in each candidate FORMAT, for the reference (ref.py) and the L300 format
grade (fmtgrade.py).  Every format is read from the bytes it would be served from and decoded exactly; nothing here
re-derives a format the engine does not read.

A weight source answers ref.Drafter's three questions, in the container's names (`drafter.<HF name>`, the target's
`model.language_model.embed_tokens.weight`):
  tensor(name)   a native tensor as stored (bf16 / i32): norms, conv base kernels, the CandidateSelector's projection
                 and codebooks, the head's draft-vocabulary ids, the target's input embedding
  linear(name)   a Linear `X.weight` as (W f32 [out][in], s2 f32 or None): y = (x @ W.T) * s2 -- s2 the nvfp4 global
                 scale the engine's kernel applies on the GEMM's output (pulsar_cuda_nvfp4.h), never folded into W
  has(name)

Formats (`Arm`: the Linears' format and the head's format chosen apart, the rest -- the CandidateSelector's projection
included -- always the BF16 checkpoint's bytes, as the served container stores them):
  bf16        the HF checkpoint's BF16 bytes (incoai/Qwen3.8-Flash-Next-DFlash2) / the target's BF16 lm_head rows at the
              draft vocabulary's ids (tools/container/drafters.py's rows_gather)
  nvfp4       a pulsar container's `nvfp4` entries as served (producers.nvfp4_decode's planes: the nibbles, the
              128x4-tiled E4M3 scale plane unswizzled, the f32 global scale kept apart)
  nvfp4q      the same layout re-produced from the BF16 source by producers.nvfp4_from_bf16 (the builder's producer:
              byte-identical to a container built by it; `validate` checks that against a container)
  exl3:FILE   an exllamav3 checkpoint's `X.trellis | X.suh | X.svh` (+ the `X.mul1` marker; mul1 is the only
              codebook pulsar reads) decoded exactly as src/engine/exl3_trellis.h: W = diag(suh) H W_hat H diag(svh),
              W_hat the mul1 codebook's fp16 of each 16-bit trellis state, the Hadamards in f64.  A head from such a
              file needs `lm_head.*` in it (our own quant, exl3_quant.py); without one the arm names another head.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "container"))
import producers as PR  # noqa: E402

BF16 = torch.bfloat16
DRAFTER = "drafter."
EMBED = "model.language_model.embed_tokens.weight"
HEAD, HEAD_IDS = DRAFTER + "lm_head.weight", DRAFTER + "lm_head.ids"


# ---- safetensors, read lazily (headers once, tensors on demand) ---------------------------------------------------
class Shards:
    """Every `*.safetensors` file of a directory (or one file): name -> (path, offset, nbytes, header entry, pulsar
    declaration)."""

    def __init__(self, path):
        files = [path] if os.path.isfile(path) else [os.path.join(path, f) for f in sorted(os.listdir(path))
                                                     if f.endswith(".safetensors")]
        self.at = {}
        for p in files:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            decl = json.loads((h.pop("__metadata__", None) or {}).get("pulsar.tensors", "{}"))
            for k, v in h.items():
                o0, o1 = v["data_offsets"]
                self.at[k] = (p, 8 + n + o0, o1 - o0, v, decl.get(k, {}))

    def has(self, name):
        return name in self.at

    def raw(self, name):
        p, off, n, _v, _d = self.at[name]
        with open(p, "rb") as f:
            f.seek(off)
            return f.read(n)

    def decl(self, name):
        return self.at[name][4]

    def entry(self, name):
        return self.at[name][3]

    def plain(self, name):
        """a BF16 / F16 / F32 / I32 tensor as stored (torch, its own dtype)"""
        v = self.entry(name)
        b = self.raw(name)
        dt = {"BF16": (np.int16, BF16), "F16": (np.float16, None), "F32": (np.float32, None),
              "I32": (np.int32, None), "I16": (np.int16, None)}.get(v["dtype"])
        if dt is None:
            raise SystemExit(f"{name}: dtype {v['dtype']} is not one the reference reads")
        a = torch.from_numpy(np.frombuffer(b, dtype=dt[0]).copy()).reshape(v["shape"])
        return a.view(dt[1]) if dt[1] is not None else a


def bf16_to_f32(b: bytes, shape) -> torch.Tensor:
    return torch.from_numpy((np.frombuffer(b, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
                            .reshape(shape).copy())


# ---- NVFP4 ---------------------------------------------------------------------------------------------------------
def nvfp4_planes(buf: bytes, out: int, inp: int):
    """An `nvfp4` entry -> (W f32 [out][in] = e2m1 x e4m3, exact; s2 f32): producers.nvfp4_decode with the global
    scale left for the GEMM's output, as the kernel applies it."""
    kb_n = inp // 16
    sf_n = PR._rup(out, 128) * kb_n
    s2 = float(np.frombuffer(buf, dtype="<f4", count=1, offset=out * inp // 2 + sf_n)[0])
    if len(buf) != out * inp // 2 + sf_n + PR.NVFP4_TAIL:
        raise SystemExit(f"nvfp4 entry of [{out}, {inp}] holds {len(buf)} bytes, the layout "
                         f"{out * inp // 2 + sf_n + PR.NVFP4_TAIL}")
    vals = np.frombuffer(buf, dtype=np.uint8, count=out * inp // 2).reshape(out, inp // 2)
    sf = PR.unswizzle_sf(np.frombuffer(buf, dtype=np.uint8, count=sf_n, offset=out * inp // 2), out, kb_n)
    code = np.empty((out, inp), dtype=np.uint8)
    code[:, 0::2] = vals & 15
    code[:, 1::2] = vals >> 4
    e2 = np.where(code & 8, -1.0, 1.0).astype(np.float32) * PR.E2M1_VALUE[code & 7]
    w = e2 * PR.E4M3_VALUE[sf].astype(np.float32).repeat(16, axis=1)
    return torch.from_numpy(w), s2


# ---- EXL3 (src/engine/exl3_trellis.h, vectorised) -----------------------------------------------------------------
def _mul1_table() -> np.ndarray:
    """fp16 value of every 16-bit state under the mul1 codebook (exl3_mul1_decode): s = 0x6400 + bytesum(x *
    0x83DCD12D), v = fp16(s * fp16(0x1eee) + fp16(0xc931)) with one rounding (both steps exact in f32)."""
    x = (np.arange(65536, dtype=np.uint64) * np.uint64(0x83DCD12D)) & np.uint64(0xFFFFFFFF)
    s = (0x6400 + (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)).astype(np.uint16)
    h = s.view(np.float16).astype(np.float32)
    c1 = np.array([0x1eee], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    c0 = np.array([0xc931], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    return (h * c1 + c0).astype(np.float16)


MUL1 = _mul1_table()


def _state_end_bit(k2: int, p: np.ndarray) -> np.ndarray:
    bits = k2 >> 1
    if not k2 & 1:
        return (p + 1) * bits
    b2 = 2 * bits + 1
    return (p >> 1) * b2 + np.where(p & 1, b2, bits)


def _tile_perm():
    """position p -> (row, col) of the 16x16 tile (exl3_tile_position): the flat index row * 16 + col"""
    p = np.arange(256)
    t, i = p >> 3, p & 7
    row = (t % 4) * 2 + (i & 1) + 8 * ((i >> 1) & 1)
    col = t // 4 + 8 * (i >> 2)
    return row * 16 + col


TILE_FLAT = _tile_perm()


def exl3_k2_from_words(words: int) -> int:
    if words <= 0 or words % 8:
        return 0
    k2 = 2 * (words // 16) if words % 16 == 0 else 2 * (words // 16) + 1
    bits = k2 >> 1
    ok = (1 <= bits <= 3) if k2 & 1 else (1 <= bits <= 8)
    return k2 if ok else 0


def exl3_what(trellis: np.ndarray) -> np.ndarray:
    """trellis i16 [kt][nt][words] -> W_hat fp16 [16 kt][16 nt] (the rotated basis), byte-exact with
    exl3_tile_dequant / exllamav3's reconstruct."""
    kt, nt, words = trellis.shape
    k2 = exl3_k2_from_words(words)
    if not k2:
        raise SystemExit(f"EXL3 trellis tile of {words} words is not an EXL3 rate")
    total = 16 * words
    end = _state_end_bit(k2, np.arange(256))
    idx = (end[:, None] - 16 + np.arange(16)[None, :]) % total          # [256][16] stream bits, MSB first
    wts = (1 << (15 - np.arange(16))).astype(np.uint32)
    t = np.ascontiguousarray(trellis.reshape(kt * nt, words)).view(np.uint16)
    out = np.empty((kt * nt, 256), dtype=np.float16)
    CH = 8192
    for a in range(0, kt * nt, CH):
        tw = t[a:a + CH].astype(np.uint32)
        w32 = tw[:, 0::2] | (tw[:, 1::2] << 16)                          # the packer's halves swapped back
        bits = np.unpackbits(w32.astype(">u4").view(np.uint8), axis=1)   # stream bit s at column s
        st = (bits[:, idx].astype(np.uint32) * wts).sum(axis=2)          # [n][256] states
        v = MUL1[st]
        o = np.empty_like(v)
        o[:, TILE_FLAT] = v
        out[a:a + CH] = o
    return out.reshape(kt, nt, 16, 16).transpose(0, 2, 1, 3).reshape(16 * kt, 16 * nt)


def _had128() -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < 128:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(128.0)


HAD = _had128()


def exl3_weight(trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray, out: int | None = None) -> torch.Tensor:
    """-> W f32 [out][in] (the Linear's orientation): W_io = diag(suh) H W_hat H diag(svh) in f64, then the logical
    out columns (exllamav3 pads out to 128; the pad's svh is 0)."""
    wh = exl3_what(trellis).astype(np.float64)                           # [in][out_stored]
    k, n = wh.shape
    if suh.shape != (k,) or svh.shape != (n,):
        raise SystemExit(f"EXL3 scales suh {suh.shape} / svh {svh.shape} do not fit the trellis [{k}, {n}]")
    w = (HAD @ wh.reshape(k // 128, 128, n)).reshape(k, n) * suh.astype(np.float64)[:, None]
    w = (w.reshape(k, n // 128, 128) @ HAD).reshape(k, n) * svh.astype(np.float64)[None, :]
    w = w[:, :out] if out is not None else w
    return torch.from_numpy(np.ascontiguousarray(w.T).astype(np.float32))


class Exl3File:
    """An exllamav3 checkpoint's EXL3 Linears by HF name (`X.trellis`, `X.suh`, `X.svh`, `X.mul1`)."""

    def __init__(self, path):
        self.s = Shards(path)
        self.path = path

    def has(self, hf):
        return self.s.has(hf + ".trellis")

    def k2(self, hf):
        return exl3_k2_from_words(self.s.entry(hf + ".trellis")["shape"][-1])

    def linear(self, hf, out=None):
        if not self.s.has(hf + ".mul1") or self.s.has(hf + ".mcg"):
            raise SystemExit(f"{self.path}: {hf} is not on the mul1 codebook (the only one pulsar reads)")
        tr = self.s.plain(hf + ".trellis").numpy()
        suh = self.s.plain(hf + ".suh").numpy()
        svh = self.s.plain(hf + ".svh").numpy()
        return exl3_weight(tr, suh, svh, out), None


# ---- the sources ---------------------------------------------------------------------------------------------------
class Arm:
    """One graded configuration: the Linears' format, the head's format, everything else the BF16 checkpoint.

    drafter_hf  the BF16 checkpoint dir (config.json + model.safetensors)
    target_hf   the target's HF checkpoint (its BF16 lm_head and input embedding)
    container   a pulsar container (the served nvfp4 entries and the draft vocabulary's ids); required by nvfp4 and
                for the ids (the head is always the draft-vocabulary head the engine serves)
    """

    def __init__(self, name, linears, head, drafter_hf, target_hf, container, cache):
        self.name, self.lfmt, self.hfmt = name, linears, head
        self.d = cache.shards(drafter_hf)
        self.t = cache.target(target_hf)
        self.c = cache.shards(container) if container else None
        self.cache = cache
        if self.c is None or not self.c.has(HEAD_IDS):
            raise SystemExit("the grade serves the draft-vocabulary head: --container (its drafter.lm_head.ids)")
        self.ids = self.c.plain(HEAD_IDS).long()
        if "nvfp4q" in (linears, head):   # the producer is numpy, one tensor a thread: make them all at once
            hfs = [n for n in self.d.at if n.endswith(".weight") and len(self.d.entry(n)["shape"]) == 2
                   and not n.startswith("candidate_selector.")] if linears == "nvfp4q" else []
            hfs += ["lm_head.weight"] if head == "nvfp4q" else []
            with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as ex:
                list(ex.map(lambda h: cache.nvfp4q_entry(h, self), hfs))

    # native tensors: the BF16 checkpoint's own bytes (validate checks the container stores the same)
    def has(self, name):
        if name == HEAD_IDS:
            return True
        return name == EMBED or (name.startswith(DRAFTER) and self.d.has(name[len(DRAFTER):]))

    def tensor(self, name):
        if name == HEAD_IDS:
            return self.ids.int()
        if name == EMBED:
            return self.cache.embed(self.t)
        return self.d.plain(name[len(DRAFTER):])

    def linear(self, name):
        """name = drafter.X.weight"""
        hf = name[len(DRAFTER):]
        # the CandidateSelector's projection is BF16 in every arm: the served container stores it so (its rank-256
        # read is small), and the grade isolates the drafter's Linears and head
        fmt = self.hfmt if name == HEAD else "bf16" if hf.startswith("candidate_selector.") else self.lfmt
        return self.cache.linear(fmt, hf, self)


class Cache:
    """Decoded weights shared between arms (an arm naming the same format of the same tensor reuses it)."""

    def __init__(self):
        self._sh, self._lin, self._ex, self._nvq, self._emb = {}, {}, {}, {}, None

    def shards(self, path):
        if path not in self._sh:
            self._sh[path] = Shards(path)
        return self._sh[path]

    def target(self, target_hf):
        idx = json.load(open(os.path.join(target_hf, "model.safetensors.index.json")))["weight_map"]
        return {n: os.path.join(target_hf, idx[n]) for n in ("lm_head.weight", EMBED)}

    def embed(self, t):
        if self._emb is None:
            s = self.shards(t[EMBED])
            self._emb = s.plain(EMBED)
        return self._emb

    def exl3(self, path):
        if path not in self._ex:
            self._ex[path] = Exl3File(path)
        return self._ex[path]

    def nvfp4q_entry(self, hf, arm: Arm):
        """(out, in, the `nvfp4` entry producers.nvfp4_from_bf16 makes of the Linear's BF16 source), made once"""
        if hf not in self._nvq:
            if hf == "lm_head.weight":
                W = self.linear("bf16", hf, arm)[0]
                out, inp = W.shape
                b = (W.numpy().view(np.uint32) >> 16).astype("<u2").tobytes()   # the BF16 rows, exactly
            else:
                out, inp = arm.d.entry(hf)["shape"]
                b = arm.d.raw(hf)
            self._nvq[hf] = (out, inp, PR.nvfp4_from_bf16(b, out, inp))
        return self._nvq[hf]

    def linear(self, fmt, hf, arm: Arm):
        key = (fmt, hf)
        if key in self._lin:
            return self._lin[key]
        head = hf == "lm_head.weight"
        if fmt == "bf16":
            if head:
                s = self.shards(arm.t["lm_head.weight"])
                v = s.entry("lm_head.weight")
                full = s.raw("lm_head.weight")
                V, H = v["shape"]
                rows = PR.rows_gather(full, 2 * H, arm.ids.tolist())
                r = (bf16_to_f32(rows, (len(arm.ids), H)), None)
            else:
                r = (arm.d.plain(hf).float(), None)
        elif fmt == "nvfp4":
            name = DRAFTER + hf
            d = arm.c.decl(name)
            if d.get("layout") != "nvfp4":
                raise SystemExit(f"{name}: the container stores {d.get('layout')}, not nvfp4")
            inp, out = d["dims_ne"]
            r = nvfp4_planes(arm.c.raw(name), out, inp)
        elif fmt == "nvfp4q":
            out, inp, b = self.nvfp4q_entry(hf, arm)
            r = nvfp4_planes(b, out, inp)
        elif fmt.startswith("exl3:"):
            ex = self.exl3(fmt[5:])
            if head:
                if not ex.has("lm_head"):
                    raise SystemExit(f"{fmt[5:]} holds no lm_head: name the arm's head format apart (head=nvfp4 ...)")
                r = ex.linear("lm_head", out=len(arm.ids))
            else:
                r = ex.linear(hf[:-len(".weight")])
                want = list(arm.d.entry(hf)["shape"])
                if list(r[0].shape) != want:
                    raise SystemExit(f"{fmt[5:]}: {hf} decodes to {list(r[0].shape)}, the checkpoint's is {want}")
        else:
            raise SystemExit(f"weight format {fmt!r}: bf16, nvfp4, nvfp4q or exl3:FILE")
        self._lin[key] = r
        return r


def parse_arm(spec: str):
    """NAME=linears:FMT,head:FMT (FMT bf16 | nvfp4 | nvfp4q | exl3:FILE) -> (name, linears, head)"""
    name, _, rest = spec.partition("=")
    kv = {}
    for part in rest.split(","):
        k, _, v = part.partition(":")
        kv[k] = v
    if not name or set(kv) != {"linears", "head"}:
        raise SystemExit(f"--arm {spec!r}: NAME=linears:FMT,head:FMT")
    return name, kv["linears"], kv["head"]
