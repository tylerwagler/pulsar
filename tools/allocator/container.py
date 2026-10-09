"""Read a Qwen pulsar container's own bytes back to weights (L251 S6's qwen_dequant.py, vendored for L286).

build.py's EXL3 repack is a slice copy: a tensor's (or one expert's) blob is exllamav3's [trellis | suh | svh] back to
back, so LinearEXL3's own decode recovers the weight exactly.  `pulsar.tensors` / `pulsar.experts` declare dims_ne and
layout per tensor; nothing is guessed.  mxfp8_lt is decoded with tools/container/producers.py's own inverse.

The U-e4-d5 container's EXL3 tensors are the full-KL pass's U-e4-d5 quantization (same rows, Hessians, seeds; S6
reproduced its record), so its dequantized weights ARE the graded U-e4-d5.
"""
import importlib.util, json, os, struct, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "container"))
import exl3_rates                                                              # noqa: E402

spec = importlib.util.spec_from_file_location("pulsar_producers", os.path.join(HERE, "..", "container", "producers.py"))
PR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PR)


class Container:
    """Name -> declared (dims_ne, layout) and raw bytes, straight from the shard headers."""

    def __init__(self, d):
        self.d = d
        self.index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.hdr, self.base, self.tensors, self.experts = {}, {}, {}, {}
        for f in sorted(os.listdir(d)):
            if not (f.startswith("model-") and f.endswith(".safetensors")):
                continue
            with open(os.path.join(d, f), "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                h = json.loads(fh.read(n))
            md = h.pop("__metadata__", None) or {}
            self.hdr[f], self.base[f] = h, 8 + n
            for nm, v in json.loads(md.get("pulsar.tensors", "{}")).items():
                self.tensors[nm] = (v["dims_ne"], v["layout"])
            for e in json.loads(md.get("pulsar.experts", "[]")):
                self.experts[e["gguf_name"]] = e

    def layout(self, name):
        dims, lay = self.tensors[name]
        return (lay, dims[0], 1) if len(dims) == 1 else (lay, dims[0], dims[1])

    def raw(self, name):
        f = self.index[name]
        o0, o1 = self.hdr[f][name]["data_offsets"]
        with open(os.path.join(self.d, f), "rb") as fh:
            fh.seek(self.base[f] + o0)
            return fh.read(o1 - o0)

    def expert_blobs(self, family):
        """All experts' blobs of one family (one contiguous read: experts are laid out base + e * expert_bytes)."""
        e = self.experts[family]
        names = [e["entry_name"].format(e=i) for i in range(e["n_experts"])]
        f = self.index[names[0]]
        assert all(self.index[n] == f for n in names), family
        offs = [self.hdr[f][n]["data_offsets"] for n in names]
        nb = e["expert_bytes"]
        assert all(o1 - o0 == nb for o0, o1 in offs), family
        assert all(offs[i + 1][0] == offs[i][1] for i in range(len(offs) - 1)), f"{family}: experts not contiguous"
        with open(os.path.join(self.d, f), "rb") as fh:
            fh.seek(self.base[f] + offs[0][0])
            buf = fh.read(offs[-1][1] - offs[0][0])
        return [buf[i * nb:(i + 1) * nb] for i in range(len(names))], e


def exl3_slices(blob, k, n, words):
    tb = (k // 16) * (n // 16) * words * 2
    if len(blob) != tb + (k + n) * 2:
        raise SystemExit(f"exl3 blob {len(blob)} != trellis {tb} + scales {(k + n) * 2} for k={k} n={n} words={words}")
    tr = np.frombuffer(blob[:tb], dtype=np.int16).reshape(k // 16, n // 16, words)
    suh = np.frombuffer(blob[tb:tb + k * 2], dtype=np.float16)
    svh = np.frombuffer(blob[tb + k * 2:], dtype=np.float16)
    return tr, suh, svh


def dequant_exl3(blob, k, n, words, dev="cuda"):
    """Container blob -> bf16 [out, in] on `dev` (LinearEXL3 owns the decode; mul1 codebook)."""
    from exllamav3.modules.quant.exl3 import LinearEXL3
    tr, suh, svh = exl3_slices(blob, k, n, words)
    lin = LinearEXL3(None, k, n, suh=torch.from_numpy(suh.copy()).to(dev), svh=torch.from_numpy(svh.copy()).to(dev),
                     trellis=torch.from_numpy(tr.copy()).to(dev), mul1=torch.zeros(1, dtype=torch.int32, device=dev))
    return lin.get_weight_tensor().t().to(torch.bfloat16).contiguous()


def dequant_mxfp8(blob, k, n, dev="cuda"):
    """Container mxfp8_lt bytes -> bf16 [out, in] (k = in, n = out), via the producer's own inverse."""
    kb_n = k // 32
    if len(blob) != n * k + PR._rup(n, 128) * PR._rup(kb_n, 4):
        raise SystemExit(f"mxfp8 blob {len(blob)} does not match in={k} out={n}")
    codes = np.frombuffer(blob[:n * k], dtype=np.uint8).reshape(n, k)
    s_exp = PR.unswizzle_sf(np.frombuffer(blob[n * k:], dtype=np.uint8), n, kb_n).astype(np.int32)
    val = PR.E4M3_VALUE[codes].reshape(n, kb_n, 32) * np.exp2(s_exp - 127)[:, :, None]
    return torch.from_numpy(np.ascontiguousarray(val).reshape(n, k)).to(dev).to(torch.bfloat16)


def words(layout):
    return exl3_rates.words_for(layout)


def quantized_dense(c, layer):
    """{hf name: bf16 [out, in]} for every layer-`layer` non-expert tensor the container stores quantized."""
    pre = f"model.language_model.layers.{layer}."
    out = {}
    for nm, (dims, lay) in c.tensors.items():
        if not nm.startswith(pre) or ".mlp.experts." in nm or lay in ("bf16", "f32", "i64", "i32", "f16"):
            continue
        _, k, n = c.layout(nm)
        if lay in exl3_rates.K2:
            out[nm] = dequant_exl3(c.raw(nm), k, n, words(lay))
        elif lay == "mxfp8_lt":
            out[nm] = dequant_mxfp8(c.raw(nm), k, n)
        else:
            raise SystemExit(f"{nm}: layout {lay} has no decode here")
    return out


def experts(c, layer, part):
    """[512, out, in] bf16 of one routed stack (part: gate_up_proj | down_proj), plus its layout."""
    fam = f"model.language_model.layers.{layer}.mlp.experts.{part}"
    blobs, e = c.expert_blobs(fam)
    lay = e["layout"]
    k, n = e["dims_per_expert_ne"][0], e["dims_per_expert_ne"][1]
    w = torch.empty(len(blobs), n, k, dtype=torch.bfloat16, device="cuda")
    for i, b in enumerate(blobs):
        w[i] = dequant_exl3(b, k, n, words(lay))
    return w, lay
