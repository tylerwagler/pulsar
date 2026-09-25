#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash routed experts -> EXL3, calibrated on the unquantized source model.

One pass over the 40 backbone layers, one layer resident at a time:

  1. every calibration row goes through layer L -- DeepSeek's own reference Block
     (inference/model.py), run unmodified on refkernels.py, the torch port of its kernels;
  2. while it runs, the layer's expert Hessians accumulate with exllamav3's recipe for a
     block-sparse MLP: ONE gate/up Hessian over every token's FFN input, and a down Hessian
     per expert with every expert run on every token (convert_model.py's
     calibration_all_experts), each from the fp8 activation the GEMM actually sees;
  3. the Hessians are written to disk, then exllamav3's quantize_exl3_batch quantizes the
     layer's 1152 expert projections at every requested K from the same Hessians;
  4. the layer's output stream -- plus the compressed KV, index keys, top-k and candidate
     blocks V4.1 hands down the stack -- is checkpointed, so a run resumes at the next layer.

Unlike exllamav3's converter, the stream carries the SOURCE model's activations (not the
already-quantized layers'), so one Hessian set serves every K and the per-layer shards of
different K can be mixed afterwards (tools/container/build.py --exl3 takes the result as it
takes the MiaAI checkpoint: layers.L.ffn.experts.E.wN.{trellis,suh,svh,mul1}).

After the last backbone layer the run reports perplexity over the calibration text rows --
an end-to-end check that the streamed forward is the model.

    python tools/exl3quant/v41_stream.py --ref /models/DeepSeek-V4.1-Flash \\
        --exllamav3 ~/exllamav3 --out /scratch/v41-exl3 --k 2,3 --device cuda:0

Output (under --out):
    calib_rows.safetensors            the token rows (built once, reused on resume)
    hessians/layerNN.safetensors      gate_up [5120,5120] and down [384, 2304*2305/2] (packed
                                      upper triangle), fp32 sums; metadata carries the count
    exl3-kK/model-layerNN.safetensors the layer's experts at K, exllamav3 tensor format
    exl3-kK/proxy-layerNN.json        proxy_err per tensor (the K-mix ordering signal)
    state/after.pt                    the stream after the last completed layer (torch.save)
    log.jsonl                         one record per completed stage
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import shutil
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refkernels  # noqa: E402

N_BACKBONE = 40
ACT_BLOCK = 32  # model.py fp8_block_size: every GEMM's activation block


# ------------------------------------------------------------------------------------------
# the reference model, on refkernels
# ------------------------------------------------------------------------------------------

def load_reference(ref_dir):
    """Import the snapshot's inference/model.py with refkernels standing in for its kernel module."""
    sys.modules["kernel"] = refkernels
    sys.path.insert(0, os.path.join(ref_dir, "inference"))
    import model as ref
    ref.world_size, ref.rank = 1, 0
    ref.default_dtype = torch.float8_e4m3fn  # what Transformer.__init__ sets for dtype "fp8"
    return ref


def model_args(ref, ref_dir, batch, seqlen):
    cfg = json.load(open(os.path.join(ref_dir, "inference", "config.json")))
    names = {f.name for f in dataclasses.fields(ref.ModelArgs)}
    unknown = sorted(set(cfg) - names)
    if unknown:
        raise SystemExit(f"inference/config.json keys ModelArgs does not know: {unknown}")
    args = ref.ModelArgs(**cfg)
    args.max_batch_size, args.max_seq_len = batch, seqlen
    if args.dtype != "fp8" or args.expert_dtype != "fp4":
        raise SystemExit(f"expected dtype fp8 / expert_dtype fp4, config says {args.dtype} / {args.expert_dtype}")
    return args


class Checkpoint:
    """The HF snapshot's shards, by tensor name."""

    def __init__(self, ref_dir):
        self.dir = ref_dir
        self.weight_map = json.load(open(os.path.join(ref_dir, "model.safetensors.index.json")))["weight_map"]
        self.handles = {}
        self.headers = {}

    def names(self, prefix):
        return sorted(n for n in self.weight_map if n.startswith(prefix))

    def get(self, name):
        """The tensor in owned host memory: safetensors hands back a view of its mmap, and a
        page-faulting copy from that into CUDA memory runs at ~5 MB/s on GB10."""
        shard = self.weight_map[name]
        h = self.handles.get(shard)
        if h is None:
            h = self.handles[shard] = safe_open(os.path.join(self.dir, shard), framework="pt", device="cpu")
        return h.get_tensor(name).clone()

    def raw(self, name):
        """(path, absolute byte offset, dtype string, shape) of a tensor's bytes in its shard."""
        path = os.path.join(self.dir, self.weight_map[name])
        hdr = self.headers.get(path)
        if hdr is None:
            with open(path, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                hdr = (8 + n, json.loads(fh.read(n)))
            self.headers[path] = hdr
        base, entries = hdr
        e = entries[name]
        return path, base + e["data_offsets"][0], e["dtype"], e["shape"]


def layer_state_dict(ckpt, layer):
    """Layer `layer`'s tensors under Block's names, with convert.py's two transforms: wo_a
    dequantized to bf16 over its weight blocks, routed experts' int8 bytes read as e2m1 pairs.
    The Engram row table is not included (EngramTable reads it in place)."""
    pre = f"layers.{layer}."
    sd = {n[len(pre):]: ckpt.get(n) for n in ckpt.names(pre) if ".engram.embed." not in n}
    w, s = sd["attn.wo_a.weight"], sd.pop("attn.wo_a.scale")
    bo, bi = w.size(0) // s.size(0), w.size(1) // s.size(1)
    if (bo, bi) not in ((32, 32), (128, 128)):
        raise SystemExit(f"layers.{layer}.attn.wo_a: block {bo}x{bi}")
    sd["attn.wo_a.weight"] = (w.unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)).float()
                              * s[:, None, :, None].float()).flatten(2, 3).flatten(0, 1).bfloat16()
    for k in sd:
        if ".experts." in k and "shared_experts" not in k and k.endswith(".weight"):
            if sd[k].dtype != torch.int8:
                raise SystemExit(f"layers.{layer}.{k}: {sd[k].dtype}, expected int8-packed e2m1")
            sd[k] = sd[k].view(torch.float4_e2m1fn_x2)
    return sd


def load_into(module, sd, what):
    """Copy a state dict into a module's existing parameters in place (keeping their dtypes --
    the fp32 compressor/head parameters take the checkpoint's bf16 -- and the Linear
    `weight.scale` aliases).  e2m1 parameters are copied as bytes."""
    own = dict(module.named_parameters())
    missing, unexpected = sorted(set(own) - set(sd)), sorted(set(sd) - set(own))
    if missing or unexpected:
        raise SystemExit(f"{what}: missing {missing[:8]} ({len(missing)}), unexpected {unexpected[:8]} ({len(unexpected)})")
    with torch.no_grad():
        for name, p in own.items():
            t = sd[name]
            if tuple(t.shape) != tuple(p.shape):
                raise SystemExit(f"{what}.{name}: shape {tuple(t.shape)}, the module has {tuple(p.shape)}")
            if p.dtype == torch.float4_e2m1fn_x2:
                p.view(torch.uint8).copy_(t.view(torch.uint8))
            else:
                p.copy_(t)


class EngramTable(torch.nn.Module):
    """ParallelEngramEmbedding at world size 1 over the shard's row table (2 x 94 GiB): e4m3
    rows times their e8m0 scale per 32, as bf16.  `prefetch` reads the table ONCE, sequentially,
    in large chunks, keeping only the rows the calibration set will look up -- random row reads
    run at ~86 rows/s over NFS, and a layer needs millions."""

    CHUNK_ROWS = 1 << 20  # 256 MiB of rows per sequential read

    def __init__(self, ckpt, layer, block):
        super().__init__()
        self.wpath, self.woff, wdt, wsh = ckpt.raw(f"layers.{layer}.engram.embed.weight")
        self.spath, self.soff, sdt, ssh = ckpt.raw(f"layers.{layer}.engram.embed.scale")
        if wdt != "F8_E4M3" or sdt != "F8_E8M0" or ssh != [wsh[0], wsh[1] // block]:
            raise SystemExit(f"layers.{layer}.engram.embed: {wdt}{wsh} / {sdt}{ssh}")
        self.rows, self.dim, self.block = wsh[0], wsh[1], block
        self.ids = None

    def prefetch(self, indices):
        """Hold every row `indices` names (all the lookups this layer will make)."""
        uniq = np.unique(indices.reshape(-1).cpu().numpy())
        if uniq[0] < 0 or uniq[-1] >= self.rows:
            raise SystemExit(f"engram row index out of range [0, {self.rows})")
        sdim = self.dim // self.block
        w = np.empty((len(uniq), self.dim), np.uint8)
        s = np.empty((len(uniq), sdim), np.uint8)
        with open(self.wpath, "rb") as fw, open(self.spath, "rb") as fs:
            for c0 in range(0, self.rows, self.CHUNK_ROWS):
                n = min(self.CHUNK_ROWS, self.rows - c0)
                lo, hi = np.searchsorted(uniq, [c0, c0 + n])
                if lo == hi:
                    continue
                pick = uniq[lo:hi] - c0
                fw.seek(self.woff + c0 * self.dim)
                w[lo:hi] = np.frombuffer(fw.read(n * self.dim), np.uint8).reshape(n, self.dim)[pick]
                fs.seek(self.soff + c0 * sdim)
                s[lo:hi] = np.frombuffer(fs.read(n * sdim), np.uint8).reshape(n, sdim)[pick]
        self.ids = uniq
        self.w_rows = torch.from_numpy(w).view(torch.float8_e4m3fn)
        self.s_rows = torch.from_numpy(s).view(torch.float8_e8m0fnu)

    def forward(self, indices):
        flat = indices.reshape(-1).cpu().numpy()
        pos = np.searchsorted(self.ids, flat)
        if pos.max() >= len(self.ids) or not np.array_equal(self.ids[pos], flat):
            raise SystemExit("engram lookup of a row the prefetch did not hold")
        pos = torch.from_numpy(pos)
        w, s = self.w_rows[pos].float(), self.s_rows[pos].float()
        vals = (w.unflatten(-1, (-1, self.block)) * s.unsqueeze(-1)).flatten(-2).bfloat16()
        return vals.view(*indices.shape, self.dim).to(indices.device)


def load_layer(ref, args, layout, ckpt, layer, dev):
    """(Block, Engram or None) for backbone layer `layer`, weights on `dev`."""
    with torch.device(dev):
        block = ref.Block(layer, args, None)
    sd = layer_state_dict(ckpt, layer)
    load_into(block, {k: v for k, v in sd.items() if not k.startswith("engram.")}, f"layers.{layer}")
    engram = None
    if layer in args.engram_layer_ids:
        table = EngramTable(ckpt, layer, ref.fp8_block_size)

        def table_for(rows, dim):  # Engram.__init__ builds its row table through this name
            if (rows, dim) != (table.rows, table.dim):
                raise SystemExit(f"layers.{layer}.engram: layout [{rows}, {dim}], shard [{table.rows}, {table.dim}]")
            return table

        table_class, ref.ParallelEngramEmbedding = ref.ParallelEngramEmbedding, table_for
        try:
            with torch.device(dev):
                engram = ref.Engram(args, layer, layout)
        finally:
            ref.ParallelEngramEmbedding = table_class
        load_into(engram, {k[len("engram."):]: v for k, v in sd.items() if k.startswith("engram.")},
                  f"layers.{layer}.engram")
    return block, engram


# ------------------------------------------------------------------------------------------
# calibration rows
# ------------------------------------------------------------------------------------------

class Exl3Tokenizer:
    """What exllamav3's calibration_data.py asks of a tokenizer, over the V4.1 HF tokenizer."""

    def __init__(self, hf_tok):
        self.t = hf_tok
        self.actual_vocab_size = len(hf_tok)
        self.bos, self.eos = hf_tok.bos_token_id, hf_tok.eos_token_id

    def encode(self, text, add_bos=False, add_eos=False):
        ids = self.t.encode(text, add_special_tokens=False)
        ids = ([self.bos] if add_bos else []) + ids + ([self.eos] if add_eos else [])
        return torch.tensor([ids], dtype=torch.long)


def pack_articles(articles, cols, tok):
    """Every full `cols`-token row the articles fill, packed the way calibration_data.split_art
    packs them (bos/eos on alternate rows, an article's overflow past a row dropped)."""
    rows, cur = [], []
    for text in articles:
        special = len(rows) % 2 == 0
        cur = (cur + tok.encode(text, add_bos=special, add_eos=special)[0].tolist())[:cols]
        if len(cur) == cols:
            rows.append(cur)
            cur = []
    return torch.tensor(rows, dtype=torch.long).view(-1, cols)


def build_rows(ref_dir, exl3_dir, n_rows, cols, extra_corpora):
    """exllamav3's default calibration mix (what the MiaAI V4.1 quant used) at n_rows x cols,
    then every full row of each extra jsonl corpus ({"text": ...} per line).  Returns the rows
    and a per-row bool: True for text rows, False for the mix's uniform-random-token rows."""
    from transformers import AutoTokenizer
    tok = Exl3Tokenizer(AutoTokenizer.from_pretrained(ref_dir))
    spec = importlib.util.spec_from_file_location(
        "exl3_calibration_data", os.path.join(exl3_dir, "exllamav3", "conversion", "calibration_data.py"))
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    base = torch.cat(cal.get_default_calibration({"cal_rows": n_rows, "cal_cols": cols}, tok), dim=0)
    # the mix ends with random_data's rows (seeded, drawn in order): find where they start
    rand = torch.cat(cal.random_data(None, base.size(0), cols, tok), dim=0)
    n_random = next(n for n in range(base.size(0), -1, -1) if torch.equal(base[base.size(0) - n:], rand[:n]))
    parts, is_text = [base], [torch.arange(base.size(0)) < base.size(0) - n_random]
    for path in extra_corpora:
        arts = [json.loads(line)["text"] for line in open(path)]
        extra = pack_articles(arts, cols, tok)
        print(f"calibration: {path}: {len(arts)} articles -> {extra.size(0)} rows")
        parts.append(extra)
        is_text.append(torch.ones(extra.size(0), dtype=torch.bool))
    return torch.cat(parts), torch.cat(is_text)


# ------------------------------------------------------------------------------------------
# expert Hessians
# ------------------------------------------------------------------------------------------

def gemm_input(x):
    """The fp32 value a model GEMM multiplies for bf16 activation x: act_quant's e4m3 codes
    times their power-of-two scale per 32 (model.linear's fp8 / fp4 paths)."""
    q, s = refkernels.act_quant(x, ACT_BLOCK, "ue8m0", torch.float8_e8m0fnu)
    return (q.float().unflatten(-1, (-1, ACT_BLOCK)) * s.float().unsqueeze(-1)).flatten(-2)


class ExpertHessians:
    """Sum x^T x for the layer's routed experts, exllamav3's block-sparse recipe."""

    def __init__(self, moe, dev, flush_rows):
        self.moe = moe
        self.dim = moe.dim
        self.inter = moe.experts[0].w1.out_features
        self.n = moe.n_routed_experts
        self.gate_up = torch.zeros(self.dim, self.dim, dtype=torch.float32, device=dev)
        self.down = torch.zeros(self.n, self.inter, self.inter, dtype=torch.float32, device=dev)
        self.count = 0
        self.dropped = 0
        self.pending, self.pending_rows, self.flush_rows = [], 0, flush_rows

    def observe(self, x):
        """x: the MoE's input, [b, s, dim] bf16 (the ffn_norm output)."""
        x = x.reshape(-1, self.dim)
        finite = torch.isfinite(x).all(dim=1)
        if not finite.all():  # exllamav3 drops non-finite rows rather than poison H
            self.dropped += int((~finite).sum())
            x = x[finite]
        self.pending.append(x)
        self.pending_rows += x.size(0)
        if self.pending_rows >= self.flush_rows:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        x = torch.cat(self.pending)
        self.pending, self.pending_rows = [], 0
        xq = gemm_input(x)
        self.gate_up.addmm_(xq.T, xq)
        self.count += x.size(0)
        x16 = xq.bfloat16()  # exact: e4m3 x 2^k
        limit = self.moe.experts[0].swiglu_limit
        for e, ex in enumerate(self.moe.experts):
            # Expert.forward, every token, unweighted (exllamav3 activates all experts in
            # calibration; the route weight is not part of the down input it captures)
            gate = F.linear(x16, refkernels.dequant_fp4(ex.w1.weight, ex.w1.scale).bfloat16()).float()
            up = F.linear(x16, refkernels.dequant_fp4(ex.w3.weight, ex.w3.scale).bfloat16()).float()
            if limit > 0:
                up = up.clamp(-limit, limit)
                gate = gate.clamp(max=limit)
            hq = gemm_input((F.silu(gate) * up).bfloat16())
            self.down[e].addmm_(hq.T, hq)

    def save(self, path, layer):
        self.flush()
        iu = torch.triu_indices(self.inter, self.inter, device=self.down.device)
        down = self.down[:, iu[0], iu[1]].cpu()
        tmp = path + ".tmp"
        save_file({"gate_up": self.gate_up.cpu(), "down": down}, tmp,
                  metadata={"layer": str(layer), "count": str(self.count), "dropped": str(self.dropped),
                            "down_layout": f"upper triangle of [{self.inter},{self.inter}] row-major, per expert"})
        os.replace(tmp, path)


# ------------------------------------------------------------------------------------------
# EXL3 quantization
# ------------------------------------------------------------------------------------------

def quantize_layer(hess, layer, K, devices, out_dir, debug_dir):
    """The layer's routed experts at rate K from the collected Hessians, batched as exllamav3's
    group_quant_linears batches a block-sparse MLP: gate/up concatenated over the shared
    Hessian up to 32768 columns, down projections stacked 16 at a time."""
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3_batch

    dev = torch.device(devices[0])
    moe, dim, inter = hess.moe, hess.dim, hess.inter

    def qa():
        # convert_model.make_quant_args at the converter's defaults (out scales "always",
        # mul1 codebook, no output-side Hessian); seed = the layer, as the converter seeds
        # by module
        return {"seed": layer, "K": K, "devices": devices, "device_ratios": None,
                "apply_out_scales": True, "mul1": True, "debug_dir": debug_dir}

    def h_data(H, key):
        return {"H": H.clone(), "first_key": key, "count": hess.count, "finalized": False,
                "num_total": hess.count * H.size(0), "inf_nan": torch.zeros(2, dtype=torch.long, device=dev),
                "device": dev}

    def weight(lin):  # (in_features, out_features) fp32, as quantize_exl3 takes it
        return refkernels.dequant_fp4(lin.weight, lin.scale).T.contiguous()

    tensors, proxy = {}, {}

    def keep(key, result):
        err, out = result
        proxy[key] = float(err)
        for sub in ("trellis", "suh", "svh", "mul1"):
            tensors[f"{key}.{sub}"] = out[sub].cpu()

    gu_keys = [(e, w) for e in range(hess.n) for w in ("w1", "w3")]
    per_group = max(1, 32768 // inter)
    H_gu = h_data(hess.gate_up, f"layers.{layer}.ffn.experts.input")
    for g0 in range(0, len(gu_keys), per_group):
        group = gu_keys[g0:g0 + per_group]
        ws = [weight(getattr(moe.experts[e], w)) for e, w in group]
        res = quantize_exl3_batch(ws, [H_gu] * len(group), [qa() for _ in group])
        for (e, w), r in zip(group, res):
            keep(f"layers.{layer}.ffn.experts.{e}.{w}", r)
    for e0 in range(0, hess.n, 16):
        es = list(range(e0, min(hess.n, e0 + 16)))
        ws = [weight(moe.experts[e].w2) for e in es]
        hs = [h_data(hess.down[e], f"layers.{layer}.ffn.experts.{e}.w2") for e in es]
        res = quantize_exl3_batch(ws, hs, [qa() for _ in es])
        for e, r in zip(es, res):
            keep(f"layers.{layer}.ffn.experts.{e}.w2", r)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"model-layer{layer:02d}.safetensors")
    save_file(tensors, path + ".tmp", metadata={"format": "pt", "exl3_K": str(K)})
    os.replace(path + ".tmp", path)
    with open(os.path.join(out_dir, f"proxy-layer{layer:02d}.json"), "w") as f:
        json.dump(proxy, f, indent=0, sort_keys=True)
    return proxy


# ------------------------------------------------------------------------------------------
# the stream
# ------------------------------------------------------------------------------------------

SHARED = ("compress_kv", "index_k", "topk_idxs", "candidates")


@dataclasses.dataclass
class Stream:
    """Everything layer L+1 needs from layers 0..L, for every calibration row, on the host."""
    h: torch.Tensor          # [R, S, hc, dim] bf16, the hyper-connection residual stream
    pre_mix: torch.Tensor    # [R, S, hc] fp32, the next attention's collapse mix
    shared: dict             # SHARED name -> [R, ...] tensor, what the sources published
    after: int               # last layer applied (-1: embeddings only)

    def save(self, path):
        # torch.save streams the storages; safetensors' save_file would copy the whole stream
        # (tens of GB) into memory first
        torch.save({"h": self.h, "pre_mix": self.pre_mix, "shared": self.shared, "after": self.after}, path + ".tmp")
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, path):
        t = torch.load(path)
        return cls(t["h"], t["pre_mix"], t["shared"], t["after"])


def publishes(args, layer):
    """Which shared_attn slots `layer` writes during a prefill (model.py Attention/Indexer)."""
    out = []
    if layer in args.kv_source_layers:
        out += ["compress_kv", "index_k"]
    if layer in args.index_source_layers:
        out.append("topk_idxs")
    if layer == args.candidate_source_layer:
        out.append("candidates")
    return out


def run_layer(ref, args, block, engram, hashes, stream, layer, batch, dev, on_ffn_input):
    """Apply backbone layer `layer` to every row of the stream, in place."""
    ffn_forward = block.ffn.forward

    def observed(x, image_mask=None):
        on_ffn_input(x)
        return ffn_forward(x, image_mask)

    block.ffn.forward = observed
    pub = publishes(args, layer)
    R, S = stream.h.shape[:2]
    fresh = {}  # what this layer publishes; readers in this layer still see the previous source
    for r0 in range(0, R, batch):
        r1 = min(R, r0 + batch)
        sa = ref.shared_attn
        for k in SHARED:
            setattr(sa, k, stream.shared[k][r0:r1].to(dev) if k in stream.shared else None)
        h = stream.h[r0:r1].to(dev)
        pre_mix = stream.pre_mix[r0:r1].to(dev)
        with torch.device(dev):  # model.py builds index tensors on the default device
            if engram is not None:
                idx = args.engram_layer_ids.index(layer)
                h = engram(h, hashes[r0:r1, :, idx].to(dev), None)
            h, pre_mix = block(h, 0, pre_mix, None)
        stream.h[r0:r1] = h.cpu()
        stream.pre_mix[r0:r1] = pre_mix.cpu()
        for k in pub:
            v = getattr(sa, k)
            if k in ("compress_kv", "index_k"):  # the source's whole cache buffer: keep this prefill's rows
                v = v[: r1 - r0, : S // args.compress_ratios[layer]]
            v = v.cpu()
            if k not in fresh:
                fresh[k] = torch.empty((R, *v.shape[1:]), dtype=v.dtype)
            fresh[k][r0:r1] = v
    stream.shared.update(fresh)
    block.ffn.forward = ffn_forward
    stream.after = layer


def embed_rows(ckpt, rows, hc_mult):
    w = ckpt.get("embed.weight")
    h = F.embedding(rows, w).unsqueeze(2).repeat(1, 1, hc_mult, 1).contiguous()
    pre_mix = torch.zeros(*rows.shape, hc_mult, dtype=torch.float32)
    pre_mix[..., 0] = 1.0  # model.make_identity_pre_mix
    return h, pre_mix


def perplexity(ref, args, ckpt, stream, rows, is_text, batch, dev):
    """Mean next-token NLL over the text rows after the last backbone layer: hc_pre with the
    final mix, the output norm, the fp32 head -- Transformer.forward's tail."""
    norm = ref.RMSNorm(args.dim, args.norm_eps).to(dev)
    norm.weight.data.copy_(ckpt.get("norm.weight"))
    head = ckpt.get("head.weight").float().to(dev)
    nll, n = 0.0, 0
    for r in torch.nonzero(is_text).flatten().tolist():
        h = stream.h[r:r + 1].to(dev)
        pm = stream.pre_mix[r:r + 1].to(dev)
        x = torch.sum(pm.unsqueeze(-1) * h.float(), dim=2).to(h.dtype)  # Block.hc_pre
        logits = F.linear(norm(x).float(), head)[0, :-1]
        tgt = rows[r, 1:].to(dev)
        nll += F.cross_entropy(logits, tgt, reduction="sum").item()
        n += tgt.numel()
    return nll / max(1, n), n


def log(out, **rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(rec), flush=True)
    with open(os.path.join(out, "log.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help="V4.1 HF snapshot: shards + index, inference/, tokenizer")
    ap.add_argument("--exllamav3", required=True, help="exllamav3 checkout (quantizer + calibration mix)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=250, help="rows of exllamav3's default mix (MiaAI: 250)")
    ap.add_argument("--cols", type=int, default=2048, help="tokens per row (MiaAI: 2048)")
    ap.add_argument("--extra-corpus", action="append", default=[], help="jsonl {\"text\"} packed into more rows")
    ap.add_argument("--k", default="2,3", help="comma list of EXL3 rates; 'none' collects Hessians only")
    ap.add_argument("--forward-only", action="store_true",
                    help="no Hessians, no quantization: the forward and its perplexity (the gate)")
    ap.add_argument("--layers", type=int, default=N_BACKBONE, help="stop after this many backbone layers")
    ap.add_argument("--until", default=None,
                    help="HH:MM local: start no layer the last layer's duration says would end after it")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="start no layer with less free disk than this under --out")
    ap.add_argument("--batch", type=int, default=8, help="rows per forward")
    ap.add_argument("--flush-rows", type=int, default=16384, help="rows per all-experts Hessian pass")
    ap.add_argument("--device", default="cuda:0", help="the forward's device")
    ap.add_argument("--quant-devices", default=None,
                    help="comma list of CUDA indices the quantizer splits tiles over (default: the forward's)")
    a = ap.parse_args()

    ks = [] if a.k == "none" or a.forward_only else [int(k) for k in a.k.split(",")]
    deadline = None
    if a.until:
        hh, mm = (int(x) for x in a.until.split(":"))
        now = time.localtime()
        deadline = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hh, mm, 0, 0, 0, -1))
        if deadline <= time.time():
            deadline += 86400
    dev = torch.device(a.device)
    quant_devices = [int(d) for d in a.quant_devices.split(",")] if a.quant_devices else [dev.index or 0]
    if ks and (dev.type != "cuda" or quant_devices[0] != (dev.index or 0)):
        raise SystemExit(f"--quant-devices must start with the forward's CUDA device (the Hessians live there); "
                         f"forward {dev}, quant {quant_devices}")
    os.makedirs(a.out, exist_ok=True)
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if ks:
        sys.path.insert(0, a.exllamav3)

    ref = load_reference(a.ref)
    args = model_args(ref, a.ref, a.batch, a.cols)
    layout = ref.EngramLayout.from_args(args)
    ckpt = Checkpoint(a.ref)

    rows_path = os.path.join(a.out, "calib_rows.safetensors")
    if os.path.exists(rows_path):
        t = load_file(rows_path)
        rows, is_text = t["rows"], t["is_text"]
    else:
        rows, is_text = build_rows(a.ref, a.exllamav3, a.rows, a.cols, a.extra_corpus)
        save_file({"rows": rows, "is_text": is_text}, rows_path)
    if rows.max().item() >= args.vocab_size:
        raise SystemExit(f"calibration rows hold token id {rows.max().item()} >= vocab {args.vocab_size}")
    log(a.out, stage="rows", rows=rows.size(0), cols=rows.size(1), text_rows=int(is_text.sum()))

    from transformers import AutoTokenizer
    hasher = ref.NgramHashState(dataclasses.replace(args, max_batch_size=rows.size(0)), layout,
                                AutoTokenizer.from_pretrained(a.ref))
    hashes = hasher(rows, 0)  # [R, S, n_engram_layers, n_hash_cols]
    del hasher

    state_path = os.path.join(a.out, "state", "after.pt")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    if os.path.exists(state_path):
        stream = Stream.load(state_path)
        log(a.out, stage="resume", after=stream.after)
    else:
        h, pre_mix = embed_rows(ckpt, rows, args.hc_mult)
        stream = Stream(h, pre_mix, {}, -1)

    last = None  # seconds the previous layer took, the estimate for the next
    for layer in range(stream.after + 1, a.layers):
        if deadline is not None and last is not None and time.time() + last > deadline:
            log(a.out, stage="stop", reason=f"layer {layer} would end after {a.until}", after=stream.after)
            break
        free = shutil.disk_usage(a.out).free / 1e9
        if free < a.min_free_gb:
            log(a.out, stage="stop", reason=f"{free:.1f} GB free under --out", after=stream.after)
            break
        t0 = time.time()
        block, engram = load_layer(ref, args, layout, ckpt, layer, dev)
        if engram is not None:
            engram.embed.prefetch(hashes[:, :, args.engram_layer_ids.index(layer)])
            log(a.out, stage="engram", layer=layer, rows=len(engram.embed.ids), seconds=round(time.time() - t0, 1))
        hess = None if a.forward_only else ExpertHessians(block.ffn, dev, a.flush_rows)
        run_layer(ref, args, block, engram, hashes, stream, layer, a.batch, dev,
                  (lambda x: None) if hess is None else hess.observe)
        if hess is not None:
            hpath = os.path.join(a.out, "hessians", f"layer{layer:02d}.safetensors")
            os.makedirs(os.path.dirname(hpath), exist_ok=True)
            hess.save(hpath, layer)
        log(a.out, stage="forward", layer=layer, seconds=round(time.time() - t0, 1),
            tokens=hess.count if hess else None, dropped=hess.dropped if hess else None)
        for K in ks:
            t2 = time.time()
            proxy = quantize_layer(hess, layer, K, quant_devices, os.path.join(a.out, f"exl3-k{K}"),
                                   os.path.join(a.out, "debug"))
            errs = sorted(proxy.values())
            log(a.out, stage="quant", layer=layer, K=K, seconds=round(time.time() - t2, 1),
                proxy_median=errs[len(errs) // 2], proxy_max=errs[-1])
        del block, engram, hess
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        stream.save(state_path)
        last = time.time() - t0

    if stream.after == N_BACKBONE - 1:
        nll, n = perplexity(ref, args, ckpt, stream, rows, is_text, a.batch, dev)
        log(a.out, stage="perplexity", nll=round(nll, 4), ppl=round(float(np.exp(nll)), 3), tokens=n)


if __name__ == "__main__":
    main()
