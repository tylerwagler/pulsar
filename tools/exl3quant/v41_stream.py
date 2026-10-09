#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash routed experts -> EXL3, calibrated on the unquantized source model, one layer resident.

One pass over the 40 trunk layers, then the DSpark drafter's 3 stages:

  1. every calibration row goes through layer L -- DeepSeek's own reference Block (inference/model.py), run
     unmodified on refkernels.py, the torch port of its kernels;
  2. while it runs, the layer's routed-expert Hessians accumulate with exllamav3's recipe for a block-sparse MLP:
     ONE gate/up Hessian over every token's FFN input, and a down Hessian per expert with every expert run on
     every token (convert_model.py's calibration_all_experts), each from the fp8 activation the GEMM sees;
  3. the Hessians are written to disk, then exllamav3's quantize_exl3_batch quantizes the layer's 1152 expert
     projections at the layer's K (the K map) and any extra --k, from the same Hessians;
  4. the layer's output stream -- plus the compressed KV, index keys, top-k and candidate blocks V4.1 hands down
     the stack, and the drafter's target-layer inputs -- is checkpointed, so a run resumes at the next layer.

The drafter (mtp.0-2, 128 routed experts each) is calibrated as it is served: at sampled positions p of every
row, the block [token p+1, 4 noise tokens] runs through the three stages with main_x = main_norm(main_proj(the
attention inputs of layers 37-39 at p)) and each stage's window KV holding main_x's KV for positions p-127..p.

Unlike exllamav3's converter, the stream carries the SOURCE model's activations (not the already-quantized
layers'), so one Hessian set serves every K and per-layer shards of different K mix freely afterwards
(kmap.py assembles them; tools/container/build.py --exl3 reads `<block>.L.ffn.experts.E.wN.{trellis,suh,svh,
mul1}`).  Rows calib.py marks held out are forwarded but kept out of every Hessian; at the quantized layers their
FFN inputs and routes are saved for grade_layer.py.  The config, layer schedule and quantization plan come from
deepseek_v41.py (checked against the reference's inference/config.json).

    python tools/exl3quant/v41_stream.py --ref $V41 --exllamav3 ~/exllamav3 --calib calib.safetensors \\
        --kmap kmap.json --out /mnt/models/v41-exl3-ours --device cuda:0

Output (under --out):
    hessians/<block>NN.safetensors   gate_up [5120,5120] and down [E, 2304*2305/2] (packed upper triangle),
                                     fp32 sums; metadata carries the count
    exl3-kK/model-<block>NN.safetensors   the block's routed experts at K, exllamav3 tensor format
    exl3-kK/proxy-<block>NN.json     exllamav3's proxy error per tensor (the K-map ordering signal)
    holdout/<block>NN.safetensors    held-out rows' FFN inputs (bf16), route indices and weights
    state/after.pt                   the stream after the last completed layer (torch.save)
    log.jsonl                        one record per stage: seconds, peak memory
"""

import argparse
import dataclasses
import json
import os
import resource
import shutil
import struct
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kmap as KM  # noqa: E402
import refkernels  # noqa: E402

ACT_BLOCK = 32  # model.py fp8_block_size: every GEMM's activation block


# ------------------------------------------------------------------------------------------
# measurement: wall time and peak memory per stage
# ------------------------------------------------------------------------------------------

class MemWatch:
    """Peak system memory in use (MemTotal - MemAvailable: on GB10 the GPU's allocations are host memory too),
    sampled every 0.2 s, plus torch's peak CUDA allocation, both reset per stage."""

    def __init__(self):
        self.total = self._meminfo()["MemTotal"]
        self.peak = 0
        self.stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _meminfo():
        out = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                out[k] = int(v.split()[0]) * 1024
        return out

    def used(self):
        m = self._meminfo()
        return m["MemTotal"] - m["MemAvailable"]

    def _run(self):
        while not self.stop.wait(0.2):
            self.peak = max(self.peak, self.used())

    def stage(self):
        """Start a stage: returns a function that closes it as a dict of seconds and peaks (GiB)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.peak = self.used()
        t0 = time.time()

        def done():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.peak = max(self.peak, self.used())
            rec = {"seconds": round(time.time() - t0, 1), "sys_peak_gib": round(self.peak / 2**30, 2),
                   "rss_peak_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
            if torch.cuda.is_available():
                rec["cuda_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            return rec
        return done


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
        """The tensor in owned host memory: safetensors hands back a view of its mmap, and a page-faulting copy
        from that into CUDA memory runs at ~5 MB/s on GB10."""
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


def block_state_dict(ckpt, prefix):
    """A block's tensors (`layers.L.` / `mtp.S.`) under Block's names, with convert.py's two transforms: wo_a
    dequantized to bf16 over its weight blocks, routed experts' int8 bytes read as e2m1 pairs.  The Engram row
    table is not included (EngramTable reads it in place)."""
    sd = {n[len(prefix):]: ckpt.get(n) for n in ckpt.names(prefix) if ".engram.embed." not in n}
    w, s = sd["attn.wo_a.weight"], sd.pop("attn.wo_a.scale")
    bo, bi = w.size(0) // s.size(0), w.size(1) // s.size(1)
    if (bo, bi) not in ((32, 32), (128, 128)):
        raise SystemExit(f"{prefix}attn.wo_a: block {bo}x{bi}")
    sd["attn.wo_a.weight"] = (w.unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)).float()
                              * s[:, None, :, None].float()).flatten(2, 3).flatten(0, 1).bfloat16()
    for k in sd:
        if ".experts." in k and "shared_experts" not in k and k.endswith(".weight"):
            if sd[k].dtype != torch.int8:
                raise SystemExit(f"{prefix}{k}: {sd[k].dtype}, expected int8-packed e2m1")
            sd[k] = sd[k].view(torch.float4_e2m1fn_x2)
    return sd


def load_into(module, sd, what):
    """Copy a state dict into a module's existing parameters in place (keeping their dtypes -- the fp32
    compressor/head parameters take the checkpoint's bf16 -- and the Linear `weight.scale` aliases).  e2m1
    parameters are copied as bytes."""
    own = dict(module.named_parameters())
    missing, unexpected = sorted(set(own) - set(sd)), sorted(set(sd) - set(own))
    if missing or unexpected:
        raise SystemExit(f"{what}: missing {missing[:8]} ({len(missing)}), unexpected {unexpected[:8]} "
                         f"({len(unexpected)})")
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
    """ParallelEngramEmbedding at world size 1 over the shard's row table (2 x 94 GiB): e4m3 rows times their
    e8m0 scale per 32, as bf16.  `prefetch` reads the table ONCE, sequentially, in large chunks, keeping only the
    rows the calibration set will look up -- random row reads run at ~86 rows/s over NFS, and a layer needs
    millions."""

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
    """(Block, Engram or None) for trunk layer `layer`, weights on `dev`."""
    with torch.device(dev):
        block = ref.Block(layer, args, None)
    sd = block_state_dict(ckpt, f"layers.{layer}.")
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


def load_drafter_stage(ref, args, ckpt, stage, dev):
    with torch.device(dev):
        block = ref.DSparkBlock(args.n_layers + stage, args)
    load_into(block, block_state_dict(ckpt, f"mtp.{stage}."), f"mtp.{stage}")
    return block


# ------------------------------------------------------------------------------------------
# expert Hessians, held-out capture
# ------------------------------------------------------------------------------------------

def gemm_input(x):
    """The fp32 value a model GEMM multiplies for bf16 activation x: act_quant's e4m3 codes times their
    power-of-two scale per 32 (model.linear's fp8 / fp4 paths)."""
    q, s = refkernels.act_quant(x, ACT_BLOCK, "ue8m0", torch.float8_e8m0fnu)
    return (q.float().unflatten(-1, (-1, ACT_BLOCK)) * s.float().unsqueeze(-1)).flatten(-2)


class ExpertHessians:
    """Sum x^T x for a block's routed experts, exllamav3's block-sparse recipe."""

    def __init__(self, moe, dev, flush_rows):
        self.moe = moe
        self.dim = moe.dim
        self.inter = moe.experts[0].w1.out_features
        self.n = moe.n_routed_experts
        self.gate_up = torch.zeros(self.dim, self.dim, dtype=torch.float32, device=dev)
        self.down = torch.zeros(self.n, self.inter, self.inter, dtype=torch.float32, device=dev)
        self.count = 0
        self.dropped = 0
        self.seconds = 0.0          # time spent in the all-experts passes (flush), the Hessian cost proper
        self.pending, self.pending_rows, self.flush_rows = [], 0, flush_rows

    def observe(self, x):
        """x: the MoE's input, [..., dim] bf16 (the ffn_norm output)."""
        x = x.reshape(-1, self.dim)
        finite = torch.isfinite(x).all(dim=1)
        if not finite.all():  # exllamav3 drops non-finite rows rather than poison H
            self.dropped += int((~finite).sum())
            x = x[finite]
        self.pending.append(x)
        self.pending_rows += x.size(0)
        if self.pending_rows >= self.flush_rows:
            self.flush()

    def sink(self, x, weights, indices):
        """observed_moe's sink signature: the route does not enter a Hessian (every expert sees every token)."""
        self.observe(x)

    def flush(self):
        if not self.pending:
            return
        torch.cuda.synchronize()
        t0 = time.time()
        x = torch.cat(self.pending)
        self.pending, self.pending_rows = [], 0
        xq = gemm_input(x)
        self.gate_up.addmm_(xq.T, xq)
        self.count += x.size(0)
        x16 = xq.bfloat16()  # exact: e4m3 x 2^k
        limit = self.moe.experts[0].swiglu_limit
        for e, ex in enumerate(self.moe.experts):
            # Expert.forward, every token, unweighted (exllamav3 activates all experts in calibration; the route
            # weight is not part of the down input it captures)
            gate = F.linear(x16, refkernels.dequant_fp4(ex.w1.weight, ex.w1.scale).bfloat16()).float()
            up = F.linear(x16, refkernels.dequant_fp4(ex.w3.weight, ex.w3.scale).bfloat16()).float()
            if limit > 0:
                up = up.clamp(-limit, limit)
                gate = gate.clamp(max=limit)
            hq = gemm_input((F.silu(gate) * up).bfloat16())
            self.down[e].addmm_(hq.T, hq)
        torch.cuda.synchronize()
        self.seconds += time.time() - t0

    def save(self, path, key):
        self.flush()
        iu = torch.triu_indices(self.inter, self.inter, device=self.down.device)
        down = self.down[:, iu[0], iu[1]].cpu()
        save_file({"gate_up": self.gate_up.cpu(), "down": down}, path + ".tmp",
                  metadata={"block": key, "count": str(self.count), "dropped": str(self.dropped),
                            "down_layout": f"upper triangle of [{self.inter},{self.inter}] row-major, per expert"})
        os.replace(path + ".tmp", path)


class HoldoutCapture:
    """The held-out rows' MoE inputs and routes at one block: what grade_layer.py scores the quantized experts on."""

    def __init__(self):
        self.x, self.idx, self.w = [], [], []

    def observe(self, x, weights, indices):
        self.x.append(x.reshape(-1, x.size(-1)).cpu())
        self.idx.append(indices.reshape(x.reshape(-1, x.size(-1)).size(0), -1).int().cpu())
        self.w.append(weights.reshape(self.idx[-1].shape).float().cpu())

    def save(self, path, key):
        if not self.x:
            return
        save_file({"x": torch.cat(self.x), "indices": torch.cat(self.idx), "weights": torch.cat(self.w)},
                  path + ".tmp", metadata={"block": key})
        os.replace(path + ".tmp", path)


def observed_moe(moe, sink):
    """Wrap a MoE so every forward hands (input, route weights, route indices) to `sink` first-hand: the gate's
    own output, not a recomputation.  Returns the undo."""
    ffn_forward, gate_forward = moe.forward, moe.gate.forward
    route = {}

    def gate(x, *a):
        w, i = gate_forward(x, *a)
        route["w"], route["i"] = w, i
        return w, i

    def ffn(x, *a):
        y = ffn_forward(x, *a)
        sink(x, route["w"], route["i"])
        return y

    moe.forward, moe.gate.forward = ffn, gate

    def undo():
        moe.forward, moe.gate.forward = ffn_forward, gate_forward
    return undo


# ------------------------------------------------------------------------------------------
# EXL3 quantization
# ------------------------------------------------------------------------------------------

def quantize_block(hess, key, seed, K, devices, out_dir, debug_dir, name):
    """A block's routed experts at rate K from the collected Hessians, batched as exllamav3's group_quant_linears
    batches a block-sparse MLP: gate/up concatenated over the shared Hessian up to 32768 columns, down
    projections stacked 16 at a time.  Returns (proxy errors, seconds quantizing, seconds writing)."""
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3_batch

    dev = torch.device(devices[0])
    moe, inter = hess.moe, hess.inter
    t0 = time.time()

    def qa():
        # convert_model.make_quant_args with out scales "always" and the mul1 codebook (the only one pulsar
        # reads), no output-side Hessian; seed = the block's layer id, as the converter seeds by module
        return {"seed": seed, "K": K, "devices": devices, "device_ratios": None,
                "apply_out_scales": True, "mul1": True, "debug_dir": debug_dir}

    def h_data(H, first_key):
        return {"H": H.clone(), "first_key": first_key, "count": hess.count, "finalized": False,
                "num_total": hess.count * H.size(0), "inf_nan": torch.zeros(2, dtype=torch.long, device=dev),
                "device": dev}

    def weight(lin):  # (in_features, out_features) fp32, as quantize_exl3 takes it
        return refkernels.dequant_fp4(lin.weight, lin.scale).T.contiguous()

    tensors, proxy = {}, {}

    def keep(k, result):
        err, out = result
        proxy[k] = float(err)
        for sub in ("trellis", "suh", "svh", "mul1"):
            tensors[f"{k}.{sub}"] = out[sub].cpu()

    gu_keys = [(e, w) for e in range(hess.n) for w in ("w1", "w3")]
    per_group = max(1, 32768 // inter)
    H_gu = h_data(hess.gate_up, f"{key}.ffn.experts.input")
    for g0 in range(0, len(gu_keys), per_group):
        group = gu_keys[g0:g0 + per_group]
        ws = [weight(getattr(moe.experts[e], w)) for e, w in group]
        res = quantize_exl3_batch(ws, [H_gu] * len(group), [qa() for _ in group])
        for (e, w), r in zip(group, res):
            keep(f"{key}.ffn.experts.{e}.{w}", r)
    for e0 in range(0, hess.n, 16):
        es = list(range(e0, min(hess.n, e0 + 16)))
        ws = [weight(moe.experts[e].w2) for e in es]
        hs = [h_data(hess.down[e], f"{key}.ffn.experts.{e}.w2") for e in es]
        res = quantize_exl3_batch(ws, hs, [qa() for _ in es])
        for e, r in zip(es, res):
            keep(f"{key}.ffn.experts.{e}.w2", r)
    t_quant = time.time() - t0

    t1 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"model-{name}.safetensors")
    save_file(tensors, path + ".tmp", metadata={"format": "pt", "exl3_K": str(K)})
    os.replace(path + ".tmp", path)
    with open(os.path.join(out_dir, f"proxy-{name}.json"), "w") as f:
        json.dump(proxy, f, indent=0, sort_keys=True)
    return proxy, t_quant, time.time() - t1


# ------------------------------------------------------------------------------------------
# the stream
# ------------------------------------------------------------------------------------------

SHARED = ("compress_kv", "index_k", "topk_idxs", "candidates")


@dataclasses.dataclass
class Stream:
    """Everything layer L+1 (and the drafter) needs from layers 0..L, for every calibration row, on the host."""
    h: torch.Tensor          # [R, S, hc, dim] bf16, the hyper-connection residual stream
    pre_mix: torch.Tensor    # [R, S, hc] fp32, the next attention's collapse mix
    shared: dict             # SHARED name -> [R, ...] tensor, what the sources published
    main_hidden: dict        # dspark target layer -> [R, S, dim] bf16, that layer's attention input (mean of copies)
    after: int               # last trunk layer applied (-1: embeddings only)

    def save(self, path):
        # torch.save streams the storages; safetensors' save_file would copy the whole stream (tens of GB) into
        # memory first
        # (not dataclasses.asdict: it deep-copies every tensor)
        torch.save({f.name: getattr(self, f.name) for f in dataclasses.fields(self)}, path + ".tmp")
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path))


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


def row_batches(n_cal, n_rows, batch):
    """Batches that never mix calibration and held-out rows."""
    return ([(r, min(n_cal, r + batch)) for r in range(0, n_cal, batch)]
            + [(r, min(n_rows, r + batch)) for r in range(n_cal, n_rows, batch)])


def run_layer(ref, args, block, engram, hashes, stream, layer, batches, n_cal, dev, sinks):
    """Apply trunk layer `layer` to every row of the stream, in place.  sinks = (calibration, held-out) MoE
    observers, either None."""
    pub = publishes(args, layer)
    R, S = stream.h.shape[:2]
    fresh = {}  # what this layer publishes; readers in this layer still see the previous source
    target = layer in args.dspark_target_layer_ids
    if target and layer not in stream.main_hidden:
        stream.main_hidden[layer] = torch.empty(R, S, args.dim, dtype=torch.bfloat16)
    current = {"sink": None}
    undo = observed_moe(block.ffn, lambda x, w, i: current["sink"] and current["sink"](x, w, i))
    try:
        for r0, r1 in batches:
            current["sink"] = sinks[0] if r0 < n_cal else sinks[1]
            sa = ref.shared_attn
            for k in SHARED:
                setattr(sa, k, stream.shared[k][r0:r1].to(dev) if k in stream.shared else None)
            h = stream.h[r0:r1].to(dev)
            pre_mix = stream.pre_mix[r0:r1].to(dev)
            with torch.device(dev):  # model.py builds index tensors on the default device
                if engram is not None:
                    idx = args.engram_layer_ids.index(layer)
                    h = engram(h, hashes[r0:r1, :, idx].to(dev), None)
                if target:  # Transformer.forward: the drafter reads the attention INPUT of its target layers
                    stream.main_hidden[layer][r0:r1] = h.mean(dim=2).cpu()
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
    finally:
        undo()
    stream.shared.update(fresh)
    stream.after = layer


def embed_rows(ckpt, rows, hc_mult):
    w = ckpt.get("embed.weight")
    h = F.embedding(rows, w).unsqueeze(2).repeat(1, 1, hc_mult, 1).contiguous()
    pre_mix = torch.zeros(*rows.shape, hc_mult, dtype=torch.float32)
    pre_mix[..., 0] = 1.0  # model.make_identity_pre_mix
    return h, pre_mix


def perplexity(ref, args, ckpt, stream, rows, dev):
    """Mean next-token NLL over every row after the last trunk layer: hc_pre with the final mix, the output norm,
    the fp32 head -- Transformer.forward's tail."""
    norm = ref.RMSNorm(args.dim, args.norm_eps).to(dev)
    norm.weight.data.copy_(ckpt.get("norm.weight"))
    head = ckpt.get("head.weight").float().to(dev)
    nll, n = 0.0, 0
    for r in range(rows.size(0)):
        h = stream.h[r:r + 1].to(dev)
        pm = stream.pre_mix[r:r + 1].to(dev)
        x = torch.sum(pm.unsqueeze(-1) * h.float(), dim=2).to(h.dtype)  # Block.hc_pre
        logits = F.linear(norm(x).float(), head)[0, :-1]
        tgt = rows[r, 1:].to(dev)
        nll += F.cross_entropy(logits, tgt, reduction="sum").item()
        n += tgt.numel()
    return nll / max(1, n), n


# ------------------------------------------------------------------------------------------
# the drafter
# ------------------------------------------------------------------------------------------

def drafter_positions(S, win, n, seed):
    """n positions p in [win - 1, S - 2] (a full window behind, a next token ahead), evenly spread, jittered."""
    lo, hi = win - 1, S - 2
    base = np.linspace(lo, hi, n, endpoint=False)
    jit = np.random.default_rng(seed).integers(0, max(1, (hi - lo) // n), n)
    return sorted({int(min(hi, b + j)) for b, j in zip(base, jit)})


def main_kv(ref, attn, main_x):
    """DSparkAttention's prefill-side KV of main_x over the whole row (wkv, kv_norm, rope at positions 0..S-1, the
    fp8 round trip) -- what its window cache holds for each position."""
    rd = attn.rope_head_dim
    kv = attn.kv_norm(attn.wkv(main_x))
    ref.apply_rotary_emb(kv[..., -rd:], attn.freqs_cis[: main_x.size(1)])
    refkernels.act_quant(kv, ACT_BLOCK, "ue8m0", torch.float8_e8m0fnu, True)
    return kv


def run_drafter(ref, args, ckpt, stream, rows, n_cal, dev, a, ks_for, quant_devices, out, mw):
    """Calibrate (and quantize) the drafter's routed experts: every stage, at sampled positions of every row."""
    R, S = rows.shape
    win = args.window_size
    dargs = dataclasses.replace(args, max_batch_size=R, max_seq_len=S + args.dspark_block_size + 1)
    stages = list(range(args.n_mtp_layers))
    done = mw.stage()
    blocks = [load_drafter_stage(ref, dargs, ckpt, s, dev) for s in stages]
    embed = ckpt.get("embed.weight").to(dev)
    log(out, stage="drafter-load", **done())

    done = mw.stage()
    main = torch.cat([stream.main_hidden[t] for t in args.dspark_target_layer_ids], dim=-1)  # [R, S, 3 * dim]
    mx = torch.empty(R, S, args.dim, dtype=torch.bfloat16)
    kv = [torch.empty(R, S, args.head_dim, dtype=torch.bfloat16) for _ in stages]
    with torch.device(dev):
        for r in range(R):
            m = blocks[0].main_norm(blocks[0].main_proj(main[r:r + 1].to(dev)))
            mx[r] = m[0].cpu()
            for s in stages:
                kv[s][r] = main_kv(ref, blocks[s].attn, m)[0].cpu()
    del main
    pos = drafter_positions(S, win, a.drafter_positions, seed=args.n_layers)
    log(out, stage="drafter-main-x", positions=len(pos), **done())

    hess = [ExpertHessians(b.ffn, dev, a.flush_rows) for b in blocks]
    hold = [HoldoutCapture() for _ in stages]
    current = {"cal": True}
    def sink(s):
        return lambda x, w, i: (hess[s].sink if current["cal"] else hold[s].observe)(x, w, i)
    undos = [observed_moe(b.ffn, sink(s)) for s, b in enumerate(blocks)]
    done = mw.stage()
    noise = args.dspark_noise_token_id
    try:
        for r0, r1 in ((0, n_cal), (n_cal, R)):
            if r0 == r1:
                continue
            current["cal"] = r0 == 0
            b = r1 - r0
            for p in pos:
                ids = torch.full((b, args.dspark_block_size), noise, dtype=torch.long, device=dev)
                ids[:, 0] = rows[r0:r1, p + 1].to(dev)        # the token after p, then the noise slots
                h = F.embedding(ids, embed).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
                pre_mix = ref.make_identity_pre_mix(h, args.hc_mult)
                m = mx[r0:r1, p:p + 1].to(dev)
                q = torch.arange(p - win + 1, p + 1)
                with torch.device(dev):
                    for s, blk in enumerate(blocks):
                        blk.attn.window_kv_cache[:b, q % win] = kv[s][r0:r1, q].to(dev)
                        h, pre_mix = blk(h, p, pre_mix, m)
    finally:
        for u in undos:
            u()
    log(out, stage="drafter-forward", tokens=[hh.count for hh in hess],
        hessian_seconds=[round(hh.seconds, 1) for hh in hess], **done())

    for s in stages:
        name = f"mtp{s:02d}"
        key = f"mtp.{s}"
        done = mw.stage()
        os.makedirs(os.path.join(out, "hessians"), exist_ok=True)
        hess[s].save(os.path.join(out, "hessians", f"{name}.safetensors"), key)
        os.makedirs(os.path.join(out, "holdout"), exist_ok=True)
        hold[s].save(os.path.join(out, "holdout", f"{name}.safetensors"), key)
        log(out, stage="hessian-save", block=key, **done())
        for K in ks_for(("mtp", s)):
            done = mw.stage()
            proxy, tq, tw = quantize_block(hess[s], key, args.n_layers + s, K, quant_devices,
                                           os.path.join(out, f"exl3-k{K}"), os.path.join(out, "debug"), name)
            errs = sorted(proxy.values())
            log(out, stage="quant", block=key, K=K, quantize_seconds=round(tq, 1), write_seconds=round(tw, 1),
                proxy_median=errs[len(errs) // 2], proxy_max=errs[-1], **done())


class SavedHessians(ExpertHessians):
    """ExpertHessians as a previous run saved them (hessians/<block>.safetensors), over the block's source experts:
    what quantize_block needs to re-quantize at another K without the forward."""

    def __init__(self, moe, path, dev):
        f = safe_open(path, framework="pt")
        meta = f.metadata()
        self.moe, self.dim = moe, moe.dim
        self.inter, self.n = moe.experts[0].w1.out_features, moe.n_routed_experts
        self.count, self.dropped = int(meta["count"]), int(meta["dropped"])
        self.gate_up = f.get_tensor("gate_up").to(dev)
        packed = f.get_tensor("down")
        if tuple(packed.shape) != (self.n, self.inter * (self.inter + 1) // 2):
            raise SystemExit(f"{path}: down {tuple(packed.shape)}, the block has {self.n} x {self.inter}")
        iu = torch.triu_indices(self.inter, self.inter, device=dev)
        self.down = torch.zeros(self.n, self.inter, self.inter, dtype=torch.float32, device=dev)
        for e in range(self.n):
            d = self.down[e]
            d[iu[0], iu[1]] = packed[e].to(dev)
            d.T[iu[0], iu[1]] = packed[e].to(dev)   # symmetric: the lower triangle mirrors the upper


class RoutedExperts(torch.nn.Module):
    """A block's routed experts alone, on `dev`: the part of MoE quantize_block reads."""

    def __init__(self, ref, args, ckpt, prefix, n, dev):
        super().__init__()
        with torch.device(dev):
            self.experts = torch.nn.ModuleList(
                ref.Expert(args.dim, args.moe_inter_dim, dtype=torch.float4_e2m1fn_x2, swiglu_limit=args.swiglu_limit)
                for _ in range(n))
        for e, ex in enumerate(self.experts):
            pre = f"{prefix}.ffn.experts.{e}."
            load_into(ex, {f"{w}.{s}": (ckpt.get(pre + f"{w}.{s}").view(torch.float4_e2m1fn_x2) if s == "weight"
                                        else ckpt.get(pre + f"{w}.{s}"))
                           for w in ("w1", "w2", "w3") for s in ("weight", "scale")}, pre)
        self.dim, self.n_routed_experts = args.dim, n


def requantize(ref, args, ckpt, a, quant_layers, ks_for, quant_devices, mw):
    blocks = [("layers", L, args.n_routed_experts) for L in sorted(quant_layers)]
    if a.drafter:
        blocks += [("mtp", s, args.dspark_n_routed_experts) for s in range(args.n_mtp_layers)]
    dev = torch.device(a.device)
    for comp, i, n in blocks:
        key = f"{comp}.{i}"
        name = f"{'layer' if comp == 'layers' else 'mtp'}{i:02d}"
        done = mw.stage()
        moe = RoutedExperts(ref, args, ckpt, key, n, dev)
        hess = SavedHessians(moe, os.path.join(a.out, "hessians", f"{name}.safetensors"), dev)
        log(a.out, stage="load-saved", block=key, tokens=hess.count, **done())
        seed = i if comp == "layers" else args.n_layers + i
        for K in ks_for((comp, i)):
            done = mw.stage()
            proxy, tq, tw = quantize_block(hess, key, seed, K, quant_devices, os.path.join(a.out, f"exl3-k{K}"),
                                           os.path.join(a.out, "debug"), name)
            errs = sorted(proxy.values())
            log(a.out, stage="quant", block=key, K=K, quantize_seconds=round(tq, 1), write_seconds=round(tw, 1),
                proxy_median=errs[len(errs) // 2], proxy_max=errs[-1], **done())
        del moe, hess
        torch.cuda.empty_cache()


def log(out, **rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(rec), flush=True)
    with open(os.path.join(out, "log.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help="V4.1 HF snapshot: shards + index, inference/, tokenizer")
    ap.add_argument("--exllamav3", required=True, help="exllamav3 checkout (the quantizer; config conventions)")
    ap.add_argument("--calib", required=True, help="calib.py's rows file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kmap", default=None, help="kmap.py JSON: each block's K (the run quantizes each block at it)")
    ap.add_argument("--container", default=None, help="the builder's tools/container whose exl3_rates grades the K map")
    ap.add_argument("--k", default="", help="comma list of EXL3 rates every quantized block ALSO gets")
    ap.add_argument("--quant-layers", default=None,
                    help="trunk layers to collect Hessians for and quantize (ints / a-b ranges; default all); "
                         "the others are only forwarded")
    ap.add_argument("--drafter", action="store_true", help="after layer 39, calibrate and quantize mtp.0-2")
    ap.add_argument("--drafter-positions", type=int, default=256, help="sampled draft positions per row")
    ap.add_argument("--from-hessians", action="store_true",
                    help="no forward: re-quantize the --quant-layers (and with --drafter the drafter) from the "
                         "Hessians a previous run saved under --out, at the K map's / --k rates")
    ap.add_argument("--forward-only", action="store_true",
                    help="no Hessians, no quantization: the forward and its perplexity (the gate)")
    ap.add_argument("--layers", type=int, default=None, help="stop after this many trunk layers")
    ap.add_argument("--until", default=None,
                    help="HH:MM local: start no layer the last layer's duration says would end after it")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="start no layer with less free disk than this under --out")
    ap.add_argument("--save-every", type=int, default=1,
                    help="checkpoint the stream after every Nth layer and after the last (0: never)")
    ap.add_argument("--batch", type=int, default=4, help="rows per forward")
    ap.add_argument("--flush-rows", type=int, default=16384, help="rows per all-experts Hessian pass")
    ap.add_argument("--device", default="cuda:0", help="the forward's device")
    ap.add_argument("--quant-devices", default=None,
                    help="comma list of CUDA indices the quantizer splits tiles over (default: the forward's)")
    a = ap.parse_args()

    deadline = None
    if a.until:
        hh, mm = (int(x) for x in a.until.split(":"))
        now = time.localtime()
        deadline = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hh, mm, 0, 0, 0, -1))
        if deadline <= time.time():
            deadline += 86400
    dev = torch.device(a.device)
    quant_devices = [int(d) for d in a.quant_devices.split(",")] if a.quant_devices else [dev.index or 0]
    if dev.type != "cuda" or quant_devices[0] != (dev.index or 0):
        raise SystemExit(f"--quant-devices must start with the forward's CUDA device (the Hessians live there); "
                         f"forward {dev}, quant {quant_devices}")
    os.makedirs(a.out, exist_ok=True)
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    sys.path.insert(0, a.exllamav3)
    mw = MemWatch()

    ref = load_reference(a.ref)
    rows_file = safe_open(a.calib, framework="pt")
    rows_all, holdout = rows_file.get_tensor("rows"), rows_file.get_tensor("holdout")
    order = torch.cat([torch.nonzero(~holdout).flatten(), torch.nonzero(holdout).flatten()])
    rows, n_cal = rows_all[order].contiguous(), int((~holdout).sum())
    args = model_args(ref, a.ref, a.batch, rows.size(1))

    from deepseek_v41 import DeepseekV41Config
    cfg = DeepseekV41Config(a.ref)
    cfg.check_reference(args)
    n_exl3 = sum(v == "exl3" for v in cfg.plan().values()) // 2
    n_trunk = cfg.num_hidden_layers
    quant_layers = set(range(n_trunk)) if a.quant_layers is None else set(KM.parse_layers(a.quant_layers.split(",")))
    kmap = KM.load(a.kmap, KM.Shape(a.ref), a.container) if a.kmap else None
    extra = [int(k) for k in a.k.split(",") if k]
    if not a.forward_only and kmap is None and not extra:
        raise SystemExit("nothing to quantize at: pass --kmap and/or --k (or --forward-only)")

    def ks_for(block):
        return sorted(set(([kmap[block]] if kmap else []) + extra))

    if rows.max().item() >= args.vocab_size:
        raise SystemExit(f"calibration rows hold token id {rows.max().item()} >= vocab {args.vocab_size}")
    log(a.out, stage="rows", rows=rows.size(0), cols=rows.size(1), calibration=n_cal, holdout=rows.size(0) - n_cal,
        mix=json.loads(rows_file.metadata()["mix"]), exl3_projections=n_exl3, sys_used_gib=round(mw.used() / 2**30, 2),
        sys_total_gib=round(mw.total / 2**30, 2))

    ckpt = Checkpoint(a.ref)
    if a.from_hessians:
        requantize(ref, args, ckpt, a, quant_layers, ks_for, quant_devices, mw)
        mw.stop.set()
        return
    layout = ref.EngramLayout.from_args(args)
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
        stream = Stream(h, pre_mix, {}, {}, -1)
    batches = row_batches(n_cal, rows.size(0), a.batch)

    last = None  # seconds the previous layer took, the estimate for the next
    stop = n_trunk if a.layers is None else a.layers
    for layer in range(stream.after + 1, stop):
        if deadline is not None and last is not None and time.time() + last > deadline:
            log(a.out, stage="stop", reason=f"layer {layer} would end after {a.until}", after=stream.after)
            break
        free = shutil.disk_usage(a.out).free / 1e9
        if free < a.min_free_gb:
            log(a.out, stage="stop", reason=f"{free:.1f} GB free under --out", after=stream.after)
            break
        t0 = time.time()
        name, key = f"layer{layer:02d}", f"layers.{layer}"
        done = mw.stage()
        block, engram = load_layer(ref, args, layout, ckpt, layer, dev)
        log(a.out, stage="load", block=key, **done())
        if engram is not None:
            done = mw.stage()
            engram.embed.prefetch(hashes[:, :, args.engram_layer_ids.index(layer)])
            log(a.out, stage="engram-prefetch", block=key, rows=len(engram.embed.ids), **done())
        quant = layer in quant_layers and not a.forward_only
        hess = ExpertHessians(block.ffn, dev, a.flush_rows) if quant else None
        hold = HoldoutCapture() if quant else None
        sinks = (hess.sink if quant else None, hold.observe if quant else None)
        done = mw.stage()
        run_layer(ref, args, block, engram, hashes, stream, layer, batches, n_cal, dev, sinks)
        if hess is not None:
            hess.flush()
        log(a.out, stage="forward", block=key, tokens=hess.count if hess else None,
            dropped=hess.dropped if hess else None, hessian_seconds=round(hess.seconds, 1) if hess else None,
            **done())
        if quant:
            done = mw.stage()
            os.makedirs(os.path.join(a.out, "hessians"), exist_ok=True)
            hess.save(os.path.join(a.out, "hessians", f"{name}.safetensors"), key)
            os.makedirs(os.path.join(a.out, "holdout"), exist_ok=True)
            hold.save(os.path.join(a.out, "holdout", f"{name}.safetensors"), key)
            log(a.out, stage="hessian-save", block=key, **done())
            for K in ks_for(("layers", layer)):
                done = mw.stage()
                proxy, tq, tw = quantize_block(hess, key, layer, K, quant_devices, os.path.join(a.out, f"exl3-k{K}"),
                                               os.path.join(a.out, "debug"), name)
                errs = sorted(proxy.values())
                log(a.out, stage="quant", block=key, K=K, quantize_seconds=round(tq, 1), write_seconds=round(tw, 1),
                    proxy_median=errs[len(errs) // 2], proxy_max=errs[-1], **done())
        del block, engram, hess, hold
        torch.cuda.empty_cache()
        if a.save_every and ((layer + 1) % a.save_every == 0 or layer == stop - 1):
            done = mw.stage()
            stream.save(state_path)
            log(a.out, stage="stream-save", after=layer, bytes=os.path.getsize(state_path), **done())
        last = time.time() - t0
        log(a.out, stage="layer-total", block=key, seconds=round(last, 1))

    if stream.after == n_trunk - 1:
        done = mw.stage()
        nll, n = perplexity(ref, args, ckpt, stream, rows, dev)
        log(a.out, stage="perplexity", nll=round(nll, 4), ppl=round(float(np.exp(nll)), 3), tokens=n, **done())
        if a.drafter and not a.forward_only:
            run_drafter(ref, args, ckpt, stream, rows, n_cal, dev, a, ks_for, quant_devices, a.out, mw)
    mw.stop.set()


if __name__ == "__main__":
    main()
