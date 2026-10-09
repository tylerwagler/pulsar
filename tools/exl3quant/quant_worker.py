#!/usr/bin/env python3
"""Quantize saved V4.1 expert Hessians in units, one worker process per GPU, resumable -- then assemble the units
into the block shards v41_stream.py would have written.

The calibration stream is the SOURCE model's (v41_stream.py: every layer's Hessians come from the unquantized
layers below it, never from quantized output), so no block's quantization depends on another's.  The job splits
in two: `v41_stream.py --hessians-only` saves every block's Hessians in one streamed forward, and this tool
quantizes them as units -- one quantize_exl3_batch call each (v41_stream.Unit: 14 gate/up projections over the
block's shared Hessian, or 16 down projections; 79 units per trunk block, 27 per drafter stage) -- on as many GPUs
as there are workers.  A unit's numerics depend only on the unit (its tensors, Hessians, seed and the process-wide
torch state v41_stream.torch_setup sets), so the shard `assemble` writes is the shard quantize_block writes in
one process (test_units.py proves it on real units).

    # one worker per GPU (run.sh starts them); --wait keeps polling while the forward is still saving Hessians
    CUDA_VISIBLE_DEVICES=3 python quant_worker.py work --ref $V41 --exllamav3 $EXL3 --run $OUT \\
        --kmap kmaps/v41-k3-uniform.json --container $L279/tools/container --worker 3 --wait
    python quant_worker.py assemble --ref $V41 --run $OUT --kmap ... --container ... --prune
    python quant_worker.py status   --ref $V41 --run $OUT --kmap ... --container ...

Work layout (under --run):
    units/<block>-kK/<unit>.safetensors   one unit's tensors (the shard's names) + its proxy errors (metadata)
    units/<block>-kK/<unit>.sha256        written after the unit file: a unit is done iff both exist
    units/.claims/<block>-kK-<unit>       O_EXCL claim of the worker running it (run.sh clears them at start)
    exl3-kK/model-<block>.safetensors     assembled shard (+ .sha256: the block is done, its units pruned)
    exl3-kK/proxy-<block>.json            its proxy errors
    logs/worker-<id>.jsonl                one record per unit: seconds, GPU, proxy errors, CUDA peak
    state/forward.done | forward.failed   written by run.sh when the forward exits (ends a --wait worker)
"""

import argparse
import gc
import hashlib
import json
import os
import socket
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kmap as KM  # noqa: E402
import v41_stream as vs  # noqa: E402


# ------------------------------------------------------------------------------------------
# the work list
# ------------------------------------------------------------------------------------------

class Block:
    """One block's routed experts at one K: its units, Hessian file, shard and seed."""

    def __init__(self, run, comp, i, n_experts, inter, n_layers, K, experts_limit):
        self.comp, self.i, self.K = comp, i, K
        self.key = f"{comp}.{i}"
        self.name = f"{'layer' if comp == 'layers' else 'mtp'}{i:02d}"
        self.seed = i if comp == "layers" else n_layers + i      # v41_stream seeds by layer id
        self.n = n_experts if experts_limit is None else min(n_experts, experts_limit)
        self.n_full = n_experts
        self.units = vs.block_units(self.n, inter)
        self.hessians = os.path.join(run, "hessians", f"{self.name}.safetensors")
        self.unit_dir = os.path.join(run, "units", f"{self.name}-k{K}")
        self.shard_dir = os.path.join(run, f"exl3-k{K}")
        self.shard = os.path.join(self.shard_dir, f"model-{self.name}.safetensors")

    def unit_path(self, u):
        return os.path.join(self.unit_dir, f"{u.id}.safetensors")

    def unit_sha(self, u):
        return os.path.join(self.unit_dir, f"{u.id}.sha256")

    def unit_done(self, u):
        return os.path.exists(self.unit_path(u)) and os.path.exists(self.unit_sha(u))

    def assembled(self):
        return os.path.exists(self.shard) and os.path.exists(self.shard + ".sha256")


def work_list(a):
    """Every Block the run quantizes, in order: trunk layers ascending (the forward saves them in that order),
    then the drafter stages."""
    shape = KM.Shape(a.ref)
    kmap = KM.load(a.kmap, shape, a.container)
    layers = (range(shape.n_layers) if a.quant_layers is None
              else sorted(set(KM.parse_layers(x for x in a.quant_layers.split(",") if x))))
    blocks = [Block(a.run, "layers", L, shape.experts["layers"], shape.mid, shape.n_layers, kmap[("layers", L)],
                    a.experts) for L in layers]
    if a.drafter:
        blocks += [Block(a.run, "mtp", s, shape.experts["mtp"], shape.mid, shape.n_layers, kmap[("mtp", s)],
                         a.experts) for s in range(shape.n_mtp)]
    return blocks


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def write_sha(path, side):
    """path's sha256 into `side`, in sha256sum's format (`sha256sum -c` checks it), written atomically."""
    with open(side + ".tmp", "w") as f:
        f.write(f"{sha256(path)}  {os.path.basename(path)}\n")
    os.replace(side + ".tmp", side)
    return side


def check_sha(path, side):
    want = open(side).read().split()[0]
    got = sha256(path)
    if got != want:
        raise SystemExit(f"{path}: sha256 {got}, {side} says {want}")


# ------------------------------------------------------------------------------------------
# work
# ------------------------------------------------------------------------------------------

class HessianSource:
    """One block's saved Hessians, opened lazily: the gate/up matrix kept on the device while the worker stays on
    the block, the down Hessians read per unit (only the unit's experts)."""

    def __init__(self, block, inter, dev):
        self.block, self.inter, self.dev = block, inter, dev
        self.f = safe_open(block.hessians, framework="pt")
        meta = self.f.metadata()
        if meta.get("block") != block.key:
            raise SystemExit(f"{block.hessians}: holds {meta.get('block')}, expected {block.key}")
        self.count = int(meta["count"])
        n = self.f.get_slice("down").get_shape()[0]
        if n != block.n_full:
            raise SystemExit(f"{block.hessians}: {n} down Hessians, {block.key} has {block.n_full} experts")
        self.gate_up = None

    def h_gu(self):
        if self.gate_up is None:
            self.gate_up = self.f.get_tensor("gate_up").to(self.dev)
        return vs.hessian_data(self.gate_up, f"{self.block.key}.ffn.experts.input", self.count, self.dev)

    def h_down(self, experts):
        e0, e1 = experts[0], experts[-1] + 1
        if experts != list(range(e0, e1)):
            raise SystemExit(f"{self.block.key}: a down unit's experts are not contiguous: {experts}")
        down = vs.unpack_down(self.f.get_slice("down")[e0:e1], self.inter, self.dev)
        return {e: vs.hessian_data(down[e - e0], f"{self.block.key}.ffn.experts.{e}.w2", self.count, self.dev)
                for e in experts}


def claim(a, block, u):
    path = os.path.join(a.run, "units", ".claims", f"{block.name}-k{block.K}-{u.id}")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"worker": a.worker, "pid": os.getpid(), "host": socket.gethostname(),
                   "t": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
    return True


def wlog(a, **rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec["worker"] = a.worker
    print(json.dumps(rec), flush=True)
    with open(os.path.join(a.run, "logs", f"worker-{a.worker}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_unit(a, ref, args, ckpt, block, u, src, dev):
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    experts = {e: vs.load_expert(ref, args, ckpt, block.key, e, dev) for e in u.experts()}
    h_gu = src.h_gu() if u.kind == "gu" else None
    h_dn = src.h_down(u.experts()) if u.kind == "dn" else {}
    t_load = time.time() - t0
    results = vs.quantize_unit(u, block.key, experts, h_gu, h_dn.__getitem__, block.seed, block.K, [dev.index],
                               os.path.join(a.run, "debug"))
    t_quant = time.time() - t0 - t_load
    tensors = {f"{k}.{sub}": t for k, (_, parts) in results.items() for sub, t in parts.items()}
    proxy = {k: err for k, (err, _) in results.items()}
    os.makedirs(block.unit_dir, exist_ok=True)
    path = block.unit_path(u)
    save_file(tensors, path + ".tmp", metadata={"format": "pt", "exl3_K": str(block.K), "block": block.key,
                                                "unit": u.id, "proxy": json.dumps(proxy)})
    os.replace(path + ".tmp", path)
    write_sha(path, block.unit_sha(u))
    errs = sorted(proxy.values())
    wlog(a, stage="unit", block=block.key, K=block.K, unit=u.id, tensors=len(proxy),
         load_seconds=round(t_load, 1), quantize_seconds=round(t_quant, 1), seconds=round(time.time() - t0, 1),
         proxy_median=errs[len(errs) // 2], proxy_max=errs[-1],
         cuda_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
         gpu=os.environ.get("CUDA_VISIBLE_DEVICES", str(dev.index)), gpu_name=torch.cuda.get_device_name(dev))
    del experts, h_gu, h_dn, results, tensors


def forward_finished(a):
    return any(os.path.exists(os.path.join(a.run, "state", f)) for f in ("forward.done", "forward.failed"))


@torch.inference_mode()
def cmd_work(a):
    vs.torch_setup(a.exllamav3)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    ref = vs.load_reference(a.ref)
    args = vs.model_args(ref, a.ref, 1, 128)
    ckpt = vs.Checkpoint(a.ref)
    for d in ("logs", os.path.join("units", ".claims"), "debug"):
        os.makedirs(os.path.join(a.run, d), exist_ok=True)
    blocks = work_list(a)
    only = set(a.units.split(",")) if a.units else None
    pending = [(b, u) for b in blocks for u in b.units if only is None or u.id in only]
    wlog(a, stage="start", units=len(pending), blocks=len(blocks), pid=os.getpid(), host=socket.gethostname(),
         gpu=os.environ.get("CUDA_VISIBLE_DEVICES", str(dev.index)), gpu_name=torch.cuda.get_device_name(dev),
         torch=torch.__version__)
    src, ran, t0 = None, 0, time.time()
    while True:
        pending = [(b, u) for b, u in pending if not (b.assembled() or b.unit_done(u))]
        if not pending:
            break
        pick = next(((b, u) for b, u in pending if os.path.exists(b.hessians) and claim(a, b, u)), None)
        if pick is None:
            waiting = [b.key for b in dict.fromkeys(b for b, _ in pending) if not os.path.exists(b.hessians)]
            if not waiting:
                break  # everything left is claimed by another worker
            if not a.wait or forward_finished(a):
                wlog(a, stage="stop", reason="no Hessians for blocks the forward will not write", blocks=waiting)
                sys.exit(2)
            time.sleep(a.poll)
            continue
        b, u = pick
        if src is None or src.block is not b:
            src = None  # release the previous block's gate/up Hessian first
            torch.cuda.empty_cache()
            src = HessianSource(b, args.moe_inter_dim, dev)
        run_unit(a, ref, args, ckpt, b, u, src, dev)
        ran += 1
    wlog(a, stage="done", units_run=ran, seconds=round(time.time() - t0, 1))


# ------------------------------------------------------------------------------------------
# assemble, status
# ------------------------------------------------------------------------------------------

def cmd_assemble(a):
    blocks = work_list(a)
    missing = [f"{b.key} k{b.K} {u.id}" for b in blocks if not b.assembled() for u in b.units if not b.unit_done(u)]
    if missing:
        raise SystemExit(f"{len(missing)} units not done, e.g. {missing[:5]}")
    for b in blocks:
        if b.assembled():
            check_sha(b.shard, b.shard + ".sha256")
            print(f"{b.key} k{b.K}: assembled (sha256 verified)", flush=True)
            continue
        t0 = time.time()
        results = {}
        for u in b.units:
            path = b.unit_path(u)
            check_sha(path, b.unit_sha(u))
            with safe_open(path, framework="pt") as f:
                meta = f.metadata()
                if (meta["block"], meta["unit"], meta["exl3_K"]) != (b.key, u.id, str(b.K)):
                    raise SystemExit(f"{path}: holds {meta['block']} {meta['unit']} K{meta['exl3_K']}")
                proxy = json.loads(meta["proxy"])
                for e, w in u.members:
                    k = f"{b.key}.ffn.experts.{e}.{w}"
                    results[k] = (proxy[k], {sub: f.get_tensor(f"{k}.{sub}") for sub in vs.EXL3_PARTS})
        meta = None if b.n == b.n_full else {"experts": f"{b.n} of {b.n_full} (a dry run's partial block)"}
        vs.write_block(results, b.K, b.shard_dir, b.name, meta)
        write_sha(b.shard, b.shard + ".sha256")
        del results
        gc.collect()   # drop the unit files' last references before they are unlinked
        if a.prune:
            for u in b.units:
                os.remove(b.unit_path(u))
                os.remove(b.unit_sha(u))
            try:
                os.rmdir(b.unit_dir)
            except OSError as e:   # NFS keeps a .nfsXXXX for a file unlinked while mapped; harmless, logged
                print(f"{b.unit_dir}: not removed ({e.strerror}): {os.listdir(b.unit_dir)}", flush=True)
        print(f"{b.key} k{b.K}: {len(b.units)} units -> {b.shard} ({os.path.getsize(b.shard) / 1e9:.2f} GB, "
              f"{time.time() - t0:.0f} s)", flush=True)


def cmd_status(a):
    blocks = work_list(a)
    total = sum(len(b.units) for b in blocks)
    assembled = [b for b in blocks if b.assembled()]
    done = sum(len(b.units) if b.assembled() else sum(b.unit_done(u) for u in b.units) for b in blocks)
    hess = sum(os.path.exists(b.hessians) for b in blocks)
    print(json.dumps({"blocks": len(blocks), "hessians_saved": hess, "units": total, "units_done": done,
                      "blocks_assembled": len(assembled)}))


def cmd_blocks(a):
    """`<key> <name>` per block (each block once, whatever its K count): what run.sh grades."""
    for key, name in dict.fromkeys((b.key, b.name) for b in work_list(a)):
        print(key, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(required=True)
    for name, fn in (("work", cmd_work), ("assemble", cmd_assemble), ("status", cmd_status),
                     ("blocks", cmd_blocks)):
        p = sub.add_parser(name)
        p.add_argument("--ref", required=True, help="V4.1 HF snapshot")
        p.add_argument("--run", required=True, help="v41_stream.py's --out (hessians/ in, units/ and exl3-kK/ out)")
        p.add_argument("--kmap", required=True, help="each block's K")
        p.add_argument("--container", default=None, help="L279's tools/container (exl3_rates.admit grades the K map)")
        p.add_argument("--quant-layers", default=None, help="trunk layers (ints / a-b ranges; default all)")
        p.add_argument("--drafter", action="store_true", help="also the drafter's mtp.0-2")
        p.add_argument("--experts", type=int, default=None,
                       help="DRY RUN ONLY: each block's first N experts (the shards say so in their metadata)")
        if name == "work":
            p.add_argument("--exllamav3", required=True)
            p.add_argument("--worker", required=True, help="this worker's name (its log, its claims)")
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--wait", action="store_true",
                           help="poll for Hessians the forward has not saved yet until state/forward.done|failed")
            p.add_argument("--poll", type=float, default=15.0)
            p.add_argument("--units", default=None, help="only these unit ids (e.g. gu01,dn11): tests")
        if name == "assemble":
            p.add_argument("--prune", action="store_true", help="delete a block's units once its shard is written")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
