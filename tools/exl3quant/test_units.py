#!/usr/bin/env python3
"""A unit quantized by a quant_worker.py process is byte-identical to the same unit inside quantize_block.

On one block's saved Hessians (a real run's hessians/<block>.safetensors), two units -- by default gu01 (a gate/up
unit that, inside quantize_block, reuses the shared Hessian gu00 finalized; a worker finalizes its own copy) and
dn11 (layer 20's down unit whose expert 180 needed exllamav3's Cholesky damping retry) -- are quantized

  A. in this process, as quantize_block runs them: gu00 then gu01 over ONE shared gate/up H_data, dn11 after;
  B. by two quant_worker.py `work` processes started together, each claiming units from the same list (on a
     multi-GPU host pass --devices 0,1 to put them on different GPUs; on one GPU both share it);

and every tensor of B's unit files must equal A's byte for byte.  With --reference-shard (the block's shard from a
run of the original single-process path) A is also compared to it, and reported.

    python tools/exl3quant/test_units.py --ref $V41 --exllamav3 $EXL3 --container $L279/tools/container \\
        --hessians $RUN/hessians/layer20.safetensors --reference-shard $RUN/exl3-k3/model-layer20.safetensors \\
        --scratch DIR [--units gu01,dn11] [--devices 0,0]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kmap as KM  # noqa: E402
import v41_stream as vs  # noqa: E402

FAILS = []


def check(cond, what):
    print(("PASS  " if cond else "FAIL  ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.reshape(-1).view(torch.uint8),
                                                                     b.reshape(-1).view(torch.uint8))


@torch.inference_mode()
def sequential(a, block_key, seed, K, units, dev):
    """A: quantize_block's loop body over the units it needs, one process, one shared gate/up H_data."""
    vs.torch_setup(a.exllamav3)
    ref = vs.load_reference(a.ref)
    args = vs.model_args(ref, a.ref, 1, 128)
    ckpt = vs.Checkpoint(a.ref)
    f = safe_open(a.hessians, framework="pt")
    count = int(f.metadata()["count"])
    all_units = {u.id: u for u in vs.block_units(args.n_routed_experts, args.moe_inter_dim)}
    order = [all_units["gu00"]] if any(u.startswith("gu") for u in units) else []
    order += [all_units[u] for u in units if u != "gu00"]
    gate_up = f.get_tensor("gate_up").to(dev)
    h_gu = vs.hessian_data(gate_up, f"{block_key}.ffn.experts.input", count, dev)
    out = {}
    for u in order:
        t0 = time.time()
        experts = {e: vs.load_expert(ref, args, ckpt, block_key, e, dev) for e in u.experts()}
        down = None
        if u.kind == "dn":
            e0 = u.experts()[0]
            down = vs.unpack_down(f.get_slice("down")[e0:u.experts()[-1] + 1], args.moe_inter_dim, dev)

        def h_down(e, down=down, e0=u.experts()[0]):
            return vs.hessian_data(down[e - e0], f"{block_key}.ffn.experts.{e}.w2", count, dev)
        res = vs.quantize_unit(u, block_key, experts, h_gu, h_down, seed, K, [dev.index],
                               os.path.join(a.scratch, "debug-seq"))
        print(f"A: {u.id} {time.time() - t0:.1f} s", flush=True)
        if u.id in units:
            out[u.id] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--exllamav3", required=True)
    ap.add_argument("--container", required=True, help="L279's tools/container (the K map's admit)")
    ap.add_argument("--kmap", default=os.path.join(HERE, "kmaps", "v41-k3-uniform.json"))
    ap.add_argument("--hessians", required=True, help="a run's hessians/<block>.safetensors")
    ap.add_argument("--reference-shard", default=None, help="the block's shard from the single-process path")
    ap.add_argument("--units", default="gu01,dn11")
    ap.add_argument("--devices", default="0,0", help="the two workers' CUDA devices")
    ap.add_argument("--scratch", required=True, help="a directory for the workers' run (not tmpfs on GB10)")
    a = ap.parse_args()
    units = a.units.split(",")

    meta = safe_open(a.hessians, framework="pt").metadata()
    block_key = meta["block"]
    comp, i = block_key.split(".")
    i = int(i)
    shape = KM.Shape(a.ref)
    K = KM.load(a.kmap, shape, a.container)[(comp, i)]
    name = f"{'layer' if comp == 'layers' else 'mtp'}{i:02d}"
    seed = i if comp == "layers" else shape.n_layers + i

    # B: two workers on a run directory holding only this block's Hessians
    run = os.path.join(a.scratch, "run")
    os.makedirs(os.path.join(run, "hessians"), exist_ok=True)
    link = os.path.join(run, "hessians", f"{name}.safetensors")
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(a.hessians), link)
    sel = ["--quant-layers", str(i)] if comp == "layers" else ["--quant-layers", "", "--drafter"]
    procs = []
    for w, d in enumerate(a.devices.split(",")):
        cmd = [sys.executable, os.path.join(HERE, "quant_worker.py"), "work", "--ref", a.ref, "--run", run,
               "--kmap", a.kmap, "--container", a.container, "--exllamav3", a.exllamav3, "--worker", f"t{w}",
               "--units", a.units, *sel]
        procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": d}))
    rcs = [p.wait() for p in procs]
    check(all(rc == 0 for rc in rcs), f"both workers exit 0 ({rcs})")
    by_worker = {}
    for w in range(len(procs)):
        log = os.path.join(run, "logs", f"worker-t{w}.jsonl")
        for line in open(log) if os.path.exists(log) else []:
            r = json.loads(line)
            if r["stage"] == "unit":
                by_worker[r["unit"]] = (r["worker"], r["gpu"], r["seconds"])
    print(f"B: units by worker {by_worker}", flush=True)
    check(len({w for w, _, _ in by_worker.values()}) == len(procs) or len(units) < len(procs),
          "the units were split across both workers")

    dev = torch.device("cuda:0")
    seq = sequential(a, block_key, seed, K, units, dev)
    ref_f = safe_open(a.reference_shard, framework="pt") if a.reference_shard else None
    for u in units:
        path = os.path.join(run, "units", f"{name}-k{K}", f"{u}.safetensors")
        with safe_open(path, framework="pt") as f:
            proxy = json.loads(f.metadata()["proxy"])
            n_t, n_same, n_ref, n_ref_same = 0, 0, 0, 0
            for k, (err, parts) in seq[u].items():
                for sub, t in parts.items():
                    n_t += 1
                    n_same += same(f.get_tensor(f"{k}.{sub}"), t)
                    if ref_f is not None:
                        n_ref += 1
                        n_ref_same += same(ref_f.get_tensor(f"{k}.{sub}"), t)
            check(n_same == n_t, f"{block_key} {u}: worker {by_worker.get(u, ('?',))[0]} == in-process quantize_block "
                                 f"order, {n_same}/{n_t} tensors byte-identical")
            check(all(proxy[k] == err for k, (err, _) in seq[u].items()), f"{block_key} {u}: proxy errors identical")
            if ref_f is not None:
                print(f"INFO  {block_key} {u}: vs the reference shard {n_ref_same}/{n_ref} tensors byte-identical",
                      flush=True)
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
