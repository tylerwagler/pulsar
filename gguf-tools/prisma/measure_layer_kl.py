#!/usr/bin/env python3
"""measure_layer_kl.py - measured per-(layer, parent) expert-promotion KL (engine-side).

For each unit U = (layer L, parent P in {all, gate_up, down}), measures the real
end-to-end KL between the all-cheap base checkpoint and the same checkpoint with
U's routed experts swapped to the donor's lossless MXFP4 tensors.  That KL is the
distortion U's cheap quantization contributes to the output distribution == the
benefit of promoting it, which is exactly what an allocator's knapsack trades
against bytes.

WHY SHARD SWAPS AND NOT --expert-overlay.  The overlay was the GGUF-era way to
compose a donor into a base, and it does give tensor-level granularity through
its name prefix.  It also REFUSES CUTLASS MXFP4 donors outright:

    pulsar: expert overlay does not support CUTLASS type-40 donor tensors yet
            (the grouped-GEMM prefill path device-asserts on an overlay range;
             observed 2026-07-02, root cause never chased)

MXFP4 is the donor this measures, so that path cannot measure the promotion at
all.  A safetensors checkpoint is one shard per layer, so a promotion is a FILE
SWAP -- which sidesteps the device-assert permanently rather than chasing it.
Granularity comes from what is IN the swapped shard: a (layer, gate_up) shard
carries gate/up at MXFP4 and down at its base layout, and the engine binds the
mix (its rule is "gate and up must match; each of gate/up and down is
independently 40 or 44, and the combo may differ per layer").

THE HARD LINK TRAP, written down because it already bit once: the composed
checkpoint is built with hard links, so a plain `cp` over the target name writes
THROUGH the link and silently rewrites the BASE artifact's shard.  compose()
therefore removes the name before copying.

Method (AURA-era, measured-not-proxied): one --kl-ref-dump run on the base (the
anchor), then one --kl-score run per unit with its shard swapped in.  Runs are
strictly serial (pulsar holds an instance lock).  Resumable: units already in
--out are skipped.

Usage:
  measure_layer_kl.py --pulsar ./pulsar --base /srv/models/vexp-safetensors \\
      --donor-root /srv/models/parent-donors --calib calib.txt \\
      --out kl.json [--parents all,gate_up,down] [--layers 0-42]

  Expects, per unit, <donor-root>/L<L>-<parent>/<shard>, i.e. what
  `build_mxfp4_layer.py --layer L --parents P --outdir <donor-root>/L<L>-<P>`
  writes.

Output (--out): {"30:all": {"dkl_promote": .., "stderr": ..}, "30:gate_up": {...}}
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time


def parse_layers(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


MIN_AVAIL_GB = 100  # ~90GiB device weight cache + margin


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576.0
    return 0.0


def run_all(cmds, what):
    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"error: {' '.join(cmd)} failed: {r.stderr.strip()} ({what})")


def uvm_reload():
    """Reclaim unified memory the GB10 driver leaks on every pulsar exit, drop
    page caches, and VERIFY enough memory is available before launching -- a
    SIGKILLed CUDA process can leak into the nvidia CORE module where the uvm
    reload cannot reach; then escalate to a full driver teardown.  Launching
    anyway would OOM, SIGKILL pulsar mid-CUDA, and leak even more."""
    run_all((["sudo", "-n", "rmmod", "nvidia_uvm"],
             ["sudo", "-n", "modprobe", "nvidia_uvm"],
             ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"]),
            "--uvm-reload needs passwordless sudo")
    if mem_available_gb() >= MIN_AVAIL_GB:
        return
    print(f"[uvm-reload] only {mem_available_gb():.0f}GB available; escalating to "
          f"full nvidia driver teardown", flush=True)
    subprocess.run(["sudo", "-n", "systemctl", "stop", "nvidia-persistenced"],
                   capture_output=True)
    run_all((["sudo", "-n", "modprobe", "-r", "nvidia_uvm", "nvidia_drm",
              "nvidia_modeset", "nvidia"],
             ["sudo", "-n", "modprobe", "nvidia"],
             ["sudo", "-n", "modprobe", "nvidia_uvm"],
             ["sudo", "-n", "modprobe", "nvidia_modeset"],
             ["sudo", "-n", "modprobe", "nvidia_drm"]),
            "full driver teardown")
    subprocess.run(["sudo", "-n", "systemctl", "start", "nvidia-persistenced"],
                   capture_output=True)
    if mem_available_gb() < MIN_AVAIL_GB:
        sys.exit(f"error: {mem_available_gb():.0f}GB available even after full driver "
                 f"teardown (< {MIN_AVAIL_GB}GB); refusing to launch a doomed run")


def run_pulsar(args, cmd, log_name):
    if args.uvm_reload:
        uvm_reload()
    log_path = os.path.join(args.log_dir, log_name)
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True)
    if proc.returncode != 0:
        sys.exit(f"error: pulsar failed (exit {proc.returncode}); see {log_path}")
    return proc.stdout


INDEX = "model.safetensors.index.json"


def shard_name_for_layer(base, L):
    """The shard file a layer lives in, READ FROM THE INDEX rather than guessed:
    a checkpoint's shard order is its own, and the donor builder names its output
    after the base's file, so the base is the one authority."""
    idx = json.load(open(os.path.join(base, INDEX)))
    for name, f in idx["weight_map"].items():
        if name.startswith(f"layers.{L}.ffn.experts."):
            return f
    sys.exit(f"error: no shard in {base} declares layers.{L}.ffn.experts")


def materialize(work, base):
    """A read-only VIEW of `base` in `work`: every shard and the index as a
    SYMLINK.  Idempotent, and it must run before the reference dump -- the dump
    loads the composed checkpoint, so a fresh work dir would otherwise be an
    empty directory.

    SYMLINKS, not hard links.  A hard link into the base is one careless `cp`
    away from rewriting the base's shard in place -- that happened, and it cost
    a regeneration of the delivered artifact -- and it also pins the work dir to
    the base's filesystem (hard-linking /srv/models into /tmp is a cross-device
    error).  A symlink is read-only by construction.

    SELF-HEALING, and it has to be: a swapped shard is a real file in `work`, so
    a REUSED work dir still carries whichever unit ran last.  Dumping a fresh
    reference against that would anchor the whole comparison on a promoted
    checkpoint instead of the base -- silently, since every score would then be
    against the wrong reference.  Any non-symlink entry is therefore restored."""
    os.makedirs(work, exist_ok=True)
    for f in os.listdir(base):
        if not (f.endswith(".safetensors") or f == INDEX):
            continue
        dst = os.path.join(work, f)
        if os.path.islink(dst):
            continue
        if os.path.exists(dst):
            # a shard left swapped by an earlier unit: restore the anchor
            os.remove(dst)
        os.symlink(os.path.join(base, f), dst)
    open(os.path.join(work, ".composed"), "w").close()
    return work


def compose(work, base, shard, donor_path):
    """Point the view at `base` with ONE shard replaced by `donor_path`.  The
    swapped shard is a REAL file here, so nothing written in `work` can reach
    the base."""
    materialize(work, base)
    dst = os.path.join(work, shard)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(donor_path, dst)
    return work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulsar", default="./pulsar")
    ap.add_argument("--base", required=True,
                    help="all-cheap base CHECKPOINT DIRECTORY (the anchor)")
    ap.add_argument("--donor-root", required=True,
                    help="dir holding L<L>-<parent>/ drop-in donor shards")
    ap.add_argument("--builder", default=None,
                    help="build_mxfp4_layer.py: build a unit's donor shard on demand "
                         "instead of pre-materializing all of them (86 shards is ~232 GB "
                         "and only one is ever in use at a time)")
    ap.add_argument("--source", default=None,
                    help="source checkpoint dir for --builder")
    ap.add_argument("--cleanup-donors", action="store_true",
                    help="remove each donor shard once its unit is scored (pair with "
                         "--builder to hold peak disk to one shard)")
    ap.add_argument("--calib", required=True, help="calibration text file")
    ap.add_argument("--out", required=True, help="kl.json output (resumable)")
    ap.add_argument("--parents", default="all",
                    help="comma list of all|gate_up|down (default all)")
    ap.add_argument("--ref-dump", help="reference logit dump (default <out>.ref.bin)")
    ap.add_argument("--layers", default="0-42")
    ap.add_argument("--tokens", type=int, default=2048, help="scored tokens per run")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--work", default=None,
                    help="where the composed checkpoint lives (default: a temp dir)")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--uvm-reload", action="store_true",
                    help="reload nvidia_uvm before each run (the GB10 leaks memory "
                         "per exit; needs passwordless sudo)")
    a = ap.parse_args()

    if not os.path.isdir(a.base):
        sys.exit(f"error: --base must be a checkpoint DIRECTORY (got {a.base!r}).  The "
                 f"overlay path that accepted a GGUF is gone: it refuses CUTLASS MXFP4 "
                 f"donors, which is the donor this measures.")
    ref_dump = a.ref_dump or (a.out + ".ref.bin")
    a.log_dir = a.log_dir or tempfile.mkdtemp(prefix="measure_layer_kl_")
    a.work = a.work or tempfile.mkdtemp(prefix="measure_kl_ckpt_")
    layers = parse_layers(a.layers)
    parents = [p.strip() for p in a.parents.split(",") if p.strip()]
    for p in parents:
        if p not in ("all", "gate_up", "down"):
            sys.exit(f"error: unknown parent {p!r}")
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}

    materialize(a.work, a.base)      # the ref dump loads the UNswapped view
    common = [a.pulsar, "-m", a.work, "--kl-file", a.calib,
              "-n", str(a.tokens), "-c", str(a.ctx), "--kl-stride", str(a.stride)]

    if not os.path.exists(ref_dump):
        print(f"[ref] dumping base logits -> {ref_dump}", flush=True)
        t0 = time.time()
        out = run_pulsar(a, common + ["--kl-ref-dump", ref_dump], "ref_dump.log")
        print(f"[ref] {out.strip()}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"[ref] reusing {ref_dump}", flush=True)

    kl_re = re.compile(r"kl_mean=([0-9.eE+-]+) kl_stderr=([0-9.eE+-]+)")
    units = [(L, p) for L in layers for p in parents]
    for (L, p) in units:
        key = f"{L}:{p}"
        if key in results:
            print(f"[{key}] cached: {results[key]}", flush=True)
            continue
        shard = shard_name_for_layer(a.base, L)
        donor_dir = os.path.join(a.donor_root, f"L{L}-{p}")
        donor = os.path.join(donor_dir, shard)
        if not os.path.exists(donor):
            if not a.builder or not a.source:
                sys.exit(f"error: no donor shard for {key}: expected {donor}\n"
                         f"       build it with: build_mxfp4_layer.py --layer {L} "
                         f"--parents {p} --outdir {donor_dir}\n"
                         f"       or pass --builder/--source to build it on demand")
            print(f"[{key}] building donor shard", flush=True)
            os.makedirs(donor_dir, exist_ok=True)
            subprocess.run([sys.executable, a.builder, "--layer", str(L),
                            "--parents", p, "--ours", a.base, "--source", a.source,
                            "--outdir", donor_dir], check=True)
        compose(a.work, a.base, shard, donor)
        t0 = time.time()
        out = run_pulsar(a, common + ["--kl-score", ref_dump], f"unit_{L}_{p}.log")
        m = kl_re.search(out)
        if not m:
            sys.exit(f"error: could not parse kl output for {key}: {out!r}")
        results[key] = {"dkl_promote": float(m.group(1)), "stderr": float(m.group(2)),
                        "shard": shard, "donor": donor}
        json.dump(results, open(a.out, "w"), indent=1)
        print(f"[{key}] dkl_promote={m.group(1)} stderr={m.group(2)} "
              f"({time.time()-t0:.0f}s, {len(results)}/{len(units)} done)", flush=True)
        if a.cleanup_donors:
            shutil.rmtree(donor_dir, ignore_errors=True)

    print(f"done: {len(results)} units -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
