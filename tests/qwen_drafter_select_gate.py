#!/usr/bin/env python3
"""L299: Qwen's drafter selection at open, on the real container -- host-only (`pulsar --inspect`: the family's load
binds every tensor and exits before the GPU).

From a container holding both drafters (the MTP sidecar and the DFlash2 block drafter) and --base, the same model built
without a block drafter, the gate makes link farms in --work (every shard a symlink; nothing copied): both; the MTP
sidecar removed; the base plus the MTP sidecar (an MTP-only container, as served before L299); the drafter shard
removed from the both-container (pulsar.drafter names a drafter the shards do not hold).  Then:

  both, no --drafter        DFlash2 by default: the default line, the MTP module unloaded, the drafter bound
  both, --drafter mtp       the MTP module: the DFlash2 drafter unloaded
  both, --drafter dflash2   DFlash2
  both, --drafter lilicorr  refused by name (it holds: mtp dflash2)
  mtp only, no --drafter    the MTP module, no default line
  dflash2 only, none        DFlash2, no default line
  mtp only, --drafter dflash2   refused by name
  the drafter shard missing     refused: the declaration's tensors are not in the container

    python3 tests/qwen_drafter_select_gate.py --open ./pulsar --work DIR --base BASE [--ple-rows FILE] CONTAINER
"""
import argparse
import os
import shutil
import subprocess
import sys

FAILS = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)


def farm(src, dst, drop, ple):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for f in sorted(os.listdir(src)):
        if f in drop:
            continue
        target = ple if f == "ple-l1.rows" and ple else os.path.realpath(os.path.join(src, f))
        os.symlink(target, os.path.join(dst, f))


def run(binary, model, drafter):
    args = [binary, "--inspect", "-m", model] + (["--drafter", drafter] if drafter else [])
    r = subprocess.run(args, capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", required=True, help="the pulsar CLI")
    ap.add_argument("--work", required=True)
    ap.add_argument("--ple-rows", help="the PLE row file to link (a host whose path differs from the container's link)")
    ap.add_argument("--base", required=True, help="the same model's container without a block drafter (and no MTP)")
    ap.add_argument("container", help="a Qwen container holding the MTP sidecar and the DFlash2 drafter")
    a = ap.parse_args()
    files = os.listdir(a.container)
    mtp = [f for f in files if f == "model-mtp.safetensors"]
    shards = sorted(f for f in files if f.endswith(".safetensors") and f.startswith("model-0"))
    drafter = [shards[-1]] if shards else []          # the build's last shard: `drafter`
    if not mtp or not drafter:
        sys.exit(f"{a.container}: not a container holding both drafters")
    both, only_mtp, only_df2, broken = (os.path.join(a.work, n) for n in ("both", "mtp-only", "dflash2-only", "no-shard"))
    farm(a.container, both, [], a.ple_rows)
    farm(a.base, only_mtp, ["SHA256SUMS", "model.safetensors.index.json"], a.ple_rows)
    os.symlink(os.path.realpath(os.path.join(a.container, mtp[0])), os.path.join(only_mtp, mtp[0]))
    farm(a.container, only_df2, mtp, a.ple_rows)
    farm(a.container, broken, drafter, a.ple_rows)
    DEFAULT = "dflash2 by default (--drafter mtp selects the MTP module)"
    MTP_OFF = "the MTP module (34 tensors) is not the selected drafter: unloaded"
    DF2_OFF = "the dflash2 block drafter (83 tensors) is not the selected drafter: unloaded"
    DF2_ON = "83 of 83 tensors bound in 23 roles"
    print("qwen drafter selection (L299):")
    rc, out = run(a.open, both, None)
    check(rc == 0 and DEFAULT in out and MTP_OFF in out and DF2_ON in out,
          "both drafters, no --drafter: DFlash2 by default (announced), the MTP module unloaded")
    rc, out = run(a.open, both, "mtp")
    check(rc == 0 and DF2_OFF in out and MTP_OFF not in out and DEFAULT not in out,
          "both drafters, --drafter mtp: the MTP module, the DFlash2 drafter unloaded")
    rc, out = run(a.open, both, "dflash2")
    check(rc == 0 and MTP_OFF in out and DF2_ON in out and DEFAULT not in out, "both drafters, --drafter dflash2: DFlash2")
    rc, out = run(a.open, both, "lilicorr")
    check(rc != 0 and "--drafter names no drafter the container holds (it holds: mtp dflash2)" in out,
          "both drafters, --drafter lilicorr: refused by name")
    rc, out = run(a.open, only_mtp, None)
    check(rc == 0 and DEFAULT not in out and MTP_OFF not in out and "block drafter" not in out,
          "the MTP module only, no --drafter: the MTP module")
    rc, out = run(a.open, only_df2, None)
    check(rc == 0 and DEFAULT not in out and DF2_ON in out, "DFlash2 only, no --drafter: DFlash2")
    rc, out = run(a.open, only_mtp, "dflash2")
    check(rc != 0 and "--drafter names no drafter the container holds (it holds: mtp)" in out,
          "the MTP module only, --drafter dflash2: refused by name")
    rc, out = run(a.open, broken, None)
    check(rc != 0 and "the container has no tensor for role fc, block drafter" in out,
          "pulsar.drafter dflash2 with its shard missing: refused, naming the missing tensors")
    print(f"qwen-drafter-select gate: {'PASS' if not FAILS else 'FAIL'} ({len(FAILS)} failing)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
