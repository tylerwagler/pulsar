#!/usr/bin/env python3
"""Per-kernel census of ONE single-stream decode step (L210's last open item).

The L208 decomposition priced the "glue" (hc mix + compressors + router) at
-1.8 ms/token and L210 closed every other bucket except that one, so the number
had to be measured.  This is the instrument: it profiles two CLI runs on the same
prompt with nsys -- one that prefills and generates a single token, one that
prefills and generates N+1 -- and differences the per-kernel GPU totals, so the
model load and the prefill cancel and everything left is divided by N.

    make decode-kernel-census                 # 24 decode steps, production artifact
    python3 tests/decode_kernel_census.py --help

Requires nsys (CUDA toolkit) and a free GPU: it refuses while a pulsar process is
live, because a second mapping of the model does not fit on the GB10.  It drops
the page cache first and prints SM/mem clocks before and after, so a reading
names its conditions (house protocol, L222).  Reports per-KERNEL-FAMILY ms/token
plus the launch count, and prints the glue, dense, bf16 and expert totals the
rows quote.
"""
import argparse, collections, csv, os, shutil, subprocess, sys

FAMILIES = [
    ("dense projections (mxfp8 mmvq)", lambda n: "mxfp8_mmvq_deint_a8" in n),
    ("attn-output grouped fp8",        lambda n: "grouped_fp8mx_a" in n),
    ("bf16 matmuls (hc/shared/head)",  lambda n: "matmul_bf16" in n),
    ("routed expert GEMV (IQ2)",       lambda n: "gateup_iq2_decode_gemv" in n),
    ("attention (f16 + combine)",      lambda n: "attn_f16" in n),
    ("HC mix (split weighted sum)",    lambda n: "hc_split_weighted_sum_norm" in n),
    ("HC expand / weights",            lambda n: "hc_expand" in n or "hc_weighted_sum" in n or "output_hc_weights" in n),
    ("compressor (csa2)",              lambda n: "csa2_comp" in n),
    ("indexer (rope/hadamard pack)",   lambda n: "indexer_" in n),
    ("router (topk + hash)",           lambda n: "router_select" in n),
    ("KV store / scatter / pack",      lambda n: "attn_pack_store" in n or "winkv_scatter" in n or "kv_store" in n),
    ("norms (rms/rows)",               lambda n: "rms_norm" in n or "dsv4_qkv_rms_norm" in n),
    ("activation quantise (mxfp8)",    lambda n: "mxfp8_quant" in n),
    ("swiglu + fold",                  lambda n: "swiglu" in n),
    ("moe gather/sum/ids",             lambda n: "moe_" in n or "mm_ids_helper" in n or "ds4_gather" in n),
    ("rope tail",                      lambda n: "rope_tail" in n),
    ("cutlass / nvjet / misc",         lambda n: "cutlass" in n or "nvjet" in n or "fill_f32" in n
                                                or "f32_to_bf16" in n or "embed_tokens" in n),
]
GLUE = ("HC mix (split weighted sum)", "HC expand / weights", "compressor (csa2)",
        "indexer (rope/hadamard pack)", "router (topk + hash)")


def kern_summary(path):
    """(per-kernel total ns, per-kernel instances) from an nsys .nsys-rep."""
    csv_path = path + ".csv"
    if os.path.exists(path + ".sqlite"):
        os.unlink(path + ".sqlite")
    subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                    path + ".nsys-rep"], stdout=open(csv_path, "w"), stderr=subprocess.DEVNULL,
                   check=False)
    rows = list(csv.reader(open(csv_path)))
    hdr = next((i for i, r in enumerate(rows)
                if r and r[0].strip().lower().startswith("time")), None)
    total, count = collections.Counter(), collections.Counter()
    if hdr is None:
        return total, count
    it = {c.strip(): j for j, c in enumerate(rows[hdr])}
    nj = next(j for c, j in it.items() if c.lower().startswith("name"))
    tj = next(j for c, j in it.items() if "total time" in c.lower())
    cj = next(j for c, j in it.items() if c.lower().startswith("instances"))
    for r in rows[hdr + 1:]:
        if len(r) <= max(nj, tj, cj) or not r[nj].strip():
            continue
        total[r[nj].strip()] += float(r[tj])
        count[r[nj].strip()] += int(r[cj])
    return total, count


def clocks():
    r = subprocess.run(["nvidia-smi",
                        "--query-gpu=clocks.sm,clocks.mem,temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip() or "clocks unavailable"


def drop_caches():
    """The house protocol is COLD (L222).  Best-effort: a box without passwordless
    sudo still profiles, but says so rather than implying the reading was cold."""
    subprocess.run(["sync"], check=False)
    r = subprocess.run(["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"],
                       input="3\n", capture_output=True, text=True)
    return r.returncode == 0


def report(model, base_ns, base_ct, dec_ns, dec_ct, N):
    """Difference two run summaries and print the per-family table."""
    delta, launches = {}, {}
    for k, v in dec_ns.items():
        dv = (v - base_ns.get(k, 0.0)) / N / 1e6          # ns -> ms per decode step
        if dv > 0:
            delta[k] = dv
            launches[k] = (dec_ct[k] - base_ct.get(k, 0)) / N
    total = sum(delta.values())
    print(f"model {os.path.basename(model)}")
    print(f"TOTAL {total:.2f} ms/token of GPU kernel time over "
          f"{sum(launches.values()):.0f} launches/step\n")

    seen, sums = set(), {}
    for label, pred in FAMILIES:
        sub = {k: v for k, v in delta.items() if pred(k) and k not in seen}
        seen |= set(sub)
        sums[label] = sum(sub.values())
    rest = {k: v for k, v in delta.items() if k not in seen}
    for label, v in sorted(sums.items(), key=lambda kv: -kv[1]):
        if v > 0:
            print(f"  {v:7.3f} ms/token {100 * v / total:5.1f}%   {label}")
    if rest:
        print(f"  {sum(rest.values()):7.3f} ms/token {100 * sum(rest.values()) / total:5.1f}%   other")
        for k, v in sorted(rest.items(), key=lambda kv: -kv[1]):
            print(f"      {v:7.3f} ms  {launches[k]:5.0f}/step  {k[:84]}")
    glue = sum(sums[g] for g in GLUE)
    expert = sums["routed expert GEMV (IQ2)"]
    print(f"\nGLUE (hc mix + compressors + router/indexer) = {glue:.3f} ms/token "
          f"({100 * glue / total:.1f}%)")
    for k, v in sorted(((k, v) for k, v in delta.items()
                        if any(p in k for p in ("hc_split_weighted_sum_norm", "router_select",
                                                "csa2_comp", "indexer_"))),
                       key=lambda kv: -kv[1])[:6]:
        print(f"    {v:7.3f} ms  {launches[k]:5.0f}/step  {k[:80]}")
    if expert > 0:
        print(f"expert GEMV {expert:.3f} ms/token for the 1.674 GB/row routed set "
              f"= {1.674 / expert * 1000:.0f} GB/s")
    return delta, launches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/models/v5-vexp-full256-iq2-t46.gguf")
    ap.add_argument("--binary", default="./pulsar")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--prompt", default="Explain why the sky is blue, in two sentences.")
    ap.add_argument("--work", default="/tmp/decode-kernel-census")
    a = ap.parse_args()

    for p in ("pulsar", "pulsar-bench", "pulsar-server"):
        if subprocess.run(["pgrep", "-x", p], capture_output=True).returncode == 0:
            sys.exit(f"GUARD: {p} is live -- one pulsar process at a time on the GB10")
    shutil.rmtree(a.work, ignore_errors=True)
    os.makedirs(a.work, exist_ok=True)

    if not drop_caches():
        print("WARNING: page cache NOT dropped (no passwordless sudo) -- these are WARM readings")
    print(f"clocks before: {clocks()}")

    def run(tag, n):
        subprocess.run(["nsys", "profile", "-o", f"{a.work}/{tag}", "--force-overwrite=true",
                        "-t", "cuda", "--stats=false", a.binary, "-m", a.model, "-p", a.prompt,
                        "--nothink", "--temp", "0", "--no-dspark", "-n", str(n)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return kern_summary(f"{a.work}/{tag}")

    base_ns, base_ct = run("base", 1)
    dec_ns, dec_ct = run("decode", a.steps + 1)
    print(f"clocks after:  {clocks()}\n")
    report(a.model, base_ns, base_ct, dec_ns, dec_ct, a.steps)


if __name__ == "__main__":
    main()
