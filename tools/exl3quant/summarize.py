#!/usr/bin/env python3
"""One table for a quantization run: per block its K, the held-out grade (grade_layer.py), the proxy errors and the
GPU time its units took; plus the forward's per-layer seconds and perplexity.  Writes <run>/summary.md and
<run>/summary.json and prints the table.

    python tools/exl3quant/summarize.py --run $OUT
"""

import argparse
import glob
import json
import os


def records(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    a = ap.parse_args()

    fwd = records(os.path.join(a.run, "log.jsonl"))
    units = [r for p in sorted(glob.glob(os.path.join(a.run, "logs", "worker-*.jsonl"))) for r in records(p)
             if r["stage"] == "unit"]
    layer_s = {r["block"]: r["seconds"] for r in fwd if r["stage"] == "layer-total"}
    hess_s = {r["block"]: r.get("hessian_seconds") for r in fwd if r["stage"] == "forward"}
    ppl = next((r for r in reversed(fwd) if r["stage"] == "perplexity"), None)
    gpu_s, n_units, gpus = {}, {}, {}
    for r in units:
        gpu_s[r["block"]] = gpu_s.get(r["block"], 0.0) + r["seconds"]
        n_units[r["block"]] = n_units.get(r["block"], 0) + 1
        gpus.setdefault(r["block"], set()).add(r["gpu"])

    rows = []
    for g in sorted(glob.glob(os.path.join(a.run, "grade-*.json"))):
        rep = json.load(open(g))
        for K, r in sorted(rep["by_k"].items()):
            rows.append({
                "block": rep["block"], "K": int(K), "experts": rep.get("experts"),
                "holdout_tokens": rep["holdout_tokens"], "moe_out_rel": r["moe_out_rel"],
                **{f"func_{p}": r["func_routed"][p]["median"] for p in ("w1", "w3", "w2") if p in r["func_routed"]},
                **{f"fro_{p}": r["rel_fro"][p]["median"] for p in ("w1", "w3", "w2")},
                **({f"proxy_{p}": r["proxy_err"][p]["median"] for p in ("w1", "w3", "w2")} if r["proxy_err"] else {}),
                "units": n_units.get(rep["block"]), "unit_gpu_seconds": round(gpu_s.get(rep["block"], 0.0), 1),
                "gpus": sorted(gpus.get(rep["block"], [])),
                "forward_seconds": layer_s.get(rep["block"]), "hessian_seconds": hess_s.get(rep["block"]),
            })

    def key(r):
        c, i = r["block"].split(".")
        return (c != "layers", int(i), r["K"])
    rows.sort(key=key)
    cols = [("block", "{}"), ("K", "{}"), ("experts", "{}"), ("moe_out_rel", "{:.4f}"), ("func_w1", "{:.4f}"),
            ("func_w3", "{:.4f}"), ("func_w2", "{:.4f}"), ("fro_w2", "{:.4f}"), ("proxy_w2", "{:.5f}"),
            ("units", "{}"), ("unit_gpu_seconds", "{}"), ("forward_seconds", "{}")]

    def cell(r, c, f):
        v = r.get(c)
        return "-" if v is None else f.format(v)
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(r, c, f) for c, f in cols) + " |" for r in rows]
    worst = max(rows, key=lambda r: r["moe_out_rel"]) if rows else None
    head = [f"# {os.path.basename(os.path.abspath(a.run))}: {len(rows)} graded blocks, {len(units)} units",
            "",
            "- forward perplexity (calibration + held-out rows, source model): "
            + (f"{ppl['ppl']} over {ppl['tokens']} tokens" if ppl else "not reached (partial forward)"),
            f"- unit GPU time: {sum(gpu_s.values()) / 3600:.2f} h over {len({r['gpu'] for r in units})} GPU(s)",
            f"- worst moe_out_rel: {worst['moe_out_rel']:.4f} ({worst['block']})" if worst else "- nothing graded",
            ""]
    text = "\n".join(head + lines) + "\n"
    open(os.path.join(a.run, "summary.md"), "w").write(text)
    json.dump({"perplexity": ppl, "blocks": rows}, open(os.path.join(a.run, "summary.json"), "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
