#!/usr/bin/env python3
"""The K map: which EXL3 rate each block's routed experts take -- and from it the pulsar.recipe.v1 rows, the
per-rank bytes on the pair, and the EXL3 source directory tools/container/build.py --exl3 reads.

A K map is JSON:

    {"about": "...", "layers": {"0-39": 3}, "mtp": {"0-2": 3}}          (kmaps/v41-k3-uniform.json)

Uniform is the default and the only map this tool ships.  A mixed map (some layers at another K) is expressible,
but on Qwen every surrogate-chosen mixed-K recipe lost to uniform on full-model KL (L251 2026-09-27: early layers
priced cheap, error propagation made them expensive), so a mixed map is a candidate until a full-model KL grade
says otherwise -- this tool does not choose one.

every trunk layer and every drafter stage named exactly once (ints or "a-b" ranges as keys), each at an integer
K the engine's routed arms read for gate / up (the pair arm) AND down (the down arm) -- the builder's
exl3_rates.admit is the authority, so this refuses what the builder would.  v41_stream.py --kmap quantizes each
block at its K.

    kmap.py budget   --kmap K.json --ref $V41 --dump plan-dump.json     per-rank GiB on the pair
    kmap.py recipe   --kmap K.json --default default-recipe.json        the build's pulsar.recipe.v1
    kmap.py assemble --kmap K.json --run $OUT --out $OUT/exl3-mixed     symlinks build.py --exl3 reads

`--container DIR` is the builder (tools/container, L279's pulsar.recipe.v1 builder: exl3_rates.admit by engine
arm); `default-recipe.json` is `build.py recipe --hf $V41` and `plan-dump.json` is `build.py plan --hf $V41
--recipe default-recipe.json --dump plan-dump.json` (the non-expert bytes in their served formats).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROLES = {"w1": "expert_gate", "w3": "expert_up", "w2": "expert_down"}
GIB = 2 ** 30


def parse_layers(items) -> list:
    """Layer indices from ints and 'a-b' ranges (strings or ints)."""
    out = []
    for it in items:
        it = str(it).strip()
        if "-" in it:
            a, b = it.split("-", 1)
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(it))
    return out


def compress(layers) -> list:
    out, run = [], []
    for x in sorted(layers):
        if run and x == run[-1] + 1:
            run.append(x)
            continue
        if run:
            out.append(run[0] if len(run) == 1 else f"{run[0]}-{run[-1]}")
        run = [x]
    if run:
        out.append(run[0] if len(run) == 1 else f"{run[0]}-{run[-1]}")
    return out


class Shape:
    """What the K map needs of V4.1, from config.json (deepseek_v41.py reads the same keys)."""

    def __init__(self, ref):
        cfg = json.load(open(os.path.join(ref, "config.json")))
        t = cfg["text_config"]
        self.dim, self.mid = t["hidden_size"], t["moe_intermediate_size"]
        self.n_layers, self.n_mtp = t["num_hidden_layers"], t["num_nextn_predict_layers"]
        self.experts = {"layers": t["n_routed_experts"], "mtp": t.get("dspark_n_routed_experts", t["n_routed_experts"])}
        self.blocks = {"layers": self.n_layers, "mtp": self.n_mtp}


def load(path, shape, container=None) -> dict:
    """{(block, index): K} for every trunk layer and drafter stage, refused unless complete, single and admitted."""
    data = json.load(open(path))
    unknown = set(data) - {"about", "layers", "mtp"}
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}")
    out = {}
    for block in ("layers", "mtp"):
        for spec, K in data[block].items():
            for i in parse_layers(spec.split(",")):
                if (block, i) in out:
                    raise SystemExit(f"{path}: {block} {i} named twice")
                out[(block, i)] = K
        want = set(range(shape.blocks[block]))
        have = {i for b, i in out if b == block}
        if have != want:
            raise SystemExit(f"{path}: {block} covers {compress(have)}, the model has {compress(want)}")
    admit = rates_authority(container)
    for (block, i), K in sorted(out.items()):
        for role in ROLES.values():
            admit(role, f"exl3m_k{K}", f"{path}: {block} {i}")
    return out


def rates_authority(container=None):
    """The builder's exl3_rates.admit (refuses a layout no engine arm reads for the role)."""
    sys.path.insert(0, container or os.path.join(HERE, "..", "container"))
    import exl3_rates
    if not hasattr(exl3_rates, "admit"):
        raise SystemExit(f"{exl3_rates.__file__}: no admit() -- the K map is graded by the engine-arm rate table of "
                         "L279's builder (exl3_rates.ARMS / admit); pass --container <an L279 tools/container>")
    return exl3_rates.admit


def projection_bytes(k_in, n_out, K, container=None, half=None):
    """Container bytes of one EXL3 projection (the builder's byte model, hf_source.exl3_expert_bytes: trellis +
    suh + svh).  half = "out" (gate / up: the rank's output columns) or "in" (down: its input rows) is one TP
    rank's cut (L269 W1 exl3_expert_half_cut: the cut side's scales halve, the other side's stay whole)."""
    rates_authority(container)
    import exl3_rates
    import hf_source
    if half == "out":
        n_out //= 2
    elif half == "in":
        k_in //= 2
    trellis, scales = hf_source.exl3_expert_bytes(k_in, n_out, exl3_rates.words_for(f"exl3m_k{K}"))
    return trellis + scales


def expert_bytes(shape, kmap, per_rank, container=None):
    """{(block, i): bytes} of each block's routed experts, whole or one rank's half."""
    out = {}
    for (block, i), K in kmap.items():
        gu = projection_bytes(shape.dim, shape.mid, K, container, "out" if per_rank else None)
        dn = projection_bytes(shape.mid, shape.dim, K, container, "in" if per_rank else None)
        out[(block, i)] = shape.experts[block] * (2 * gu + dn)
    return out


def non_expert_bytes(dump):
    """Bytes of every container tensor that is not a routed expert, from a builder plan dump (its exact shard
    headers), by top / layers / mtp / vision."""
    out = {}
    with open(dump) as f:
        next(f)                                         # the model line
        for line in f:
            rec = json.loads(line)
            if "header" not in rec:
                continue
            hdr = json.loads(rec["header"])
            hdr.pop("__metadata__", None)
            for name, h in hdr.items():
                if ".ffn.experts." in name:                # routed experts (the container keeps HF names)
                    continue
                part = rec["shard"].split(".")[0]
                o0, o1 = h["data_offsets"]
                out[part] = out.get(part, 0) + o1 - o0
    return out


def cmd_budget(a):
    shape = Shape(a.ref)
    km = load(a.kmap, shape, a.container)
    whole = expert_bytes(shape, km, False, a.container)
    rank = expert_bytes(shape, km, True, a.container)
    n_par = sum(shape.experts[b] for b, _ in km) * 3 * shape.dim * shape.mid
    bits = 8 * sum(whole.values()) / n_par
    trunk_bits = 8 * sum(v for (b, _), v in whole.items() if b == "layers") / (
        shape.n_layers * shape.experts["layers"] * 3 * shape.dim * shape.mid)
    print(f"K map {a.kmap}")
    by_k = {}
    for (b, i), K in sorted(km.items()):
        by_k.setdefault((b, K), []).append(i)
    for (b, K), ls in sorted(by_k.items()):
        print(f"  {b:6s} K{K}: {len(ls):2d} blocks {compress(ls)}  {sum(rank[(b, i)] for i in ls) / GIB:7.2f} GiB/rank")
    ex_rank = sum(rank.values()) / GIB
    print(f"routed experts: {sum(whole.values()) / GIB:.2f} GiB whole, {ex_rank:.2f} GiB per rank; "
          f"{trunk_bits:.3f} bpw trunk, {bits:.3f} bpw with the drafter (scales included)")
    if a.dump:
        ne = non_expert_bytes(a.dump)
        tot = sum(ne.values()) / GIB
        print("non-expert tensors (served formats, the builder's plan): "
              + ", ".join(f"{k} {v / GIB:.2f}" for k, v in sorted(ne.items())) + f" -- {tot:.2f} GiB")
        no_vis = tot - ne.get("vision", 0) / GIB
        print(f"per rank, non-expert replicated (upper bound; TP splits attention heads, the vocab and the drafter "
              f"heads): {ex_rank + tot:.2f} GiB with the tower, {ex_rank + no_vis:.2f} without; plus KV "
              f"(~0.9 GiB per 1M tokens) and the runtime's buffers, against 110-115 GiB usable")


def cmd_recipe(a):
    km = load(a.kmap, Shape(a.ref), a.container)
    default = json.load(open(a.default))
    rows = [r for r in default["rows"] if not str(r.get("role", "")).startswith("expert_")]
    if len(rows) == len(default["rows"]):
        raise SystemExit(f"{a.default}: no expert_* rows -- not the builder's DeepSeek default recipe")
    for block in ("layers", "mtp"):
        by_k = {}
        for (b, i), K in km.items():
            if b == block:
                by_k.setdefault(K, []).append(i)
        for K, ls in sorted(by_k.items()):
            sel = {"block": block} if len(by_k) == 1 else {"block": block, "layers": compress(ls)}
            rows += [{"role": role, **sel, "format": f"exl3m_k{K}", "source": "exl3"} for role in ROLES.values()]
    recipe = {"schema": "pulsar.recipe.v1", "recipe": a.name, "model_type": default["model_type"],
              "about": f"V4.1 routed experts in EXL3 by the K map {os.path.basename(a.kmap)} (our quantization, "
                       "tools/exl3quant); every other tensor as the builder's default recipe says",
              "rows": rows}
    out = json.dumps(recipe, indent=1)
    if a.out:
        open(a.out, "w").write(out + "\n")
    print(out)


def cmd_assemble(a):
    km = load(a.kmap, Shape(a.ref), a.container)
    os.makedirs(a.out, exist_ok=True)
    for (block, i), K in sorted(km.items()):
        name = f"model-{'layer' if block == 'layers' else 'mtp'}{i:02d}.safetensors"
        src = os.path.join(a.run, f"exl3-k{K}", name)
        if not os.path.exists(src):
            raise SystemExit(f"{src}: missing -- {block} {i} was not quantized at K{K}")
        dst = os.path.join(a.out, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.relpath(src, a.out), dst)
    print(f"{len(km)} blocks -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(required=True)
    for name, fn in (("budget", cmd_budget), ("recipe", cmd_recipe), ("assemble", cmd_assemble)):
        p = sub.add_parser(name)
        p.add_argument("--kmap", required=True)
        p.add_argument("--ref", required=True, help="the V4.1 snapshot (config.json)")
        p.add_argument("--container", default=None, help="the builder's tools/container (default: this tree's)")
        if name == "budget":
            p.add_argument("--dump", default=None, help="build.py plan --dump of the default recipe")
        if name == "recipe":
            p.add_argument("--default", required=True, help="build.py recipe --hf $V41 output")
            p.add_argument("--name", default="deepseek_v41-exl3-ours")
            p.add_argument("--out", default=None)
        if name == "assemble":
            p.add_argument("--run", required=True, help="v41_stream.py's --out")
            p.add_argument("--out", required=True)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
