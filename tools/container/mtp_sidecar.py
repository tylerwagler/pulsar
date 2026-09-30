#!/usr/bin/env python3
"""mtp_sidecar.py -- the MTP sidecar of a qwen4_exp container (L251): ONE extra shard, `model-mtp.safetensors`,
holding the drafter's `mtp.*` tensors in the container's own formats and names, beside a container whose recipe
omitted them.  The base container's shards are never touched or re-emitted.

    python3 mtp_sidecar.py emit --hf BF16_SNAPSHOT --exl3 TURBODERP_EXL3_SNAPSHOT --base CONTAINER_DIR \\
        --recipe format-maps/qwen38fn-u-e4-d5-mtp.json --out OUT_DIR [--emitter-rev 9596802e]

The codecs are the EMITTER'S, not copies: the base container was built by tools/container at git rev
9596802e (branch work/l251-container: qwen.py, producers.mxfp8_lt_from_bf16, hf_source.Exl3Checkpoint.linear,
build.write_shard), which this branch does not carry.  `--emitter-rev` extracts that rev's tools/container
(read-only, `git archive`) into a scratch directory and imports those modules, so the sidecar's bytes come from
the same code as the container's.  The pin is checked, not trusted: the base container's recorded
pulsar.recipe.sha256 must equal the rev's own format-maps/qwen38fn-u-e4-d5.json.

What the sidecar holds (the loader's contract, src/engine/safetensors.cpp):
  * names ARE the BF16 checkpoint's HF names (`mtp.fc_embedding.weight`, `mtp.layers.0.self_attn.q_proj.weight`,
    ...), gguf_name == that name, exactly as qwen.py names the main layers; the Qwen binder (family_qwen.cpp
    qbind) looks tensors up by these names.
  * routed experts SPLIT (turboderp's layout, recipe `expert_gate_up: split`): three families per layer,
    `mtp.layers.L.mlp.experts.{gate_proj,up_proj,down_proj}` (gguf_name), one U8 entry per expert
    `mtp.layers.L.mlp.experts.<e>.<part>.weight`, contiguous in expert order; each family in pulsar.experts
    carries entry_name (with {e}), layer, and block "mtp" -- the same shape qwen._plan_experts writes for a split
    recipe.  gate = rows [0, I) of the HF gate_up_proj stack, up = rows [I, 2I) (checked by the test's fidelity
    pass).
  * layouts: bf16 (the BF16 source's bytes), mxfp8_lt (producers.mxfp8_lt_from_bf16 of the BF16 source),
    exl3m_k3 / k4 / k5 (turboderp's trellis | suh | svh, verbatim; the rate is the trellis width's, per tensor,
    and must equal the recipe's row).
  * metadata: the shard carries NO pulsar.kv (the base primary keeps that role -- the loader takes the first
    shard carrying it); its drafter facts ride in `pulsar.mtp.kv` (same entry spelling as pulsar.kv) with
    pulsar.mtp_present = true, the MTP shape keys of the text_config, provenance, and the base container's
    identity (recipe sha256, primary shard sha256) so a sidecar laid beside the wrong container can be refused.
  * the loader lists every *.safetensors in the directory, so serving = the base shards + this shard in one
    directory; `OUT/combined/` is exactly that, as symlinks, with a merged model.safetensors.index.json.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SIDECAR = "model-mtp.safetensors"
SIDECAR_INDEX = "model-mtp.index.json"
SHARD_KEY = "mtp"
EMITTER_REV = "9596802e"
BASE_RECIPE = "format-maps/qwen38fn-u-e4-d5.json"
_EMITTER_MODULES = ("hf_source", "producers", "kv", "qwen", "build")


# ---------------------------------------------------------------------------
# the emitter's modules, pinned by git rev
# ---------------------------------------------------------------------------
class Emitter:
    def __init__(self, rev: str = EMITTER_REV):
        self.sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--verify", rev + "^{commit}"],
                                           text=True).strip()
        self.dir = os.path.join(tempfile.gettempdir(), f"pulsar-emitter-{self.sha[:12]}")
        root = os.path.join(self.dir, "tools", "container")
        if not os.path.exists(os.path.join(root, "qwen.py")):
            os.makedirs(self.dir, exist_ok=True)
            arch = subprocess.check_output(["git", "-C", REPO, "archive", self.sha, "tools/container"])
            subprocess.run(["tar", "-x", "-C", self.dir], input=arch, check=True)
        self.root = root
        for m in _EMITTER_MODULES:
            if m in sys.modules and not os.path.abspath(sys.modules[m].__file__).startswith(root + os.sep):
                raise SystemExit(f"module {m} already imported from {sys.modules[m].__file__}, not the emitter "
                                 f"rev {self.sha[:12]} -- import mtp_sidecar before any tools/container module")
        sys.path.insert(0, root)
        for m in _EMITTER_MODULES:
            mod = importlib.import_module(m)
            if not os.path.abspath(mod.__file__).startswith(root + os.sep):
                raise SystemExit(f"{m} resolved to {mod.__file__}, not the emitter rev {self.sha[:12]}")
            setattr(self, m, mod)
        self.base_recipe_sha256 = hashlib.sha256(open(os.path.join(root, BASE_RECIPE), "rb").read()).hexdigest()
        # words per 16x16 tile -> layout, from the emitter's own rate table (producers._EXL3_K2, the k2 = 2K
        # half-bit units of exl3_trellis.h) and exl3_words_per_tile (16K, +8 for a half-integer K)
        self.rates = {16 * (k2 >> 1) + (8 if k2 & 1 else 0): lay for lay, k2 in self.producers._EXL3_K2.items()}
        if len(self.rates) != len(self.producers._EXL3_K2):
            raise SystemExit("two EXL3 layouts share a tile width")


# ---------------------------------------------------------------------------
# the recipe (qwen.Recipe's rule: every tensor exactly one row, every row used; its FORMATS predate exl3m_k3)
# ---------------------------------------------------------------------------
class Recipe:
    def __init__(self, path, em: Emitter):
        self.path = path
        r = json.load(open(path))
        if r.get("model_type") != em.qwen.FAMILY:
            raise SystemExit(f"{path}: recipe is for {r.get('model_type')!r}, not {em.qwen.FAMILY}")
        self.name = r["recipe"]
        self.gate_up = r["expert_gate_up"]
        if self.gate_up not in ("fused", "split"):
            raise SystemExit(f"{path}: expert_gate_up {self.gate_up!r}")
        self.base_recipe_sha256 = r["base_recipe_sha256"]
        self.formats = {"bf16", "mxfp8_lt"} | set(em.rates.values())
        self.rows = [tuple(x) for x in r["rows"]]
        for pat, fmt in self.rows:
            if not pat.startswith("mtp."):
                raise SystemExit(f"{path}: row {pat!r} is not an mtp.* pattern")
            if fmt not in self.formats:
                raise SystemExit(f"{path}: row {pat!r} names {fmt!r}; this sidecar writes {sorted(self.formats)}")
        self.sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()

    def resolve(self, names):
        out, used = {}, set()
        for n in names:
            hits = [(p, f) for p, f in self.rows if fnmatch.fnmatchcase(n, p)]
            if len(hits) != 1:
                raise SystemExit(f"{n}: {len(hits)} rows of {self.path} match {[p for p, _ in hits]} -- refusing")
            out[n] = hits[0][1]
            used.add(hits[0][0])
        dead = [p for p, _ in self.rows if p not in used]
        if dead:
            raise SystemExit(f"{self.path}: rows that match no mtp.* tensor: {dead}")
        return out


# ---------------------------------------------------------------------------
# the base container
# ---------------------------------------------------------------------------
def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    return hdr, hdr.pop("__metadata__", {}) or {}, 8 + n


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Base:
    def __init__(self, d):
        self.dir = os.path.abspath(d)
        self.shards = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
        if not self.shards or SIDECAR in self.shards:
            raise SystemExit(f"{d}: expected the base container's shards only, found {len(self.shards)} "
                             f"({'including ' + SIDECAR if SIDECAR in self.shards else 'none'})")
        self.primary = None
        for f in self.shards:
            _h, md, _o = read_header(os.path.join(d, f))
            if "pulsar.kv" in md:
                self.primary, self.kv = f, {e["key"]: e for e in json.loads(md["pulsar.kv"])}
                self.n_shards = md.get("pulsar.n_shards")
                break
        if self.primary is None:
            raise SystemExit(f"{d}: no shard carries pulsar.kv")
        self.index = json.load(open(os.path.join(d, "model.safetensors.index.json")))

    def kv_value(self, key):
        if key not in self.kv:
            raise SystemExit(f"{self.dir}: pulsar.kv has no {key}")
        return self.kv[key]["value"]


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
_EXPERT_STACK = re.compile(r"^(mtp\.layers\.(\d+)\.mlp\.experts\.)(gate_up_proj|down_proj)$")


def _exl3(em, ex, key, k, n, want):
    ranges, words = ex.linear(key, k, n, em.rates)          # every refusal hf_source allows (mcg, dims, mul1)
    if em.rates[words] != want:
        raise SystemExit(f"{key}: the recipe names {want}, the EXL3 source holds {em.rates[words]} "
                         f"({words} words per tile) -- refusing")
    nbytes = em.producers.bytes_for(want, [k, n])
    if sum(r[2] for r in ranges) != nbytes:
        raise SystemExit(f"{key}: EXL3 source spans {sum(r[2] for r in ranges)} bytes, {want} on [{k}, {n}] is {nbytes}")
    return ranges, nbytes


def plan(em, hf, ex, recipe):
    PR = em.producers
    names = [n for n in hf.names() if n.startswith("mtp.")]
    if not names:
        raise SystemExit(f"{hf.dir}: no mtp.* tensors in the BF16 checkpoint")
    fmt = recipe.resolve(names)
    entries, tensors, experts, used_exl3 = [], {}, [], set()
    for name in names:
        f = fmt[name]
        dtype, shape = hf.dtype(name), hf.shape(name)
        if dtype != "BF16":
            raise SystemExit(f"{name}: the BF16 checkpoint holds {dtype}")
        if f == "bf16":
            path, off, n = hf.span(name)
            entries.append({"name": name, "layout": "bf16", "gguf_name": name, "dtype": "BF16", "shape": list(shape),
                            "nbytes": n, "src": ("ranges", [(path, off, n)])})
            tensors[name] = {"layout": "bf16", "dims_ne": list(reversed(shape)), "gguf_name": name}
        elif f == "mxfp8_lt":
            if len(shape) != 2:
                raise SystemExit(f"{name}: mxfp8_lt from a BF16 matrix, got {shape}")
            out, inp = shape
            nbytes = PR.bytes_for("mxfp8_lt", [inp, out])
            entries.append({"name": name, "layout": "mxfp8_lt", "gguf_name": name, "dtype": "U8", "shape": [nbytes],
                            "nbytes": nbytes, "bf16_src": name,
                            "src": ("produce", (lambda w=name, o=out, i=inp: PR.mxfp8_lt_from_bf16(hf.raw(w), o, i)))})
            tensors[name] = {"layout": "mxfp8_lt", "dims_ne": [inp, out], "gguf_name": name}
        elif _EXPERT_STACK.match(name):
            m = _EXPERT_STACK.match(name)
            pre, layer, stack = m.group(1), int(m.group(2)), m.group(3)
            E, out, inp = shape
            if stack == "gate_up_proj":
                if recipe.gate_up != "split" or out % 2:
                    raise SystemExit(f"{name}: this sidecar writes turboderp's split gate / up (width {out})")
                parts = [("gate_proj", out // 2), ("up_proj", out // 2)]
            else:
                parts = [("down_proj", out)]
            for part, n in parts:
                eb = PR.bytes_for(f, [inp, n])
                entry_name = pre + "{e}." + part + ".weight"
                for e in range(E):
                    key = f"{pre}{e}.{part}"
                    ranges, nbytes = _exl3(em, ex, key, inp, n, f)
                    used_exl3.add(key)
                    entries.append({"name": entry_name.replace("{e}", str(e)), "layout": f, "gguf_name": pre + part,
                                    "dtype": "U8", "shape": [eb], "nbytes": nbytes, "src": ("ranges", ranges)})
                experts.append({"gguf_name": pre + part, "part": part, "n_experts": E, "expert_bytes": eb,
                                "layout": f, "contiguous": True, "dims_per_expert_ne": [inp, n],
                                "entry_name": entry_name, "layer": layer, "block": "mtp"})
        else:                                               # an EXL3 dense Linear
            if len(shape) != 2 or not name.endswith(".weight"):
                raise SystemExit(f"{name}: EXL3 dense Linear from a BF16 [out, in] .weight, got {shape}")
            out, inp = shape
            key = name[:-len(".weight")]
            ranges, nbytes = _exl3(em, ex, key, inp, out, f)
            used_exl3.add(key)
            entries.append({"name": name, "layout": f, "gguf_name": name, "dtype": "U8", "shape": [nbytes],
                            "nbytes": nbytes, "src": ("ranges", ranges)})
            tensors[name] = {"layout": f, "dims_ne": [inp, out], "gguf_name": name}

    # the EXL3 source closed both ways: every mtp.* Linear it quantized is consumed, and every tensor it keeps
    # unquantized is one the recipe takes from BF16 (else the two checkpoints disagree on what a Linear is)
    # -- a Linear the recipe deliberately takes from BF16 instead (mxfp8_lt) is the one allowed non-use
    src_lin = {k[:-len(".trellis")] for k in ex.entries if k.startswith("mtp.") and k.endswith(".trellis")}
    unused = sorted(k for k in src_lin - used_exl3 if fmt.get(k + ".weight") != "mxfp8_lt")
    if unused or used_exl3 - src_lin:
        raise SystemExit(f"EXL3 source mtp Linears not consumed: {unused[:8]}; "
                         f"consumed but absent: {sorted(used_exl3 - src_lin)[:8]}")
    for k in sorted(src_lin - used_exl3):
        print(f"  {k}: the EXL3 source quantized it; the recipe takes mxfp8_lt from BF16", flush=True)
    src_plain = {k for k in ex.entries if k.startswith("mtp.") and not k.endswith((".trellis", ".suh", ".svh", ".mul1"))}
    bad = sorted(k for k in src_plain if fmt.get(k) not in ("bf16", "mxfp8_lt"))
    if bad:
        raise SystemExit(f"EXL3 source keeps these unquantized, the recipe does not take them from BF16: {bad}")
    return names, fmt, entries, tensors, experts


def mtp_kvs(em, hf, recipe, base, ex_dir, base_primary_sha):
    """pulsar.mtp.kv: the drafter's facts in pulsar.kv's own spelling (kv.entry)."""
    text = hf.config["top_level"]["text_config"]
    cfg = []
    for k in sorted(text):
        if "mtp" in k:
            em.qwen.config_kvs(f"{em.qwen.FAMILY}.{k}", text[k], cfg)
    if not cfg:
        raise SystemExit("text_config carries no mtp keys")
    for key, typ, val in cfg:                                # the base already records them: they must agree
        want = em.kv.entry(key, typ, val)
        if base.kv.get(key) != want:
            raise SystemExit(f"{key}: base pulsar.kv holds {base.kv.get(key)}, the checkpoint says {want}")
    qc = json.load(open(os.path.join(ex_dir, "config.json"))).get("quantization_config") or {}
    kvs = [("pulsar.mtp_present", "bool", True)] + cfg + [
        ("pulsar.mtp.recipe", "string", recipe.name),
        ("pulsar.mtp.recipe.sha256", "string", recipe.sha256),
        ("pulsar.mtp.expert_gate_up", "string", recipe.gate_up),
        ("pulsar.mtp.emitter_rev", "string", em.sha),
        ("pulsar.mtp.source.hf", "string", os.path.basename(os.path.normpath(hf.dir))),
        ("pulsar.mtp.source.exl3", "string", os.path.basename(os.path.normpath(ex_dir))),
        ("pulsar.mtp.source.exl3.version", "string", str(qc.get("version"))),
        ("pulsar.mtp.source.exl3.codebook", "string", str(qc.get("codebook"))),
        ("pulsar.mtp.source.exl3.mtp_bits", "u32", int(qc.get("mtp_bits"))),
        ("pulsar.mtp.base.recipe", "string", base.kv_value("pulsar.recipe")),
        ("pulsar.mtp.base.recipe.sha256", "string", base.kv_value("pulsar.recipe.sha256")),
        ("pulsar.mtp.base.primary", "string", base.primary),
        ("pulsar.mtp.base.primary_sha256", "string", base_primary_sha),
        ("pulsar.mtp.base.n_shards", "u32", int(base.n_shards)),
    ]
    return [em.kv.entry(k, t, v) for k, t, v in kvs]


def cmd_emit(a):
    em = Emitter(a.emitter_rev)
    hf = em.hf_source.HFCheckpoint(a.hf)
    if hf.config["top_level"].get("model_type") != em.qwen.FAMILY:
        raise SystemExit(f"{a.hf}: not a {em.qwen.FAMILY} checkpoint")
    ex = em.hf_source.Exl3Checkpoint(a.exl3)
    recipe = Recipe(a.recipe, em)
    base = Base(a.base)
    # the pins: base <- emitter rev's base recipe, base <- this BF16 snapshot, sidecar recipe <- base recipe
    base_sha = base.kv_value("pulsar.recipe.sha256")
    if base_sha != em.base_recipe_sha256 or base_sha != recipe.base_recipe_sha256:
        raise SystemExit(f"base recipe sha256 {base_sha}; emitter rev {em.sha[:12]} has {em.base_recipe_sha256}, "
                         f"{a.recipe} names {recipe.base_recipe_sha256} -- not the container this sidecar is for")
    if base.kv_value("pulsar.source.hf") != os.path.basename(os.path.normpath(hf.dir)):
        raise SystemExit(f"base was built from HF {base.kv_value('pulsar.source.hf')}, not {hf.dir}")
    if base.kv_value("pulsar.mtp_present") is not False:
        raise SystemExit("the base container already declares MTP")
    names, fmt, entries, tensors, experts = plan(em, hf, ex, recipe)
    clash = sorted(set(e["name"] for e in entries) & set(base.index["weight_map"]))
    if clash:
        raise SystemExit(f"sidecar names already in the base container: {clash[:8]}")
    print(f"emitter {em.sha[:12]}  base {base.dir} ({len(base.shards)} shards, primary {base.primary})", flush=True)
    base_primary_sha = sha256_file(os.path.join(base.dir, base.primary))
    kvs = mtp_kvs(em, hf, recipe, base, a.exl3, base_primary_sha)
    meta = {
        "format": "pt",
        "pulsar.format": "pulsar-safetensors-v1",
        "pulsar.family": em.qwen.FAMILY,
        "pulsar.alignment": str(em.build.ALIGN),
        "pulsar.shard": SIDECAR,
        "pulsar.shard_key": SHARD_KEY,
        "pulsar.sidecar": SHARD_KEY,
        "pulsar.primary": base.primary,
        "pulsar.tensors": json.dumps(tensors, separators=(",", ":"), sort_keys=True),
        "pulsar.experts": json.dumps(experts, separators=(",", ":")),
        "pulsar.expert_dtype": "exl3" if experts and all(x["layout"].startswith("exl3m_") for x in experts) else "mixed",
        "pulsar.mtp.kv": json.dumps(kvs, separators=(",", ":")),
    }
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, SIDECAR)
    header, total = em.build.write_shard(path, entries, meta)
    size = os.path.getsize(path)
    print(f"{path}: {len(header)} tensors, payload {total} B, file {size} B ({size / 2**30:.3f} GiB)", flush=True)
    json.dump({"metadata": {"total_size": size, "base": base.dir, "base_primary_sha256": base_primary_sha,
                            "sha256": sha256_file(path)},
               "weight_map": {k: SIDECAR for k in sorted(header)}},
              open(os.path.join(a.out, SIDECAR_INDEX), "w"), indent=0, sort_keys=True)
    # combined/: the base shards + the sidecar as one loadable directory (symlinks), and a merged index
    comb = os.path.join(a.out, "combined")
    os.makedirs(comb, exist_ok=True)
    links = {f: os.path.join(base.dir, f) for f in base.shards}
    links[SIDECAR] = os.path.join("..", SIDECAR)
    for f in ("tokenizer.json", "generation_config.json"):   # the Qwen family reads these from the model dir
        links[f] = os.path.join(hf.dir, f)
    for f, tgt in links.items():
        p = os.path.join(comb, f)
        if os.path.islink(p):
            os.unlink(p)
        os.symlink(tgt, p)
    wm = dict(base.index["weight_map"])
    wm.update({k: SIDECAR for k in header})
    json.dump({"metadata": {"total_size": base.index["metadata"]["total_size"] + size},
               "weight_map": dict(sorted(wm.items()))},
              open(os.path.join(comb, "model.safetensors.index.json"), "w"), indent=0, sort_keys=True)
    by_fmt = {}
    for n in names:
        by_fmt[fmt[n]] = by_fmt.get(fmt[n], 0) + 1
    print(f"formats over the {len(names)} mtp.* HF tensors: {by_fmt}; {len(experts)} expert families; "
          f"combined dir {comb}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    e.add_argument("--hf", required=True)
    e.add_argument("--exl3", required=True)
    e.add_argument("--base", required=True)
    e.add_argument("--recipe", default=os.path.join(HERE, "format-maps", "qwen38fn-u-e4-d5-mtp.json"))
    e.add_argument("--out", required=True)
    e.add_argument("--emitter-rev", default=EMITTER_REV)
    a = ap.parse_args()
    return cmd_emit(a)


if __name__ == "__main__":
    sys.exit(main())
