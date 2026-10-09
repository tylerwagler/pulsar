"""recipe.py -- pulsar.recipe.v1: the one recipe schema, every family (L279 step 5).

A recipe names, for every tensor the family's naming table emits, the FORMAT it is written in and the SOURCE it is
read from.  A JSON object:

  schema         "pulsar.recipe.v1" (optional: the files that predate the key are v1)
  recipe         its name (Qwen carries it, and the file's SHA-256, in pulsar.kv)
  model_type     the checkpoint's config.json model_type; any other checkpoint refuses
  about, formats free text, never read
  <setting>      a family setting the family declares (qwen4_exp: expert_gate_up = fused | split)
  rows           a list; each row is
                   {"role": R, ["block": B], ["layers": [...]], "format": F, ["source": S]}   by ROLE (the family
                       naming table's word for what a tensor is), optionally restricted to a block ("layers" /
                       "mtp") and a set of that block's layer indices (ints or "a-b" ranges; layers alone implies
                       the "layers" block);
                   {"name": P, ...} or the compact [P, F]                                     by HF-name fnmatch
                       pattern: the escape hatch for what no role says.

Rules (Qwen's, kept and now everyone's): every emitted tensor matches EXACTLY ONE row; a row that matches nothing
refuses.  FORMAT is an engine layout name (bf16, mxfp8_lt, cutlass_mxfp4, exl3m_k2 .. k8, fp8_e4m3_soa_k, i32 ...),
`native` (the source's own dtype: bf16 / f32 / i32), `tessera` (Tessera planes; the producer lands with L255), or a
consumed format that writes no container tensor (ple_rows, kv, omit).  SOURCE names a source the build was given:
hf (default), exl3 (default for the EXL3 formats), exl3_experts, tessera (default for tessera).  Which formats a
role admits is the family's (what its loader binds) and, for EXL3, the engine arm's (exl3_rates.admits); the
bytes come from the one producer table (producers.PRODUCER_FOR).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json

import exl3_rates

SCHEMA = "pulsar.recipe.v1"
CONSUMED = ("ple_rows", "kv", "omit")
FORMATS = {"native", "bf16", "f32", "i32", "mxfp8_lt", "fp8_e4m3_soa_k", "cutlass_mxfp4", "tessera",
           *exl3_rates.K2, *CONSUMED}
SOURCES = ("hf", "exl3", "exl3_experts", "tessera")
BLOCKS = ("layers", "mtp")
TOP_KEYS = {"schema", "recipe", "model_type", "about", "formats", "rows"}


def block_of(m) -> str | None:
    """The block a mapped tensor lives in: its shard's namespace (layers.N / mtp.N), None for top / vision."""
    ns = m.shard.split(".")[0]
    return ns if ns in BLOCKS else None


def parse_layers(items) -> frozenset:
    out = set()
    for it in items:
        if isinstance(it, int) and not isinstance(it, bool):
            out.add(it)
        elif isinstance(it, str) and "-" in it:
            a, b = it.split("-", 1)
            out |= set(range(int(a), int(b) + 1))
        else:
            raise SystemExit(f"layers: {it!r} is neither a layer index nor an 'a-b' range")
    return frozenset(out)


def compress_layers(layers) -> list:
    """Sorted layer indices -> ints and 'a-b' runs."""
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


class Row:
    def __init__(self, raw, where):
        if isinstance(raw, list):                       # the compact name row
            if len(raw) != 2:
                raise SystemExit(f"{where}: a compact row is [pattern, format], got {raw!r}")
            raw = {"name": raw[0], "format": raw[1]}
        unknown = set(raw) - {"role", "name", "block", "layers", "format", "source"}
        if unknown or ("role" in raw) == ("name" in raw) or "format" not in raw:
            raise SystemExit(f"{where}: a row is {{role | name, [block], [layers], format, [source]}}, got {raw!r}")
        self.raw = raw
        self.role, self.name = raw.get("role"), raw.get("name")
        self.format, self.source = raw["format"], raw.get("source")
        self.layers = parse_layers(raw["layers"]) if "layers" in raw else None
        self.block = raw.get("block", "layers" if self.layers is not None else None)
        if self.format not in FORMATS:
            raise SystemExit(f"{where}: row {self.label} names format {self.format!r} ({sorted(FORMATS)})")
        if self.source is not None and self.source not in SOURCES:
            raise SystemExit(f"{where}: row {self.label} names source {self.source!r} ({list(SOURCES)})")
        if self.block is not None and self.block not in BLOCKS:
            raise SystemExit(f"{where}: row {self.label} names block {self.block!r} ({list(BLOCKS)})")

    @property
    def label(self) -> str:
        if self.name is not None and set(self.raw) == {"name", "format"}:
            return self.name
        return json.dumps(self.raw, separators=(",", ":"))

    def matches(self, name, m) -> bool:
        if self.name is not None:
            if not fnmatch.fnmatchcase(name, self.name):
                return False
        elif m.role != self.role:
            return False
        if self.block is not None and block_of(m) != self.block:
            return False
        return self.layers is None or m.layer in self.layers


class Recipe:
    def __init__(self, data: dict, where: str, settings=(), sha256: str | None = None):
        if not isinstance(data, dict):
            raise SystemExit(f"{where}: a recipe is a JSON object")
        if data.get("schema", SCHEMA) != SCHEMA:
            raise SystemExit(f"{where}: schema {data['schema']!r}; this builder reads {SCHEMA}")
        unknown = set(data) - TOP_KEYS - set(settings)
        if unknown:
            raise SystemExit(f"{where}: keys {sorted(unknown)} are neither {SCHEMA} keys nor settings of this family "
                             f"({sorted(settings) or 'none'})")
        self.path = where
        self.data = data
        self.name = data["recipe"]
        self.model_type = data["model_type"]
        self.settings = {k: data[k] for k in settings if k in data}
        self.rows = [Row(r, where) for r in data["rows"]]
        self.sha256 = sha256

    @classmethod
    def load(cls, path, settings=()):
        raw = open(path, "rb").read()
        return cls(json.loads(raw), path, settings, hashlib.sha256(raw).hexdigest())

    def to_json(self) -> str:
        return json.dumps({"schema": SCHEMA, **self.data}, indent=1)

    def resolve(self, mapped: dict) -> dict:
        """name -> (format, source or None) for every emitted tensor (the family's companions and drop rules are
        not the recipe's), refusing unnamed / doubly-named tensors and dead rows."""
        out, used = {}, set()
        for n, m in mapped.items():
            if not m.emit or m.is_scale:
                continue
            hits = [i for i, r in enumerate(self.rows) if r.matches(n, m)]
            if not hits:
                raise SystemExit(f"{n}: no row of {self.path} names this tensor's format -- refusing")
            if len(hits) > 1:
                raise SystemExit(f"{n}: {len(hits)} rows of {self.path} match ({[self.rows[i].label for i in hits]}) "
                                 "-- refusing")
            r = self.rows[hits[0]]
            out[n] = (r.format, r.source)
            used.add(hits[0])
        dead = [r.label for i, r in enumerate(self.rows) if i not in used]
        if dead:
            raise SystemExit(f"{self.path}: rows that match no checkpoint tensor: {dead}")
        return out


def rows_for(mapped: dict, assign: dict) -> list:
    """The fewest role rows that say `assign` ({name: (format, source)}) exactly: one row per role where the role is
    uniform, else one per (block, format) with the layer set, else name rows for what no role row can say."""
    by_role = {}
    for n, (fmt, src) in assign.items():
        by_role.setdefault(mapped[n].role, {}).setdefault((fmt, src), []).append(n)
    rows = []

    def row(sel, fmt, src, **extra):
        r = {**sel, **extra, "format": fmt}
        if src is not None:
            r["source"] = src
        return r
    for role in sorted(by_role):
        groups = by_role[role]
        if len(groups) == 1:
            (fmt, src), = groups
            rows.append(row({"role": role}, fmt, src))
            continue
        per_block = {}
        for (fmt, src), names in groups.items():
            for n in names:
                per_block.setdefault(block_of(mapped[n]), {}).setdefault((fmt, src), []).append(n)
        for block in sorted(per_block, key=lambda b: (b is None, b or "")):
            bg = per_block[block]
            if block is None:
                rows += [row({"name": n}, fmt, src) for (fmt, src), names in sorted(bg.items()) for n in sorted(names)]
                continue
            if len(bg) == 1:
                (fmt, src), = bg
                rows.append(row({"role": role}, fmt, src, block=block))
                continue
            layer_fmt = {}
            for (fmt, src), names in bg.items():
                for n in names:
                    layer_fmt.setdefault(mapped[n].layer, set()).add((fmt, src))
            if any(len(v) > 1 for v in layer_fmt.values()):
                rows += [row({"name": n}, fmt, src) for (fmt, src), names in sorted(bg.items()) for n in sorted(names)]
                continue
            for (fmt, src), names in sorted(bg.items()):
                rows.append(row({"role": role}, fmt, src, block=block,
                                layers=compress_layers({mapped[n].layer for n in names})))
    return rows
