"""entries.py -- one container entry's payload from its source, through the ONE producer table (L279 step 1).

Both families' plans call these.  What decides the bytes is (source kind, target format) -> producer
(producers.PRODUCER_FOR); a pair the table lacks refuses by name.  The source kind is the HF tensor's stored dtype,
`+scale` when it carries an FP8 / FP4 block-scale companion (`X.scale` beside `X.weight`), or `exl3` for an
exllamav3 Linear (trellis | suh | svh, copied verbatim)."""
from __future__ import annotations

import producers as PR


def scale_of(name: str) -> str | None:
    return name[:-len(".weight")] + ".scale" if name.endswith(".weight") else None


def kind_of(hf, name: str) -> str:
    """The source kind of HF tensor `name`."""
    s = scale_of(name)
    return PR.source_kind(hf.dtype(name), s is not None and hf.has(s))


def dense(hf, name: str, layout: str, dshape: list, mxfp8_mode: str = "rederive") -> dict:
    """{dtype, shape, nbytes, src} of the entry writing HF tensor `name` as `layout`, declared `dshape` (the
    shape the container holds; dims_ne is its reverse)."""
    kind = kind_of(hf, name)
    prod = PR.producer_for(kind, layout, name)
    hshape = hf.shape(name)
    if prod == PR.COPY:
        path, off, n = hf.span(name)
        return {"dtype": hf.dtype(name), "shape": list(dshape), "nbytes": n, "src": ("ranges", [(path, off, n)])}
    if prod == "i64_to_i32":
        n_el = 1
        for d in hshape:
            n_el *= d
        return {"dtype": "I32", "shape": list(dshape), "nbytes": 4 * n_el, "src": ("produce", PR.spec(prod, [name]))}
    if len(hshape) != 2:
        raise SystemExit(f"{name}: {prod} writes a matrix, the source is {hf.dtype(name)} {hshape}")
    nbytes = PR.bytes_for(layout, list(reversed(dshape)))
    if prod == "mxfp8_lt":
        out, inp = hshape
        s = scale_of(name)
        desc = PR.spec(prod, [name, s], out=out, inp=inp, block=out // hf.shape(s)[0], mode=mxfp8_mode)
    elif prod == "mxfp8_lt_from_bf16":
        out, inp = hshape
        desc = PR.spec(prod, [name], out=out, inp=inp)
    elif prod == "fp8_e4m3_soa_k_from_bf16":
        rows, cols = hshape
        desc = PR.spec(prod, [name], rows=rows, cols=cols)
    else:
        raise SystemExit(f"{name}: {prod} writes one routed expert, not a dense entry")
    return {"dtype": "U8", "shape": [nbytes], "nbytes": nbytes, "src": ("produce", desc)}


def expert_src(hf, name: str, layout: str, out: int, inp: int):
    """The source descriptor of one routed expert-projection read from the HF checkpoint."""
    prod = PR.producer_for(kind_of(hf, name), layout, name)
    if prod != "cutlass_mxfp4":
        raise SystemExit(f"{name}: {prod} does not write a routed expert")
    return ("produce", PR.spec(prod, [name, scale_of(name)], out=out, inp=inp))


def exl3_ranges(src, key: str, layout: str, k: int, n: int, rates: dict):
    """One EXL3 Linear `key` of the exllamav3 checkpoint `src` as `layout`: its byte ranges and their count."""
    PR.producer_for("exl3", layout, key)
    ranges, words = src.linear(key, k, n, rates)
    if rates[words] != layout:
        raise SystemExit(f"{key}: the recipe names {layout}, the EXL3 source holds {rates[words]} -- refusing")
    nbytes = PR.bytes_for(layout, [k, n])
    if sum(r[2] for r in ranges) != nbytes:
        raise SystemExit(f"{key}: EXL3 source spans {sum(r[2] for r in ranges)} bytes, {layout} on [{k}, {n}] is {nbytes}")
    return ranges, nbytes
