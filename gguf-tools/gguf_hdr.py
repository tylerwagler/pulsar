"""Minimal read-only GGUF header parser, shared by the in-place splice tools.

Standalone so tools that only need to read the header do not inherit a numpy
dependency.

The one thing this does that a general GGUF reader would not bother with:
scan() records where each tensor's TYPE FIELD sits in the file (`type_pos`), so
a caller can retype a tensor by writing 4 bytes rather than rebuilding and
reflowing the whole header.  That is what makes size-preserving splices -- IQ2
SoA permutation, F16 -> BF16 -- a copy plus a few small writes instead of a
full re-quantize pass.
"""
import struct


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _skip_value(f, t):
    if t in (0, 1, 7):   f.read(1)
    elif t in (2, 3):    f.read(2)
    elif t in (4, 5, 6): f.read(4)
    elif t in (10, 11, 12): f.read(8)
    elif t == 8:         _read_str(f)
    elif t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _skip_value(f, et)
    else:
        raise SystemExit(f"unknown GGUF value type {t}")


def scan(path):
    """Return (tensors, alignment, data_start). Each tensor records where its
    TYPE field sits in the header so it can be patched without reflowing."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit("not a GGUF")
        struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        alignment = 32
        for _ in range(n_kv):
            key = _read_str(f)
            (vt,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment" and vt == 4:
                (alignment,) = struct.unpack("<I", f.read(4))
            else:
                _skip_value(f, vt)
        tensors = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            type_pos = f.tell()
            (ttype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            ne = 1
            for d in dims:
                ne *= d
            tensors.append(dict(name=name, dims=dims, type=ttype, type_pos=type_pos,
                                offset=offset, ne=ne))
        data_start = f.tell()
    pad = (alignment - (data_start % alignment)) % alignment
    return tensors, alignment, data_start + pad
