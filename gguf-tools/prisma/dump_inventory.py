#!/usr/bin/env python3
"""dump_inventory.py — emit tensors.json (name/shape/family/type) from a GGUF.
Structural tensor inventory (family buckets: routed_expert
(the allocation lever) vs attn/shared_expert/output/embed/mla_aux/norm (pinned).
Usage: python3 dump_inventory.py MODEL.gguf > tensors.json   (needs `gguf` pkg)
"""
import sys, json, gguf

def family(n):
    if "exps" in n: return "routed_expert"
    if "shexp" in n: return "shared_expert"
    if "attn" in n: return "attn"
    if n.startswith("output"): return "output"
    if "token_embd" in n: return "embed"
    if any(k in n for k in ("indexer", "compress", "_hc", "index")): return "mla_aux"
    if "norm" in n: return "norm"
    return "other"

r = gguf.GGUFReader(sys.argv[1])
rows = [dict(name=t.name, shape=[int(s) for s in t.shape],
             type=t.tensor_type.name, family=family(t.name)) for t in r.tensors]
json.dump(rows, sys.stdout)
