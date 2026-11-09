#!/usr/bin/env python3
"""The Engram hash layout of a DeepSeek V4.1 checkpoint, for the container (L242 slice 5).

    engram_layout.py SNAPSHOT_DIR OUT.json        # needs `tokenizers`, `sympy`, `numpy`

The checkpoint's inference/engram.py derives everything the n-gram hash needs from the tokenizer and config.json at
run time: the compressed token map (build_compressed_token_map), the per-(layer, n-gram, head) bucket primes
(EngramLayout.from_args), their offsets, the per-layer hash multipliers (compute_hash_multipliers) and the pad id.
The engine must not re-derive them (a mismatch silently rehashes the whole 189 GiB table), so this tool does it ONCE,
asserts the two checks the checkpoint itself makes (the compressed vocab size, and that the bucket layout fills each
table to the row), and the builder carries the result in the container (tools/container/kv.py
_engram_layout_triples; `build.py --engram-layout OUT.json`).  The engine reads it at load
(weights.cpp engram_bind) and hashes with pulsar_engram_hash_pos.

A port of notes/gate-baseline/l218-v41/engram_hash.py (whose output tests/engram_hash_test.cpp pins) onto the HF
config.json (text_config) the builder reads, so the layout comes from the same snapshot as the container."""
import json
import os
import sys

import numpy as np
from sympy import isprime
from tokenizers import Regex, Tokenizer, normalizers


def compressed_token_map(tok, n_vocab):
    """engram.py build_compressed_token_map, over the raw Rust tokenizer."""
    sentinel = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    key_to_new, lookup = {}, [0] * n_vocab
    for tid in range(n_vocab):
        text = tok.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(tid)
        else:
            n = norm.normalize_str(text)
            key = n if n else text
        lookup[tid] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    snap, out = sys.argv[1], sys.argv[2]
    top = json.load(open(os.path.join(snap, "config.json")))
    cfg = top.get("text_config", top)
    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    n_vocab = tok.get_vocab_size(with_added_tokens=True)
    if n_vocab != cfg["vocab_size"]:
        raise SystemExit(f"tokenizer has {n_vocab} tokens, config.json vocab_size {cfg['vocab_size']}")
    lookup, cv = compressed_token_map(tok, n_vocab)
    if cv != cfg["engram_compressed_vocab_size"]:
        raise SystemExit(f"compressed vocab {cv} != config {cfg['engram_compressed_vocab_size']}: the hash would "
                         "rehash the whole table")

    layer_ids, max_ngram, n_heads = list(cfg["engram_layer_ids"]), cfg["engram_max_ngram_size"], cfg["engram_n_heads"]
    primes, seen = [], set()
    for _ in layer_ids:                                   # EngramLayout.from_args: drawn in order, never reused
        per_ngram = []
        for _ in range(max_ngram - 1):
            sizes, cur = [], cfg["engram_vocab_size"] - 1
            for _ in range(n_heads):
                cur += 1
                while not isprime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    flat = [[p for per in layer for p in per] for layer in primes]
    offsets = [np.cumsum([0, *sizes[:-1]]).tolist() for sizes in flat]
    if [sum(s) for s in flat] != list(cfg["engram_num_embeddings"]):
        raise SystemExit(f"bucket layout fills {[sum(s) for s in flat]} rows, config.json "
                         f"engram_num_embeddings {cfg['engram_num_embeddings']}")
    bound = max(1, (np.iinfo(np.int64).max // cv) // 2)  # compute_hash_multipliers
    mult = []
    for lid in layer_ids:
        v = np.random.default_rng(10007 * lid).integers(low=0, high=bound, size=(max_ngram,), dtype=np.int64)
        mult.append((v * 2 + 1).tolist())
    pad_id = lookup[cfg["engram_pad_token_id"]]
    json.dump(dict(compressed_vocab_size=cv, token_map=lookup, layer_ids=layer_ids, max_ngram_size=max_ngram,
                   n_heads=n_heads, head_dim=cfg["engram_head_dim"], primes=primes, offsets=offsets,
                   multipliers=mult, pad_compressed_id=pad_id, num_embeddings=list(cfg["engram_num_embeddings"])),
              open(out, "w"))
    print(f"{out}: vocab {n_vocab} -> compressed {cv}, layers {layer_ids}, rows {cfg['engram_num_embeddings']}, "
          f"pad -> {pad_id}")


if __name__ == "__main__":
    main()
