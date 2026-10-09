"""The tokenizer.* entries of the container's pulsar.kv block, from the checkpoint's own tokenizer files -- ONE core
for every family (L279 step 6); a family supplies only its Settings.

The walk is mechanical and shared: tokenizer.json's BPE vocab by id with the added tokens overlaid (an added token
overrides its vocab id), refusing holes in the id space; the merges as "a b" strings; the special ids by content
(added tokens first, then the vocab; a tokenizer_config entry may be a string or {"content": ...}); add_bos /
add_eos from tokenizer_config.  What differs is a family's decision, so it is a setting, never a second copy of
the walk:

  added_type          an added token's GGUF type (record-only; the engine reads tokens + merges): DeepSeek marks six
                      tokens its parsers read USER_DEFINED and the rest CONTROL (kv.py); qwen4_exp marks HF-special
                      CONTROL and the rest USER_DEFINED
  template            the chat template: a repo-side file (DeepSeek ships its format as Python; kv.py) or None for
                      the checkpoint's own chat_template.jinja (qwen4_exp)
  pre                 tokenizer.ggml.pre, a llama.cpp pretokenizer id (DeepSeek's), or None
  pretokenize_regex   carry tokenizer_config's pretokenize_regex (qwen4_exp)
  bos_optional        False: bos is required and rides after the merges (DeepSeek); True: written last, and only
                      when the config names one (qwen4_exp, which has none)

`kvs` returns (key, type, value) triples in the engine's type spelling (arrays as (elem_type, list)); kv.py is the
one place a triple becomes the JSON entry.  The entry ORDER is each family's served order (the bytes depend on it).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

# GGUF token-type codes, kept because the served blocks carry them.
TT_NORMAL, TT_CONTROL, TT_USER_DEFINED = 1, 3, 4


@dataclass(frozen=True)
class Settings:
    added_type: Callable[[dict], int]
    template: str | None = None
    pre: str | None = None
    pretokenize_regex: bool = False
    bos_optional: bool = False


def kvs(tok_dir, st: Settings):
    with open(os.path.join(tok_dir, "tokenizer.json"), encoding="utf-8") as f:
        tok = json.load(f)
    with open(os.path.join(tok_dir, "tokenizer_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if tok["model"]["type"] != "BPE":
        raise SystemExit(f"tokenizer model {tok['model']['type']}: BPE expected")

    vocab = tok["model"]["vocab"]                 # token -> id
    added = tok["added_tokens"]                   # explicit ids, may overlay vocab
    by_id = {i: t for t, i in vocab.items()}
    types = {i: TT_NORMAL for i in by_id}
    for a in added:                               # added_tokens OVERRIDE vocab ids
        by_id[a["id"]] = a["content"]
        types[a["id"]] = st.added_type(a)
    n = max(by_id) + 1
    missing = [i for i in range(n) if i not in by_id]
    if missing:
        raise SystemExit("tokenizer id space has %d holes (first: %d)" % (len(missing), missing[0]))
    merges = tok["model"]["merges"]
    if merges and not isinstance(merges[0], str):
        merges = [" ".join(m) for m in merges]

    def tok_id(name):
        t = cfg.get(name)
        if isinstance(t, dict):
            t = t.get("content")
        if t is None:
            raise SystemExit("tokenizer_config.json has no %s" % name)
        for a in added:
            if a["content"] == t:
                return a["id"]
        if t in vocab:
            return vocab[t]
        raise SystemExit("%s %r is not in the vocabulary" % (name, t))

    template = st.template or os.path.join(tok_dir, "chat_template.jinja")
    with open(template, "rb") as f:
        chat_template = f.read().decode("utf-8")

    out = [("tokenizer.ggml.model", "string", "gpt2")]
    if st.pre is not None:
        out.append(("tokenizer.ggml.pre", "string", st.pre))
    out += [
        ("tokenizer.ggml.tokens", "array", ("string", [by_id[i] for i in range(n)])),
        ("tokenizer.ggml.token_type", "array", ("i32", [types[i] for i in range(n)])),
        ("tokenizer.ggml.merges", "array", ("string", merges)),
    ]
    if st.pretokenize_regex:
        out.append(("tokenizer.pretokenize_regex", "string", cfg["pretokenize_regex"]))
    if not st.bos_optional:
        out.append(("tokenizer.ggml.bos_token_id", "u32", tok_id("bos_token")))
    out += [
        ("tokenizer.ggml.eos_token_id", "u32", tok_id("eos_token")),
        ("tokenizer.ggml.padding_token_id", "u32", tok_id("pad_token")),
        ("tokenizer.ggml.add_bos_token", "bool", bool(cfg.get("add_bos_token", False))),
        ("tokenizer.ggml.add_eos_token", "bool", bool(cfg.get("add_eos_token", False))),
        ("tokenizer.chat_template", "string", chat_template),
    ]
    if st.bos_optional and cfg.get("bos_token") is not None:
        out.append(("tokenizer.ggml.bos_token_id", "u32", tok_id("bos_token")))
    return out
