#!/usr/bin/env python3
"""Build the GGUF tokenizer.* KV block from the HF checkpoint.

Removes build_main_template.py's --splice-tokenizer-from bootstrap: that flag
copies tokenizer KVs verbatim out of a previously-good GGUF, which means a
"rebuild from source weights" has never actually been possible -- you always
needed one earlier artifact to seed the next.

Nine of the eleven KVs derive mechanically from tokenizer.json +
tokenizer_config.json. THREE DO NOT, and pretending otherwise is how this stays
broken, so they live here as explicit, documented repo-side constants. They are
OUR decisions, not borrowed artifact bytes -- which is exactly why checking them
in breaks the circularity instead of relocating it:

  1. PRE_TOKENIZER_ID -- a llama.cpp-side pretokenizer identifier. Not present
     in, and not derivable from, any HF file.
  2. chat_template.jinja -- the checkpoint ships its chat format as PYTHON
     (encoding/encoding_dsv4.py), not Jinja. Our template is a translation of
     it, so it is a source artifact in its own right.
  3. USER_DEFINED_TOKENS -- see below. Hand-curated; the HF `special` flag does
     NOT reproduce it (1230/53 there vs the required 1277/6).

Verify with --verify-against, which byte-compares every emitted KV against the
blobs dumped from a known-good artifact. That check is the whole point: a
tokenizer that is *nearly* right produces an artifact that loads and then
mis-tokenizes, which no other gate in this repo would catch.

  build_tokenizer_kvs.py --hf DIR --out KVDIR [--verify-against REFDIR]
"""
import argparse
import json
import os
import struct
import sys

# GGUF value type ids
T_UINT32, T_BOOL, T_STRING, T_ARRAY, T_INT32 = 4, 7, 8, 9, 5
# GGUF token types
TT_NORMAL, TT_CONTROL, TT_USER_DEFINED = 1, 3, 4

# (1) llama.cpp pretokenizer identifier. A classification, not checkpoint data.
PRE_TOKENIZER_ID = "joyai-llm"

# (3) Added tokens that must be USER_DEFINED rather than CONTROL. CONTROL tokens
# are skipped during detokenization; these six have to survive round-trip as
# literal text because the engine's own parsers consume them -- the reasoning
# split reads <think>/</think>, and the DSML tool grammar reads the dsml markers.
# Marking them CONTROL yields a model whose reasoning and tool-call output are
# silently swallowed. Every other added token (including special=False ones like
# <|fim_hole|>) is CONTROL.
USER_DEFINED_TOKENS = frozenset({
    "<think>", "</think>", "｜DSML｜", "<dsml:", "</dsml:", "<｜/table>｜",
})


def w_str(b):
    if isinstance(b, str):
        b = b.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv(key, type_id, payload):
    return w_str(key) + struct.pack("<I", type_id) + payload


def kv_string(key, s):
    return kv(key, T_STRING, w_str(s))


def kv_u32(key, v):
    return kv(key, T_UINT32, struct.pack("<I", v))


def kv_bool(key, v):
    return kv(key, T_BOOL, struct.pack("<B", 1 if v else 0))


def kv_arr_str(key, items):
    body = struct.pack("<I", T_STRING) + struct.pack("<Q", len(items))
    return kv(key, T_ARRAY, body + b"".join(w_str(x) for x in items))


def kv_arr_i32(key, items):
    body = struct.pack("<I", T_INT32) + struct.pack("<Q", len(items))
    return kv(key, T_ARRAY, body + b"".join(struct.pack("<i", x) for x in items))


def build(hf_dir, template_path):
    tok = json.load(open(os.path.join(hf_dir, "tokenizer.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(hf_dir, "tokenizer_config.json"), encoding="utf-8"))

    vocab = tok["model"]["vocab"]                 # token -> id
    added = tok["added_tokens"]                   # explicit ids, may overlay vocab
    merges = tok["model"]["merges"]               # already "a b" space-joined

    by_id = {i: t for t, i in vocab.items()}
    types = {i: TT_NORMAL for i in by_id}
    for a in added:                               # added_tokens OVERRIDE vocab ids
        by_id[a["id"]] = a["content"]
        types[a["id"]] = (TT_USER_DEFINED if a["content"] in USER_DEFINED_TOKENS
                          else TT_CONTROL)

    n = max(by_id) + 1
    missing = [i for i in range(n) if i not in by_id]
    if missing:
        raise SystemExit("tokenizer id space has %d holes (first: %d)" % (len(missing), missing[0]))

    tokens = [by_id[i] for i in range(n)]
    token_types = [types[i] for i in range(n)]

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

    chat_template = open(template_path, "rb").read()

    return {
        "tokenizer.ggml.model":           kv_string("tokenizer.ggml.model", "gpt2"),
        "tokenizer.ggml.pre":             kv_string("tokenizer.ggml.pre", PRE_TOKENIZER_ID),
        "tokenizer.ggml.tokens":          kv_arr_str("tokenizer.ggml.tokens", tokens),
        "tokenizer.ggml.token_type":      kv_arr_i32("tokenizer.ggml.token_type", token_types),
        "tokenizer.ggml.merges":          kv_arr_str("tokenizer.ggml.merges", merges),
        "tokenizer.ggml.bos_token_id":    kv_u32("tokenizer.ggml.bos_token_id", tok_id("bos_token")),
        "tokenizer.ggml.eos_token_id":    kv_u32("tokenizer.ggml.eos_token_id", tok_id("eos_token")),
        "tokenizer.ggml.padding_token_id": kv_u32("tokenizer.ggml.padding_token_id", tok_id("pad_token")),
        "tokenizer.ggml.add_bos_token":   kv_bool("tokenizer.ggml.add_bos_token", cfg.get("add_bos_token", False)),
        "tokenizer.ggml.add_eos_token":   kv_bool("tokenizer.ggml.add_eos_token", cfg.get("add_eos_token", False)),
        "tokenizer.chat_template":        kv("tokenizer.chat_template", T_STRING, w_str(chat_template)),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="HF checkpoint directory")
    ap.add_argument("--out", help="directory to write one .kv blob per entry")
    ap.add_argument("--template", default=os.path.join(here, "chat_template.jinja"))
    ap.add_argument("--verify-against", metavar="REFDIR",
                    help="directory of reference .kv blobs; byte-compare and exit nonzero on any mismatch")
    a = ap.parse_args()

    kvs = build(a.hf, a.template)

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for key, blob in kvs.items():
            open(os.path.join(a.out, key.replace(".", "_") + ".kv"), "wb").write(blob)
        print("wrote %d KV blobs -> %s" % (len(kvs), a.out))

    if a.verify_against:
        bad = 0
        for key, blob in sorted(kvs.items()):
            p = os.path.join(a.verify_against, key.replace(".", "_") + ".kv")
            if not os.path.exists(p):
                print("  MISSING REF  %s" % key); bad += 1; continue
            ref = open(p, "rb").read()
            if ref == blob:
                print("  ok    %-40s %9d B" % (key, len(blob)))
            else:
                print("  MISMATCH %-37s got %d B, ref %d B" % (key, len(blob), len(ref)))
                bad += 1
        print()
        if bad:
            print("tokenizer KV verify: FAIL (%d of %d differ)" % (bad, len(kvs)), file=sys.stderr)
            return 1
        print("tokenizer KV verify: PASS -- all %d byte-identical to the reference" % len(kvs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
