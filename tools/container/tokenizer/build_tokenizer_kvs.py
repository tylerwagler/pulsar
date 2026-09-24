"""The tokenizer.* entries of the container's pulsar.kv block, from the HF checkpoint.

Nine of the eleven entries derive mechanically from tokenizer.json +
tokenizer_config.json.  THREE DO NOT, and pretending otherwise is how this stays
broken, so they live here as explicit, documented repo-side constants.  They are
OUR decisions, not borrowed artifact bytes -- which is exactly why checking them
in breaks the circularity instead of relocating it:

  1. PRE_TOKENIZER_ID -- a llama.cpp-side pretokenizer identifier.  Not present
     in, and not derivable from, any HF file.
  2. chat_template.jinja -- the checkpoint ships its chat format as PYTHON
     (encoding/encoding_dsv4.py), not Jinja.  Our template is a translation of
     it, so it is a source artifact in its own right.
  3. USER_DEFINED_TOKENS -- see below.  Hand-curated; the HF `special` flag does
     NOT reproduce it (1230/53 there vs the required 1277/6).

The engine reads `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` only
(src/engine/tokenizer.cpp, vocab_load); special ids are resolved by string and
the chat template is the engine's own renderer.  The other nine entries are
carried for the record, in the served block's order.

`build` returns (key, type, value) triples in the engine's type spelling; array
values are (elem_type, list) and kv.py is the one place that turns a triple
into the JSON entry.  tools/container/test_kv.py grades the whole block
entry-for-entry against the served artifact.
"""
import json
import os

# (1) llama.cpp pretokenizer identifier.  A classification, not checkpoint data.
PRE_TOKENIZER_ID = "joyai-llm"

# GGUF token-type codes, kept because the served block carries them.
TT_NORMAL, TT_CONTROL, TT_USER_DEFINED = 1, 3, 4

# (3) Added tokens that must be USER_DEFINED rather than CONTROL.  CONTROL tokens
# are skipped during detokenization; these six have to survive round-trip as
# literal text because the engine's own parsers consume them -- the reasoning
# split reads <think>/</think>, and the DSML tool grammar reads the dsml markers.
# Marking them CONTROL yields a model whose reasoning and tool-call output are
# silently swallowed.  Every other added token (including special=False ones like
# <|fim_hole|>) is CONTROL.
USER_DEFINED_TOKENS = frozenset({
    "<think>", "</think>", "｜DSML｜", "<dsml:", "</dsml:", "<｜/table>｜",
})

DEFAULT_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_template.jinja")


def build(hf_dir, template_path=DEFAULT_TEMPLATE):
    """The eleven tokenizer.* triples, in the served block's order."""
    with open(os.path.join(hf_dir, "tokenizer.json"), encoding="utf-8") as f:
        tok = json.load(f)
    with open(os.path.join(hf_dir, "tokenizer_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

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

    with open(template_path, "rb") as f:
        chat_template = f.read().decode("utf-8")

    return [
        ("tokenizer.ggml.model", "string", "gpt2"),
        ("tokenizer.ggml.pre", "string", PRE_TOKENIZER_ID),
        ("tokenizer.ggml.tokens", "array", ("string", tokens)),
        ("tokenizer.ggml.token_type", "array", ("i32", token_types)),
        ("tokenizer.ggml.merges", "array", ("string", merges)),
        ("tokenizer.ggml.bos_token_id", "u32", tok_id("bos_token")),
        ("tokenizer.ggml.eos_token_id", "u32", tok_id("eos_token")),
        ("tokenizer.ggml.padding_token_id", "u32", tok_id("pad_token")),
        ("tokenizer.ggml.add_bos_token", "bool", bool(cfg.get("add_bos_token", False))),
        ("tokenizer.ggml.add_eos_token", "bool", bool(cfg.get("add_eos_token", False))),
        ("tokenizer.chat_template", "string", chat_template),
    ]
