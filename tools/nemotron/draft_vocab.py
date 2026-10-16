#!/usr/bin/env python3
"""Generate src/families/nemotron_h/nemotron_h_draft_vocab.inc: the Nemotron-H MTP draft head's ids (L296 draft-cost).

The drafter's head (nemotron_h_forward.cpp, the draft head) scores a SUBSET of the 131,072-id vocabulary: the named added
tokens (the chat / reasoning / tool-call markers, ids 0..29 -- the <SPECIAL_n> placeholders are left out) and the most
frequent ids of the ranked list below.  Drafts are proposals the target verifies, so the list moves only how often a
token CAN be drafted, never the output.  The tokenizer's id order is not a frequency order (a merge rank over many
scripts: a prefix of 16,384 ids covers 80 % of the corpus below, a ranked 16,384 98.7 %), so the subset is a list, not
a range.

The ranking counts the tokens of the corpus tools/qwen/draft_vocab.py ranks Qwen's with (the CPython standard
library's *.py, the upstream clones' Markdown and sources, 28 Project Gutenberg books) AND the assistant side of four
dataset shards (a *.parquet pattern: ultrachat_200k's assistant turns, hermes_reasoning_tool_use's reasoning + tool
calls, OpenCodeInstruct's answers, wikitext-103).  The first alone missed modern expository English: this tokenizer
spells many whole words in one id (" technology", " maritime"), and on the model's own served replies a 20,480-id head
from it left 25 % of a prose reply's tokens undraftable (tokens / round 2.90 -> 2.29); with the datasets a 40,960-id head
leaves < 1 % of held-out ultrachat replies undraftable.  Every non-added id seen at least --min-count times is ranked,
most frequent first (ties by id).  The list is written FIXED ids first, then the ranked ids; the engine keeps a PREFIX
of it (NEMOTRON_DRAFT_ROWS_DEFAULT rows, the sweep's knob PULSAR_NEMOTRON_MTP_DRAFT_ROWS) and sorts it ascending.

The header carries the FNV-1a 64 digest of every listed id's raw token bytes (each id's bytes, then a 0xff separator)
under the tokenizer: the engine recomputes it from the checkpoint's tokenizer at load and refuses a mismatch (a list
made for another tokenizer would rank the wrong tokens).

  U=~/Projects/AI/upstream; R=/mnt/models/reap-corpus/raw   (tokenizers + pyarrow)
  python3 tools/nemotron/draft_vocab.py TOKENIZER_JSON src/families/nemotron_h/nemotron_h_draft_vocab.inc \
      --default 40960 --max 65536 '/usr/lib/python3.13/**/*.py' "$U/**/*.md" "$U/**/*.py" "$U/**/*.cpp" "$U/**/*.h" \
      "$U/**/*.rs" "$U/**/*.ts" "$U/**/*.js" "$U/**/*.sh" "$U/**/*.c" "$U/**/*.cu" 'GUTENBERG_DIR/*.txt' \
      $R/ultrachat_200k/data/train_sft-00000-of-00003-*.parquet $R/hermes_reasoning_tool_use/data/train-00000-of-00001.parquet \
      $R/OpenCodeInstruct/data/train-00000-of-00050.parquet $R/wikitext/wikitext-103-raw-v1/train-00000-of-00002.parquet
  (the Gutenberg ids of tools/qwen/draft_vocab.py)
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

from tokenizers import Tokenizer

PLACEHOLDER = re.compile(r"<SPECIAL_\d+>")


def byte_decoder() -> dict[str, int]:
    """GPT-2 byte-level BPE: the printable stand-in of each byte, reversed."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def parquet_texts(path: str):
    """A dataset shard's ASSISTANT-side text, by its schema: chat `messages` (role assistant: ultrachat_200k),
    `conversations` (from gpt: hermes_reasoning_tool_use), an `output` column (OpenCodeInstruct), a `text` column
    (wikitext)."""
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    cols = t.column_names
    if "messages" in cols:
        for conv in t.column("messages").to_pylist():
            yield from (m["content"] for m in conv if m["role"] == "assistant" and m["content"])
    elif "conversations" in cols:
        for conv in t.column("conversations").to_pylist():
            yield from (m["value"] for m in conv if m["from"] == "gpt" and m["value"])
    elif "output" in cols:
        yield from (x for x in t.column("output").to_pylist() if x)
    elif "text" in cols:
        yield from (x for x in t.column("text").to_pylist() if x)
    else:
        raise SystemExit(f"{path}: no messages / conversations / output / text column")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer")
    p.add_argument("out")
    p.add_argument("patterns", nargs="+")
    p.add_argument("--max", type=int, default=49152, help="rows listed (fixed + ranked; the engine keeps a prefix)")
    p.add_argument("--default", type=int, required=True, help="the engine's default rows (NEMOTRON_DRAFT_ROWS_DEFAULT)")
    p.add_argument("--min-count", type=int, default=4)
    p.add_argument("--max-bytes", type=int, default=2_000_000, help="skip files larger than this")
    a = p.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    added = {t["id"]: t["content"] for t in json.load(open(a.tokenizer))["added_tokens"]}
    fixed = sorted(i for i, c in added.items() if not PLACEHOLDER.fullmatch(c))
    counts: collections.Counter = collections.Counter()
    files = sorted({f for pat in a.patterns for f in glob.glob(os.path.expanduser(pat), recursive=True)})
    used = total = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal total
        for enc in tok.encode_batch(batch, add_special_tokens=False):
            counts.update(enc.ids)
            total += len(enc.ids)
        batch.clear()

    for path in files:
        if path.endswith(".parquet"):
            for text in parquet_texts(path):
                batch.append(text)
                if len(batch) >= 4096:
                    flush()
            used += 1
            continue
        try:
            if os.path.getsize(path) > a.max_bytes:
                continue
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        if not text:
            continue
        batch.append(text)
        used += 1
        if len(batch) >= 256:
            flush()
    flush()
    ranked = [t for t, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
              if t not in added and c >= a.min_count][: a.max - len(fixed)]
    rows = fixed + ranked
    if not len(fixed) <= a.default <= len(rows):
        raise SystemExit(f"--default {a.default} is outside [{len(fixed)}, {len(rows)}]")
    dec = byte_decoder()
    h = 1469598103934665603
    for t in rows:
        raw = added[t].encode() if t in added else bytes(dec[ch] for ch in tok.id_to_token(t))
        for b in raw + b"\xff":
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    def cover(k: int) -> float:
        return sum(counts[t] for t in rows[:k]) / max(1, total)

    stats = {"files": used, "tokens": total, "fixed": len(fixed), "rows": len(rows),
             "coverage_default": round(cover(a.default), 5), "coverage_max": round(cover(len(rows)), 5)}
    with open(a.out, "w") as fh:
        fh.write("/* nemotron_h_draft_vocab.inc -- GENERATED by tools/nemotron/draft_vocab.py; do not edit.  The Nemotron-H\n"
                 " * MTP draft head's ids: the named added tokens, then the ranked ids, most frequent first\n"
                 " * (nemotron_h_family.cpp draft_vocab keeps a prefix).\n"
                 f" * Corpus: {used} files, {total} tokens; coverage of the corpus's tokens: the default {a.default} "
                 f"rows {stats['coverage_default']:.4%},\n"
                 f" * all {len(rows)} {stats['coverage_max']:.4%}. */\n")
        fh.write(f"#define NEMOTRON_DRAFT_FIXED {len(fixed)}u\n")
        fh.write(f"#define NEMOTRON_DRAFT_ROWS_DEFAULT {a.default}u\n")
        fh.write(f"#define NEMOTRON_DRAFT_VOCAB_DIGEST 0x{h:016x}ull\n")
        fh.write("static const int32_t k_nemotron_draft_ids[] = {\n")
        for i in range(0, len(rows), 12):
            fh.write("    " + ", ".join(str(t) for t in rows[i:i + 12]) + ",\n")
        fh.write("};\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
