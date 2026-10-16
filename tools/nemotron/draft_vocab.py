#!/usr/bin/env python3
"""Generate src/families/nemotron_h/nemotron_h_draft_vocab.inc: the Nemotron-H MTP draft head's ids (L296 draft-cost).

The drafter's head (nemotron_h_forward.cpp, the draft head) scores a SUBSET of the 131,072-id vocabulary: the named added
tokens (the chat / reasoning / tool-call markers, ids 0..29 -- the <SPECIAL_n> placeholders are left out) and the most
frequent ids of the ranked list below.  Drafts are proposals the target verifies, so the list moves only how often a
token CAN be drafted, never the output.  The tokenizer's id order is not a frequency order (a merge rank over many
scripts: a prefix of 16,384 ids covers 80 % of the corpus below, a ranked 16,384 98.7 %), so the subset is a list, not
a range.

The ranking counts the tokens of the corpus tools/qwen/draft_vocab.py ranks Qwen's with (the CPython standard
library's *.py, the upstream clones' Markdown and sources, 28 Project Gutenberg books): technical English, lists, code
in many languages, general prose -- what a chat assistant writes.  Every non-added id seen at least --min-count times is
ranked, most frequent first (ties by id).  The list is written FIXED ids first, then the ranked ids; the engine keeps a
PREFIX of it (NEMOTRON_DRAFT_ROWS_DEFAULT rows, the sweep's knob PULSAR_NEMOTRON_MTP_DRAFT_ROWS) and sorts it ascending.

The header carries the FNV-1a 64 digest of every listed id's raw token bytes (each id's bytes, then a 0xff separator)
under the tokenizer: the engine recomputes it from the checkpoint's tokenizer at load and refuses a mismatch (a list
made for another tokenizer would rank the wrong tokens).

  U=~/Projects/AI/upstream
  python3 tools/nemotron/draft_vocab.py TOKENIZER_JSON src/families/nemotron_h/nemotron_h_draft_vocab.inc \
      --default 20480 --max 49152 '/usr/lib/python3.13/**/*.py' "$U/**/*.md" "$U/**/*.py" "$U/**/*.cpp" "$U/**/*.h" \
      "$U/**/*.rs" "$U/**/*.ts" "$U/**/*.js" "$U/**/*.sh" "$U/**/*.c" "$U/**/*.cu" 'GUTENBERG_DIR/*.txt'
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
