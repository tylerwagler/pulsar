#!/usr/bin/env python3
"""Generate src/engine/qwen_draft_vocab.inc: the Qwen MTP draft head's frequency-ranked ids beyond the 64 Ki prefix.

The draft head (family_qwen.cpp qwen_draft_vocab, L251 / L288) scores a subset of the vocabulary: every id below 65,536, the added /
special ids, and -- from this list -- the most frequent ids above the prefix.  Drafts are proposals the target
verifies, so the list moves only how often a token CAN be drafted, never the output.  An id outside the head is
undraftable: on dev 2.1 % of the generated tokens of the L288 matrix were (```` ``` ````, ' doubly', '**.' ...).

The ranking counts the tokens of public text: the CPython standard library's *.py, the upstream clones' Markdown
and sources (*.py *.md *.c *.h *.cpp *.cu *.rs *.ts *.js *.sh: technical English, lists, code in many languages) and
28 Project Gutenberg books (general English prose) -- what a chat assistant writes.  Only ids >= 65,536 that are not
added tokens are ranked; ids seen fewer than --min-count times are left out.  The list is written most frequent
first, so the engine keeps a PREFIX of it (PULSAR_QWEN_MTP_DRAFT_EXTRA, default QWEN_DRAFT_EXTRA_DEFAULT).  On
held-out replies (24 chat prompts disjoint from serve_matrix's, 8,881 tokens) the undraftable share is 3.04 % with
the prefix alone, 0.50 % with 16,384 ranked ids (TensorFold's 79,591-id list: 0.90 %).

The header also carries the FNV-1a 64 digest of the listed ids' raw token bytes (each id's bytes, then a 0xff
separator) under the tokenizer: the engine recomputes it from the checkpoint's tokenizer at load and refuses a
mismatch (a list made for another tokenizer would rank the wrong tokens).

  U=~/Projects/AI/upstream
  python3 tools/qwen/draft_vocab.py TOKENIZER_JSON src/engine/qwen_draft_vocab.inc --default 16384 --max 32768 \
      '/usr/lib/python3.13/**/*.py' "$U/**/*.md" "$U/**/*.py" "$U/**/*.cpp" "$U/**/*.h" "$U/**/*.rs" "$U/**/*.ts" \
      "$U/**/*.js" "$U/**/*.sh" "$U/**/*.c" "$U/**/*.cu" 'GUTENBERG_DIR/*.txt'
  (Gutenberg ebook ids 1342 84 1661 98 11 345 174 2600 1400 768 1260 5200 4300 76 1952 16328 205 3207 1497 2554
   6130 1080 25344 158 36 46 219 1232, https://www.gutenberg.org/cache/epub/ID/pgID.txt)
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

from tokenizers import Tokenizer

PREFIX = 65536


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
    p.add_argument("--max", type=int, default=32768, help="ids written (the engine keeps a prefix)")
    p.add_argument("--default", type=int, required=True, help="the engine's default prefix (QWEN_DRAFT_EXTRA_DEFAULT)")
    p.add_argument("--min-count", type=int, default=4)
    p.add_argument("--max-bytes", type=int, default=2_000_000, help="skip files larger than this")
    a = p.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    added = {t["id"] for t in json.load(open(a.tokenizer))["added_tokens"]}
    counts: collections.Counter = collections.Counter()
    files = sorted({f for pat in a.patterns for f in glob.glob(os.path.expanduser(pat), recursive=True)})
    used = total = 0
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
        ids = tok.encode(text, add_special_tokens=False).ids
        counts.update(ids)
        total += len(ids)
        used += 1
    ranked = [t for t, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
              if t >= PREFIX and t not in added and c >= a.min_count][: a.max]
    if a.default > len(ranked):
        raise SystemExit(f"--default {a.default} exceeds the {len(ranked)} ranked ids")
    dec = byte_decoder()
    h = 1469598103934665603
    for t in ranked:
        for b in bytes(dec[ch] for ch in tok.id_to_token(t)) + b"\xff":
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    below = sum(c for t, c in counts.items() if t < PREFIX or t in added)

    def cover(k: int) -> float:
        keep = set(ranked[:k])
        return (below + sum(counts[t] for t in keep)) / max(1, total)

    stats = {"files": used, "tokens": total, "ranked": len(ranked), "coverage_prefix": round(cover(0), 5),
             "coverage_default": round(cover(a.default), 5), "coverage_max": round(cover(len(ranked)), 5)}
    with open(a.out, "w") as fh:
        fh.write("/* qwen_draft_vocab.inc -- GENERATED by tools/qwen/draft_vocab.py; do not edit.  The Qwen MTP draft\n"
                 " * head's ids beyond [0, 65536), most frequent first (family_qwen.cpp qwen_draft_vocab keeps a prefix).\n"
                 f" * Corpus: {used} files, {total} tokens; coverage of the corpus's tokens: prefix + specials\n"
                 f" * {stats['coverage_prefix']:.4%}, + the default {a.default} {stats['coverage_default']:.4%}, "
                 f"+ all {len(ranked)} {stats['coverage_max']:.4%}. */\n")
        fh.write(f"#define QWEN_DRAFT_PREFIX {PREFIX}u\n")
        fh.write(f"#define QWEN_DRAFT_EXTRA_DEFAULT {a.default}u\n")
        fh.write(f"#define QWEN_DRAFT_VOCAB_DIGEST 0x{h:016x}ull\n")
        fh.write("static const int32_t k_qwen_draft_extra[] = {\n")
        for i in range(0, len(ranked), 12):
            fh.write("    " + ", ".join(str(t) for t in ranked[i:i + 12]) + ",\n")
        fh.write("};\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
