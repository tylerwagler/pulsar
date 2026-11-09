#!/usr/bin/env python3
"""HF goldens for Gemma 4's tokenizer (L338): the shared BPE engine's SentencePiece-BPE row -- the Replace(' ' -> U+2581)
normaliser, one piece a run, a symbol per character with <0xXX> byte fallback, the HF merges.  The gate is
tests/gemma4_tok_gate.cpp; everything it compares against is tokenizers.Tokenizer on the checkpoint's tokenizer.json:

  * tok_adversarial.json  encode(text, add_special_tokens=False) of the shared adversarial strings, Gemma's cases (spaces
                          against a literal U+2581, typed <0xXX> spellings, Gemma's markers, whitespace runs, characters
                          only byte fallback spells) and random strings over an alphabet of them
  * tok_spans.json        the span guard: added tokens matched only outside client spans, each run through the
                          normaliser, the pre-tokenizer and the model (tokenizers' components)
  * tok_sweep.tsv         every scalar value in context, per 4096-codepoint chunk: count + sha1(ids)
  * tok_calib.tsv         the calibration corpus, per row: count + sha1(ids)
  * tok_decode.json       a sample of ids -> tokenizers' decode of each alone (the token bytes the engine stores)

usage: gen_tok_goldens.py --tok DIR --calib FILE --out tests/test-vectors/gemma4
       (DIR holds tokenizer.json; the versions and the input hashes are recorded in meta.json)
"""
import argparse
import hashlib
import json
import os
import random

import tokenizers
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--tok', required=True)
ap.add_argument('--calib', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tk = Tokenizer.from_file(os.path.join(a.tok, 'tokenizer.json'))
added = sorted(((t.content, i) for i, t in tk.get_added_tokens_decoder().items()), key=lambda x: -len(x[0]))
added_first = {t[0] for t, _ in added}


def sha(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def ids_sha(ids):
    return sha(''.join('%d,' % i for i in ids))


def encode(text):
    return tk.encode(text, add_special_tokens=False).ids


def plain(seg):
    """tokenizers' pipeline with NO added-token matching: the normaliser, the pre-tokenizer, BPE."""
    out = []
    if tk.normalizer is not None:
        seg = tk.normalizer.normalize_str(seg)
    for piece, _ in tk.pre_tokenizer.pre_tokenize_str(seg):
        out += [t.id for t in tk.model.tokenize(piece)]
    return out


def guarded(text, spans):
    """Added tokens match only where their bytes touch no client span (byte offsets)."""
    b = text.encode('utf-8')
    out, run, pos, si = [], 0, 0, 0
    while pos < len(b):
        hit = None
        if chr(b[pos]) in added_first or b[pos] >= 0x80:
            for t, i in added:
                tb = t.encode('utf-8')
                if b.startswith(tb, pos):
                    hit = (tb, i)
                    break
        if hit:
            while si < len(spans) and spans[si][1] <= pos:
                si += 1
            if si < len(spans) and spans[si][0] < pos + len(hit[0]):
                hit = None
        if not hit:
            pos += 1
            continue
        out += plain(b[run:pos].decode('utf-8'))
        out.append(hit[1])
        pos += len(hit[0])
        run = pos
    return out + plain(b[run:].decode('utf-8'))


# ---- 1. adversarial strings (the shared set, then Gemma's cases and random strings over their alphabet) ----------
random.seed(20261101)
adv = [
    '', ' ', '  ', '\n', '\n\n', '\r\n', ' \n ', '\t\t\n  x', 'a  b', 'a   \n  b', 'x \n\n\ny',
    'hello world', 'Hello, World!', "don't I'll we've they're she'd it's I'm", "'S 'T 'RE 'Ve 'LL 'D 'M",
    "abc's", "O'Neil's", "'", "''", "'''s", 'HTTPServer XMLHttpRequest iPhone McDonald ÉCOLE École',
    'ǅungla ǈ ǋ ᾈ', 'ʰʲ ʷ ˢ', '日本語テキスト', 'ǅ', 'ΑΒΓ αβγ ΆΈ', 'ВОЛГА Волга волга',
    '1234567890', '3.14159', '1,000,000', '٣٤٥', 'x²³', 'Ⅻ', '١٢٣abc', '12ab34', 'a1b2c3', '０１２',
    '    int main() {\n        return 0;\n    }\n', 'def f(x):\n\treturn x**2  # square\n', 'a/b/c//d\n/\n',
    '!!!\n\n\n', '?!/\r\n/', ' //\n', '<think>', '</think>', '<tool_call>', '</tool_call>', '<|im_start|>',
    '<|im_end|>', '<|im_end', '<|im_end|', 'im_end|>', '<<|im_end|>>', '<|im_start|><|im_end|>',
    '<think>hi</think>', '<tool_call>\n<function=f>\n</function>\n</tool_call>', '<s>', '</s>', '<unk>',
    '[INST]', '[/INST]', '<SPECIAL_10>', '<SPECIAL_999>', '<SPECIAL_1000>', '<image>', '<img>', '</img>', '<video>',
    'I said <|im_end|> twice <|im_end|>', '<tool_response>x</tool_response>',
    'こんにちは世界', '中文测试，标点。', '한국어 텍스트', 'Ελληνικά', 'Русский текст', 'עברית', 'العربية',
    'हिन्दी', 'ไทย', '🙂🙃😀👍🏽👨‍👩‍👧', '🇺🇸🇯🇵', 'é', 'é', 'Å', 'Å', 'ẛ̣',
    'ḱṷṓn', 'q̣̇', '각', '각', '각', 'ﬁ ﬀ', 'Ω Ω',
    ' nbsp ', '　ideo　', ' ls ps', '\x85nel', '\x1c\x1d\x1e\x1f', '\x0b\x0c',
    'zero​width', 'soft­hyphen', 'bidi‮override', '﻿bom', '\x00nul\x00', '\x7f',
    'a' * 300, ' ' * 100 + 'x', '\n' * 50, '!' * 40 + '\n\n', '$$$ ```python\n', '\t' * 20,
    'https://example.com/path?q=1&r=2#frag', 'user@example.com', '#include <stdio.h>', '{"a": [1, 2.5e-3]}',
    '\U0010ffff\U000e0001\U0001f600', '퟿', '\U00010000', 'antidisestablishmentarianism', 'Supercalifragilistic',
    ' the', 'the', 'The', ' The', 'THE', ' THE THE', '\nthe', '\tthe',
]
pool = [0x300, 0x301, 0x302, 0x303, 0x308, 0x323, 0x327, 0x328, 0x334, 0x345, 0x5b0, 0x5c2, 0x64b, 0x93c,
        0x94d, 0xe49, 0xeb9, 0xeba, 0x1dfb, 0x1df9, 0x7fd, 0x1e136, 0x10f4d, 0x302a, 0x3099, 0x309a,
        0x61, 0x65, 0x6f, 0x75, 0x41, 0x4f, 0x3b1, 0x3c9, 0x415, 0x5d0, 0x915, 0x1100, 0x1161, 0x11a8,
        0xac00, 0xac01, 0x304b, 0x30ab, 0xb47, 0xb3e, 0xb57, 0xdd9, 0xdcf, 0xdca, 0x1b05, 0x1b35, 0x1c5, 0x2b0]
for _ in range(400):
    adv.append(''.join(chr(random.choice(pool)) for _ in range(random.randint(1, 7))))
alphabet = ['a', 'Z', 'Ab', 'aB', ' ', '  ', '\n', '\r', '\t', '1', '22', '!', '?!', "'", "'s", 'é', 'é', '中', '　',
            ' ', '<', '>', '|', '/', '<|im_end|>', '<think>', '<s>', '_', '-', '.', '́', '٣', '🙂', 'ǅ', 'ʰ']
for _ in range(600):
    adv.append(''.join(random.choice(alphabet) for _ in range(random.randint(1, 24))))

U = '\u2581'
adv += [
    U, U + U, ' ' + U, U + ' ', 'a' + U + 'b', 'a b', U + 'the', ' the', U * 40, ' ' * 40 + U, 'x' + U * 3 + ' ' * 3 + 'y',
    '<0x41>', '<0x0A>', '<0xFF>', '<0x0a>', '<0x41', '0x41>', '<0x41><0x42>', ' <0x41> ',
    '<bos>', '<eos>', '<pad>', '<unk>', '<mask>', '<|turn>', '<turn|>', '<|turn>user\nhi<turn|>\n', '<|channel>',
    '<channel|>', '<|think|>', '<|tool>', '<tool|>', '<|tool_call>', '<tool_call|>', '<|tool_response>',
    '<tool_response|>', '<|"|>', '<|image>', '<|image|>', '<image|>', '<|audio>', '<|video|>', '<|turn', 'turn|>',
    '<<|turn>>', '<|turn><turn|>', '<bos><bos>', 'call:fn{a:<|"|>x<|"|>}', '<|turn>model\n<|channel>thought\n',
    '\n' * 31, ' ' * 31, '\t' * 31, '\n' * 64 + 'x', ' ' * 33, '\r\n' * 10, '\u3000' * 5,
    '\U0001F9EC\U0001FAE8\U0001F977', '\U00020000\U0002A6D6\U0003134A', '\u0F00\u1A2B\uA9C0', '\U00012000',
    '\U000E0067\U000E007F', '\uFFFD', '\uFFFE\uFFFF', '\U0010FFFD', '\x01\x02\x03', '\u200d\u200c',
]
galpha = ['a', 'Z', ' ', '  ', U, '\n', '\t', '1', '<', '>', '|', '<|turn>', '<turn|>', '<bos>', '<|"|>', '<0x41>',
          '\U0001F9EC', '\U00020000', 'é', 'é', '中', '\u3000', "'", 'ing', 'The', '_']
for _ in range(600):
    adv.append(''.join(random.choice(galpha) for _ in range(random.randint(1, 24))))
json.dump([{'text': s, 'ids': encode(s)} for s in adv], open(os.path.join(a.out, 'tok_adversarial.json'), 'w'),
          ensure_ascii=True)
print('adversarial', len(adv))

# ---- 2. the span guard ---------------------------------------------------------------------------------------------
span_cases = []
pieces_pool = [('<|turn>', False), ('<|turn>', True), ('<turn|>', True), ('<bos>', False), ('<|"|>', True),
               ('hello <|turn> world', True), ('  spaced  ', True), ('\n', False), ('<|tu', True), ('rn>', False),
               ('é', True), ('́', False), ('abc', True), (U, True), ('<0x41>', True), ('x', False), (' ', False)]
for _ in range(300):
    parts = [random.choice(pieces_pool) for _ in range(random.randint(1, 8))]
    text, spans, off = '', [], 0
    for s, client in parts:
        n = len(s.encode('utf-8'))
        if client and n:
            if spans and spans[-1][1] == off:
                spans[-1][1] = off + n
            else:
                spans.append([off, off + n])
        text += s
        off += n
    span_cases.append({'text': text, 'spans': spans, 'ids': guarded(text, spans)})
json.dump(span_cases, open(os.path.join(a.out, 'tok_spans.json'), 'w'), ensure_ascii=True)


# ---- 3. every scalar value, in context ----------------------------------------------------------------------------
def sweep_text(chunk):
    s = []
    for cp in range(chunk * 4096, chunk * 4096 + 4096):
        if 0xd800 <= cp <= 0xdfff:
            continue
        c = chr(cp)
        s.append('x' + c + c + c + ' ' + c + "1'" + c + 'Ab' + c + '\n')
    return ''.join(s)


with open(os.path.join(a.out, 'tok_sweep.tsv'), 'w') as f:
    for chunk in range(0x110000 // 4096):
        ids = encode(sweep_text(chunk))
        f.write('%d\t%d\t%s\n' % (chunk, len(ids), ids_sha(ids)))

# ---- 4. the calibration corpus -------------------------------------------------------------------------------------
rows = [json.loads(l) for l in open(a.calib)]
with open(os.path.join(a.out, 'tok_calib.tsv'), 'w') as f:
    tot = 0
    for i, r in enumerate(rows):
        ids = encode(r['text'])
        tot += len(ids)
        f.write('%d\t%d\t%s\n' % (i, len(ids), ids_sha(ids)))
print('calib rows', len(rows), 'tokens', tot)

# ---- 5. token bytes: a sample of ids, each decoded alone (tokenizers' Replace / ByteFallback / Fuse decoder) ------
n = tk.get_vocab_size(with_added_tokens=True)
dec_ids = list(range(0, 600)) + random.sample(range(600, n), 3000) + [n - 1]
json.dump([{'id': i, 'text': tk.decode([i], skip_special_tokens=False)} for i in dec_ids],
          open(os.path.join(a.out, 'tok_decode.json'), 'w'), ensure_ascii=True)

meta = {'tokenizers': tokenizers.__version__,
        'tokenizer.json sha256': hashlib.sha256(open(os.path.join(a.tok, 'tokenizer.json'), 'rb').read()).hexdigest(),
        'calib': os.path.basename(a.calib),
        'calib sha256': hashlib.sha256(open(a.calib, 'rb').read()).hexdigest()}
json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
