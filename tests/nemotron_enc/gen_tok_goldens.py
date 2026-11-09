#!/usr/bin/env python3
"""HF goldens for the Nemotron-3.5 encoders' tokenizer (L321): tekken plus the Replace pair that drops a zero-width
non-joiner (U+200C) after '[' or '<'.  The gate is tests/nemotron_enc_tok_gate.cpp; everything it compares against is
tokenizers.Tokenizer on the checkpoint's own tokenizer.json:

  * tok_adversarial.json  encode(text, add_special_tokens=False) of the Nemotron-H gate's adversarial strings, the
                          ZWNJ cases (after '[', '<', other characters, doubled, at run edges, beside added tokens)
                          and random strings over an alphabet that includes U+200C
  * tok_spans.json        the span guard: added tokens matched only outside client spans, each run through the
                          normaliser, the pre-tokenizer and the model (tokenizers' components)
  * tok_sweep.tsv         every scalar value in context, per 4096-codepoint chunk: count + sha1(ids)
  * tok_calib.tsv         the calibration corpus, per row: count + sha1(ids)

usage: gen_tok_goldens.py --tok DIR --calib FILE --out tests/test-vectors/nemotron_enc
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
Z = '‌'


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


# ---- 1. adversarial strings (the Nemotron-H gate's, then the ZWNJ cases, then random over an alphabet with U+200C) --
random.seed(20261024)
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

adv += [
    '[' + Z, '<' + Z, '[' + Z + 'x', '<' + Z + 'x', 'a[' + Z + 'b<' + Z + 'c', '[' + Z + Z, '<' + Z + Z, Z,
    Z + '[', Z + '<', 'x' + Z + 'y', '(' + Z, '{' + Z, ']' + Z, '>' + Z, '[[' + Z, '<<' + Z, '[' + Z + '[' + Z,
    '[' + Z + 'INST]', '<' + Z + 's>', '<' + Z + '/s>', '[' + Z + '/INST]', 'query: [' + Z + 'tag] <' + Z + 'b>',
    '[INST]' + Z, Z + '[INST]', '<s>' + Z, '[' + Z + '\n', '\n<' + Z, ' [' + Z + ' ', '<' + Z + '‍', '[‍',
    '[ ' + Z, '< ' + Z, '[' + Z * 5, '<' + Z * 3 + '<' + Z,
]
zalpha = ['[', '<', ']', '>', Z, Z, 'a', ' ', '\n', '[INST]', '<s>', 'é', '中', '‍']
for _ in range(400):
    adv.append(''.join(random.choice(zalpha) for _ in range(random.randint(1, 16))))
json.dump([{'text': s, 'ids': encode(s)} for s in adv], open(os.path.join(a.out, 'tok_adversarial.json'), 'w'),
          ensure_ascii=True)
print('adversarial', len(adv))

# ---- 2. the span guard ---------------------------------------------------------------------------------------------
span_cases = []
pieces_pool = [('[INST]', False), ('[INST]', True), ('</s>', True), ('<s>', False), ('[' + Z, True), ('<' + Z, True),
               ('hello [INST] world', True), ('  spaced  ', True), ('\n', False), ('[INS', True), ('T]', False),
               ('é', True), ('́', False), ('abc', True), ('<SPECIAL_1', True), ('2>', False), (Z, True), ('x', False)]
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

meta = {'tokenizers': tokenizers.__version__,
        'tokenizer.json sha256': hashlib.sha256(open(os.path.join(a.tok, 'tokenizer.json'), 'rb').read()).hexdigest(),
        'calib': os.path.basename(a.calib),
        'calib sha256': hashlib.sha256(open(a.calib, 'rb').read()).hexdigest()}
json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
