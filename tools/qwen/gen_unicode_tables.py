#!/usr/bin/env python3
"""Generate src/lib/qwen_unicode_tables.inc -- the Unicode data the Qwen tokenizer needs (L251 S5).

HF `tokenizers` (0.23.2, the version transformers main pins for Qwen3.8-Flash-Next) runs TWO Unicode
databases, and they are NOT the same version.  Both were measured, codepoint by codepoint, against the
library itself (research/l251/render/README.md in pulsar-notes):

  * the pre-tokenizer regex runs in Oniguruma: \\p{L} \\p{M} \\p{N} and \\s are UCD **16.0.0**
    (General_Category L*/M*/N*, PropList White_Space) -- 0 mismatches over every scalar value in
    13 contexts; 17.0.0 differs at 1,450 codepoints, 15.1 at ~1,230;
  * the NFC normalizer is the `unicode-normalization-alignments` crate: UCD **9.0.0** -- 0 mismatches
    on 70k random mark/decomposable/Hangul strings; 10.0.0 already differs (U+1DF9's ccc).

So this script takes each table from ITS version, and refuses UCD files whose SHA-256 is not the one
pinned here (the inputs are part of the result).  Fetch them with --fetch or point --ucd at a dir laid
out as <dir>/<version>/<file>.

usage: gen_unicode_tables.py --ucd DIR [--fetch] [--out src/lib/qwen_unicode_tables.inc]
"""
import argparse, hashlib, os, sys, urllib.request

CLASS_VERSION = '16.0.0'
NFC_VERSION = '9.0.0'
PINNED = {
    ('16.0.0', 'UnicodeData.txt'): 'ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f',
    ('16.0.0', 'PropList.txt'): '53d614508e2a0b2305a8aa21cd60d993de9326cdf65993660dfcce4503548583',
    ('9.0.0', 'UnicodeData.txt'): '68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f',
    ('9.0.0', 'CompositionExclusions.txt'): '5623df16856ad4007c60bdfff6f054e087521becd24cb4006be69c3a1d851aee',
}


def ucd_file(root, ver, name, fetch):
    path = os.path.join(root, ver, name)
    if not os.path.exists(path):
        if not fetch:
            sys.exit(f'missing {path} (pass --fetch to download it from unicode.org)')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(f'https://www.unicode.org/Public/{ver}/ucd/{name}', path)
    data = open(path, 'rb').read()
    got = hashlib.sha256(data).hexdigest()
    if got != PINNED[(ver, name)]:
        sys.exit(f'{path}: sha256 {got} is not the pinned {PINNED[(ver, name)]}')
    return data.decode('utf-8')


def unicode_data(text):
    """(general category, ccc, canonical decomposition) per codepoint, ranges expanded."""
    cat, ccc, dec = {}, {}, {}
    first = None
    for ln in text.splitlines():
        f = ln.split(';')
        cp = int(f[0], 16)
        if f[1].endswith(', First>'):
            first = cp
            continue
        span = range(first, cp + 1) if f[1].endswith(', Last>') else (cp,)
        for x in span:
            cat[x] = f[2]
            if int(f[3]):
                ccc[x] = int(f[3])
        if f[5] and not f[5].startswith('<'):
            dec[cp] = [int(x, 16) for x in f[5].split()]
    return cat, ccc, dec


def ranges_of(text, prop):
    out = set()
    for ln in text.splitlines():
        ln = ln.split('#')[0].strip()
        if not ln:
            continue
        r, p = [x.strip() for x in ln.split(';')]
        if p != prop:
            continue
        a, _, b = r.partition('..')
        out.update(range(int(a, 16), int(b or a, 16) + 1))
    return out


def code_points(text):
    out = set()
    for ln in text.splitlines():
        ln = ln.split('#')[0].strip()
        if not ln:
            continue
        a, _, b = ln.partition('..')
        out.update(range(int(a, 16), int(b or a, 16) + 1))
    return out


def runs(values):
    """[(lo, hi, v)] for maximal runs of equal non-zero v over sorted codepoints."""
    out = []
    for cp in sorted(values):
        v = values[cp]
        if out and out[-1][1] == cp - 1 and out[-1][2] == v:
            out[-1][1] = cp
        else:
            out.append([cp, cp, v])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ucd', required=True)
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'lib',
                                                  'qwen_unicode_tables.inc'))
    a = ap.parse_args()

    # --- regex classes (16.0.0) ---
    cat16, _, _ = unicode_data(ucd_file(a.ucd, CLASS_VERSION, 'UnicodeData.txt', a.fetch))
    ws16 = ranges_of(ucd_file(a.ucd, CLASS_VERSION, 'PropList.txt', a.fetch), 'White_Space')
    flags = {}
    for cp, c in cat16.items():
        f = {'L': 1, 'M': 2, 'N': 4}.get(c[0], 0)
        if f:
            flags[cp] = f
    for cp in ws16:
        flags[cp] = flags.get(cp, 0) | 8
    cls = runs(flags)

    # --- NFC (9.0.0) ---
    cat9, ccc9, dec9 = unicode_data(ucd_file(a.ucd, NFC_VERSION, 'UnicodeData.txt', a.fetch))
    excl9 = code_points(ucd_file(a.ucd, NFC_VERSION, 'CompositionExclusions.txt', a.fetch))
    full = {}

    def fd(cp):
        if cp not in full:
            full[cp] = sum((fd(x) for x in dec9[cp]), []) if cp in dec9 else [cp]
        return full[cp]
    for cp in dec9:
        fd(cp)
    decomp = sorted((cp, full[cp]) for cp in dec9)
    comp = sorted((d[0], d[1], cp) for cp, d in dec9.items()
                  if len(d) == 2 and cp not in excl9 and not ccc9.get(cp) and not ccc9.get(d[0]))
    cccr = runs(ccc9)
    maxlen = max(len(d) for _, d in decomp)

    o = []
    o.append('/* GENERATED by tools/qwen/gen_unicode_tables.py -- do not edit.  L251 S5.')
    o.append(f' * Regex classes: UCD {CLASS_VERSION} (Oniguruma in HF tokenizers 0.23.2).')
    o.append(f' * NFC: UCD {NFC_VERSION} (unicode-normalization-alignments in HF tokenizers 0.23.2).')
    o.append(' * Both versions were measured against the library; see the generator. */')
    o.append(f'#define QWEN_UNI_CLASS_VERSION "{CLASS_VERSION}"')
    o.append(f'#define QWEN_UNI_NFC_VERSION "{NFC_VERSION}"')
    o.append(f'#define QWEN_UNI_DECOMP_MAX {maxlen}')
    o.append('/* {lo, hi, flags}: 1 = \\p{L}, 2 = \\p{M}, 4 = \\p{N}, 8 = \\s (White_Space) */')
    o.append(f'static const uint32_t qwen_uni_class[{len(cls)}][3] = {{')
    for i in range(0, len(cls), 4):
        o.append('    ' + ' '.join('{0x%x,0x%x,%d},' % tuple(r) for r in cls[i:i + 4]))
    o.append('};')
    o.append('/* {lo, hi, canonical combining class} */')
    o.append(f'static const uint32_t qwen_uni_ccc[{len(cccr)}][3] = {{')
    for i in range(0, len(cccr), 4):
        o.append('    ' + ' '.join('{0x%x,0x%x,%d},' % tuple(r) for r in cccr[i:i + 4]))
    o.append('};')
    pool = []
    rows = []
    for cp, d in decomp:
        rows.append((cp, len(pool), len(d)))
        pool += d
    o.append('/* {codepoint, offset into qwen_uni_decomp_pool, length}: the FULL canonical decomposition */')
    o.append(f'static const uint32_t qwen_uni_decomp[{len(rows)}][3] = {{')
    for i in range(0, len(rows), 4):
        o.append('    ' + ' '.join('{0x%x,%d,%d},' % r for r in rows[i:i + 4]))
    o.append('};')
    o.append(f'static const uint32_t qwen_uni_decomp_pool[{len(pool)}] = {{')
    for i in range(0, len(pool), 8):
        o.append('    ' + ' '.join('0x%x,' % x for x in pool[i:i + 8]))
    o.append('};')
    o.append('/* {first, second, primary composite}, sorted by (first, second) */')
    o.append(f'static const uint32_t qwen_uni_comp[{len(comp)}][3] = {{')
    for i in range(0, len(comp), 4):
        o.append('    ' + ' '.join('{0x%x,0x%x,0x%x},' % r for r in comp[i:i + 4]))
    o.append('};')
    with open(a.out, 'w') as f:
        f.write('\n'.join(o) + '\n')
    print(f'{a.out}: {len(cls)} class ranges, {len(cccr)} ccc ranges, {len(rows)} decompositions '
          f'(pool {len(pool)}, max {maxlen}), {len(comp)} compositions')


if __name__ == '__main__':
    main()
