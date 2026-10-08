#!/usr/bin/env python3
"""HF-generated goldens for the Qwen3.8-Flash-Next tokenizer, renderer and span map (L251 S5).

Everything the gate (tests/qwen_chat_gate.cpp) compares against comes from HF here -- transformers'
apply_chat_template with the checkpoint's own chat_template.jinja, and tokenizers' Tokenizer on the
checkpoint's tokenizer.json.  Nothing is computed by a second implementation of our own:

  * TOKENS: tokenizers.Tokenizer.encode(text, add_special_tokens=False).
  * CLIENT SPANS (the L223 map): the conversation is rendered a second time with every client string
    wrapped in two private-use sentinels (U+E000 / U+E001) -- non-string argument values become the
    string of their own tojson, which the template writes raw, so the bytes are unchanged -- and the
    sentinel positions ARE the spans.  The tools block is located by its json.dumps text.
  * GUARDED TOKENS (markers suppressed inside client spans): when no client span spells an added token
    they equal HF's encode of the whole text; otherwise they come from tokenizers' own components --
    the text cut at the added tokens that lie outside every span, and each run through the normalizer,
    the pre-tokenizer and the BPE model (no added-token matching at all).

usage: gen_qwen_goldens.py --tok DIR --cases DIR --calib FILE --out tests/test-vectors/qwen38
       (DIR holds tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json;
        needs transformers + tokenizers -- the versions are recorded in meta.json)
"""
import argparse, glob, hashlib, json, os, random, sys

import tokenizers
import transformers
from tokenizers import Tokenizer
from transformers import AutoTokenizer

O, C = '', ''
ap = argparse.ArgumentParser()
ap.add_argument('--tok', required=True)
ap.add_argument('--cases', required=True)
ap.add_argument('--calib', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

hf = AutoTokenizer.from_pretrained(a.tok)
tk = Tokenizer.from_file(os.path.join(a.tok, 'tokenizer.json'))
added = sorted(((t.content, i) for i, t in tk.get_added_tokens_decoder().items()), key=lambda x: -len(x[0]))


def sha(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def ids_sha(ids):
    return sha(''.join('%d,' % i for i in ids))


def encode(text):
    return tk.encode(text, add_special_tokens=False).ids


def plain(seg):
    """tokenizers' pipeline with NO added-token matching: normalizer, pre-tokenizer, BPE."""
    out = []
    for piece, _ in tk.pre_tokenizer.pre_tokenize_str(tk.normalizer.normalize_str(seg)):
        out += [t.id for t in tk.model.tokenize(piece)]
    return out


def guarded(text, spans):
    """Added tokens match only where their bytes touch no client span (byte offsets)."""
    b = text.encode('utf-8')
    out, run, pos, si = [], 0, 0, 0
    while pos < len(b):
        hit = None
        if b[pos:pos + 1] == b'<':
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


def spells_marker(text, spans):
    b = text.encode('utf-8')
    return any(t.encode('utf-8') in b[lo:hi] for lo, hi in spans for t, _ in added)


# ---- rendering ---------------------------------------------------------------------------------
def kwargs_for(effort, gen):
    k = {'add_generation_prompt': gen}
    if effort == 'none':
        k['enable_thinking'] = False
    elif effort != 'default':
        k['reasoning_effort'] = effort
    return k


class Refused(Exception):
    pass


def api_to_hf(messages, wrap):
    """API messages (arguments as JSON text) -> what the template takes (arguments decoded ONCE)."""
    def w(s):   # content / reasoning: the template trims them
        if isinstance(s, list):   # L268: content parts -- text wrapped, images (the template's vision literal) not
            out_parts = []
            for i, part in enumerate(s):
                if part.get('type') == 'image':
                    out_parts.append({'type': 'image'})
                    continue
                t = part['text']
                if wrap:   # |trim acts on the whole rendered content: only the outer parts lose whitespace
                    if i == 0:
                        t = t.lstrip()
                    if i == len(s) - 1:
                        t = t.rstrip()
                    t = O + t + C if t else t
                out_parts.append({'type': 'text', 'text': t})
            return out_parts
        if not wrap or not isinstance(s, str):
            return s
        s = s.strip()
        return O + s + C if s else s

    def wr(s):  # names, keys, values: written raw
        return O + s + C if wrap and isinstance(s, str) and s else s
    out = []
    for m in messages:
        n = {'role': m['role'], 'content': w(m.get('content'))}
        if 'reasoning_content' in m:
            n['reasoning_content'] = w(m['reasoning_content'])
        if m.get('tool_calls'):
            calls = []
            for tc in m['tool_calls']:
                fn = tc.get('function', tc)
                args = fn.get('arguments')
                if isinstance(args, str) and args != '':
                    try:
                        args = json.loads(args)
                    except ValueError:
                        raise Refused('tool_arguments_not_json')
                if wrap and isinstance(args, dict):
                    args = {wr(k): (wr(v) if isinstance(v, str) else wr(json.dumps(v, ensure_ascii=False)))
                            for k, v in args.items()}
                calls.append({'type': 'function', 'function': {'name': wr(fn['name']), 'arguments': args}})
            n['tool_calls'] = calls
        out.append(n)
    return out


HF_ERRORS = {
    'No messages provided.': 'no_messages',
    'Cannot apply chat template to an empty conversation': 'no_messages',
    'System message must be at the beginning.': 'system_not_first',
    'Unexpected message role.': 'unexpected_role',
    'No user query found in messages.': 'no_user_query',
    'Can only get item pairs from a mapping.': 'tool_arguments_not_object',
    'System message cannot contain images.': 'image_position',
}


def hf_render(messages, tools_text, effort, gen):
    """(text, spans) or Refused(key)."""
    tools = json.loads(tools_text) if tools_text else None
    k = kwargs_for(effort, gen)
    try:
        text = hf.apply_chat_template(api_to_hf(messages, False), tools=tools, tokenize=False, **k)
    except Refused:
        raise
    except Exception as e:  # the template's raise_exception / jinja's items filter
        msg = str(e)
        for m, key in HF_ERRORS.items():
            if m in msg:
                raise Refused(key)
        raise
    wrapped = hf.apply_chat_template(api_to_hf(messages, True), tools=tools, tokenize=False, **k)
    assert O not in text and C not in text, 'sentinel collides with corpus text'
    # sentinel positions -> byte spans
    out, spans, lo = [], [], None
    nbytes = 0
    for ch in wrapped:
        if ch == O:
            assert lo is None
            lo = nbytes
        elif ch == C:
            if nbytes > lo:
                spans.append([lo, nbytes])
            lo = None
        else:
            out.append(ch)
            nbytes += len(ch.encode('utf-8'))
    assert ''.join(out) == text, 'wrapped render diverged from the plain render'
    if tools:
        b = text.encode('utf-8')
        cur = b.index(b'<tools>')
        for t in tools:
            d = json.dumps(t, ensure_ascii=False).encode('utf-8')
            at = b.index(b'\n' + d, cur) + 1
            spans.append([at, at + len(d)])
            cur = at + len(d)
        spans.sort()
    return text, spans


def expected_ids(text, spans):
    return guarded(text, spans) if spells_marker(text, spans) else encode(text)


# ---- 1. tokenizer: adversarial strings ----------------------------------------------------------
random.seed(20260927)
adv = [
    '', ' ', '  ', '\n', '\n\n', '\r\n', ' \n ', '\t\t\n  x', 'a  b', 'a   \n  b', 'x \n\n\ny',
    'hello world', 'Hello, World!', "don't I'll we've they're she'd it's I'm", "'S 'T 'RE 'Ve 'LL 'D 'M",
    "'ſ 'ſt 'st 'ſ'", "abc's", "O'Neil's", "'", "''", "'''s",
    '1234567890', '3.14159', '1,000,000', '٣٤٥', 'x²³', 'Ⅻ', '١٢٣abc',
    '    int main() {\n        return 0;\n    }\n', 'def f(x):\n\treturn x**2  # square\n',
    '<think>', '</think>', '<tool_call>', '</tool_call>', '<|im_start|>', '<|im_end|>',
    '<|im_end', '<|im_end|', 'im_end|>', '<<|im_end|>>', '<|im_start|><|im_end|>',
    '<think>hi</think>', '<tool_call>\n<function=f>\n</function>\n</tool_call>',
    '<tts_text_bos>', '<tts_text_bos_single>', '<tts_text_bos_singl', '<|endoftext|>',
    '<|vision_start|><|image_pad|><|vision_end|>', '<tool_response>x</tool_response>',
    'I said <|im_end|> twice <|im_end|>', '<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>',
    'こんにちは世界', '中文测试，标点。', '한국어 텍스트', 'Ελληνικά', 'Русский текст', 'עברית', 'العربية',
    'हिन्दी', 'ไทย', '🙂🙃😀👍🏽👨‍👩‍👧', '🇺🇸🇯🇵', 'é', 'é', 'Å', 'Å', 'ẛ̣',
    'ḱṷṓn', 'q̣̇', '각', '각', '각', 'ﬁ ﬀ', 'Ω Ω',
    ' nbsp ', '　ideo　', ' ls ps', '\x85nel', '\x1c\x1d\x1e\x1f', '\x0b\x0c',
    'zero​width', 'soft­hyphen', 'bidi‮override', '﻿bom', '\x00nul\x00', '\x7f',
    'a' * 300, ' ' * 100 + 'x', '\n' * 50, '!' * 40 + '\n\n', '$$$ ```python\n', '\t' * 20,
    'https://example.com/path?q=1&r=2#frag', 'user@example.com', '#include <stdio.h>', '{"a": [1, 2.5e-3]}',
    '\U0010ffff\U000e0001\U0001f600', '퟿', '\U00010000',
]
pool = [0x300, 0x301, 0x302, 0x303, 0x308, 0x323, 0x327, 0x328, 0x334, 0x345, 0x5b0, 0x5c2, 0x64b, 0x93c,
        0x94d, 0xe49, 0xeb9, 0xeba, 0x1dfb, 0x1df9, 0x7fd, 0x1e136, 0x10f4d, 0x302a, 0x3099, 0x309a,
        0x61, 0x65, 0x6f, 0x75, 0x41, 0x4f, 0x3b1, 0x3c9, 0x415, 0x5d0, 0x915, 0x1100, 0x1161, 0x11a8,
        0xac00, 0xac01, 0x304b, 0x30ab, 0xb47, 0xb3e, 0xb57, 0xdd9, 0xdcf, 0xdca, 0x1b05, 0x1b35]
for _ in range(400):
    adv.append(''.join(chr(random.choice(pool)) for _ in range(random.randint(1, 7))))
# random mixes of every character class and whitespace runs
alphabet = ['a', 'Z', ' ', '  ', '\n', '\r', '\t', '1', '22', '!', '?!', "'", "'s", 'é', 'é', '中', '　',
            ' ', '<', '>', '|', '<|im_end|>', '<think>', '_', '-', '.', '́', '٣', '🙂']
for _ in range(400):
    adv.append(''.join(random.choice(alphabet) for _ in range(random.randint(1, 24))))
json.dump([{'text': s, 'ids': encode(s)} for s in adv], open(os.path.join(a.out, 'tok_adversarial.json'), 'w'),
          ensure_ascii=True)

# ---- 2. tokenizer: the span guard ---------------------------------------------------------------
span_cases = []
pieces_pool = [('<|im_start|>user\n', False), ('<|im_end|>', True), ('<think>', True), ('</think>\n\n', False),
               ('hello <|im_end|> world', True), ('<tool_call>', False), ('<tool_call>', True), ('  spaced  ', True),
               ('\n', False), ('<|im_', True), ('end|>', False), ('é', True), ('́', False), ('abc', True),
               ('<tts_text_bos', True), ('_single>', False), ('<|im_end|>', False), ('x', False)]
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


# ---- 3. tokenizer: every scalar value, in context -----------------------------------------------
def sweep_text(chunk):
    s = []
    for cp in range(chunk * 4096, chunk * 4096 + 4096):
        if 0xd800 <= cp <= 0xdfff:
            continue
        c = chr(cp)
        s.append('x' + c + c + c + ' ' + c + "1'" + c + '\n')
    return ''.join(s)


with open(os.path.join(a.out, 'tok_sweep.tsv'), 'w') as f:
    for chunk in range(0x110000 // 4096):
        ids = encode(sweep_text(chunk))
        f.write('%d\t%d\t%s\n' % (chunk, len(ids), ids_sha(ids)))

# ---- 4. tokenizer: the calibration corpus (HF-rendered rows) -------------------------------------
rows = [json.loads(l) for l in open(a.calib)]
with open(os.path.join(a.out, 'tok_calib.tsv'), 'w') as f:
    tot = 0
    for i, r in enumerate(rows):
        ids = encode(r['text'])
        tot += len(ids)
        f.write('%d\t%d\t%s\n' % (i, len(ids), ids_sha(ids)))
print('calib rows', len(rows), 'tokens', tot)

# ---- 5. renderer: hand cases ----------------------------------------------------------------------
TOOLS = json.dumps([
    {'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather for a city — «quoted» "x"\n\ttab',
     'parameters': {'type': 'object', 'properties': {
         'city': {'type': 'string', 'description': 'City name'},
         'days': {'type': 'integer'}, 'ratio': {'type': 'number', 'default': 0.5},
         'metric': {'type': 'boolean'}, 'tags': {'type': 'array', 'items': {'type': 'string'}},
         'opts': {'type': 'object'}, 'note': {'type': ['string', 'null']}},
         'required': ['city']}, 'strict': True}},
    {'type': 'function', 'function': {'name': 'run', 'parameters': {'type': 'object', 'properties': {
        'cmd': {'type': 'string'}, 'timeout': {'type': 'number', 'minimum': 1e-07, 'maximum': 1e+16}}}}},
], ensure_ascii=False)
U = lambda c: {'role': 'user', 'content': c}
S = lambda c: {'role': 'system', 'content': c}
A = lambda c, r=None, calls=None: dict({'role': 'assistant', 'content': c},
                                       **({'reasoning_content': r} if r is not None else {}),
                                       **({'tool_calls': [{'type': 'function', 'id': 'call_%d' % i,
                                                           'function': {'name': n, 'arguments': g}}
                                                          for i, (n, g) in enumerate(calls)]} if calls else {}))
T = lambda c: {'role': 'tool', 'tool_call_id': 'call_0', 'content': c}
hand = []


def case(name, msgs, tools=None, effort='default', gen=True):
    hand.append({'name': name, 'messages': msgs, 'tools': tools, 'effort': effort, 'gen': gen})


case('user_only', [U('Hi')])
for e in ('default', 'none', 'low', 'medium', 'xhigh'):
    case('sys_user_' + e, [S('You are terse.'), U('What is 2+2?')], effort=e)
    case('tools_user_' + e, [U('Weather in Paris?')], tools=TOOLS, effort=e)
    case('tools_sys_user_' + e, [S('Be careful.'), U('Weather?')], tools=TOOLS, effort=e)
case('empty_system', [S(''), U('x')])
case('ws_system', [S(' \n　 '), U('x')], effort='medium')
case('ws_system_tools', [S('\t'), U('x')], tools=TOOLS)
case('no_gen_prompt', [S('s'), U('u'), A('a')], gen=False)
case('multi_turn_reasoning', [S('sys'), U('q1'), A('a1', '  r1 thinking\n'), U('q2'), A('a2'), U('q3')])
case('reasoning_none_vs_empty', [U('q'), A('a', ''), U('q2'), A(None, None), U('q3')], effort='low')
case('tool_round', [S('agent'), U('Weather in Paris for 3 days?'),
                    A('Let me check.', 'Need the tool.', [('get_weather', json.dumps(
                        {'city': 'Paris', 'days': 3, 'ratio': 0.25, 'metric': True, 'tags': ['a', 'ü'],
                         'opts': {'k': [1, 2.0, None]}, 'note': None}))]),
                    T('{"temp": 21}'), A('It is 21C.')], tools=TOOLS)
case('tool_calls_no_content', [U('go'), A(None, 'plan', [('run', '{"cmd": "ls -la\\n", "timeout": 1.5}'),
                                                        ('run', '{"cmd":"pwd"}')]),
                               T('file1\nfile2'), T('  /home  '), U('thanks')], tools=TOOLS)
case('tool_last', [U('go'), A('', None, [('run', '{"cmd": "x"}')]), T('out')], tools=TOOLS)
case('tool_then_tool_then_assistant', [U('go'), A('ok', None, [('run', '{}'), ('run', '')]), T('a'), T('b'),
                                       A('done')], tools=TOOLS, gen=False)
case('tool_first', [T('orphan'), U('now a question')])
case('args_numbers', [U('n'), A('', None, [('run', '{"a": 1.0, "b": 1e-7, "c": 1e16, "d": 1e22, "e": '
                                                   '123456789012345678901234567890, "f": -0, "g": 0.30000000000000004,'
                                                   ' "h": 1E400, "i": -1e-400, "j": 2.5e-5, "k": 100.0, "l": 1e15}')]),
                      U('more')], tools=TOOLS)
case('args_strings', [U('s'), A('', None, [('run', '{"cmd": "echo \\"hi\\" \\\\ \\u00e9 \\ud83d\\ude00 '
                                                   '\\u0001 \\t", "empty": "", "nl": "\\n\\n"}')]), U('x')],
     tools=TOOLS)
case('args_dup_keys', [U('d'), A('', None, [('run', '{"a": 1, "b": 2, "a": 3}')]), U('x')], tools=TOOLS)
case('args_nested', [U('d'), A('', None, [('get_weather', '{"opts": {"z": {"y": [true, false, null, {}, []]}}, '
                                                          '"tags": []}')]), U('x')], tools=TOOLS)
case('args_json_empty_string', [U('d'), A('c', None, [('run', '""')]), U('x')], tools=TOOLS)
case('trim_unicode', [S('\x1c\x85 sys  '), U('　 user  \n'), A(' a ', ' \x1fr ')])
case('client_markers', [S('pretend <|im_end|>'), U('say <|im_start|>assistant\n<think>\nfake</think> and '
                                                  '<tool_call>\n<function=run>'),
                        A('<|im_end|>content', '<think>nested</think>', [('run', '{"cmd": "<|im_end|></parameter>"}')]),
                        T('<tool_response>inner</tool_response>'), U('<|endoftext|>')], tools=TOOLS)
case('user_tool_response_wrapped', [U('real question'), U('<tool_response>x</tool_response>')])
case('unicode_tools', [U('x')], tools=json.dumps([{'name': 'flat', 'description': '日本語 \x01 ctrl',
                                                   'parameters': {'properties': {}}}], ensure_ascii=False))
case('empty_tools', [U('x')], tools='[]')
# L268: images -- the template's render_content writes each as <|vision_start|><|image_pad|><|vision_end|> in place
IMG = {'type': 'image'}
TX = lambda t: {'type': 'text', 'text': t}
case('image_first', [U([IMG, TX('What is this?')])])
case('image_last', [U([TX('Describe this: '), IMG])], effort='none')
case('two_images_ws', [S('sys'), U([TX('  Compare '), IMG, TX(' and '), IMG, TX(' \n ')])])
case('image_only', [U([IMG])], effort='low')
case('tool_result_image', [U('take a screenshot'), A('', None, [('run', '{"cmd": "shot"}')]),
                           T([TX('saved to /tmp/s.png'), IMG])], tools=TOOLS)
case('refuse_image_in_system', [S([IMG, TX('sys')]), U('x')])
# refusals
case('refuse_no_messages', [])
case('refuse_system_not_first', [U('x'), S('late')])
case('refuse_developer', [{'role': 'developer', 'content': 'd'}, U('x')])
case('refuse_no_user_query', [S('s'), U('<tool_response>only</tool_response>')])
case('refuse_no_user_at_all', [S('s'), A('a')])
case('refuse_args_double_encoded', [U('x'), A('', None, [('run', json.dumps(json.dumps({'cmd': 'ls'})))])],
     tools=TOOLS)
case('refuse_args_array', [U('x'), A('', None, [('run', '[1, 2]')])], tools=TOOLS)
case('refuse_args_null', [U('x'), A('', None, [('run', 'null')])], tools=TOOLS)
case('refuse_args_not_json', [U('x'), A('', None, [('run', '{cmd: ls}')])], tools=TOOLS)

for h in hand:
    try:
        text, spans = hf_render(h['messages'], h['tools'], h['effort'], h['gen'])
        ids = expected_ids(text, spans)
        h['expect'] = {'text': text, 'spans': spans, 'ids': ids, 'hf_ids_equal': ids == encode(text)}
    except Refused as r:
        h['expect'] = {'refusal': str(r)}
json.dump(hand, open(os.path.join(a.out, 'render_cases.json'), 'w'), ensure_ascii=True, indent=0)
print('hand cases', len(hand), 'refusals', sum('refusal' in h['expect'] for h in hand))

# ---- 6. renderer: the L216 corpus, two configurations per case ------------------------------------
EFFORTS = ['none', 'low', 'medium', 'xhigh']
with open(os.path.join(a.out, 'render_corpus.tsv'), 'w') as f:
    f.write('# file\tconfig\tcases\trendered\trefusals(key=n)\tsha1(text+spans)\tsha1(ids)\ttokens\tguarded\n')
    for path in sorted(glob.glob(os.path.join(a.cases, '*.json'))):
        cases = json.load(open(path))
        for cfg in ('A', 'B'):
            ht, hi, ok, refused, ntok, nguard = hashlib.sha1(), hashlib.sha1(), 0, {}, 0, 0
            for idx, c in enumerate(cases):
                effort = 'default' if cfg == 'A' else EFFORTS[idx % 4]
                tools = json.dumps(c['tools'], ensure_ascii=False) if c.get('tools') else None
                try:
                    text, spans = hf_render(c['messages'], tools, effort, cfg == 'B')
                except Refused as r:
                    refused[str(r)] = refused.get(str(r), 0) + 1
                    ht.update(('%d refuse %s\n' % (idx, r)).encode())
                    continue
                ok += 1
                ht.update(('%d %s %s\n' % (idx, sha(text), ','.join('%d-%d' % tuple(s) for s in spans))).encode())
                guard = spells_marker(text, spans)
                nguard += guard
                ids = guarded(text, spans) if guard else encode(text)
                ntok += len(ids)
                hi.update(('%d %s\n' % (idx, ids_sha(ids))).encode())
            f.write('%s\t%s\t%d\t%d\t%s\t%s\t%s\t%d\t%d\n' % (
                os.path.basename(path), cfg, len(cases), ok,
                ','.join('%s=%d' % kv for kv in sorted(refused.items())) or '-', ht.hexdigest(), hi.hexdigest(),
                ntok, nguard))
        print(os.path.basename(path), flush=True)

gen_cfg = json.load(open(os.path.join(a.tok, 'generation_config.json')))
meta = {
    'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
    'python': sys.version.split()[0],
    'sha256': {n: hashlib.sha256(open(os.path.join(a.tok, n), 'rb').read()).hexdigest()
               for n in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json')},
    'calib': {'path': a.calib, 'sha256': hashlib.sha256(open(a.calib, 'rb').read()).hexdigest()},
    'stop_ids': gen_cfg['eos_token_id'],
    'n_tokens': tk.get_vocab_size(with_added_tokens=True),
}
json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
print(json.dumps(meta, indent=1))
