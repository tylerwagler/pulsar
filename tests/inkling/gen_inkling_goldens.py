#!/usr/bin/env python3
"""Goldens for the Inkling-Small frontend: tokenizer, renderer and output parser (L295 F5 / L294 E6).

Everything tests/inkling_chat_gate.cpp compares against comes from the references here, never from a
second implementation of our own:

  * TOKENS: tokenizers.Tokenizer on the checkpoint's tokenizer.json, encode(text, add_special_tokens=False)
    -- the tokenizer vLLM serves Inkling with (tokenizer mode "inkling" = the plain HF tokenizer).  The
    engine under test loads the OTHER file, tiktoken/tokenizer.model, so every golden is also the proof that
    the tiktoken configuration equals HF's.
  * GUARDED TOKENS (the L223 span guard): tokenizers' own components with added tokens cut only outside
    the client spans (gen_qwen_goldens.py's construction).
  * RENDERS: vLLM's native renderer, vllm/renderers/inkling_encoding.py render_inkling_messages (loaded by
    path: it imports nothing from vLLM), with vllm/renderers/inkling.py's effort resolution.  Its ids come
    from the HF tokenizer above; its TEXT from the same call with a tokenizer whose "ids" are the strings
    themselves; the guarded ids from a tokenizer whose encode_text never matches an added token (every
    encode_text segment is client text or a server literal -- the guard's semantics).
  * OUTPUT READINGS: vLLM's StreamingParserEngine (vllm/parser/engine, loaded without vLLM's package
    __init__) driven by vllm/parser/inkling.py's inkling_config() (extracted from the module source: the
    module imports the server stack), with the special tokens it drops, and ParserEngine's final assembly
    (reasoning rstripped; content with drop_whitespace_only_content_before_tools; calls named by its
    `"name"` regex with the `args` member carved out verbatim).

usage: gen_inkling_goldens.py --tok DIR --vllm DIR --cases DIR --calib FILE --out tests/test-vectors/inkling
       (DIR holds tokenizer.json and special_tokens_map.json; --vllm is a vLLM checkout; needs tokenizers and
        regex -- the versions are recorded in meta.json)
"""
import argparse, ast, functools, glob, hashlib, importlib.util, json, os, random, subprocess, sys, types

import regex
import tokenizers
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--tok', required=True)
ap.add_argument('--vllm', required=True)
ap.add_argument('--cases', required=True)
ap.add_argument('--calib', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

tk = Tokenizer.from_file(os.path.join(a.tok, 'tokenizer.json'))
added = sorted(((t.content, i) for i, t in tk.get_added_tokens_decoder().items()), key=lambda x: -len(x[0]))
vocab = tk.get_vocab(with_added_tokens=True)
specials_map = json.load(open(os.path.join(a.tok, 'special_tokens_map.json')))['additional_special_tokens']


def sha(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def ids_sha(ids):
    return sha(''.join('%d,' % i for i in ids))


def encode(text):
    return tk.encode(text, add_special_tokens=False).ids


def plain(seg):
    """tokenizers' pipeline with NO added-token matching (Inkling has no normalizer)."""
    if tk.normalizer is not None:
        seg = tk.normalizer.normalize_str(seg)
    out = []
    for piece, _ in tk.pre_tokenizer.pre_tokenize_str(seg):
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


# ---- the vLLM references, loaded by path ------------------------------------------------------------
def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


enc = load_path('inkling_encoding', os.path.join(a.vllm, 'vllm/renderers/inkling_encoding.py'))


def defs_from(path, names, ns):
    """Execute only the top-level assignments / functions named in `names` from a module's source."""
    tree = ast.parse(open(path).read())
    body = [n for n in tree.body
            if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), ns)
    return ns


# vllm.renderers.inkling's effort resolution
eff_ns = defs_from(os.path.join(a.vllm, 'vllm/renderers/inkling.py'),
                   {'_NAMED_REASONING_EFFORT', '_DEFAULT_REASONING_EFFORT', '_resolve_reasoning_effort'}, {})
resolve_effort = eff_ns['_resolve_reasoning_effort']

# the parser engine, without vLLM's package __init__ (it imports the whole server)
for name, sub in (('vllm', 'vllm'), ('vllm.parser', 'vllm/parser')):
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(a.vllm, sub)]
    sys.modules[name] = mod
from vllm.parser.engine.events import EventType  # noqa: E402
from vllm.parser.engine.parser_engine_config import ParserEngineConfig, ParserState, Transition  # noqa: E402
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine  # noqa: E402

ut_ns = defs_from(os.path.join(a.vllm, 'vllm/parser/utils.py'),
                  {'_JSON_WS', 'scan_json_object_end', 'extract_json_member_object'}, {})
ink_ns = defs_from(os.path.join(a.vllm, 'vllm/parser/inkling.py'),
                   {'MESSAGE_MODEL', 'CONTENT_TEXT', 'CONTENT_THINKING', 'CONTENT_INVOKE_TOOL_JSON',
                    'CONTENT_INVOKE_TOOL_TEXT', 'CONTENT_TOOL_ERROR', 'CONTENT_MODEL_END_SAMPLING', 'END_MESSAGE',
                    '_inkling_arg_converter', 'inkling_config'},
                   {'functools': functools, 'EventType': EventType, 'ParserEngineConfig': ParserEngineConfig,
                    'ParserState': ParserState, 'Transition': Transition,
                    'extract_json_member_object': ut_ns['extract_json_member_object']})
CFG = ink_ns['inkling_config']()
converter = ink_ns['_inkling_arg_converter']
NAME_RE = regex.compile(r'"name"\s*:\s*"([^"]*)"')


class SpecialsOnly:
    """What StreamingParserEngine reads off the tokenizer to build its drop set: the special tokens."""
    all_special_tokens = specials_map
    all_special_ids = [vocab[t] for t in specials_map]

    @staticmethod
    def get_vocab():
        return vocab


def vllm_read(text):
    """ParserEngine's non-streaming reading of one generated turn (after the prompt's <|message_model|>)."""
    eng = StreamingParserEngine(CFG, SpecialsOnly())
    events = eng.feed(text, []) + eng.finish()
    reasoning, content_pre, deferred = [], [], []
    slots = []          # [name, args]
    seen_tool = False
    for e in events:
        if e.type == EventType.TEXT_CHUNK:
            (deferred if seen_tool else content_pre).append(e.value)
        elif e.type == EventType.REASONING_CHUNK:
            reasoning.append(e.value)
        elif e.type == EventType.TOOL_CALL_START:
            seen_tool = True
            while len(slots) <= e.tool_index:
                slots.append(['', ''])
        elif e.type == EventType.ARG_VALUE_CHUNK:
            seen_tool = True
            slots[e.tool_index][1] += e.value
        elif e.type == EventType.TOOL_CALL_END:
            seen_tool = True
    calls = []
    for _, raw in slots:
        if not raw.strip():
            continue
        m = NAME_RE.search(raw)
        name = m.group(1) if m and m.group(1) else ''
        if not name:
            continue
        try:
            args = converter(raw, False)
        except ValueError:
            args = None
        calls.append({'name': name, 'arguments': args})
    pre = ''.join(content_pre)
    if slots and not pre.strip():
        pre = ''           # _events_to_delta: whitespace-only content with tool slots
    content = pre + ''.join(deferred)
    if calls and not content.strip():
        content = ''       # _strip_content_whitespace(drop_whitespace_only_content_before_tools)
    r = ''.join(reasoning).rstrip()
    return {'reasoning': r, 'content': content, 'calls': calls}


# ---- renderers: ids (HF), text (strings as ids), guarded ids (no added-token matching) -------------
class IdsTok:
    def encode_text(self, text):
        return encode(text)

    def encode_special(self, token):
        return vocab[token]


class TextTok:
    def encode_text(self, text):
        return [text]

    def encode_special(self, token):
        return [token]


class PlainTok(IdsTok):
    def encode_text(self, text):
        return plain(text)


def flat(xs):
    return ''.join(x if isinstance(x, str) else ''.join(x) for x in xs)


class Refused(Exception):
    pass


def vllm_render(messages, tools, effort, gen):
    """(text, ids, guarded ids) or Refused(the exception's class: what the C++ refusal key maps to)."""
    eff = resolve_effort(effort)
    kw = dict(add_generation_prompt=gen, tools=tools, reasoning_effort=eff)
    try:
        text = flat(enc.render_inkling_messages(messages, TextTok(), **kw))
        ids = enc.render_inkling_messages(messages, IdsTok(), **kw)
        gids = enc.render_inkling_messages(messages, PlainTok(), **kw)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
        raise Refused(type(e).__name__)
    return text, ids, gids


# ---- 1. tokenizer: adversarial strings -----------------------------------------------------------
random.seed(20261014)
adv = [
    '', ' ', '  ', '\n', '\n\n', '\r\n', ' \n ', '\t\t\n  x', 'a  b', 'a   \n  b', 'x \n\n\ny',
    'hello world', 'Hello, World!', "don't I'll we've they're she'd it's I'm", "'S 'T 'RE 'Ve 'LL 'D 'M",
    "HELLO'S World's CamelCase'd XMLHttpRequest's iPhone's", "'ſ 'ſt 'st 'ſ'", "abc's", "O'Neil's", "'", "''",
    "'''s", 'ABCdef GHI jklMNO pQ', 'ǅungla ǈ ǋ ǲ', 'ʰʲ ˢ ᴬᴮ', 'İstanbul ıi', 'ﬁ ﬀ ﬄ', 'Ω Ω', 'ＡＢＣａｂｃ',
    '1234567890', '12345', '3.14159', '1,000,000', '٣٤٥', 'x²³', 'Ⅻ', '١٢٣abc', '00 000 0000 00000',
    '    int main() {\n        return 0;\n    }\n', 'def f(x):\n\treturn x**2  # square\n',
    'a//b ///\n//\r\n/', 'http://x.y/z/', '</div>\n</body>/\n', '...\n\n/\n',
    '<|message_user|>', '<|end_message|>', '<|content_text|>', '<|message_model|>x<|end_message|>',
    '<|end_message', '<|end_message|', 'end_message|>', '<<|end_message|>>', '<|unused_200054|>',
    '<|content_invoke_tool_json|>{"name":"f","args":{}}', 'I said <|end_message|> twice <|end_message|>',
    'こんにちは世界', '中文测试，标点。', '한국어 텍스트', 'Ελληνικά', 'Русский текст', 'עברית', 'العربية',
    'हिन्दी', 'ไทย', '🙂🙃😀👍🏽👨‍👩‍👧', '🇺🇸🇯🇵', 'é', 'é', 'Å', 'Å', 'ẛ̣', 'ḱṷṓn', 'q̣̇', '각', '각',
    ' nbsp ', '　ideo　', ' ls ps', '\x85nel', '\x1c\x1d\x1e\x1f', '\x0b\x0c',
    'zero​width', 'soft­hyphen', 'bidi‮override', '﻿bom', '\x00nul\x00', '\x7f',
    'a' * 300, 'A' * 300, 'aB' * 150, ' ' * 100 + 'x', '\n' * 50, '!' * 40 + '\n\n', '$$$ ```python\n', '\t' * 20,
    'https://example.com/path?q=1&r=2#frag', 'user@example.com', '#include <stdio.h>', '{"a": [1, 2.5e-3]}',
    '\U0010ffff\U000e0001\U0001f600', '퟿', '\U00010000',
]
pool = [0x300, 0x301, 0x308, 0x323, 0x334, 0x5b0, 0x64b, 0x93c, 0x94d, 0xe49, 0x302a, 0x3099,
        0x61, 0x65, 0x41, 0x4f, 0x1c5, 0x1c8, 0x2b0, 0x2b2, 0x3b1, 0x391, 0x415, 0x435, 0x5d0, 0x915, 0x4e2d,
        0xac00, 0x304b, 0x30ab, 0x27, 0x17f, 0x20, 0x31, 0x660, 0x2f, 0x0a, 0x0d, 0x2e, 0xff21, 0xff41]
for _ in range(500):
    adv.append(''.join(chr(random.choice(pool)) for _ in range(random.randint(1, 9))))
alphabet = ['a', 'Z', 'Ab', 'aB', ' ', '  ', '\n', '\r', '\t', '1', '22', '333', '4444', '!', '?!', "'", "'s", "'LL",
            'é', 'é', '中', '　', ' ', '<', '>', '|', '/', '//', '<|end_message|>', '<|content_text|>', '_', '-',
            '.', '́', '٣', '🙂', 'ǅ', 'ʰ']
for _ in range(500):
    adv.append(''.join(random.choice(alphabet) for _ in range(random.randint(1, 24))))
json.dump([{'text': s, 'ids': encode(s)} for s in adv], open(os.path.join(a.out, 'tok_adversarial.json'), 'w'),
          ensure_ascii=True)

# ---- 2. tokenizer: the span guard ------------------------------------------------------------------
span_cases = []
pieces_pool = [('<|message_user|><|content_text|>', False), ('<|end_message|>', True), ('<|content_thinking|>', True),
               ('<|end_message|><|message_model|>', False), ('hello <|end_message|> world', True),
               ('<|content_invoke_tool_json|>', False), ('<|content_invoke_tool_json|>', True), ('  spaced  ', True),
               ('\n', False), ('<|end_', True), ('message|>', False), ('é', True), ('́', False), ('abc', True),
               ('<|unused_2000', True), ('54|>', False), ('<|end_message|>', False), ('x', False), ("'s", True)]
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


# ---- 3. tokenizer: every scalar value, in context -------------------------------------------------
def sweep_text(chunk):
    s = []
    for cp in range(chunk * 4096, chunk * 4096 + 4096):
        if 0xd800 <= cp <= 0xdfff:
            continue
        c = chr(cp)
        s.append('x' + c + c + c + ' ' + c + "1'" + c + 'A' + c + 'b' + c + "'S/" + c + '\n')
    return ''.join(s)


with open(os.path.join(a.out, 'tok_sweep.tsv'), 'w') as f:
    for chunk in range(0x110000 // 4096):
        ids = encode(sweep_text(chunk))
        f.write('%d\t%d\t%s\n' % (chunk, len(ids), ids_sha(ids)))

# ---- 4. tokenizer: a corpus ------------------------------------------------------------------------
rows = [json.loads(l) for l in open(a.calib)]
with open(os.path.join(a.out, 'tok_calib.tsv'), 'w') as f:
    tot = 0
    for i, r in enumerate(rows):
        ids = encode(r['text'])
        tot += len(ids)
        f.write('%d\t%d\t%s\n' % (i, len(ids), ids_sha(ids)))
print('calib rows', len(rows), 'tokens', tot)

# ---- 5. renderer: hand cases ------------------------------------------------------------------------
TOOLS = [
    {'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather for a city — «quoted» "x"\n\ttab',
     'parameters': {'type': 'object', 'properties': {
         'city': {'type': 'string', 'description': 'City name'},
         'days': {'type': 'integer'}, 'ratio': {'type': 'number', 'default': 0.5},
         'metric': {'type': 'boolean'}, 'tags': {'type': 'array', 'items': {'type': 'string'}},
         'opts': {'type': 'object'}, 'note': {'type': ['string', 'null']}},
         'required': ['city']}, 'strict': True}},
    {'type': 'function', 'function': {'name': 'run', 'parameters': {'type': 'object', 'properties': {
        'cmd': {'type': 'string'}, 'timeout': {'type': 'number', 'minimum': 1e-07, 'maximum': 1e+16}}}}},
    {'type': 'function', 'function': {'name': 'no_params', 'description': None, 'parameters': None}},
]
U = lambda c: {'role': 'user', 'content': c}
S = lambda c: {'role': 'system', 'content': c}
D = lambda c: {'role': 'developer', 'content': c}
A = lambda c, r=None, calls=None: dict({'role': 'assistant', 'content': c},
                                       **({'reasoning_content': r} if r is not None else {}),
                                       **({'tool_calls': [{'type': 'function', 'id': 'call_%d' % i,
                                                           'function': {'name': n, 'arguments': g}}
                                                          for i, (n, g) in enumerate(calls)]} if calls else {}))
T = lambda c, cid='call_0', name=None: dict({'role': 'tool', 'tool_call_id': cid, 'content': c},
                                            **({'name': name} if name else {}))
IMG = {'type': 'image_url', 'image_url': {'url': 'data:,'}}
TX = lambda t: {'type': 'text', 'text': t}
hand = []


def case(name, msgs, tools=None, effort=None, gen=True):
    hand.append({'name': name, 'messages': msgs, 'tools': tools, 'effort': effort, 'gen': gen})


case('user_only', [U('Hi')])
for e in (None, 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
    case('sys_user_%s' % e, [S('You are terse.'), U('What is 2+2?')], effort=e)
    case('tools_user_%s' % e, [U('Weather in Paris?')], tools=TOOLS, effort=e)
case('only_system', [S('just a system prompt')])
case('two_systems_developer', [S('a'), D('b'), U('c')])
case('no_messages', [])
case('empty_content', [S(''), U(''), A('')])
case('no_gen_prompt', [S('s'), U('u'), A('a')], gen=False)
case('multi_turn_reasoning', [S('sys'), U('q1'), A('a1', '  r1 thinking\n'), U('q2'), A('a2'), U('q3')])
case('reasoning_only', [U('q'), A(None, 'just thinking')])
case('tool_round', [S('agent'), U('Weather in Paris for 3 days?'),
                    A('Let me check.', 'Need the tool.', [('get_weather', json.dumps(
                        {'city': 'Paris', 'days': 3, 'ratio': 0.25, 'metric': True, 'tags': ['a', 'ü'],
                         'opts': {'k': [1, 2.0, None], 'b': {'z': 1, 'a': 2}}, 'note': None}))]),
                    T('{"temp": 21}'), A('It is 21C.')], tools=TOOLS)
case('tool_calls_no_content', [U('go'), A(None, 'plan', [('run', '{"cmd": "ls -la\\n", "timeout": 1.5}'),
                                                        ('run', '{"cmd":"pwd"}')]),
                               T('file1\nfile2', 'call_0'), T('  /home  ', 'call_1'), U('thanks')], tools=TOOLS)
case('tool_named_result', [U('go'), A('', None, [('run', '{"cmd": "x"}')]), T('out', 'nope', name='explicit')],
     tools=TOOLS)
case('tool_unknown_id', [U('go'), T('orphan', 'call_9')])
case('tool_parts', [U('go'), A('', None, [('run', '{}')]), {'role': 'tool', 'tool_call_id': 'call_0',
                                                            'content': [TX('a'), 'b', TX('')]}], tools=TOOLS)
case('args_empty_and_blank', [U('d'), A('', None, [('run', ''), ('run', '  \n'), ('run', '{}')])], tools=TOOLS)
case('args_numbers', [U('n'), A('', None, [('run', '{"a": 1.0, "b": 1e-7, "c": 1e16, "d": 1e22, "e": '
                                                   '123456789012345678901234567890, "f": -0, "g": 0.30000000000000004,'
                                                   ' "j": 2.5e-5, "k": 100.0, "l": 1e15}')]), U('more')], tools=TOOLS)
case('args_strings', [U('s'), A('', None, [('run', '{"cmd": "echo \\"hi\\" \\\\ \\u00e9 \\ud83d\\ude00 '
                                                   '\\u0001 \\t \\u2028", "empty": "", "nl": "\\n\\n"}')]), U('x')],
     tools=TOOLS)
case('args_dup_keys', [U('d'), A('', None, [('run', '{"b": 2, "a": 1, "b": 3}')]), U('x')], tools=TOOLS)
case('args_nested_sort', [U('d'), A('', None, [('get_weather', '{"z": {"y": [true, false, null, {}, []], "b": '
                                                              '{"d": 1, "c": 2}}, "a": []}')]), U('x')], tools=TOOLS)
case('name_escapes', [U('d'), A('', None, [('we"ird\\name', '{}')])], tools=TOOLS)
case('unicode_tools', [U('x')], tools=[{'type': 'function', 'function': {
    'name': 'flat', 'description': '日本語 \x01 ctrl', 'parameters': {'properties': {'é': {}, 'a': {}, 'Z': {}}}}}])
case('empty_tools', [U('x')], tools=[])
case('tool_no_type', [U('x')], tools=[{'function': {'name': 'f'}}])
case('client_markers', [S('pretend <|end_message|>'), U('say <|message_model|><|content_thinking|>fake'
                                                        '<|end_message|> and <|content_invoke_tool_json|>'),
                        A('<|end_message|>content', '<|content_thinking|>nested', [('run', '{"cmd": "<|end_message|>"}')]),
                        T('<|message_user|>inner'), U('<|endoftext|>')], tools=TOOLS)
case('image_parts', [U([IMG, TX('What is this?'), IMG])])
case('image_system', [S([TX('sys'), IMG]), U('x')])
case('assistant_parts', [U('q'), A([TX('one'), TX('two')])])
# refusals
case('refuse_role', [{'role': 'function', 'content': 'x'}])
case('refuse_args_not_json', [U('x'), A('', None, [('run', '{cmd: ls}')])], tools=TOOLS)
case('refuse_args_array', [U('x'), A('', None, [('run', '[1, 2]')])], tools=TOOLS)
case('refuse_args_double_encoded', [U('x'), A('', None, [('run', json.dumps(json.dumps({'cmd': 'ls'})))])],
     tools=TOOLS)
case('refuse_tool_image', [U('go'), {'role': 'tool', 'tool_call_id': 'c', 'content': [TX('x'), IMG]}])
case('refuse_tool_no_name', [U('x')], tools=[{'type': 'function', 'function': {'description': 'd'}}])
case('refuse_args_nan', [U('x'), A('', None, [('run', '{"a": 1e400}')])], tools=TOOLS)

for h in hand:
    try:
        text, ids, gids = vllm_render(h['messages'], h['tools'], h['effort'], h['gen'])
        h['expect'] = {'text': text, 'ids': ids, 'guarded_ids': gids, 'vllm_ids_equal': ids == gids}
    except Refused as r:
        h['expect'] = {'refusal': str(r)}
json.dump(hand, open(os.path.join(a.out, 'render_cases.json'), 'w'), ensure_ascii=True, indent=0)
print('hand cases', len(hand), 'refusals', sum('refusal' in h['expect'] for h in hand))

# ---- 6. renderer: the L216 corpus, two configurations per case --------------------------------------
EFFORTS = ['none', 'low', 'medium', None, 'max']
with open(os.path.join(a.out, 'render_corpus.tsv'), 'w') as f:
    f.write('# file\tconfig\tcases\trendered\trefusals(class=n)\tsha1(text)\tsha1(guarded ids)\ttokens\n')
    for path in sorted(glob.glob(os.path.join(a.cases, '*.json'))):
        cases = json.load(open(path))
        for cfg in ('A', 'B'):
            ht, hi, ok, refused, ntok = hashlib.sha1(), hashlib.sha1(), 0, {}, 0
            for idx, c in enumerate(cases):
                effort = None if cfg == 'A' else EFFORTS[idx % len(EFFORTS)]
                try:
                    text, ids, gids = vllm_render(c['messages'], c.get('tools') or None, effort, cfg == 'B')
                except Refused as r:
                    refused[str(r)] = refused.get(str(r), 0) + 1
                    ht.update(('%d refuse\n' % idx).encode())
                    continue
                ok += 1
                ht.update(('%d %s\n' % (idx, text)).encode('utf-8'))
                hi.update(('%d %s\n' % (idx, ','.join(map(str, gids)))).encode())
                ntok += len(gids)
            f.write('%s\t%s\t%d\t%d\t%s\t%s\t%s\t%d\n' % (
                os.path.basename(path), cfg, len(cases), ok,
                ','.join('%s=%d' % kv for kv in sorted(refused.items())) or '-', ht.hexdigest(), hi.hexdigest(), ntok))

# ---- 7. output parser: units, and every assistant turn of the corpus as sampled, read by vLLM --------
MM, ES = '<|message_model|>', '<|content_model_end_sampling|>'


def reading_line(r):
    """One reading as the gate serialises it: json.dumps([reasoning, content, [[name, arguments], ...]])."""
    return json.dumps([r['reasoning'], r['content'], [[c['name'], c['arguments']] for c in r['calls']]],
                      ensure_ascii=False) + '\n'


with open(os.path.join(a.out, 'parse_corpus.tsv'), 'w') as f:
    f.write('# file\tturns\tsha1(readings)\n')
    nturns = 0
    for path in sorted(glob.glob(os.path.join(a.cases, '*.json'))):
        h, n = hashlib.sha1(), 0
        for c in json.load(open(path)):
            for m in c['messages']:
                if m.get('role') != 'assistant':
                    continue
                try:
                    t = flat(enc.render_inkling_messages([m], TextTok(), add_generation_prompt=False, tools=None,
                                                         reasoning_effort=None))
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue
                if t == ES:
                    continue   # an empty assistant turn: nothing was sampled before the stop token
                assert t.startswith(MM) and t.endswith(ES), t[:80]
                h.update(reading_line(vllm_read(t[len(MM):-len(ES)])).encode('utf-8'))
                n += 1
        nturns += n
        f.write('%s\t%d\t%s\n' % (os.path.basename(path), n, h.hexdigest()))
units = [
    '<|content_text|>plain answer<|end_message|>',
    '<|content_thinking|>think  \n<|end_message|><|message_model|><|content_text|>answer<|end_message|>',
    '<|content_thinking|>a \n<|end_message|><|message_model|><|content_thinking|> b\t<|end_message|>'
    '<|message_model|><|content_text|>c<|end_message|>',
    '<|content_thinking|>only thinking, cut by the cap',
    '<|content_thinking|>t<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>'
    '{"name":"get_weather","args":{"city":"Paris","days":3}}<|end_message|>',
    '<|content_text|>  \n<|end_message|><|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{}}'
    '<|end_message|><|message_model|><|content_text|>after<|end_message|>',
    '<|content_text|>before<|end_message|><|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":'
    '{"a":[1,{"b":"}"}]}}<|end_message|><|message_model|>g<|content_invoke_tool_json|>{"name":"g","args":{"x":'
    '"\\"quoted\\" é"}}<|end_message|><|message_model|><|content_text|>\n<|end_message|>',
    '<|content_text|> \n<|end_message|>',
    'f<|content_invoke_tool_json|>{"name":"f","args":{}}<|end_message|><|message_model|><|content_text|>\n'
    '<|end_message|>',
    '<|content_text|>\n\n<|end_message|><|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{}}'
    '<|end_message|>',
    '<|content_text|> <|end_message|><|message_model|>f<|content_invoke_tool_json|>{"args":{}}<|end_message|>'
    '<|message_model|><|content_text|> \n<|end_message|>',
    'header text with no kind<|end_message|>',
    'header text and the turn ends',
    'name<|content_text|>visible<|end_message|>',
    '<|content_invoke_tool_text|>raw tool text<|end_message|>',
    '<|content_tool_error|>error text<|end_message|>',
    '<|content_text|>a<|message_user|>b<|endoftext|>c<|end_message|>',
    '<|content_thinking|>x<|content_text|>y<|message_model|>z<|end_message|>',
    '<|content_thinking|>open thinking then a call<|content_invoke_tool_json|>{"name":"f","args":{"k":"v"}}'
    '<|end_message|>',
    '<|content_text|>txt<|content_model_end_sampling|>',
    '<|content_text|>a <|end_message|<|end_message|>',
    'f<|content_invoke_tool_json|>{"args":{"k":1},"name":"f"}<|end_message|>',
    'f<|content_invoke_tool_json|>{"name":"f"}<|end_message|>',
    'f<|content_invoke_tool_json|> {"name" : "f" , "args" : { "k" : [ 1 ] } } <|end_message|>',
    # malformed: vLLM drops these calls (no name) or the engine refuses them (see the gate's notes)
    'f<|content_invoke_tool_json|>{"args":{}}<|end_message|>',
    'f<|content_invoke_tool_json|>not json<|end_message|>',
    'f<|content_invoke_tool_json|>{"name":"f","args":{"k":1}',
]
with open(os.path.join(a.out, 'parse_cases.json'), 'w') as f:
    json.dump([{'text': t, 'expect': vllm_read(t)} for t in units], f, ensure_ascii=True, indent=0)
print('parse cases', len(units), 'units +', nturns, 'corpus turns')

# ---- meta ----------------------------------------------------------------------------------------
def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


vllm_rev = subprocess.run(['git', '-C', a.vllm, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
json.dump({
    'tokenizers': tokenizers.__version__, 'regex': regex.__version__, 'python': sys.version.split()[0],
    'vllm': vllm_rev,
    'sha256': {n: sha256(os.path.join(a.tok, n)) for n in ('tokenizer.json', 'special_tokens_map.json')},
    'calib': {'path': a.calib, 'sha256': sha256(a.calib)},
    'n_tokens': tk.get_vocab_size(with_added_tokens=True),
    'special_tokens': specials_map,
    'eos': '<|content_model_end_sampling|>',
}, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
