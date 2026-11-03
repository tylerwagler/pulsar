#!/usr/bin/env python3
"""HF goldens for the Gemma 4 renderer and its span map (L338).  The gate is tests/gemma4_chat_gate.cpp (R1, R2).

  * RENDERS: transformers' apply_chat_template with the checkpoint's own chat_template.jinja (enable_thinking,
    add_generation_prompt as the template takes them).
  * CLIENT SPANS (the L223 map) of the messages: the conversation rendered a second time with every message-level
    client string wrapped in two private-use sentinels (U+E000 / U+E001) -- contents (wrapped already trimmed: the
    template trims them), reasoning, call names, argument keys and string values (nested too), tool-result bodies and
    names -- and the sentinel positions ARE the spans.  The tools block's spans are not derived (the gate compares the
    spans outside it); a case whose wrapped render does not unwrap to the plain one records no spans.
  * GUARDED TOKENS: when no client string spells an added token they equal HF's encode of the whole text; with message
    spans (and no marker in the tools) they come from tokenizers' own components -- the text cut at the added tokens
    outside every span, each run through the normaliser, the pre-tokenizer and the BPE model; otherwise none.

  render_cases.json   hand cases: the template's edges, each {name, messages, tools, thinking, gen, expect}
  render_corpus.tsv   the L216 corpus, two configurations per file: A (thinking off, no generation prompt), B (thinking
                      on for even cases, the generation prompt): sha1 of the texts and of the ids

usage: gen_render_goldens.py --tok DIR --cases DIR --out tests/test-vectors/gemma4
       (DIR holds tokenizer.json, tokenizer_config.json, chat_template.jinja; the versions go to render_meta.json)
"""
import argparse, glob, hashlib, json, os, shutil, tempfile

import transformers
from tokenizers import Tokenizer
from transformers import AutoTokenizer

O, C = '', ''
ap = argparse.ArgumentParser()
ap.add_argument('--tok', required=True)
ap.add_argument('--cases', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
FILES = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')
tokdir = tempfile.mkdtemp(prefix='gemma4-tok-')
for f in FILES:
    shutil.copy(os.path.join(a.tok, f), tokdir)
hf = AutoTokenizer.from_pretrained(tokdir)
tk = Tokenizer.from_file(os.path.join(a.tok, 'tokenizer.json'))
added = sorted(((t.content, i) for i, t in tk.get_added_tokens_decoder().items()), key=lambda x: -len(x[0]))
added_first = {t[0] for t, _ in added}
added_set = {t for t, _ in added}


def sha(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def ids_sha(ids):
    return sha(''.join('%d,' % i for i in ids))


def encode(text):
    return tk.encode(text, add_special_tokens=False).ids


def plain(seg):
    out = []
    if tk.normalizer is not None:
        seg = tk.normalizer.normalize_str(seg)
    for piece, _ in tk.pre_tokenizer.pre_tokenize_str(seg):
        out += [t.id for t in tk.model.tokenize(piece)]
    return out


def guarded(text, spans):
    b = text.encode('utf-8')
    out, run, pos, si = [], 0, 0, 0
    while pos < len(b):
        hit = None
        if chr(b[pos]) in added_first:
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


def spells_marker(s):
    return isinstance(s, str) and any(t in s for t in added_set)


class Refused(Exception):
    pass


class NoSpans(Exception):
    pass


def wrap_value(v, w):
    if isinstance(v, str):
        return w(v)
    if isinstance(v, dict):
        return {w(k): wrap_value(x, w) for k, x in v.items()}
    if isinstance(v, list):
        return [wrap_value(x, w) for x in v]
    return v


def api_to_hf(messages, wrap):
    """API messages (arguments as JSON text) -> what the template takes (arguments decoded ONCE)."""
    def w(s):
        return O + s + C if wrap and isinstance(s, str) and s else s
    out = []
    for m in messages:
        role = m['role']
        content = m.get('content')
        if isinstance(content, list):
            raise Refused('content_parts')
        n = {'role': role}
        if not isinstance(content, str) or not wrap:
            n['content'] = content
        elif role == 'tool':
            n['content'] = w(content)          # a tool body is written as sent
        elif role == 'assistant' and ('<|channel>' in content or '<channel|>' in content):
            raise NoSpans()                    # strip_thinking cuts inside the client's text: no spans derived
        else:
            n['content'] = w(content.strip())  # the template trims every other content
        r = m.get('reasoning_content')
        if r is not None:
            n['reasoning_content'] = w(r)
        if m.get('tool_calls'):
            calls = []
            for tc in m['tool_calls']:
                fn = tc.get('function', tc)
                if not fn.get('name'):
                    raise Refused('tool_name_missing')
                args = fn.get('arguments')
                if args is None or args == '':
                    raise Refused('tool_arguments_missing')
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        raise Refused('tool_arguments_not_json')
                if args is not None and not isinstance(args, dict):
                    raise Refused('tool_arguments_not_object')
                call = {'type': 'function', 'function': {'name': w(fn['name']), 'arguments': wrap_value(args, w) if wrap else args}}
                if tc.get('id') is not None:
                    call['id'] = tc['id']
                calls.append(call)
            n['tool_calls'] = calls
        if role == 'tool':
            if m.get('tool_call_id') is not None:
                n['tool_call_id'] = m['tool_call_id']
            if m.get('name'):
                n['name'] = w(m['name'])
        out.append(n)
    return out


def check_tools(tools):
    if tools is None:
        return
    if not isinstance(tools, list):
        raise Refused('tools_not_array')
    for t in tools:
        if not isinstance(t, dict) or not isinstance(t.get('function'), dict):
            raise Refused('tool_function_missing')
        p = t['function'].get('parameters')
        if isinstance(p, dict) and p.get('properties') and not isinstance(p['properties'], dict):
            raise Refused('tool_schema_invalid')


def hf_render(messages, tools, thinking, gen):
    """(text, spans or None) or Refused(key)."""
    if not messages:
        raise Refused('no_messages')
    check_tools(tools)
    k = {'add_generation_prompt': gen, 'enable_thinking': thinking}
    text = hf.apply_chat_template(api_to_hf(messages, False), tools=tools, tokenize=False, **k)
    try:
        wrapped = hf.apply_chat_template(api_to_hf(messages, True), tools=tools, tokenize=False, **k)
    except NoSpans:
        return text, None
    assert O not in text and C not in text, 'sentinel collides with corpus text'
    out, spans, lo, nbytes, ok = [], [], None, 0, True
    for ch in wrapped:
        if ch == O:
            ok &= lo is None
            lo = nbytes
        elif ch == C:
            ok &= lo is not None
            if lo is not None and nbytes > lo:
                if spans and spans[-1][1] == lo:
                    spans[-1][1] = nbytes
                else:
                    spans.append([lo, nbytes])
            lo = None
        else:
            out.append(ch)
            nbytes += len(ch.encode('utf-8'))
    if not ok or lo is not None or ''.join(out) != text:
        spans = None
    return text, spans


def client_strings(messages):
    out = []
    for m in messages:
        for key in ('content', 'reasoning_content', 'name'):
            if isinstance(m.get(key), str):
                out.append(m[key])
        for tc in m.get('tool_calls') or []:
            fn = tc.get('function', tc)
            out += [fn.get('name') or '', fn.get('arguments') if isinstance(fn.get('arguments'), str) else '']
    return out


def expected_ids(text, spans, messages, tools):
    """(ids, kind): 'hf' (no client string spells a marker), 'guarded' (message spans, no marker in the tools), or None."""
    tools_s = json.dumps(tools, ensure_ascii=False) if tools else ''
    if spells_marker(tools_s):
        return None, None
    if not any(spells_marker(s) for s in client_strings(messages)):
        return encode(text), 'hf'
    if spans is not None:
        return guarded(text, spans), 'guarded'
    return None, None


# ---- 1. hand cases: the template's edges ------------------------------------------------------------------------
WEATHER = [{'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'Get the current weather for a city.',
    'parameters': {'type': 'object', 'properties': {
        'city': {'type': 'string', 'description': 'City name'},
        'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
        'days': {'type': 'integer', 'description': 'Forecast days', 'nullable': True},
        'tags': {'type': 'array', 'items': {'type': 'string'}},
        'opts': {'type': 'object', 'properties': {'Zeta': {'type': 'boolean'}, 'alpha': {'type': 'number'}},
                 'required': ['alpha']},
        'Mixed': {'type': 'object', 'description': 'free-form', 'extra_key': 5}},
        'required': ['city']}}}]
RICH = [{'type': 'function', 'function': {
    'name': 'search', 'description': 'Search.',
    'parameters': {'type': 'object', 'properties': {
        'q': {'type': 'string'},
        'filters': {'type': 'array', 'items': {'type': 'object', 'properties': {'k': {'type': 'string'}},
                                               'required': ['k'], 'description': None}},
        'modes': {'type': 'array', 'items': {'type': ['string', 'null'], 'enum': ['a', 'b']}}}},
    'response': {'description': 'hits', 'type': 'object'}}},
    {'type': 'function', 'function': {'name': 'noargs', 'description': 'No parameters.'}}]


def U(t):
    return {'role': 'user', 'content': t}


def A(t, reasoning=None, calls=None):
    m = {'role': 'assistant', 'content': t}
    if reasoning is not None:
        m['reasoning_content'] = reasoning
    if calls:
        m['tool_calls'] = [{'id': cid, 'type': 'function', 'function': {'name': n, 'arguments': args}}
                           for cid, n, args in calls]
    return m


def T(t, cid=None, name=None):
    m = {'role': 'tool', 'content': t}
    if cid is not None:
        m['tool_call_id'] = cid
    if name is not None:
        m['name'] = name
    return m


hand = []


def case(name, msgs, tools=None, thinking=False, gen=True):
    try:
        text, spans = hf_render(msgs, tools, thinking, gen)
        ids, kind = expected_ids(text, spans, msgs, tools)
        expect = {'text': text, 'spans': spans}
        if ids is not None:
            expect['ids'], expect['ids_kind'] = ids, kind
    except Refused as r:
        expect = {'refusal': str(r)}
    hand.append({'name': name, 'messages': msgs, 'tools': tools, 'thinking': thinking, 'gen': gen, 'expect': expect})


W = json.dumps
case('plain', [U('Hello there')])
case('plain_nogen', [U('Hello there')], gen=False)
case('plain_thinking', [U('Hello there')], thinking=True)
case('system', [{'role': 'system', 'content': '  Be terse.  '}, U('  hi  ')])
case('developer', [{'role': 'developer', 'content': 'Dev rules.'}, U('hi')], thinking=True)
case('system_later', [U('a'), A('b'), {'role': 'system', 'content': 'late system'}, U('c')])
case('multi_turn', [U('a'), A(' b '), U('c'), A('d'), U('e')])
case('assistant_last_nogen', [U('a'), A('b')], gen=False)
case('assistant_last_gen', [U('a'), A('b')])
case('reasoning_after_last_user', [U('a'), A('ans', reasoning='thinking hard')], gen=False, thinking=True)
case('reasoning_before_last_user', [U('a'), A('ans', reasoning='old thought'), U('b')], thinking=True)
case('strip_thinking', [U('a'), A('pre<|channel>thought\nhidden<channel|>post'), U('b')])
case('strip_thinking_unclosed', [U('a'), A('x<|channel>never closed'), U('b')])
case('tools_block', [U('weather?')], tools=WEATHER)
case('tools_block_thinking', [{'role': 'system', 'content': 'sys'}, U('weather?')], tools=WEATHER, thinking=True)
case('tools_rich', [U('find')], tools=RICH)
case('call_and_result', [U('weather in Paris?'),
                         A(None, calls=[('c1', 'get_weather', W({'city': 'Paris', 'unit': 'celsius', 'days': 3,
                                                                   'opts': {'Zeta': True, 'alpha': 1.5}, 'tags': ['x', None]}))]),
                         T('{"temp": 21}', cid='c1')], tools=WEATHER)
case('call_and_result_thinking', [U('weather?'), A('', reasoning='need tool', calls=[('c1', 'get_weather', W({'city': 'Oslo'}))]),
                                  T('cold', cid='c1')], tools=WEATHER, thinking=True)
case('call_result_then_answer', [U('w?'), A(None, calls=[('c1', 'get_weather', W({'city': 'Rome'}))]), T('sunny', cid='c1'),
                                 A('It is sunny.'), U('thanks')], tools=WEATHER)
case('two_calls_two_results', [U('w?'), A('checking', calls=[('a', 'get_weather', W({'city': 'A'})),
                                                            ('b', 'get_weather', W({'city': 'B'}))]),
                               T('ra', cid='a'), T('rb', cid='b')], tools=WEATHER)
case('result_name_fallback', [U('w?'), A(None, calls=[('a', 'get_weather', W({'city': 'A'}))]), T('r', cid='zz', name='other')])
case('result_unknown', [U('w?'), A(None, calls=[('a', 'get_weather', W({'city': 'A'}))]), T('r', cid='zz')])
case('result_null_body', [U('w?'), A(None, calls=[('a', 'get_weather', W({'city': 'A'}))]), T(None, cid='a')])
case('call_no_result_gen', [U('w?'), A(None, calls=[('a', 'get_weather', W({'city': 'A'}))])])
case('call_null_args', [U('w?'), A(None, calls=[('a', 'noargs', 'null')])], tools=RICH)
case('orphan_tool', [U('x'), T('orphan', cid='q'), U('y')])
case('consecutive_assistants', [U('x'), A('one'), A('two'), U('y')])
case('args_nested', [U('x'), A(None, calls=[('a', 'f', W({'b': [1, 2.0, -3e-7, True, None, 'q'], 'A': {'Z': 'z', 'a': {}},
                                                       'c': 10 ** 20}))]), T('ok', cid='a')])
case('client_markers', [U('say <|turn> and <turn|> and <|"|>'), A('ok <|tool_call>'), U('<bos>')])
case('unicode', [U('日本語 🙂 é́ ▁ tab\tend'), A('réponse')])
case('refuse_args_string', [U('x'), A(None, calls=[('a', 'f', W(W({'k': 1})))])])
case('refuse_args_list', [U('x'), A(None, calls=[('a', 'f', W([1, 2]))])])
case('refuse_args_not_json', [U('x'), A(None, calls=[('a', 'f', '{bad')])])
case('refuse_args_missing', [U('x'), A(None, calls=[('a', 'f', '')])])
case('refuse_tool_no_function', [U('x')], tools=[{'type': 'function', 'name': 'flat'}])
case('refuse_empty', [])
json.dump(hand, open(os.path.join(a.out, 'render_cases.json'), 'w'), ensure_ascii=True, indent=0)
print('hand cases', len(hand), 'refusals', sum('refusal' in h['expect'] for h in hand))

# ---- 2. the L216 corpus, two configurations per case ------------------------------------------------------------
with open(os.path.join(a.out, 'render_corpus.tsv'), 'w') as f:
    f.write('# file\tconfig\tcases\trendered\trefusals(key=n)\tsha1(text)\tsha1(ids)\ttokens\tids_compared\t'
            'cases without derivable ids\n')
    for path in sorted(glob.glob(os.path.join(a.cases, '*.json'))):
        cases = json.load(open(path))
        for cfg in ('A', 'B'):
            ht, hi, ok, refused, ntok, ncmp, nodiv = hashlib.sha1(), hashlib.sha1(), 0, {}, 0, 0, []
            for idx, c in enumerate(cases):
                thinking = cfg == 'B' and idx % 2 == 0
                tools = c.get('tools') or None
                try:
                    text, spans = hf_render(c['messages'], tools, thinking, cfg == 'B')
                except Refused as r:
                    refused[str(r)] = refused.get(str(r), 0) + 1
                    ht.update(('%d refuse %s\n' % (idx, r)).encode())
                    continue
                ok += 1
                ht.update(('%d %s\n' % (idx, sha(text))).encode())
                ids, kind = expected_ids(text, spans, c['messages'], tools)
                if ids is None:
                    nodiv.append(str(idx))
                    continue
                ncmp += 1
                ntok += len(ids)
                hi.update(('%d %s\n' % (idx, ids_sha(ids))).encode())
            f.write('%s\t%s\t%d\t%d\t%s\t%s\t%s\t%d\t%d\t%s\n' % (
                os.path.basename(path), cfg, len(cases), ok,
                ','.join('%s=%d' % kv for kv in sorted(refused.items())) or '-', ht.hexdigest(), hi.hexdigest(),
                ntok, ncmp, ','.join(nodiv) or '-'))
        print(os.path.basename(path), flush=True)

meta = {'transformers': transformers.__version__,
        'chat_template.jinja sha256': hashlib.sha256(open(os.path.join(a.tok, 'chat_template.jinja'), 'rb').read()).hexdigest(),
        'cases': os.path.abspath(a.cases)}
json.dump(meta, open(os.path.join(a.out, 'render_meta.json'), 'w'), indent=1)
print(json.dumps(meta, indent=1))
