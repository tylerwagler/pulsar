#!/usr/bin/env python3
"""vLLM goldens for the Gemma 4 output parser (L338).  The gate is tests/gemma4_chat_gate.cpp (P1).

  parse_cases.json  sampled turns, each {start_reasoning, text, expect}: vLLM's reading of the turn -- its
                    StreamingParserEngine (vllm/parser/engine, loaded without vLLM's package __init__) in
                    vllm/parser/gemma4.py's configuration, assembled as ParserEngine._single_pass_parse does, with
                    Gemma4Parser's "thought\\n" strip, its _gemma4_arg_converter and the tool schema's coercion
                    (ParserEngine._fix_arg_types: vllm/tool_parsers/utils.py coerce_to_schema_type).

usage: gen_parse_goldens.py --tok DIR --vllm DIR --out tests/test-vectors/gemma4
"""
import argparse, ast, functools, json, math, os, sys, types
from collections.abc import Sequence

from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--tok', required=True)
ap.add_argument('--vllm', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
tk = Tokenizer.from_file(os.path.join(a.tok, 'tokenizer.json'))

for name, sub in (('vllm', 'vllm'), ('vllm.parser', 'vllm/parser')):
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(a.vllm, sub)]
    sys.modules[name] = mod
from vllm.parser.engine.events import EventType  # noqa: E402
from vllm.parser.engine.parser_engine_config import (ArgValueExtender, ParserEngineConfig, ParserState,  # noqa: E402
                                                     Transition)
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine  # noqa: E402


def defs_from(path, names, ns):
    tree = ast.parse(open(path).read())
    body = [n for n in tree.body
            if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))
            or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in names)]
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), ns)
    return ns


class _Log:
    def warning(self, *a, **k):
        pass


g4 = defs_from(os.path.join(a.vllm, 'vllm/parser/gemma4.py'),
               {'CHANNEL_START', 'CHANNEL_END', 'TOOL_CALL_START', 'TOOL_CALL_END', 'STRING_DELIM', '_DELIM_LEN',
                '_PARTIAL_DELIM_SUFFIXES', '_strip_partial_delim', '_parse_gemma4_args', '_parse_gemma4_fields',
                '_parse_gemma4_array', '_drop_call_closer', '_gemma4_arg_converter', '_gemma4_arg_value_extender',
                'gemma4_config', '_GEMMA4_THOUGHT_PREFIX'},
               {'json': json, 'functools': functools, 'EventType': EventType, 'ParserEngineConfig': ParserEngineConfig,
                'ParserState': ParserState, 'Transition': Transition, 'ArgValueExtender': ArgValueExtender,
                'logger': _Log(), 'Sequence': Sequence, '__annotations__': {}})
tu = defs_from(os.path.join(a.vllm, 'vllm/tool_parsers/utils.py'),
               {'extract_types_from_schema', '_TYPE_ALIASES', 'coerce_to_schema_type', '_is_json_finite'},
               {'json': json, 'math': math, 'Any': object})


class SpecialsOnly:
    all_special_tokens = [t.content for t in tk.get_added_tokens_decoder().values() if t.special]
    all_special_ids = [i for i, t in tk.get_added_tokens_decoder().items() if t.special]

    @staticmethod
    def get_vocab():
        return tk.get_vocab(with_added_tokens=True)


def find_tool_properties(tools, name):
    for t in tools or []:
        fn = t.get('function', t)
        if fn.get('name') == name:
            return (fn.get('parameters') or {}).get('properties', {})
    return {}


def coerce(value, schema):
    """ParserEngine._coerce_value."""
    if isinstance(value, str):
        return tu['coerce_to_schema_type'](value, tu['extract_types_from_schema'](schema))
    if isinstance(value, dict):
        props = schema.get('properties')
        if isinstance(props, dict):
            return {k: coerce(v, props[k]) if isinstance(props.get(k), dict) else v for k, v in value.items()}
        return value
    if isinstance(value, list):
        items = schema.get('items')
        if isinstance(items, dict):
            return [coerce(v, items) for v in value]
        return value
    as_str = json.dumps(value, ensure_ascii=False)
    c = tu['coerce_to_schema_type'](as_str, tu['extract_types_from_schema'](schema))
    return c if type(c) is not type(value) or c != value else value


def vllm_read(text, start_reasoning, tools):
    eng = StreamingParserEngine(g4['gemma4_config'](thinking=start_reasoning), SpecialsOnly())
    if start_reasoning:
        eng.reset(initial_state=ParserState.REASONING)
    events = eng.feed(text, []) + eng.finish()
    reasoning, content, deferred, slots, seen_tool = [], [], [], [], False
    for e in events:
        if e.type == EventType.TEXT_CHUNK:
            (deferred if seen_tool else content).append(e.value)
        elif e.type == EventType.REASONING_CHUNK:
            reasoning.append(e.value)
        elif e.type == EventType.TOOL_CALL_START:
            seen_tool = True
            while len(slots) <= e.tool_index:
                slots.append(['', ''])
        elif e.type == EventType.TOOL_NAME:
            slots[e.tool_index][0] += e.value
        elif e.type == EventType.ARG_VALUE_CHUNK:
            slots[e.tool_index][1] += e.value
    calls = []
    for name, raw in slots:
        name = name.strip()
        if not name:
            continue
        args = json.loads(g4['_gemma4_arg_converter'](raw, False))
        props = find_tool_properties(tools, name)
        if props:
            args = {k: coerce(v, props[k]) if isinstance(props.get(k), dict) else v for k, v in args.items()}
        calls.append({'name': name, 'arguments': args})
    c = ''.join(content)
    if not calls:
        c += ''.join(deferred)
    if calls:
        c = c.strip()
    r = ''.join(reasoning).rstrip()
    if r.startswith('thought\n'):
        r = r[len('thought\n'):]
    elif 'thought\n'.startswith(r):
        r = ''
    return {'reasoning': r, 'content': c, 'calls': calls}


TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {
    'city': {'type': 'string'}, 'days': {'type': 'integer'}, 'metric': {'type': 'boolean'}, 'zip': {'type': 'string'},
    'tags': {'type': 'array', 'items': {'type': 'string'}}, 'opts': {'type': 'object'}, 'ratio': {'type': 'number'}}}}},
         {'type': 'function', 'function': {'name': 'run', 'parameters': {'type': 'object', 'properties': {
             'cmd': {'type': 'string'}}}}}]
Q = '<|"|>'
units = [
    (False, 'Just content.  '),
    (False, '<|channel>thought\nlet me think\n<channel|>The answer.'),
    (False, '<|channel>thought\nr<channel|>\n\nAnswer\n'),
    (False, '<|channel>thought<channel|>Bare label.'),
    (False, '<|channel>no label here<channel|>x'),
    (False, '<|channel>thought\n<|channel>repeated open<channel|>a'),
    (False, 'answer<channel|> tail'),
    (True, 'continued reasoning<channel|>final'),
    (True, 'thought\nlabelled again<channel|>final'),
    (False, '<|tool_call>call:get_weather{city:' + Q + 'Paris' + Q + ',days:3,metric:true}<tool_call|>'),
    (False, 'Let me check.<|tool_call>call:get_weather{city:' + Q + 'Oslo' + Q + '}<tool_call|>'),
    (False, '<|channel>thought\nneed the tool<|tool_call>call:run{cmd:' + Q + 'ls -la' + Q + '}<tool_call|>'),
    (False, '<|tool_call>call:run{cmd:' + Q + 'a' + Q + '}<tool_call|><|tool_call>call:run{cmd:' + Q + 'b' + Q +
     '}<tool_call|>'),
    (False, '<|tool_call>call:get_weather{tags:[' + Q + 'x' + Q + ',' + Q + 'y' + Q + '],opts:{k:1,deep:{z:' + Q + 'q' +
     Q + '}},ratio:2.5}<tool_call|>'),
    (False, '<|tool_call>call:get_weather{zip:12345}<tool_call|>'),
    (False, '<|tool_call>call:unknown_fn{n:42,f:1.5,b:false,z:null,s:' + Q + 'v' + Q + '}<tool_call|>'),
    (False, '<|tool_call>call:run{}<tool_call|>'),
    (False, '<|tool_call>call:run<tool_call|>'),
    (False, '<|tool_call>:run{cmd:' + Q + 'bare colon' + Q + '}<tool_call|>'),
    (False, '<|tool_call>call:run{cmd:' + Q + 'has } brace and , comma' + Q + '}<tool_call|>'),
    (False, '<|tool_call>call:run{cmd:' + Q + 'multi\nline\n' + Q + '}<tool_call|> trailing words'),
    (False, '  \n<|tool_call>call:run{cmd:' + Q + 'x' + Q + '}<tool_call|>'),
    (False, 'cut <|tool_call>call:run{cmd:' + Q + 'unterminated'),
    (False, 'text with <bos> and <|turn> dropped'),
    (False, '<|channel>thought\n'),
]
cases = [{'start_reasoning': sr, 'text': t, 'tools': TOOLS, 'expect': vllm_read(t, sr, TOOLS)} for sr, t in units]
json.dump(cases, open(os.path.join(a.out, 'parse_cases.json'), 'w'), ensure_ascii=True, indent=0)
print('parse cases', len(cases))
for c in cases:
    print(json.dumps(c['text'])[:70], '->', json.dumps(c['expect'])[:160])
