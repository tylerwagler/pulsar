#!/usr/bin/env python3
"""L331: render goldens for Nemotron 3.5 Super VL's images -- the checkpoint's chat_template.jinja through transformers'
apply_chat_template, then the processor's expansion (processing_nemotron_h_omni.py __call__: each `<image>` replaced,
in order, by `<img>` + n x `<image>` + `</img>`) at n = 1: the text pulsar's renderer writes (its image front expands
the one `<image>` into the tile's block afterwards).  The gate (tests/nemotron_chat_gate.cpp R3) renders the same
conversations and compares the text, the ids (HF's encode of the expected text) and the refusals.

    ~/Projects/AI/temp/l331/venv/bin/python -I tests/nemotron/gen_nemotron_image_renders.py \
        --tok /mnt/models/nemotron35-super-vl-bf16 --out tests/test-vectors/nemotron_h/image_render_cases.json
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

from tokenizers import Tokenizer
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--tok', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
tokdir = tempfile.mkdtemp(prefix='nemotron-tok-')
for f in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json'):
    shutil.copy(os.path.join(a.tok, f), tokdir)
sys.stdout = sys.stderr   # transformers prints to stdout
hf = AutoTokenizer.from_pretrained(tokdir)
tk = Tokenizer.from_file(os.path.join(a.tok, 'tokenizer.json'))
IMG = {"type": "image"}


def T(s):
    return {"type": "text", "text": s}


CALL = {"id": "c1", "type": "function", "function": {"name": "screenshot", "arguments": {}}}
CASES = [
    ("image only", [{"role": "user", "content": [IMG]}], "default"),
    ("text then image", [{"role": "user", "content": [T("What is in this picture?"), IMG]}], "default"),
    ("image then text", [{"role": "user", "content": [IMG, T("Describe it.")]}], "default"),
    ("text ending in a newline", [{"role": "user", "content": [T("Look:\n"), IMG, T(" and say.")]}], "default"),
    ("two images numbered", [{"role": "user", "content": [IMG, IMG, T("Which is red?")]}], "default"),
    ("text between two images", [{"role": "user", "content": [T("A"), IMG, T("B"), IMG, T("C")]}], "default"),
    ("an empty text part", [{"role": "user", "content": [T(""), IMG]}], "default"),
    ("numbering per message", [{"role": "user", "content": [IMG, T("one")]},
                               {"role": "assistant", "content": "ok"},
                               {"role": "user", "content": [T("and"), IMG]}], "default"),
    ("thinking off", [{"role": "user", "content": [T("Colour?"), IMG]}], "none"),
    ("medium effort", [{"role": "user", "content": [T("Colour?"), IMG]}], "medium"),
    ("system message image", [{"role": "system", "content": [T("Reference:"), IMG]},
                              {"role": "user", "content": "What is it?"}], "default"),
    ("tool result screenshot", [{"role": "user", "content": "Take a screenshot."},
                                {"role": "assistant", "content": "", "tool_calls": [CALL]},
                                {"role": "tool", "tool_call_id": "c1", "content": [T("Here:"), IMG]}], "default"),
]
REFUSALS = [
    ("text spelling <image> beside an image", [{"role": "user", "content": [T("see <image> here"), IMG]}],
     "image_marker_in_text"),
    ("an image in an assistant turn", [{"role": "user", "content": "hi"},
                                       {"role": "assistant", "content": [T("look"), IMG]},
                                       {"role": "user", "content": "ok"}], "image_position"),
]


def kwargs(effort):
    return {"enable_thinking": False} if effort == "none" else {"medium_effort": True} if effort == "medium" else {}


def expand(text):
    """The processor's loop: each `<image>` in order -> <img> + <|placeholder|> + </img>, then the placeholder back."""
    while "<image>" in text:
        text = text.replace("<image>", "<img><|placeholder|></img>", 1)
    return text.replace("<|placeholder|>", "<image>")


out = []
for name, msgs, effort in CASES:
    raw = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs(effort))
    text = expand(raw)
    out.append({"name": name, "messages": msgs, "effort": effort, "text": text,
                "ids": tk.encode(text, add_special_tokens=False).ids, "n_images": text.count("<image>")})
    print(f"{name}: {text.count('<image>')} image(s), {len(text)} B", file=sys.stderr)
for name, msgs, key in REFUSALS:
    out.append({"name": name, "messages": msgs, "effort": "default", "refusal": key})
json.dump({"generator": "tests/nemotron/gen_nemotron_image_renders.py", "cases": out}, open(a.out, "w"), indent=1)
