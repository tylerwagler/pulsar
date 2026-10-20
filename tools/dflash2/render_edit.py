"""Render the serve_matrix edit prompt to ids (chat, enable_thinking False), after checking the renderer reproduces
~/qaccept's code ids."""
import os
import sys

import jinja2
import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import serve_matrix as sm  # noqa: E402

tk = Tokenizer.from_file(os.path.join(HERE, "tokenizer.json"))
tpl = jinja2.Environment(loader=jinja2.FileSystemLoader(HERE)).get_template("chat_template.jinja")


def render(text):
    return tpl.render(messages=[{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)


for name in ("code", "prose"):
    ids = np.fromfile(os.path.join(HERE, f"ids/ids-{name}.bin"), dtype=np.int32).tolist()
    r = render(sm.PROMPTS[name][1])
    e = tk.encode(r, add_special_tokens=False).ids
    print(name, "match:", e == ids, len(e), len(ids))
    if e != ids:
        print(repr(tk.decode(ids, skip_special_tokens=False)))
        print(repr(r))
        sys.exit(1)
er = render(sm.edit_prompt())
ee = tk.encode(er, add_special_tokens=False).ids
print("edit tokens", len(ee))
np.array(ee, dtype=np.int32).tofile(os.path.join(HERE, "ids/ids-edit.bin"))
