"""L288 acceptance curves, TensorFold side: TF 0.6.6's own single-stream engine on its weights, greedy, the
same rendered prompt ids pulsar's qwen_generate reads.  One line a round: drafts verified, drafts kept, each
draft's head confidence (incl. a stopped draft's, which TF does not verify).  Also writes the prompt ids.

  tf_trace.py MODEL_DIR OUTDIR N_TOKENS DEPTH:CONF[,DEPTH:CONF...] prompt...      (prompt in code, prose, story)
  The engine is built for the deepest config; every config runs on it (mtp_decode takes depth / confidence).
  Runs in TensorFold 0.6.6's venv (on the GB10 host: TensorFold's own venv, CC=gcc-13, TORCH_EXTENSIONS_DIR and
  TRITON_CACHE_DIR under ~/tf-run).  --render only writes the prompt ids (CPU).  TF_FORCE=DIR teacher-forces TF
  onto pulsar's reply (DIR/ours-td305-<prompt>-d6-off.ids, qwen_generate's out file) so both drafters are graded on
  the same tokens.  Pulsar's side: tests/qwen_generate with QWEN_SPEC=1 QWEN_SPEC_TRACE=... (PULSAR_SPEC_TAU=off
  for the unstopped chain); curves.py and policy.py read both traces.
"""
import json, struct, sys, time
from pathlib import Path

model_dir, out, n_tok = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
configs = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in sys.argv[4].split(",")]
names = sys.argv[5:]
out.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "serve-bench"))
from serve_matrix import PROMPTS, story_prompt   # the serving bench's prompt texts (the one copy)

from tokenizers import Tokenizer
from tensorfold.cuda.chat_template import ChatTemplate
tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
template = ChatTemplate(Path(model_dir))


def render(name):
    """As TF's CUDA server renders: chat through the checkpoint's template (thinking off), raw as given."""
    kind, text = PROMPTS[name]
    if kind == "raw":
        return [int(t) for t in tok.encode(text if text is not None else story_prompt(), add_special_tokens=True).ids]
    s = template.render([{"role": "user", "content": text}], tools=None, enable_thinking=False)
    return [int(t) for t in tok.encode(s, add_special_tokens=False).ids]


if len(names) and names[0] == "--render":
    for name in names[1:]:
        ids = render(name)
        (out / f"ids-{name}.bin").write_bytes(struct.pack(f"<{len(ids)}i", *ids))
        print(name, len(ids), ids[:12], ids[-8:])
    sys.exit(0)

from tensorfold.families.qwen4_exp.cuda import decode as D
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.engine.exact_sampling import Sampling

eng = FlashNextEngine(Path(model_dir), depth=max(d for d, _ in configs), confidence=configs[0][1])
e = eng.e
ROUNDS = []
ORIG = D.draft


def draft_rec(e_, streams, next_tokens, position, count, sampling, confidence=0.0):
    probs = []
    orig = e_.sample_draft

    def sd(logits, pos, s):
        t, p = orig(logits, pos, s)
        probs.append(p)
        return t, p

    e_.sample_draft = sd
    try:
        ds = ORIG(e_, streams, next_tokens, position, count, sampling, confidence)
    finally:
        del e_.sample_draft
    ROUNDS.append((len(ds), probs))
    return ds


D.draft = draft_rec
FORCED = None          # TF_FORCE=DIR: the text is DIR/ours-td305-<prompt>-d6-off.ids (prompt + pulsar's greedy reply);
                       # the target's picks are replaced by that text (teacher forcing), so TF's drafts are graded on
                       # the exact token sequence pulsar's were
import os
force_dir = os.environ.get("TF_FORCE")
ORIG_SAMPLE = type(e).sample


def forced_sample(self, logits, positions, sampling, *, draft=False, gathered=True):
    got = ORIG_SAMPLE(self, logits, positions, sampling, draft=draft, gathered=gathered)
    if FORCED is None or draft:
        return got
    return [FORCED[p] if p < len(FORCED) else g for p, g in zip(positions, got)]


type(e).sample = forced_sample
greedy = Sampling(seed=0, temperature=0.0)
for name in names:
    ids = render(name)
    (out / f"ids-{name}.bin").write_bytes(struct.pack(f"<{len(ids)}i", *ids))
    if force_dir:
        raw = (Path(force_dir) / f"ours-td305-{name}-d6-off.ids").read_bytes()
        FORCED = list(struct.unpack(f"<{len(raw) // 4}i", raw))
        assert FORCED[:len(ids)] == ids, "the forced text's prompt is not this prompt"
    for depth, conf in configs:
        for rep in range(2):          # rep 0 warms; rep 1 is recorded
            ROUNDS.clear()
            e.reset()
            first = D.prefill(e, ids, greedy)
            res = D.mtp_decode(e, first, n_tok, greedy, depth=depth, confidence=conf, stop_eos=True)
        tag = f"tf-{name}-d{depth}-c{conf:g}" + ("-forced" if force_dir else "")
        with open(out / f"{tag}.trace", "w") as f:
            for i, keep in enumerate(res.keeps):
                nd, probs = ROUNDS[i]
                f.write(f"{nd} {keep - 1} " + " ".join(f"{p:.6f}" for p in probs) + "\n")
        (out / f"{tag}.tokens.json").write_text(json.dumps(list(res.tokens)))
        n = len(res.tokens)
        print(f"[tf] {tag}: prompt {len(ids)} tok, {n} tokens, {res.rounds} rounds, {(n - 1) / max(1, res.rounds):.3f} "
              f"tok/round, {res.seconds / max(1, res.rounds) * 1e3:.2f} ms/round, {res.tokens_per_second:.1f} tok/s, "
              f"drafted {res.drafted} accepted {res.accepted}", flush=True)
eng.close()
