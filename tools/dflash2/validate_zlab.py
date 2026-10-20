"""Transcription check: replay.py's Drafter vs z-lab/dflash model.py's DFlash2DraftModel (the literal reference),
both f32, same synthetic context (n > window, to exercise the sliding window), several anchors and block sizes."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pylib"))
sys.path.insert(0, "/home/claude/Projects/AI/upstream/dflash")
sys.path.insert(0, HERE)
import torch  # noqa: E402
from transformers import Qwen3Config  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from dflash.model import DFlash2DraftModel  # noqa: E402
import replay  # noqa: E402

torch.manual_seed(0)
cfg = Qwen3Config(**json.load(open(os.path.join(HERE, "drafter_config.json"))))
cfg._attn_implementation = "sdpa"
m = DFlash2DraftModel(cfg).float().eval()
sd = load_file(os.path.join(HERE, "drafter.safetensors"))
sd = {("candidate_selector." + k.split(".")[1] + ".weight" if k.endswith("_codebook") else k): v.float() for k, v in sd.items()}
missing, unexpected = m.load_state_dict(sd, strict=False)
print("missing", missing, "unexpected", unexpected)
D = replay.Drafter()
n = 2300
taps = torch.randn(n, 12800) * 0.5
kv = D.context(taps)
emb = D.embed


class Head(torch.nn.Module):
    def forward(self, x):
        return x @ D.head.T


worst = 0.0
for p, B in [(40, 8), (1000, 8), (2100, 8), (2200, 4), (2290, 6)]:
    ids = [int(torch.randint(0, 200000, (1,)))] + [replay.MASK] * (B - 1)
    with torch.no_grad():
        ref = m(position_ids=torch.arange(0, p + B)[None], noise_embedding=emb[torch.tensor(ids)][None],
                target_hidden=taps[:p][None], past_key_values=None, use_cache=False)[0]
        ours = D.block(kv, p, ids)
        rel = float((ref - ours).norm() / ref.norm())
        worst = max(worst, rel)
        rt, _, _ = m.propose(ref[1:][None], torch.tensor([ids[0]]), Head(), 0.0)
        ot = D.propose(ours[1:], ids[0])["sel"]
        print(f"p {p} B {B}: rel diff {rel:.2e}; selector chain z-lab {rt[0].tolist()} ours {ot} same {rt[0].tolist() == ot}")
print("WORST rel", worst)
