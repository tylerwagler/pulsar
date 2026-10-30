#!/usr/bin/env python3
"""Real-layer fixture for the Mamba-2 block (L296 N2): tests/mamba2_gate --fixture DIR.

    python3 tests/mamba2_mini_fixture.py CHECKPOINT DIR [--rows T]

CHECKPOINT is Nemotron 3.5 Super VL in BF16 (/path/to/models/nemotron35-super-vl-bf16).  On the CPU, in fp32, from the
safetensors through the index (no model is built):

  - T real tokens (the checkpoint's README through its own tokenizer) -> embeddings;
  - HF blocks 0 .. 4 (Mamba, MoE, Mamba, MoE, Mamba): each block x += mixer(RMSNorm(x)).  The Mamba blocks are
    transformers' own NemotronHMamba2Mixer; the MoE blocks a direct transcription of NemotronHMoE /
    NemotronHTopkRouter (sigmoid, selection bias, top-22, renorm, x 5.0, latent fc1 -> relu^2 experts -> fc2, plus
    the shared expert) -- they only make the later Mamba layers' inputs real;
  - for each graded Mamba block (0, 2, 4): its in_proj output rounded to bf16 is the block's input row
    [z | x B C | dt]; the reference is NemotronHMamba2Mixer itself on those values in fp32 with in_proj and out_proj
    replaced by the identity (its output is the out_proj's input), and dt unclamped as vLLM serves it
    (time_step_limit (0, inf): transformers clamps dt at time_step_min = 0.001 in its chunked path only -- the
    count of clamped (row, head) values and the clamped output's distance are printed);
  - the final SSM state from transformers' own mamba2_chunk_scan (asserted to give the module's output);
  - transformers' bf16 run of the same module on the same bf16 inputs, the comparator.

Writes DIR/geom.txt, DIR/layers.txt, and per layer DIR/L<i>/: conv_w conv_b A_log dt_bias D norm_w (bf16),
in.bin (bf16 [T][18560]), ref32.bin (f32 [T][8192]), refbf16.bin (bf16 [T][8192]), state32.bin (f32 [128][64][128]).
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

from transformers import AutoTokenizer
from transformers.models.nemotron_h import modeling_nemotron_h as mh
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

PFX = "language_model.backbone."


class Ckpt:
    def __init__(self, root):
        self.root = root
        self.map = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def _f(self, key):
        fn = self.map[key]
        if fn not in self.files:
            self.files[fn] = safe_open(os.path.join(self.root, fn), framework="pt")
        return self.files[fn]

    def get(self, key):
        return self._f(PFX + key).get_tensor(PFX + key)

    def rows(self, key, idx):
        sl = self._f(PFX + key).get_slice(PFX + key)
        return torch.stack([sl[i] for i in idx])


def rmsnorm(x, w, eps):
    v = x.pow(2).mean(-1, keepdim=True)
    return w.float() * (x * torch.rsqrt(v + eps))


def moe(ck, li, x, cfg):
    p = f"layers.{li}.mixer."
    logits = x @ ck.get(p + "gate.weight").float().T
    scores = logits.sigmoid()
    choice = scores + ck.get(p + "gate.e_score_correction_bias").float()
    idx = torch.topk(choice, k=cfg["num_experts_per_tok"], dim=-1, sorted=False)[1]
    w = scores.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * cfg["routed_scaling_factor"]
    lat = x @ ck.get(p + "fc1_latent_proj.weight").float().T
    acc = torch.zeros_like(lat)
    for e in torch.unique(idx).tolist():
        tok, pos = torch.where(idx == e)
        up = ck.get(p + f"experts.{e}.up_proj.weight").float()
        down = ck.get(p + f"experts.{e}.down_proj.weight").float()
        h = F.relu(lat[tok] @ up.T).square() @ down.T
        acc.index_add_(0, tok, h * w[tok, pos, None])
    out = acc @ ck.get(p + "fc2_latent_proj.weight").float().T
    sh = F.relu(x @ ck.get(p + "shared_experts.up_proj.weight").float().T).square()
    return out + sh @ ck.get(p + "shared_experts.down_proj.weight").float().T


def mixer_for(ck, li, conf, dtype):
    m = mh.NemotronHMamba2Mixer(conf, layer_idx=li)
    p = f"layers.{li}.mixer."
    with torch.no_grad():
        for name, key in (("conv1d.weight", "conv1d.weight"), ("conv1d.bias", "conv1d.bias"), ("dt_bias", "dt_bias"),
                          ("A_log", "A_log"), ("D", "D"), ("norm.weight", "norm.weight")):
            dict(m.named_parameters())[name].copy_(ck.get(p + key).float())
    m.in_proj = torch.nn.Identity()
    m.out_proj = torch.nn.Identity()
    m.time_step_limit = (0.0, float("inf"))   # vLLM's: config time_step_limit [0, inf]
    return m.to(dtype).eval()


def bf16_bytes(t):
    return t.contiguous().to(torch.bfloat16).view(torch.int16).numpy().tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ck = Ckpt(a.ckpt)
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))["llm_config"]
    kw = {k: v for k, v in cfg.items() if k not in ("time_step_limit", "architectures", "dtype")}
    conf = NemotronHConfig(**kw)
    eps = cfg["layer_norm_epsilon"]
    H, P, N, G, K = cfg["mamba_num_heads"], cfg["mamba_head_dim"], cfg["ssm_state_size"], cfg["n_groups"], cfg["conv_kernel"]
    HP, CC = H * P, H * P + 2 * G * N
    NG = HP // G
    types = cfg["layers_block_type"]
    assert types[:5] == ["mamba", "moe", "mamba", "moe", "mamba"], types[:5]

    tok = AutoTokenizer.from_pretrained(a.ckpt)
    text = open(os.path.join(a.ckpt, "README.md")).read()
    ids = tok(text)["input_ids"][: a.rows]
    T = len(ids)
    assert T == a.rows, f"only {T} tokens"
    x = ck.rows("embeddings.weight", ids).float()
    print(f"{T} tokens, embeddings {tuple(x.shape)}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    graded = []
    for li in range(5):
        h = rmsnorm(x, ck.get(f"layers.{li}.norm.weight"), eps)
        if types[li] == "moe":
            x = x + moe(ck, li, h, cfg)
            print(f"block {li} (MoE) done", flush=True)
            continue
        win = ck.get(f"layers.{li}.mixer.in_proj.weight").float()
        zxbcdt = (h @ win.T).to(torch.bfloat16)          # the in_proj GEMM's bf16 output: the block's input
        m32 = mixer_for(ck, li, conf, torch.float32)
        with torch.no_grad():
            inp = zxbcdt.float()[None]
            ref = m32(inp)[0]
            # transformers as shipped clamps dt at time_step_min in its chunked path
            m32.time_step_limit = (conf.time_step_min, float("inf"))
            ref_clamped = m32(inp)[0]
            m32.time_step_limit = (0.0, float("inf"))
            # the final state from the module's own chunk scan, with the module's own arguments
            gate, xbc, dt = inp.split([HP, CC, H], dim=-1)
            xbc = mh.causal_conv1d_fn(xbc.transpose(1, 2), m32.conv1d.weight.squeeze(1), m32.conv1d.bias,
                                      activation=m32.activation).transpose(1, 2)
            xs, B, C = torch.split(xbc, [HP, G * N, G * N], dim=-1)
            y, state = mh.mamba2_chunk_scan(xs.view(1, T, H, P), dt, -torch.exp(m32.A_log.float()),
                                            B.view(1, T, G, N), C.view(1, T, G, N), chunk_size=m32.chunk_size, D=m32.D,
                                            z=None, return_final_states=True, dt_bias=m32.dt_bias, dt_softplus=True,
                                            dt_limit=(0.0, float("inf")))
            again = m32.norm(y.reshape(1, T, HP), gate)[0]
            assert torch.equal(again, ref), "the staged chunk scan is not the module's forward"
            sp = F.softplus(dt[0].float() + m32.dt_bias.float())
            n_clamped = int((sp < conf.time_step_min).sum())
            m16 = mixer_for(ck, li, conf, torch.bfloat16)
            ref16 = m16(zxbcdt[None])[0]
            # the residual stream continues through the real out_proj
            wout = ck.get(f"layers.{li}.mixer.out_proj.weight").float()
            x = x + ref @ wout.T
        d = os.path.join(a.out, f"L{li}")
        os.makedirs(d, exist_ok=True)
        p = f"layers.{li}.mixer."
        for name, key in (("conv_w", "conv1d.weight"), ("conv_b", "conv1d.bias"), ("A_log", "A_log"),
                          ("dt_bias", "dt_bias"), ("D", "D"), ("norm_w", "norm.weight")):
            t = ck.get(p + key)
            assert t.dtype == torch.bfloat16
            open(os.path.join(d, name + ".bin"), "wb").write(t.reshape(-1).view(torch.int16).numpy().tobytes())
        open(os.path.join(d, "in.bin"), "wb").write(zxbcdt.view(torch.int16).numpy().tobytes())
        open(os.path.join(d, "ref32.bin"), "wb").write(ref.contiguous().numpy().astype("float32").tobytes())
        open(os.path.join(d, "refbf16.bin"), "wb").write(bf16_bytes(ref16))
        open(os.path.join(d, "state32.bin"), "wb").write(state[0].contiguous().numpy().astype("float32").tobytes())
        rel = lambda u, v: float((u.double() - v.double()).norm() / v.double().norm())
        print(f"block {li} (Mamba-2): rms(out) {float(ref.pow(2).mean().sqrt()):.4f}; dt < time_step_min in "
              f"{n_clamped}/{T * H} (row, head); the clamped (transformers-as-shipped) output is {rel(ref_clamped, ref):.3e} "
              f"from the unclamped; transformers bf16 vs fp32 {rel(ref16.float(), ref):.3e}; dt range "
              f"[{float(sp.min()):.3e}, {float(sp.max()):.3e}]", flush=True)
        graded.append(f"L{li}")
    open(os.path.join(a.out, "geom.txt"), "w").write(f"{H} {P} {N} {G} {K} {NG} {eps} {T}\n")
    open(os.path.join(a.out, "layers.txt"), "w").write("\n".join(graded) + "\n")
    print(f"wrote {a.out}: {', '.join(graded)}")


if __name__ == "__main__":
    sys.exit(main())
