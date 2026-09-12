#!/usr/bin/env python3
"""Generate router goldens from the checkpoint's OWN Gate.forward.

The router's image-slot bias (`bias_vl`) is graded against the reference's
gating code rather than against a re-derivation of it: this instantiates
`Gate` from the snapshot's `inference/model.py`, installs a deterministic
synthetic layer (weight / bias / bias_vl / tid2eid), runs the reference's own
forward, and dumps the layer AND the reference's output.  The engine replays
the dumped logits and token ids and must reproduce the reference's experts.

Why the logits are dumped instead of the input x: the engine is fed the
gate's logits by the MoE path, so dumping them removes the matmul from the
comparison and isolates the gating arithmetic -- bias selection, the hash
escape, and the top-k -- which is what bias_vl changes.

The construction is deterministic on purpose.  With random scores the gap
between the 6th and 7th expert is occasionally smaller than the engine's
fast-math softplus noise, and the case then proves nothing while still
reporting success.  Here the 256 experts get strictly descending softplus
scores with a 5e-3 step, every expert in TEXT_SET / IMAGE_SET is ranked last,
and the two biases are +2 / +4, so a text slot must select TEXT_SET and an
image slot IMAGE_SET -- pairwise disjoint from each other, from the tid2eid
rows, and from the score leaders.  The gate asserts that per row.

`model.py` imports a `kernel` module that needs tilelang, which we do not have
and do not need: Gate.forward reaches `linear()`, and for fp32 weights that is
`F.linear`.  The stub below satisfies the import so the REAL Gate class runs.

    python3 tests/vision_router_goldens.py <snapshot-dir> > tests/test-vectors/vision-router-goldens.bin
"""
import io
import json
import struct
import sys
import types

import numpy as np
import torch

SNAP = sys.argv[1]

# Gate.forward only ever reaches F.linear (fp32 weights); the quantized GEMM,
# sparse-attention and sinkhorn entry points are never called on this path.
_kernel = types.ModuleType("kernel")
for _n in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn",
           "hc_split_sinkhorn"):
    setattr(_kernel, _n, lambda *a, **k: None)
_kernel.block_size = 128
_kernel.fp4_block_size = 32
sys.modules["kernel"] = _kernel

sys.path.insert(0, SNAP + "/inference")
import model  # noqa: E402

CFG = json.load(open(SNAP + "/config.json"))

N_EXPERT = CFG["n_routed_experts"]          # 256
TOPK = CFG["num_experts_per_tok"]           # 6
N_VOCAB = CFG["vocab_size"]                 # 129280
DIM = CFG["hidden_size"]                    # 4096
N_HASH = CFG["num_hash_layers"]             # 3
ROUTE_SCALE = CFG["routed_scaling_factor"]  # 1.5

TEXT_SET = [17, 88, 133, 200, 211, 250]
IMAGE_SET = [3, 61, 99, 150, 175, 240]
N_SET = 6
HASH_ROWS = 64          # every token id the fixtures use is < HASH_ROWS
SCORE_STEP = 5e-3
TEXT_BIAS = 2.0
IMAGE_BIAS = 4.0


def make_args(vl: bool) -> model.ModelArgs:
    a = model.ModelArgs()
    a.dim = DIM
    a.n_routed_experts = N_EXPERT
    a.n_activated_experts = TOPK
    a.score_func = CFG["scoring_func"]
    a.route_scale = ROUTE_SCALE
    a.n_hash_layers = N_HASH
    a.vocab_size = N_VOCAB
    a.vision_n_layers = CFG["vision_n_layers"] if vl else 0
    return a


def build_layer(layer_id: int, vl: bool, seed: int):
    """A Gate with a deterministic synthetic layer, and the logits it will see.

    x is the identity on the first `n_rows` axes and weight[:, r] holds the
    logits for row r, so the reference's own `linear(x, weight)` reproduces
    those logits exactly -- the scores the engine is graded on come out of the
    reference's arithmetic, not out of ours.
    """
    g = model.Gate(args=make_args(vl), layer_id=layer_id)
    rng = np.random.RandomState(seed)

    reserved = set(TEXT_SET) | set(IMAGE_SET)
    free = [e for e in range(N_EXPERT) if e not in reserved]
    rng.shuffle(free)
    order = free + sorted(reserved)          # biased experts ranked LAST
    target = 1.5 - SCORE_STEP * np.arange(N_EXPERT, dtype=np.float32)
    # p = sqrt(softplus(x))  =>  x = log(expm1(p^2))
    logits = np.log(np.expm1(target.astype(np.float64) ** 2))

    bias = np.zeros(N_EXPERT, dtype=np.float32)
    bias_vl = np.zeros(N_EXPERT, dtype=np.float32)
    for e in TEXT_SET:
        bias[e] = TEXT_BIAS
    for e in IMAGE_SET:
        bias_vl[e] = IMAGE_BIAS

    # tid2eid holds EXPERT ids, so every entry must be a valid expert: pick
    # them from the experts outside both bias sets, six consecutive by token.
    free_by_id = sorted(free)
    tid2eid = np.zeros((N_VOCAB, TOPK), dtype=np.int32)
    for t in range(HASH_ROWS):
        for j in range(TOPK):
            tid2eid[t, j] = free_by_id[(t * TOPK + j) % len(free_by_id)]

    with torch.no_grad():
        g.weight.zero_()
        row_perm = np.empty(N_EXPERT, dtype=np.int64)
        row_perm[order] = np.arange(N_EXPERT)
        g.weight[:, :N_EXPERT] = torch.from_numpy(logits[row_perm].astype(np.float32))
        if g.bias is not None:
            g.bias.copy_(torch.from_numpy(bias))
        if g.bias_vl is not None:
            g.bias_vl.copy_(torch.from_numpy(bias_vl))
        if getattr(g, "tid2eid", None) is not None:
            g.tid2eid.copy_(torch.from_numpy(tid2eid))
    return g, logits.astype(np.float32), bias, bias_vl, tid2eid


def margin_of(g, logits: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """The reference's own scores, and how much the 6th expert won the 7th by.

    Only a diagnostic: it is what tells the gate that a case is DISCRIMINATING
    -- a row whose top-6/top-7 gap is below the engine's softplus noise proves
    nothing however it comes out.  Built from torch ops, so it is the
    reference's arithmetic even though the assembly is ours.
    """
    with torch.no_grad():
        sc = torch.nn.functional.softplus(torch.from_numpy(logits)).sqrt()
        tok = torch.from_numpy(tokens)
        if g.bias_vl is not None and g.bias is not None:
            img = (tok >= g.vocab_size).unsqueeze(-1)
            b = torch.where(img, g.bias_vl, g.bias)
        elif g.bias is not None:
            b = g.bias
        else:
            b = torch.zeros(N_EXPERT)
        if g.hash:
            # a hash layer biases nothing for a text slot, and uses bias_vl for
            # an image slot
            img = (tok >= g.vocab_size).unsqueeze(-1)
            if g.bias_vl is not None:
                b = torch.where(img, g.bias_vl, torch.zeros(N_EXPERT))
        s = sc + b
        top7 = s.topk(TOPK + 1, dim=-1)[0]
        margin = top7[..., TOPK - 1] - top7[..., TOPK]
        # A hash layer's text slots select tid2eid, not a top-k of these
        # scores, so they have no boundary and no margin.
        if g.hash:
            margin = torch.where(tok >= g.vocab_size, margin,
                                 torch.full_like(margin, float("inf")))
        return margin.numpy().astype(np.float32)



CASES = []


def add(name, layer_id, vl, tokens, vl_is_text_bias=False):
    CASES.append((name, layer_id, vl, tokens, vl_is_text_bias))


TEXT_TOKENS = [7, 1, 40, 63]
MIXED_TOKENS = [7, N_VOCAB + 2, 40, N_VOCAB + 4, 1, N_VOCAB + 0]
IMAGE_TOKENS = [N_VOCAB + 2, N_VOCAB + 0]

add("learned text only, no bias_vl", 6, False, TEXT_TOKENS)
add("learned text only, bias_vl present", 6, True, TEXT_TOKENS)
add("learned mixed text+image", 6, True, MIXED_TOKENS)
add("hash text only, no bias_vl", 0, False, TEXT_TOKENS)
add("hash text only, bias_vl present", 0, True, TEXT_TOKENS)
add("hash mixed text+image", 0, True, MIXED_TOKENS)
add("learned image, bias_vl == bias", 6, True, IMAGE_TOKENS, vl_is_text_bias=True)


def main():
    out = io.BytesIO()
    out.write(b"VRT1")
    out.write(struct.pack("<6I", len(CASES), N_EXPERT, TOPK, N_VOCAB, HASH_ROWS, 0))

    for ci, (name, layer_id, vl, tokens, vl_is_text_bias) in enumerate(CASES):
        g, logits_all, bias, bias_vl, tid2eid = build_layer(layer_id, vl, 1000 + ci)
        if vl_is_text_bias:
            with torch.no_grad():
                g.bias_vl.copy_(torch.from_numpy(bias))
            bias_vl = bias
        n_rows = len(tokens)
        # x = identity: row r picks weight[:, r], i.e. logits_all.  Gate takes
        # 2-D x and 1-D input_ids -- its weights.gather(1, indices) indexes the
        # TOKEN axis, so a (bsz, seqlen) batch would be a different (broken)
        # program.
        x = torch.zeros(n_rows, DIM)
        for r in range(n_rows):
            x[r, r] = 1.0
        ids = torch.tensor(tokens, dtype=torch.long)
        with torch.no_grad():
            w, idx = g(x, ids)
        w = w.numpy().astype(np.float32)
        idx = idx.numpy().astype(np.int32)
        # every row of a case sees the SAME logits: the rows differ only in
        # their token, which is what isolates the bias choice
        logits = np.stack([logits_all] * n_rows).astype(np.float32)
        margin = margin_of(g, logits, np.asarray(tokens, dtype=np.int64))

        has_text_bias = not g.hash
        # Does this fixture give an image slot a DIFFERENT expert set than a
        # text slot?  The gate asserts this against the golden, so a fixture
        # that had collapsed to one answer could not pass as "discriminating".
        img_rows = [r for r, t in enumerate(tokens) if t >= N_VOCAB]
        txt_rows = [r for r, t in enumerate(tokens) if t < N_VOCAB]
        image_differs = 0
        if img_rows and txt_rows:
            image_differs = 1 if set(idx[img_rows[0]]) != set(idx[txt_rows[0]]) else 0
        out.write(name.encode().ljust(32, b"\0")[:32])
        out.write(struct.pack("<5I", 1 if g.hash else 0, 1 if vl else 0,
                              n_rows, 1 if has_text_bias else 0, image_differs))
        out.write(bias.astype(np.float32).tobytes())
        out.write(bias_vl.astype(np.float32).tobytes())
        out.write(tid2eid[:HASH_ROWS].astype(np.int32).tobytes())
        out.write(logits.tobytes())
        out.write(np.asarray(tokens, dtype=np.int32).tobytes())
        out.write(margin.tobytes())
        out.write(idx.tobytes())
        out.write(w.tobytes())
        print("  %-36s rows=%d idx[0]=%s" % (name, n_rows, idx[0].tolist()), file=sys.stderr)

    sys.stdout.buffer.write(out.getvalue())


main()
