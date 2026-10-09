#!/usr/bin/env python3
"""L286 step 1: per-layer / per-block propagation gains of Qwen3.8-Flash-Next, measured PAIRED against U-e4-d5.

The base model is the U-e4-d5 container (its dequantized bytes: experts EXL3 K4, dense EXL3 K5 / MXFP8, the rest BF16).
A *branch* (L, family, s) is that model with ONE family of layer L's quantized tensors moved along its own
quantization error:

    W = W_bf16 + s * (W_U - W_bf16)          (s = 2: the error doubles, its Hessian proxy x4 -- one EXL3 bit down;
                                              s = 0.5: proxy x1/4 -- one bit up; s = 1: identity, bit-exact)

Families: gu (routed gate_up, fused), dn (routed down), dense (every other tensor the container quantizes: EXL3 dense
+ MXFP8).  The perturbation keeps the quantizer's own error SHAPE (LDLQ's H-aware direction), so its local size is
known exactly in proxy units: proxy(s) = s^2 proxy_U, and the proxy at any other K is proxy_U x r(K) (r is universal,
cv <= 1% across layers in L251's record).  gain = dKL / dproxy is then the propagation factor the old surrogate lacked.

Blocks (--block B): a branch moves one family of B consecutive layers together (born at the block's first layer, its
family perturbed through the block, U's weights after).  Single layers sit inside the chaotic floor: at s = 2 their
paired dKL CI is +-5e-3 for layers 0-15 (the same as a 5e-4 multiplicative-noise control's), so L286 measured
4-layer blocks at s = 0.5 / 2 / 4 (dense: s = 4) -- the x = s^2 - 1 points of each block's response curve (model.py).

One layer-outer pass carries: REF (BF16, only from layer 0 without --no-ref), U, and one stream per branch.  After
layer 47: mixer + lm_head, per-token KL(REF || U), KL(REF || branch) and KL(U || branch) in fp32.  The REF pass also
measures every quantized dense tensor's local error on the held-out BF16 stream (||X dW^T||^2 / ||X W^T||^2, the
proxy's own definition on held-out rows) -- the MXFP8 tensors have no proxy in L251's record.  Floor controls
(--noise-at L): W_U x (1 + 5e-4 N(0,1)) on every quantized tensor of layer L.

Windows: the run from layer 0 with REF saves REF's / U's final hidden (and U's stream at --save-at layers); every later
run asserts its U final hidden bit-identical to that one; an s = 1 branch (--identity-at) asserts the swap machinery
is the identity.

    flock ~/gpu.lock -c 'timeout 10800 ~/run_gb10.sh LOG -- env PYTHONPATH=~/exllamav3 ~/venvs/q38/bin/python \
        tools/allocator/gains.py --layers 0-47 --block 4 --families gu dn --scale 2 --no-ref --work DIR'
"""
import argparse, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glue as G                                                               # noqa: E402  (refuses GPU tenants)
import container as CT                                                         # noqa: E402
import torch                                                                   # noqa: E402
import torch.nn.functional as F                                                # noqa: E402

M, DEV, PFX, log = G.M, G.DEV, G.PFX, G.log
CONTAINER = os.environ.get("ALLOC_CONTAINER", "/srv/models/qwen38fn-u-e4-d5")
EXPERT = {"gu": "mlp.experts.gate_up_proj", "dn": "mlp.experts.down_proj"}


def parse_layers(items):
    out = []
    for it in items:
        if "-" in str(it):
            a, b = map(int, str(it).split("-"))
            out += range(a, b + 1)
        else:
            out.append(int(it))
    return sorted(set(out))


def perturb(w0, wq, s):
    """bf16(W_bf16 + s (W_U - W_bf16)), in fp32, chunked along dim 0 (a routed stack is 3.4 GB in bf16)."""
    out = torch.empty_like(wq)
    step = max(1, w0.shape[0] // 16) if w0.dim() == 3 else w0.shape[0]
    for i in range(0, w0.shape[0], step):
        a, q = w0[i:i + step].float(), wq[i:i + step].float()
        out[i:i + step] = (a + s * (q - a)).to(torch.bfloat16)
    return out


def stream_rel(a, b):
    num = sum(float((x.float() - y.float()).pow(2).sum()) for x, y in zip(a, b))
    den = sum(float(y.float().pow(2).sum()) for y in b)
    return math.sqrt(num / den)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", required=True, help="branch layers (ints or a-b ranges)")
    ap.add_argument("--block", type=int, default=1, help="perturb blocks of this many consecutive layers together")
    ap.add_argument("--no-ref", action="store_true", help="skip the REF pass (ref_final.pt from an earlier run)")
    ap.add_argument("--families", nargs="+", default=["gu", "dn", "dense"], choices=["gu", "dn", "dense"])
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--identity-at", type=int, default=None, help="add an s=1 'gu' branch here (must equal U)")
    ap.add_argument("--noise-at", nargs="*", type=int, default=[], help="floor-control branches at these layers")
    ap.add_argument("--save-at", nargs="*", type=int, default=[])
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-seqs", type=int, default=48, help="smoke only")
    a = ap.parse_args()
    layers = parse_layers(a.layers)
    start = layers[0]
    tag = a.tag or f"L{layers[0]}-{layers[-1]}_b{a.block}_s{a.scale:g}_{'+'.join(a.families)}"
    os.makedirs(a.work, exist_ok=True)
    cfg, W = G.text_config(), G.Weights()
    G.install()
    tok = G.AutoTokenizer.from_pretrained(G.TOK)
    ho, _ = G.heldout(tok)
    ho = ho[:a.max_seqs]
    n_tok = sum(s.numel() for _, s in ho)
    c = CT.Container(CONTAINER)
    rotary = M.Qwen4ExpTextRotaryEmbedding(config=cfg).to(DEV)
    assert all(layers[i] + 1 == layers[i + 1] for i in range(len(layers) - 1)) and len(layers) % a.block == 0
    blocks = [(layers[i], layers[i] + a.block - 1) for i in range(0, len(layers), a.block)]
    branches = [(b0, b1, f, a.scale) for b0, b1 in blocks for f in a.families]   # (first, last layer, family, s)
    if a.identity_at is not None:
        assert a.identity_at in layers
        branches.append((a.identity_at, a.identity_at, "gu", 1.0))
    for L in a.noise_at:                    # floor control: W_U x (1 + 5e-4 N(0,1)) on every quantized tensor of L
        assert L in layers
        branches.append((L, L, "nz", 5e-4))
    bname = {b: (f"L{b[0]}" if b[0] == b[1] else f"L{b[0]}-{b[1]}") + f":{b[2]}:s{b[3]:g}" for b in branches}
    log(f"held-out {len(ho)} rows / {n_tok:,} tokens; window {start}..47; {len(branches)} branches; container {CONTAINER}")

    with_ref = start == 0 and not a.no_ref
    if start == 0:
        emb = W.get(f"{PFX}embed_tokens.weight")
        e0 = [emb[ids.to(DEV)].unsqueeze(0).repeat(1, 1, cfg.hc_count) for _, ids in ho]
        ref, u = (list(e0) if with_ref else None), list(e0)
        del emb, e0
    else:
        p = f"{a.work}/u_in_L{start}.pt"
        u = [t.to(DEV) for t in torch.load(p, map_location="cpu")][:len(ho)]
        ref = None
        log(f"U stream at layer {start} loaded from {p}")
    br = {}
    rec = {"tag": tag, "scale": a.scale, "families": a.families, "layers": layers, "container": CONTAINER,
           "src": G.SRC, "per_layer": {}, "dense_local": {}, "branch_rel": {bname[b]: {} for b in branches}}

    for L in range(start, 48):
        t0 = time.time()
        layer = G.build_layer(cfg, W, L)
        if layer.ple is not None:
            G.gather_ple(layer, ho)
        pre = f"{PFX}layers.{L}."
        dq = CT.quantized_dense(c, L)                       # {hf name: bf16 [out, in]}
        params = {}
        for nm in dq:
            p = layer.get_parameter(nm[len(pre):])
            assert tuple(p.shape) == tuple(dq[nm].shape), (nm, p.shape, dq[nm].shape)
            params[nm] = p
        for f, path in EXPERT.items():
            w, lay = CT.experts(c, L, path.rsplit(".", 1)[1])
            assert lay == "exl3m_k4", (L, f, lay)
            p = layer.get_parameter(path)
            assert tuple(p.shape) == tuple(w.shape), (L, f, p.shape, w.shape)
            params[pre + path] = p
            dq[pre + path] = w
        orig = {nm: p.data for nm, p in params.items()}
        t_load = time.time() - t0

        def set_u(keys=None):
            for nm in (keys or params):
                params[nm].data = dq[nm]

        def family_keys(f):
            if f in EXPERT:
                return [pre + EXPERT[f]]
            return [nm for nm in dq if ".mlp.experts." not in nm]

        # ---- REF (BF16) + the dense tensors' held-out local error on the BF16 input
        t1 = time.time()
        if with_ref:
            acc, hooks = {}, []
            for nm in family_keys("dense"):
                try:
                    mod = layer.get_submodule(nm[len(pre):-len(".weight")])
                except AttributeError:
                    mod = None
                if not isinstance(mod, torch.nn.Linear):
                    rec["dense_local"].setdefault("unhooked", []).append(nm)
                    continue

                def hk(m, args, nm=nm):
                    x = args[0].reshape(-1, args[0].shape[-1])
                    y = F.linear(x, orig[nm]).float()
                    e = F.linear(x, dq[nm]).float() - y
                    s = acc.setdefault(nm, [0.0, 0.0])
                    s[0] += float(e.pow(2).sum()); s[1] += float(y.pow(2).sum())
                hooks.append(mod.register_forward_pre_hook(hk))
            for j, (_, ids) in enumerate(ho):
                ref[j] = G.run_layer(cfg, rotary, layer, ref[j], ids)
            for h in hooks:
                h.remove()
            for nm, (e, y) in acc.items():
                rec["dense_local"][nm] = {"rho_heldout": e / y, "layout": c.tensors[nm][1],
                                          "rel_w": float((dq[nm].float() - orig[nm].float()).norm() / orig[nm].float().norm())}
        t_ref = time.time() - t1

        # ---- branches perturbed at L (born here from U's input, or inside their block): one family moved along its
        #      own error; every other branch and U advance through U's layer
        t1 = time.time()
        set_u()
        for b in branches:
            if not (b[0] <= L <= b[1]):
                continue
            _, _, f, s = b
            keys = list(params) if f == "nz" else family_keys(f)
            gen = torch.Generator(device=DEV).manual_seed(286000 + L)
            for nm in keys:
                if f == "nz":
                    q = dq[nm]
                    params[nm].data = (q.float() * (1 + s * torch.randn(q.shape, device=DEV, generator=gen))
                                       ).to(torch.bfloat16)
                    continue
                params[nm].data = perturb(orig[nm], dq[nm], s)
                if s == 1.0:
                    assert torch.equal(params[nm].data, dq[nm]), f"{nm}: s=1 does not reproduce U's weight"
            src = u if b[0] == L else br[b]
            br[b] = [G.run_layer(cfg, rotary, layer, src[j], ids) for j, (_, ids) in enumerate(ho)]
            set_u(keys)
        for b in list(br):
            if b[0] <= L <= b[1]:
                continue
            s_ = br[b]
            for j, (_, ids) in enumerate(ho):
                s_[j] = G.run_layer(cfg, rotary, layer, s_[j], ids)
        for j, (_, ids) in enumerate(ho):
            u[j] = G.run_layer(cfg, rotary, layer, u[j], ids)
            assert torch.isfinite(u[j]).all(), f"non-finite U stream after layer {L}, seq {j}"
        t_br = time.time() - t1
        for b, s_ in br.items():
            rec["branch_rel"][bname[b]][L] = stream_rel(s_, u)
        rec["per_layer"][L] = {"type": layer.layer_type, "t_load": t_load, "t_ref": t_ref, "t_branches": t_br,
                               "n_streams": len(br) + 1 + (1 if with_ref else 0),
                               "u_vs_ref": stream_rel(u, ref) if with_ref else None}
        log(f"layer {L:2d} {layer.layer_type:17s} load {t_load:4.0f}s ref {t_ref:4.0f}s streams {len(br) + 1} "
            f"{t_br:5.0f}s" + (f" | U/REF {rec['per_layer'][L]['u_vs_ref']:.3e}" if with_ref else ""))
        del layer, dq, params, orig
        torch.cuda.empty_cache()
        if L + 1 in a.save_at:
            p = f"{a.work}/u_in_L{L + 1}.pt"
            torch.save([t.cpu() for t in u], p + ".tmp")
            os.replace(p + ".tmp", p)
            log(f"saved U stream entering layer {L + 1} -> {p}")
        json.dump(rec, open(f"{a.work}/gains_{tag}.partial.json", "w"))

    # ---- mixer + head, per-token KLs
    mix = G.build_mixer(cfg, W)
    head = W.get("lm_head.weight")
    fin_u = [mix(h) for h in u]
    if with_ref:
        fin_ref = [mix(h) for h in ref]
        assert not os.path.exists(f"{a.work}/ref_final.pt") or a.max_seqs < 48 or all(
            torch.equal(x.cpu(), y) for x, y in zip(fin_ref, torch.load(f"{a.work}/ref_final.pt"))), "REF changed"
        if a.max_seqs == 48:
            torch.save([t.cpu() for t in fin_ref], f"{a.work}/ref_final.pt")
            torch.save([t.cpu() for t in fin_u], f"{a.work}/u_final.pt")
    else:
        fin_ref = [t.to(DEV) for t in torch.load(f"{a.work}/ref_final.pt", map_location="cpu")][:len(ho)]
    if not with_ref or os.path.exists(f"{a.work}/u_final.pt"):
        saved_u = [t.to(DEV) for t in torch.load(f"{a.work}/u_final.pt", map_location="cpu")][:len(ho)]
        assert all(torch.equal(x, y) for x, y in zip(fin_u, saved_u)), "U continuation is not bit-identical"
        log("U continuation bit-identical to the layer-0 run's U")

    def logp(h, c0, c1):
        return F.log_softmax(F.linear(h[0, c0:c1], head).float(), -1)

    names = [bname[b] for b in branches]
    fin_b = {bname[b]: [mix(h) for h in br[b]] for b in branches}
    per_tok = {"ref_u": [], **{f"ref_{n}": [] for n in names}, **{f"u_{n}": [] for n in names}}
    for j, (_, ids) in enumerate(ho):
        T = ids.numel()
        cols = {k: torch.empty(T) for k in per_tok}
        for c0 in range(0, T, 256):
            c1 = min(T, c0 + 256)
            lr, lu = logp(fin_ref[j], c0, c1), logp(fin_u[j], c0, c1)
            pr, pu = lr.exp(), lu.exp()
            cols["ref_u"][c0:c1] = (pr * (lr - lu)).sum(-1).cpu()
            for n in names:
                lb = logp(fin_b[n][j], c0, c1)
                cols[f"ref_{n}"][c0:c1] = (pr * (lr - lb)).sum(-1).cpu()
                cols[f"u_{n}"][c0:c1] = (pu * (lu - lb)).sum(-1).cpu()
        for k in per_tok:
            per_tok[k].append(cols[k])
    if a.identity_at is not None:
        n = bname[(a.identity_at, a.identity_at, "gu", 1.0)]
        exact = all(torch.equal(x, y) for x, y in zip(fin_b[n], fin_u))
        rec["identity_branch_exact"] = exact
        log(f"identity branch {n}: final hidden {'BIT-EXACT' if exact else 'DIFFERS'} vs U")
        if not exact:
            raise SystemExit("HARNESS FAIL: the s=1 branch is not U")
    torch.save(per_tok, f"{a.work}/pertok_{tag}.pt")
    kl_u = torch.cat(per_tok["ref_u"])
    rec["kl_ref_u"] = float(kl_u.mean())
    rec["summary"] = {}
    for n in names:
        kb = torch.cat(per_tok[f"ref_{n}"])
        rec["summary"][n] = {"kl_ref_b": float(kb.mean()), "dkl": float((kb - kl_u).mean()),
                             "kl_u_b": float(torch.cat(per_tok[f"u_{n}"]).mean())}
        log(f"{n:18s} KL(REF||b) {float(kb.mean()):.4e}  dKL vs U {float((kb - kl_u).mean()):+.4e}  "
            f"KL(U||b) {rec['summary'][n]['kl_u_b']:.4e}")
    log(f"KL(REF||U) {rec['kl_ref_u']:.4e} over {kl_u.numel():,} tokens")
    json.dump(rec, open(f"{a.work}/gains_{tag}.json", "w"), indent=1)
    log("GAINS DONE")


if __name__ == "__main__":
    main()
