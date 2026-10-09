#!/usr/bin/env python3
"""L286: full-model KL of per-layer expert allocations, PAIRED against U-e4-d5 on the same 73,738 held-out tokens.

An allocation moves some layers' routed experts off U-e4-d5's K4 (gate_up fused: K4 | K5 -- the engine's
gate_up_fused arm reads only those; down: K2 | K3 | K4 | K5 | K6, the down arm).  Everything else is the U-e4-d5
container's bytes.  The moved stacks are quantized here exactly as L251 quantized U (quant.py: BF16 calibration
stream, the expert's own routed rows, seed 1000 L + e) and decoded from their packed bytes like the container's.

One layer-outer pass:
  cal   the BF16 calibration stream (418 rows, 592,960 tokens), run only up to the last layer any allocation moves;
        at a moved layer it supplies each expert's Hessians
  U     held-out stream through the container's weights (must reproduce gains.py's U final hidden bit for bit)
  <a>   one held-out stream per allocation
Control: --control L quantizes layer L's experts at K4 too and asserts every expert's bytes equal the container's
(the pipeline that makes the candidates IS the one that made U).
After layer 47: KL(REF || x) per token against gains.py's saved REF final hidden; paired per-token differences vs U,
bootstrap over the 48 sequences.  Packed tensors of every moved stack are saved (exllamav3 names) for the builder.

    ... measure.py --allocs allocs.json --work /mnt/models/l286/measure --gains-work /mnt/models/l286/gains
"""
import argparse, json, math, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glue as G                                                               # noqa: E402
import container as CT                                                         # noqa: E402
import quant as Q                                                              # noqa: E402
import torch                                                                   # noqa: E402
import torch.nn.functional as F                                                # noqa: E402
from safetensors.torch import save_file                                       # noqa: E402

M, DEV, PFX, log = G.M, G.DEV, G.PFX, G.log
CONTAINER = os.environ.get("ALLOC_CONTAINER", "/srv/models/qwen38fn-u-e4-d5")
PARTS = {"gate_up": "gate_up_proj", "down": "down_proj"}
ADMIT = {"gate_up": (4, 5), "down": (2, 3, 4, 5, 6)}


def load_allocs(p):
    A = json.load(open(p))
    out = {}
    for name, a in A.items():
        m = {}
        for part in PARTS:
            for L, K in a.get(part, {}).items():
                K = int(K)
                assert K in ADMIT[part], f"{name}: {part} K{K} -- no engine arm reads it"
                if K != 4:
                    m[(int(L), part)] = K
        out[name] = m
    return out


def boot(diff_sums, counts, n=10000, seed=0):
    rnd, m = random.Random(seed), len(diff_sums)
    rs = []
    for _ in range(n):
        idx = [rnd.randrange(m) for _ in range(m)]
        rs.append(sum(diff_sums[i] for i in idx) / sum(counts[i] for i in idx))
    rs.sort()
    return rs[int(0.025 * n)], rs[int(0.975 * n)]


def boot_ratio(num, den, n=10000, seed=0):
    rnd, m = random.Random(seed), len(num)
    rs = []
    for _ in range(n):
        idx = [rnd.randrange(m) for _ in range(m)]
        rs.append(sum(num[i] for i in idx) / sum(den[i] for i in idx))
    rs.sort()
    return rs[int(0.025 * n)], rs[int(0.975 * n)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allocs", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--gains-work", required=True, help="where gains.py saved ref_final.pt / u_final.pt")
    ap.add_argument("--control", type=int, default=None, help="re-quantize this layer at K4: bytes must equal U's")
    ap.add_argument("--save-packed", default=None, help="dir for the moved stacks' packed tensors")
    a = ap.parse_args()
    allocs = load_allocs(a.allocs)
    if a.names:
        allocs = {n: allocs[n] for n in a.names}
    names = list(allocs)
    os.makedirs(a.work, exist_ok=True)
    cfg, W = G.text_config(), G.Weights()
    G.install()
    tok = G.AutoTokenizer.from_pretrained(G.TOK)
    ho, cal = G.heldout(tok)
    n_cal = sum(s.numel() for _, s in cal)
    c = CT.Container(CONTAINER)
    rotary = M.Qwen4ExpTextRotaryEmbedding(config=cfg).to(DEV)
    need = {}                                   # (L, part) -> {K}
    for m in allocs.values():
        for (L, p), K in m.items():
            need.setdefault((L, p), set()).add(K)
    if a.control is not None:
        for p in PARTS:
            need.setdefault((a.control, p), set()).add(4)
    last_cal = max([L for L, _ in need], default=-1)
    log(f"allocations {names}; moved stacks {sorted((L, p, sorted(k)) for (L, p), k in need.items())}; "
        f"calibration through layer {last_cal} ({len(cal)} rows / {n_cal:,} tokens)")

    emb = W.get(f"{PFX}embed_tokens.weight")
    cal_hs = [emb[ids.to(DEV)].unsqueeze(0).repeat(1, 1, cfg.hc_count) for _, ids in cal] if last_cal >= 0 else []
    e0 = [emb[ids.to(DEV)].unsqueeze(0).repeat(1, 1, cfg.hc_count) for _, ids in ho]
    ev = {n: list(e0) for n in ["U"] + names}
    del emb, e0
    rec = {"allocs": {n: {f"{L}.{p}": K for (L, p), K in m.items()} for n, m in allocs.items()}, "layers": {},
           "bytes_delta": {n: 0 for n in names}}

    for L in range(48):
        t0 = time.time()
        layer = G.build_layer(cfg, W, L)
        if layer.ple is not None:
            G.gather_ple(layer, ho + cal if L <= last_cal else ho)
        pre = f"{PFX}layers.{L}."
        moved = {p: sorted(need.get((L, p), ())) for p in PARTS}
        lrec = {}
        # ---- calibration (BF16) -> Hessians of this layer's moved experts
        if L <= last_cal:
            moe_x, moe_tk, hooks = [], [], []
            if any(moved.values()):
                hooks.append(layer.mlp.register_forward_pre_hook(
                    lambda m, args: moe_x.append(args[0].flatten(0, 1).to(torch.bfloat16))))
                hooks.append(layer.mlp.gate.register_forward_hook(
                    lambda m, args, out: moe_tk.append(out[2].to(torch.int16))))
            for j, (_, ids) in enumerate(cal):
                cal_hs[j] = G.run_layer(cfg, rotary, layer, cal_hs[j], ids)
            for h in hooks:
                h.remove()
            if L == last_cal:
                cal_hs.clear()
        t_cal = time.time() - t0
        recon = {}
        t1 = time.time()
        if any(moved.values()):
            x_all, tk = torch.cat(moe_x), torch.cat(moe_tk).long()
            del moe_x, moe_tk
            assert x_all.shape[0] == n_cal and tk.shape == (n_cal, 10)
            wts = {p: layer.get_parameter(f"mlp.experts.{PARTS[p]}").data for p in PARTS}
            res = Q.quantize_layer_experts(L, wts, x_all, tk, {p: Ks for p, Ks in moved.items() if Ks}, DEV)
            del x_all, tk
            for (p, K), (outs, perrs, nfb) in res.items():
                lrec[f"{p}|K{K}"] = {"proxy_mean": sum(perrs) / len(perrs), "fallback": nfb,
                                     "bytes": sum(len(Q.blob(o)) for o in outs)}
                if a.control == L and K == 4:
                    blobs, e = c.expert_blobs(f"{pre}mlp.experts.{PARTS[p]}")
                    same = sum(Q.blob(o) == b for o, b in zip(outs, blobs))
                    lrec[f"{p}|K4"]["control_equal_experts"] = same
                    log(f"CONTROL layer {L} {p} K4: {same}/512 experts byte-equal to the container")
                    if same != 512:
                        raise SystemExit("CONTROL FAIL: this pipeline does not reproduce U-e4-d5's bytes")
                    continue
                w = torch.empty_like(wts[p])
                for i, o in enumerate(outs):
                    w[i] = Q.decode(o)
                recon[(p, K)] = w
                if a.save_packed:
                    d = f"{a.save_packed}/K{K}"
                    os.makedirs(d, exist_ok=True)
                    save_file({f"{pre}mlp.experts.{i}.{PARTS[p]}.{k}": v.cpu() for i, o in enumerate(outs)
                               for k, v in o.items()}, f"{d}/layer{L:02d}-{p}.safetensors")
                del outs
        t_q = time.time() - t1
        # ---- held-out: U = the container's quantized tensors; allocations swap their moved stacks
        t1 = time.time()
        dq = CT.quantized_dense(c, L)
        for p in PARTS:
            w, lay = CT.experts(c, L, PARTS[p])
            assert lay == "exl3m_k4"
            dq[f"{pre}mlp.experts.{PARTS[p]}"] = w
        params = {nm: layer.get_parameter(nm[len(pre):]) for nm in dq}
        for nm, p_ in params.items():
            p_.data = dq[nm]
        for n in ["U"] + names:
            sw = []
            if n != "U":
                for (LL, p), K in allocs[n].items():
                    if LL == L:
                        key = f"{pre}mlp.experts.{PARTS[p]}"
                        params[key].data = recon[(p, K)]
                        sw.append(key)
                        rec["bytes_delta"][n] += lrec[f"{p}|K{K}"]["bytes"] - sum(
                            len(b) for b in c.expert_blobs(key)[0])
            s = ev[n]
            for j, (_, ids) in enumerate(ho):
                s[j] = G.run_layer(cfg, rotary, layer, s[j], ids)
                assert torch.isfinite(s[j]).all(), f"non-finite {n} after layer {L}"
            for key in sw:
                params[key].data = dq[key]
        t_ev = time.time() - t1
        lrec.update({"t_cal": t_cal, "t_quant": t_q, "t_eval": t_ev})
        rec["layers"][L] = lrec
        log(f"layer {L:2d} cal {t_cal:4.0f}s quant {t_q:5.0f}s eval {t_ev:4.0f}s moved {moved}")
        del layer, dq, params, recon
        torch.cuda.empty_cache()
        json.dump(rec, open(f"{a.work}/measure.partial.json", "w"))

    mix = G.build_mixer(cfg, W)
    head = W.get("lm_head.weight")
    fin = {n: [mix(h) for h in ev[n]] for n in ev}
    fin_ref = [t.to(DEV) for t in torch.load(f"{a.gains_work}/ref_final.pt", map_location="cpu")]
    u_saved = [t.to(DEV) for t in torch.load(f"{a.gains_work}/u_final.pt", map_location="cpu")]
    assert all(torch.equal(x, y) for x, y in zip(fin["U"], u_saved)), "U is not the gains pass's U"
    log("U final hidden bit-identical to the gains pass's U")
    per = {n: [] for n in fin}
    for j, (_, ids) in enumerate(ho):
        T = ids.numel()
        cols = {n: torch.empty(T) for n in fin}
        for c0 in range(0, T, 256):
            c1 = min(T, c0 + 256)
            lr = F.log_softmax(F.linear(fin_ref[j][0, c0:c1], head).float(), -1)
            pr = lr.exp()
            for n in fin:
                lx = F.log_softmax(F.linear(fin[n][j][0, c0:c1], head).float(), -1)
                cols[n][c0:c1] = (pr * (lr - lx)).sum(-1).cpu()
        for n in fin:
            per[n].append(cols[n])
    torch.save(per, f"{a.work}/pertok.pt")
    Ts = [int(s.numel()) for _, s in ho]
    su = [float(x.sum()) for x in per["U"]]
    out = {"U": {"kl": sum(su) / sum(Ts)}}
    log(f"U        KL/token {out['U']['kl']:.4e}")
    for n in names:
        sn = [float(x.sum()) for x in per[n]]
        d = [x - y for x, y in zip(sn, su)]
        lo, hi = boot(d, Ts)
        rlo, rhi = boot_ratio(sn, su)
        tokdiff = torch.cat(per[n]) - torch.cat(per["U"])
        out[n] = {"kl": sum(sn) / sum(Ts), "dkl": sum(d) / sum(Ts), "dkl_ci": [lo, hi],
                  "ratio": sum(sn) / sum(su), "ratio_ci": [rlo, rhi],
                  "seq_wins": sum(x < y for x, y in zip(sn, su)) / len(sn),
                  "tok_below": float((tokdiff < 0).float().mean()), "median_tok_diff": float(tokdiff.median()),
                  "bytes_delta": rec["bytes_delta"][n]}
        log(f"{n:12s} KL/token {out[n]['kl']:.4e}  vs U {out[n]['ratio']:.4f} [{rlo:.4f}, {rhi:.4f}]  "
            f"dKL {out[n]['dkl']:+.3e} [{lo:+.3e}, {hi:+.3e}]  bytes {rec['bytes_delta'][n] / 1e9:+.3f} GB  "
            f"seq wins {out[n]['seq_wins']:.2f}")
    rec["results"] = out
    json.dump(rec, open(f"{a.work}/measure.json", "w"), indent=1)
    log("MEASURE DONE")


if __name__ == "__main__":
    main()
