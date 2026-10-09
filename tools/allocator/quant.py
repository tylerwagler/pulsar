"""The Qwen routed-expert EXL3 quantization of L251 (qwen_fullkl.py / qwen_exl3_quant.py), vendored (L285 E9).

Exactly L251's: exllamav3 quantize_exl3_batch in groups of 16 with each expert's own Hessian (gate_up fused
[1280 x 2560] from the expert's routed BF16 calibration rows; down on silu(gate) * up of the BF16 source), mul1,
apply_out_scales, EXL3's default sigma_reg, seed 1000 L + e.  It keeps the PACKED output (trellis | suh | svh | mul1),
the bytes a container holds, and reconstructions are decoded from those bytes by LinearEXL3 -- the same decode the
U-e4-d5 container's tensors go through, so U and a candidate differ only in what the quantizer emitted.
"""
import torch
import torch.nn.functional as F
from exllamav3.modules.quant.exl3_lib import quantize as XQ

GROUP = 16


def qargs(K, seed):
    return {"seed": seed, "K": K, "devices": [0], "device_ratios": None,
            "apply_out_scales": True, "debug_dir": None, "mul1": True}


def hdata(H, count, key, dev):
    return {"H": H.clone(), "first_key": key, "count": count, "finalized": False,
            "num_total": count * H.shape[0], "inf_nan": torch.zeros(2), "device": dev}


def packed(out, k_in, n_out, K):
    assert set(out) == {"trellis", "suh", "svh", "mul1"}, sorted(out)
    tr, suh, svh, m1 = out["trellis"], out["suh"], out["svh"], out["mul1"]
    assert tr.dtype == torch.int16 and list(tr.shape) == [k_in // 16, n_out // 16, 16 * K], (tr.dtype, tr.shape)
    assert suh.dtype == torch.float16 and list(suh.shape) == [k_in]
    assert svh.dtype == torch.float16 and list(svh.shape) == [n_out]
    assert m1.dtype == torch.int32 and m1.numel() == 1
    return {k: v.detach().contiguous() for k, v in out.items()}


def blob(p):
    """The container's byte layout of one tensor: trellis | suh | svh (build.py's slice copy)."""
    return b"".join(p[k].cpu().numpy().tobytes() for k in ("trellis", "suh", "svh"))


def decode(p, dev="cuda"):
    from exllamav3.modules.quant.exl3 import LinearEXL3
    k, n = p["suh"].numel(), p["svh"].numel()
    lin = LinearEXL3(None, k, n, suh=p["suh"].to(dev), svh=p["svh"].to(dev), trellis=p["trellis"].to(dev),
                     mul1=torch.zeros(1, dtype=torch.int32, device=dev))
    return lin.get_weight_tensor().t().to(torch.bfloat16).contiguous()


def exl3_serial(w_oi, H, count, K, seed, dev):
    qa = qargs(K, seed)
    _, perr, out = XQ.quantize_exl3(w_oi.float().t().contiguous().clone(), hdata(H, count, "d", dev), qa, True)
    return packed(out, w_oi.shape[1], w_oi.shape[0], K), float(perr), bool(qa.get("q_fallback", False))


def exl3_expert_group(ws_oi, Hs, counts, K, seeds, dev):
    qas = [qargs(K, s) for s in seeds]
    res = XQ.quantize_exl3_batch([w.t().contiguous() for w in ws_oi],
                                 [hdata(H, n, f"e{i}", dev) for i, (H, n) in enumerate(zip(Hs, counts))], qas)
    outs, perrs, fbs = [], [], []
    for i, (w, (perr, out)) in enumerate(zip(ws_oi, res)):
        fb = bool(qas[i].get("q_fallback", False))
        if fb:      # uncalibrated fallback (zero routed rows): L251 redid these serially
            p, pe, f2 = exl3_serial(w, Hs[i], counts[i], K, seeds[i], dev)
            assert f2
            outs.append(p); perrs.append(pe)
        else:
            outs.append(packed(out, w.shape[1], w.shape[0], K)); perrs.append(float(perr))
        fbs.append(fb)
    return outs, perrs, fbs


def quantize_layer_experts(L, wts, x_all, tk, need, dev):
    """need: {part: [K, ...]} (part in gate_up | down).  Returns {(part, K): ([packed per expert], [proxy], n_fb)}."""
    res = {(p, K): ([], [], 0) for p, Ks in need.items() for K in Ks}
    for g0 in range(0, 512, GROUP):
        es = list(range(g0, g0 + GROUP))
        Hs, cnt = {"gate_up": [], "down": []}, []
        for e in es:
            x = x_all[(tk == e).any(1)].float()
            cnt.append(int(x.shape[0]))
            gu = wts["gate_up"][e].float()
            hm = gu.shape[0] // 2
            act = F.silu(x @ gu[:hm].t()) * (x @ gu[hm:].t())
            Hs["gate_up"].append(x.t() @ x)
            Hs["down"].append(act.t() @ act)
            del x, act
        for p, Ks in need.items():
            for K in Ks:
                outs, perrs, fbs = exl3_expert_group([wts[p][e] for e in es], Hs[p], cnt, K,
                                                     [1000 * L + e for e in es], dev)
                o, pe, nf = res[(p, K)]
                res[(p, K)] = (o + outs, pe + perrs, nf + sum(fbs))
        del Hs
    return res
