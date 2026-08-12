#!/usr/bin/env python3
# REAP-25 transplant (Path B layout): stamp the REAP25 survivor map onto our
# oracle-cutlass-mxfp4 intermediate, producing a ds4-compact-v1 GGUF in OUR
# formats (MXFP8 attn, MXFP4 rich, 2-bit ordinary; no Q8).
#
# Layout choice (Path B): keep the router (ffn_gate_inp) and bias (exp_probs_b)
# PADDED to the full n_expert=256 so the CUDA router_select kernels (which
# hardcode 256) stay untouched; only the expert weight tensors are physically
# DENSE-trimmed to keep_count survivors. For pruned layers:
#   ffn_gate_inp : [n_embd, 256]  rows 0..K-1 = survivor routers, K..255 = 0
#   exp_probs_b  : [256]          0..K-1 = survivor biases,        K..255 = -1e30
#   ffn_{gate,up,down}_exps : dense K survivors (sorted original-id order)
# A padded (zero) router row yields logit 0 -> prob ~0.83, but the paired
# bias = -1e30 forces its top-k score to -1e30, so it is never selected. The
# survivor experts are indexed densely 0..K-1; expert byte stride is derived
# from the intra-expert dims (count-independent) so dense indexing is correct.
#
# The survivor map itself is the REAP-25 (LiveCodeBench-50-calibrated) prune
# published as eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4. It is vendored in this
# directory as reap25-lcb50-survivors.json (default --survivors). See README.md.
#
# Usage:
#   trim_reap.py --oracle oracle-cutlass-mxfp4.gguf --out oracle-reap25-compact.gguf
#     [--survivors gguf-tools/reap/reap25-lcb50-survivors.json]  # default (vendored)
#     [--survivors path/to/eouya2-REAP.gguf]  # recover on the fly from the GGUF
import argparse, json, os, re, struct, sys
import numpy as np

ALIGN = 32
NEXP = 256
BIAS_PAD = np.float32(-1e30)
# gguf tensor type -> (block_elems, block_bytes)
BLK = {0:(1,4),1:(1,2),8:(32,34),10:(256,84),16:(256,66),38:(32,33),39:(32,17),40:(32,17),41:(32,33),26:(1,4),6:(1,4),5:(1,4),30:(1,2)}

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SURV = os.path.join(HERE, "reap25-lcb50-survivors.json")


def parse(path):
    f = open(path, "rb"); assert f.read(4) == b'GGUF'; ver = struct.unpack("<I", f.read(4))[0]
    nt = struct.unpack("<Q", f.read(8))[0]; nkv = struct.unpack("<Q", f.read(8))[0]
    def rs():
        n = struct.unpack("<Q", f.read(8))[0]; return f.read(n)
    def rv(t):
        if t == 8: return rs()
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
            return [rv(et) for _ in range(n)]
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        fm = {0:'<b',1:'<B',2:'<h',3:'<H',4:'<i',5:'<I',6:'<f',7:'<?',10:'<q',11:'<Q',12:'<d'}
        return struct.unpack(fm[t], f.read(sz[t]))[0]
    kv_start = f.tell()
    for _ in range(nkv):
        rs(); t = struct.unpack("<I", f.read(4))[0]; rv(t)
    kv_end = f.tell()
    tens = []  # [name(bytes), typ, dims, off]
    for _ in range(nt):
        nb = struct.unpack("<Q", f.read(8))[0]; name = f.read(nb)
        nd = struct.unpack("<I", f.read(4))[0]; dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        typ = struct.unpack("<I", f.read(4))[0]; off = struct.unpack("<Q", f.read(8))[0]
        tens.append([name, typ, dims, off])
    data_start = (f.tell() + ALIGN - 1) // ALIGN * ALIGN
    f.seek(kv_start); kv_raw = f.read(kv_end - kv_start)
    return dict(f=f, ver=ver, nkv=nkv, kv_raw=kv_raw, tens=tens, data_start=data_start)


def tbytes(typ, dims):
    if typ == 43:
        # IQ2_XXS_MMQ is PLANAR: one d plane then one q plane, each spanning
        # every block of every expert. The 'exp' branch below slices contiguous
        # per-expert byte ranges (correct for the expert-major types 16/40/41),
        # which for a planar tensor would splice d-plane bytes onto q-plane
        # bytes -- right size, silently corrupt data, artifact loads clean.
        #
        # This is an ORDERING constraint, not a missing table row: the MMQ
        # permutation must happen AFTER this trim (the stage-7 repack pass), or
        # REAP must move into the quantizer so it emits already-pruned tensors
        # and the tensor-global permutation applies to the final shape. Do not
        # "fix" this by adding 43 to BLK.
        raise SystemExit(
            "trim_reap: refusing type 43 (IQ2_XXS_MMQ) -- it is planar and this "
            "tool slices experts as contiguous ranges. Quantize experts as plain "
            "IQ2_XXS (16) and let the stage-7 repack convert AFTER the trim, or "
            "collapse REAP into the quantizer first.")
    be, bb = BLK[typ]; ne = 1
    for d in dims: ne *= d
    assert ne % be == 0, (typ, dims); return ne // be * bb


def read_f16(g, name):
    for nm, typ, dims, off in g['tens']:
        if nm.decode() == name:
            assert typ == 1; ne = dims[0]*dims[1]
            g['f'].seek(g['data_start']+off); return np.frombuffer(g['f'].read(ne*2), dtype=np.float16).reshape(dims[1], dims[0])
    return None


def load_survivors_json(path):
    """Vendored survivor map: reap25-lcb50-survivors.json."""
    d = json.load(open(path))
    policy = [int(x) for x in d["policy"]]
    keep = [int(x) for x in d["keep_count"]]
    survmap = {int(L): [int(e) for e in s] for L, s in d["survivors"].items()}
    for L, s in survmap.items():
        assert len(s) == keep[L] and sorted(s) == s and len(set(s)) == len(s), L
    return policy, keep, survmap


def recover_survivors_from_gguf(reap_path, og):
    """Fallback: read keep/policy from a REAP GGUF's KV and recover survivor ids
    by byte-matching its (possibly compacted) F16 router rows against the full
    256-row routers in the oracle `og`."""
    def reap_kv(path):
        f=open(path,"rb"); f.read(4); struct.unpack("<I",f.read(4)); struct.unpack("<Q",f.read(8)); nkv=struct.unpack("<Q",f.read(8))[0]
        def rs():
            n=struct.unpack("<Q",f.read(8))[0]; return f.read(n).decode()
        def rv(t):
            if t==8:
                n=struct.unpack("<Q",f.read(8))[0]; return f.read(n).decode()
            if t==9:
                et=struct.unpack("<I",f.read(4))[0]; n=struct.unpack("<Q",f.read(8))[0]; return [rv(et) for _ in range(n)]
            sz={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}; fm={0:'<b',1:'<B',2:'<h',3:'<H',4:'<i',5:'<I',6:'<f',7:'<?',10:'<q',11:'<Q',12:'<d'}
            return struct.unpack(fm[t],f.read(sz[t]))[0]
        kv={}
        for _ in range(nkv):
            k=rs(); t=struct.unpack("<I",f.read(4))[0]; kv[k]=rv(t)
        return kv
    rkv = reap_kv(reap_path)
    keep = [int(x) for x in rkv['reap.layer.keep_count']]
    policy = [int(x) for x in rkv['reap.layer.policy']]
    rg = parse(reap_path)
    nlayer = len(policy)
    survmap = {}
    for L in range(nlayer):
        if policy[L] == 1: continue
        our = read_f16(og, f"blk.{L}.ffn_gate_inp.weight")
        rp  = read_f16(rg, f"blk.{L}.ffn_gate_inp.weight")
        idx = {}
        for i, r in enumerate(our): idx.setdefault(r.tobytes(), i)
        surv = [idx[rp[j].tobytes()] for j in range(rp.shape[0])]  # KeyError on mismatch
        assert len(surv) == keep[L], (L, len(surv), keep[L])
        survmap[L] = sorted(surv)
    return policy, keep, survmap


def main():
    ap = argparse.ArgumentParser(description="REAP-25 Path-B transplant onto an oracle GGUF.")
    ap.add_argument("--oracle", required=True, help="oracle-cutlass-mxfp4 intermediate GGUF (full 256 experts)")
    ap.add_argument("--out", required=True, help="output ds4-compact-v1 GGUF")
    ap.add_argument("--survivors", default=DEFAULT_SURV,
                    help="vendored survivor map JSON (default) OR a REAP GGUF to recover from")
    args = ap.parse_args()

    print("parsing oracle...", flush=True)
    og = parse(args.oracle)

    if args.survivors.endswith(".json"):
        policy, keep, survmap = load_survivors_json(args.survivors)
        print(f"loaded vendored survivor map: {len(survmap)} pruned layers", flush=True)
    else:
        policy, keep, survmap = recover_survivors_from_gguf(args.survivors, og)
        print(f"recovered survivor map from GGUF: {len(survmap)} pruned layers", flush=True)
    NLAYER = len(policy)
    print("keep_count:", keep); print("policy:", policy, flush=True)

    # ---- classify tensors ----
    def L_of(nm, suffix_re):
        m = re.match(suffix_re, nm)
        return int(m.group(1)) if (m and int(m.group(1)) in survmap) else None
    RE_EXP  = r"blk\.(\d+)\.ffn_(?:gate|up|down)_exps\.weight$"
    RE_ROUT = r"blk\.(\d+)\.ffn_gate_inp\.weight$"
    RE_BIAS = r"blk\.(\d+)\.exp_probs_b\.bias$"

    # ---- build output tensor plan ----
    plan = []  # [name, typ, newdims, out_off(filled later), op]
    for name, typ, dims, off in og['tens']:
        nm = name.decode()
        Le = L_of(nm, RE_EXP)
        if Le is not None:
            assert dims[-1] == NEXP
            newdims = dims[:-1] + [keep[Le]]
            plan.append([name, typ, newdims, 0, ('exp', Le, dims)]); continue
        Lr = L_of(nm, RE_ROUT)
        if Lr is not None:
            assert typ == 1 and dims[1] == NEXP   # router stays 256 (Path B), zero-pad tail
            plan.append([name, typ, list(dims), 0, ('router', Lr, dims)]); continue
        Lb = L_of(nm, RE_BIAS)
        if Lb is not None:
            assert typ == 0 and dims[0] == NEXP   # bias stays 256, survivor then -1e30 tail
            plan.append([name, typ, list(dims), 0, ('bias', Lb, dims)]); continue
        plan.append([name, typ, list(dims), 0, ('copy', None, dims)])

    # ---- serialize header + KV(+reap) + tensor infos ----
    def u32(x): return struct.pack("<I", x)
    def u64(x): return struct.pack("<Q", x)
    def kv_str(k, v):
        kb=k.encode(); vb=v.encode(); return u64(len(kb))+kb+u32(8)+u64(len(vb))+vb
    def kv_bool(k, v):
        kb=k.encode(); return u64(len(kb))+kb+u32(7)+struct.pack("<?", v)
    def kv_arr_u32(k, vals):
        kb=k.encode(); b=u64(len(kb))+kb+u32(9)+u32(5)+u64(len(vals))
        for v in vals: b+=u32(int(v))
        return b
    reap_kv_bytes = kv_bool("reap.enabled", True) + kv_str("reap.layout", "ds4-compact-v1") \
                  + kv_arr_u32("reap.layer.expert_count", [NEXP]*NLAYER) \
                  + kv_arr_u32("reap.layer.keep_count", keep) \
                  + kv_arr_u32("reap.layer.policy", policy)
    new_nkv = og['nkv'] + 5

    infos = b""; off = 0
    for i, (name, typ, dims, _o, op) in enumerate(plan):
        infos += u64(len(name)) + name + u32(len(dims)) + b"".join(u64(d) for d in dims) + u32(typ) + u64(off)
        plan[i][3] = off
        sz = tbytes(typ, dims)
        off = (off + sz + ALIGN - 1) // ALIGN * ALIGN
    hdr = b"GGUF" + u32(og['ver']) + u64(len(plan)) + u64(new_nkv) + og['kv_raw'] + reap_kv_bytes
    pre = hdr + infos
    pad = (-len(pre)) % ALIGN

    print(f"writing {args.out} ... ({len(plan)} tensors, {new_nkv} kv)", flush=True)
    srcf = open(args.oracle, "rb"); DS = og['data_start']
    with open(args.out, "wb") as o:
        o.write(pre); o.write(b"\x00"*pad)
        written = 0
        for name, typ, dims, out_off, op in plan:
            kind, L, srcdims = op[0], op[1], op[2]
            src_off = None
            for nm, t2, d2, so in og['tens']:
                if nm == name: src_off = so; break
            if kind == 'copy':
                sz = tbytes(typ, srcdims); srcf.seek(DS+src_off); rem=sz
                while rem: c=srcf.read(min(rem, 64<<20)); o.write(c); rem-=len(c)
            elif kind == 'exp':
                per = tbytes(typ, srcdims)//NEXP
                for e in survmap[L]:
                    srcf.seek(DS+src_off + e*per); o.write(srcf.read(per))
            elif kind == 'router':
                # F16 [n_embd, 256]; write survivor rows (sorted) then zero-pad to 256
                rowb = srcdims[0]*2
                for e in survmap[L]:
                    srcf.seek(DS+src_off + e*rowb); o.write(srcf.read(rowb))
                o.write(b"\x00" * (rowb * (NEXP - keep[L])))
            elif kind == 'bias':
                # F32 [256]; survivor biases (sorted) then -1e30 tail
                srcf.seek(DS+src_off); src = np.frombuffer(srcf.read(NEXP*4), dtype=np.float32)
                out = np.empty(NEXP, dtype=np.float32)
                for j, e in enumerate(survmap[L]): out[j] = src[e]
                out[keep[L]:] = BIAS_PAD
                o.write(out.tobytes())
            need = tbytes(typ, dims); padn = (-need) % ALIGN
            if padn: o.write(b"\x00"*padn)
            written += 1
            if written % 200 == 0: print(f"  {written}/{len(plan)} tensors", flush=True)
    print("DONE:", args.out, os.path.getsize(args.out)/1e9, "GB", flush=True)


if __name__ == "__main__":
    main()
