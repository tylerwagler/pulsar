#!/usr/bin/env python3
# Regenerate reap25-lcb50-survivors.json.
#
# The eouya2 REAP GGUF stores only reap.layer.{policy,keep_count} in its KV, NOT
# the survivor expert ids. The survivor identity is implicit in the compacted
# F16 router (ffn_gate_inp.weight) rows: for each pruned layer the compacted
# file keeps K=192 of the original 256 router rows, byte-for-byte. We recover the
# original expert id of each survivor by matching those rows against the full
# 256-row router in the GGUF eouya2 pruned from (antirez source), which shares
# byte-identical F16 router rows.
#
# Only the small router tensors are range-fetched (~140 MB total), never the
# 60+ GB expert data. Requires network + curl. Output is deterministic.
import json, os, struct, subprocess, sys

SRC_URL = "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
REAP_URL = "https://huggingface.co/eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4/resolve/main/DeepSeek-V4-Flash-REAP25-LCB50-DS4-compact-IQ2XXS.gguf"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reap25-lcb50-survivors.json")
HDR_BYTES = 24 << 20  # first 24 MB captures GGUF header (KV + all tensor infos)


def rng(url, start, length, tries=4):
    end = start + length - 1
    got = b""
    for _ in range(tries):
        p = subprocess.run(["curl", "-sL", "-r", f"{start}-{end}", url],
                           capture_output=True, timeout=300)
        got = p.stdout
        if len(got) == length:
            return got
    raise RuntimeError(f"range fetch short: got {len(got)} want {length} @ {start} {url}")


def parse_header(buf):
    """Return (kv, routers, data_start). routers[name] = (dims, off, typ)."""
    o = [0]
    def rd(n):
        s = buf[o[0]:o[0]+n]; o[0] += n
        if len(s) != n: raise EOFError("header truncated; increase HDR_BYTES")
        return s
    assert rd(4) == b'GGUF'; struct.unpack("<I", rd(4))
    nt = struct.unpack("<Q", rd(8))[0]; nkv = struct.unpack("<Q", rd(8))[0]
    def rs(): return rd(struct.unpack("<Q", rd(8))[0])
    def rv(t):
        if t == 8: return rs().decode()
        if t == 9:
            et = struct.unpack("<I", rd(4))[0]; n = struct.unpack("<Q", rd(8))[0]
            return [rv(et) for _ in range(n)]
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        fm = {0:'<b',1:'<B',2:'<h',3:'<H',4:'<i',5:'<I',6:'<f',7:'<?',10:'<q',11:'<Q',12:'<d'}
        return struct.unpack(fm[t], rd(sz[t]))[0]
    kv = {}
    for _ in range(nkv):
        k = rs().decode(); t = struct.unpack("<I", rd(4))[0]; kv[k] = rv(t)
    routers = {}
    for _ in range(nt):
        name = rd(struct.unpack("<Q", rd(8))[0]).decode()
        nd = struct.unpack("<I", rd(4))[0]
        dims = [struct.unpack("<Q", rd(8))[0] for _ in range(nd)]
        typ = struct.unpack("<I", rd(4))[0]; off = struct.unpack("<Q", rd(8))[0]
        if name.endswith("ffn_gate_inp.weight"):
            routers[name] = (dims, off, typ)
    data_start = (o[0] + 31) // 32 * 32
    return kv, routers, data_start


def main():
    print("fetching headers...", flush=True)
    src_kv, src_r, src_ds = parse_header(rng(SRC_URL, 0, HDR_BYTES))
    rp_kv, rp_r, rp_ds = parse_header(rng(REAP_URL, 0, HDR_BYTES))
    policy = [int(x) for x in rp_kv["reap.layer.policy"]]
    keep = [int(x) for x in rp_kv["reap.layer.keep_count"]]
    NLAYER = len(policy)
    ROWB = src_r["blk.3.ffn_gate_inp.weight"][0][0] * 2  # F16 n_embd * 2

    survmap = {}
    for L in range(NLAYER):
        if policy[L] == 1:
            continue
        name = f"blk.{L}.ffn_gate_inp.weight"
        sdims, soff, styp = src_r[name]; rdims, roff, rtyp = rp_r[name]
        assert styp == 1 and rtyp == 1, (L, styp, rtyp)  # F16 routers
        assert sdims[1] == 256 and rdims[1] == keep[L], (L, sdims, rdims, keep[L])
        full = rng(SRC_URL, src_ds + soff, 256 * ROWB)
        comp = rng(REAP_URL, rp_ds + roff, keep[L] * ROWB)
        idx = {}
        for i in range(256):
            idx.setdefault(full[i*ROWB:(i+1)*ROWB], i)
        surv = sorted(idx[comp[j*ROWB:(j+1)*ROWB]] for j in range(keep[L]))
        assert len(surv) == keep[L] and len(set(surv)) == keep[L], L
        survmap[L] = surv
        print(f"layer {L:2d}: matched {keep[L]}/{keep[L]}", flush=True)

    doc = {
        "_about": "REAP-25 (LiveCodeBench-50-calibrated) routed-expert survivor map for "
                  "DeepSeek-V4-Flash, used by the ds4 v5mx REAP prune (quantizer --reap-survivors).",
        "_source": {
            "artifact": "eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4",
            "file": "DeepSeek-V4-Flash-REAP25-LCB50-DS4-compact-IQ2XXS.gguf",
            "url": "https://huggingface.co/eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4",
            "reap_plan_path_in_source": "artifacts/ds4-original/reap-observations/reap25_livecodebench50_plan.json",
            "calibration": "LiveCodeBench, 50 samples (easy 17 / medium 17 / hard 16), seed 42",
            "method": "REAP (Router-weighted Expert Activation Pruning), https://arxiv.org/abs/2510.13999",
        },
        "_recovery": "Recovered by byte-matching eouya2's compacted F16 ffn_gate_inp rows against "
                     "the full 256-row reference router in antirez/deepseek-v4-gguf "
                     "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf "
                     "(the GGUF eouya2 pruned from). See recover_survivors.py.",
        "_license_note": "eouya2 HF tag is 'license: other'; the bundled ds4 runtime LICENSE is MIT "
                         "and the underlying DeepSeek-V4-Flash weights are MIT (GGUF general.license='mit'). "
                         "Vendored here is DERIVED INTEGER INDEX METADATA, not model weights, with attribution.",
        "layout": "ds4-compact-v1",
        "n_layers": NLAYER,
        "n_expert": 256,
        "top_k": 6,
        "compression_ratio": 0.25,
        "policy": policy,
        "keep_count": keep,
        "expert_count": [256] * NLAYER,
        "survivors": {str(L): survmap[L] for L in sorted(survmap)},
    }
    json.dump(doc, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(survmap)} pruned layers", flush=True)


if __name__ == "__main__":
    main()
