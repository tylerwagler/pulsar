#!/usr/bin/env python3
"""fmtgrade.py -- the L300 drafter weight-format grade: one block drafter (DFlash2) in several weight formats, replayed
on the engine's own taps (a PULSAR_BLOCK_DUMP, greedy) in ref.py's arithmetic and graded on the target's own tokens.
CPU only.

    validate  the loaders, before any grade: the container's native tensors are the BF16 checkpoint's bytes; the
              builder's producer re-makes the container's nvfp4 bytes (nvfp4q == nvfp4, the head included); every
              arm's Linears against BF16 per tensor; each arm's draft against the reference arm's on a small context
              (synthetic taps, and a dump's real ones with --dump)
    grade     per arm on a dump, two ways:
                served   the arm's OWN greedy rounds over the dump's known target tokens (replay.py's served rounds:
                         anchor p, accept up to the first mismatch, next anchor p + accepted + 1; the full block, no
                         stop rule) -- the L292 / L299 tokens-per-round figure
                anchors  the arm's draft at the ENGINE's anchors (every dumped pass) against the target's tokens, with
                         KL(reference || arm) and top-1 agreement per draft slot over the draft vocabulary, and how
                         often the arm's chain is the one the engine drafted (the served format's fidelity)
              plus the engine's own served rounds as dumped (its stop rule and pricing included), and `cost`
    cost      the byte model: bytes each format moves per draft pass and per absorb (one verify round)

The dump (src/engine/spec_block.cpp, PULSAR_BLOCK_DUMP=<dir> PULSAR_BLOCK_DUMP_PASSES=N) holds every absorbed context
row (the prompt's -- cold -- and each round's carry + accepted rows) and the first N draft passes (bank, anchor P,
carry, chain).  The target's tokens come out of it: a pass at P whose round absorbed n rows fixes S[P] = carry and
S[P+1 .. P+n-1] = chain[:n-1] (greedy: the accepted rows are the chain's prefix); the next round's pass fixes the
correction token.  A round with no pass (a copy draft, or past the N passes) leaves its tokens unknown, so the known
tokens fall into spans; a draft running past a span's end is censored (counted, not graded).  A bank's rows going
back to an earlier position start a new epoch (the next request on that bank), which inherits the rows below it.

Arms: NAME=linears:FMT,head:FMT with FMT bf16 | nvfp4 | nvfp4q | exl3:FILE (weights.py); the first arm is the KL
reference.  Everything but the Linears and the head is the BF16 checkpoint's bytes in every arm (norms, conv base
kernels, the CandidateSelector), as the served container stores them; the head is always the draft-vocabulary head
(the container's drafter.lm_head.ids).

    python3 tools/blockdraft/fmtgrade.py validate --drafter /mnt/models/qwen38fn-dflash2 --target-hf HF \
        --container CONTAINER [--arm ...] [--dump DIR]
    python3 tools/blockdraft/fmtgrade.py grade --dump DIR --drafter /mnt/models/qwen38fn-dflash2 --target-hf HF \
        --container CONTAINER [--arm ...] --out OUT.json
    python3 tools/blockdraft/fmtgrade.py cost --drafter ... --target-hf HF --container CONTAINER [--arm ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ref as R  # noqa: E402
import weights as WS  # noqa: E402

torch.set_grad_enabled(False)
BF16 = torch.bfloat16
DEFAULT_ARMS = ["bf16=linears:bf16,head:bf16", "nvfp4=linears:nvfp4,head:nvfp4"]
GB10_BW = 273e9   # GB10 LPDDR5x peak bytes/s: the cost model's floor, not a measurement
L292_TAPS, L292_H = (5, 15, 25, 35, 45), 2560   # replay.py's tap layers and hidden width (the L292 tap files)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---- the dump ------------------------------------------------------------------------------------------------------
def read_passes(d):
    """draft.bin's passes (ref.read_dump's records without the final hidden, log-probs and scores: not graded here)"""
    passes = []
    with open(os.path.join(d, "draft.bin"), "rb") as f:
        while True:
            h = f.read(16)
            if not h:
                break
            nb, B, E, k = struct.unpack("<4I", h)
            S = B - 1
            rec = dict(nb=nb, B=B, E=E, k=k)
            rec["bank"] = np.frombuffer(f.read(4 * nb), dtype=np.int32)
            rec["carry"] = np.frombuffer(f.read(4 * nb), dtype=np.int32)
            rec["P"] = np.frombuffer(f.read(4 * nb), dtype=np.int32)
            f.seek(2 * nb * B * E, 1)
            rec["ids"] = np.frombuffer(f.read(4 * nb * S * k), dtype=np.int32).reshape(nb, S, k)
            f.seek(4 * nb * S * k + 4 * nb * S * k * k, 1)
            rec["chain"] = np.frombuffer(f.read(4 * nb * S), dtype=np.int32).reshape(nb, S)
            passes.append(rec)
    return passes


class Dump:
    """ctx.bin indexed by its headers (taps read on demand), the passes, the epochs and the target's tokens."""

    def __init__(self, d):
        self.d = d
        self.recs = []          # (file offset of the taps, n, pos[n], bank[n], width)
        with open(os.path.join(d, "ctx.bin"), "rb") as f:
            while True:
                h = f.read(4)
                if not h:
                    break
                n = struct.unpack("<I", h)[0]
                pos = np.frombuffer(f.read(4 * n), dtype=np.int32).copy()
                bank = np.frombuffer(f.read(4 * n), dtype=np.int32).copy()
                w = struct.unpack("<I", f.read(4))[0]
                self.recs.append((f.tell(), n, pos, bank, w))
                f.seek(2 * n * w, 1)
        if not self.recs:
            raise SystemExit(f"{d}/ctx.bin holds no rows")
        self.width = self.recs[0][4]
        self.passes = read_passes(d)
        if not self.passes:
            raise SystemExit(f"{d}/draft.bin holds no passes")
        self.B, self.k = self.passes[0]["B"], self.passes[0]["k"]
        self._epochs()

    def _epochs(self):
        runs = {}
        for ri, (_off, n, pos, bank, _w) in enumerate(self.recs):
            i = 0
            while i < n:   # the absorb's contract: each bank's rows one run of consecutive positions
                b, j = int(bank[i]), i
                while j + 1 < n and bank[j + 1] == b and pos[j + 1] == pos[j] + 1:
                    j += 1
                runs.setdefault(b, []).append((int(pos[i]), j - i + 1, ri, i))
                i = j + 1
        self.epochs = []
        for b, rl in runs.items():
            cur = None
            for r in rl:
                if cur is None or r[0] <= cur["runs"][-1][0] + cur["runs"][-1][1] - 1:
                    cur = dict(bank=b, runs=[], parent=cur, id=len(self.epochs), passes=[])
                    self.epochs.append(cur)
                cur["runs"].append(r)
        for e in self.epochs:
            e["at"] = {r[0]: k for k, r in enumerate(e["runs"])}
            # a speculative epoch absorbs multi-row rounds; one of single rows only is a plain decode (no passes),
            # which a pass whose P it also covers must not be placed in (L299 w1: plain greedy, then spec, one bank)
            e["spec"] = any(1 < r[1] <= self.B for r in e["runs"][1:])
        by_bank = {}
        for e in self.epochs:
            by_bank.setdefault(e["bank"], []).append(e)
        ptr = {}
        self.unplaced = 0
        for pi, ps in enumerate(self.passes):
            for bi in range(ps["nb"]):
                b, P = int(ps["bank"][bi]), int(ps["P"][bi])
                es, k0 = by_bank.get(b, []), ptr.get(b, 0)
                fits = [k for k in range(k0, len(es)) if P in es[k]["at"] and
                        (not es[k]["passes"] or es[k]["passes"][-1]["P"] < P)]
                k = next((k for k in fits if es[k]["spec"]), fits[0] if fits else len(es))
                if k == len(es):
                    self.unplaced += 1
                    continue
                ptr[b] = k
                es[k]["passes"].append(dict(pi=pi, bi=bi, P=P, carry=int(ps["carry"][bi]),
                                            chain=ps["chain"][bi].tolist(), top1=ps["ids"][bi][:, 0].tolist()))
        self.inconsistent = 0
        for e in self.epochs:
            S = {}
            for p in e["passes"]:
                n = e["runs"][e["at"][p["P"]]][1]
                p["absorbed"] = n
                if n - 1 > len(p["chain"]):
                    self.inconsistent += 1       # more rows accepted than drafted: not this pass's round
                    continue
                S[p["P"]] = p["carry"]
                for j in range(n - 1):
                    S[p["P"] + 1 + j] = p["chain"][j]
            e["S"] = S

    def rows(self, e, lo, hi):
        """taps (u16 bf16 bits [hi - lo][width], present [hi - lo]) of epoch e's positions lo .. hi - 1"""
        out = np.zeros((hi - lo, self.width), dtype=np.uint16)
        have = np.zeros(hi - lo, dtype=bool)
        with open(os.path.join(self.d, "ctx.bin"), "rb") as f:
            ep, top = e, hi
            while ep is not None and lo < top:
                for start, n, ri, r0 in ep["runs"]:
                    a, z = max(start, lo), min(start + n, top)
                    if a >= z:
                        continue
                    off, _n, _pos, _bank, w = self.recs[ri]
                    f.seek(off + 2 * w * (r0 + a - start))
                    out[a - lo:z - lo] = np.frombuffer(f.read(2 * w * (z - a)), dtype=np.uint16).reshape(z - a, w)
                    have[a - lo:z - lo] = True
                top = min(top, ep["runs"][0][0])   # an earlier request supplies only the rows below this one's
                ep = ep["parent"]
        return out, have

    @staticmethod
    def spans(e):
        """maximal runs of consecutive known target positions [(s0, s1)], inclusive"""
        out = []
        for x in sorted(e["S"]):
            if out and x == out[-1][1] + 1:
                out[-1][1] = x
            else:
                out.append([x, x])
        return [tuple(s) for s in out]

    def summary(self):
        np_ = sum(len(e["passes"]) for e in self.epochs)
        per = ", ".join("bank %d: %d passes, %d known tokens" % (e["bank"], len(e["passes"]), len(e["S"]))
                        for e in self.epochs if e["passes"])
        return (f"{len(self.recs)} ctx records, {sum(r[1] for r in self.recs)} rows, {len(self.passes)} passes "
                f"(block {self.B}, top-{self.k}), {len(self.epochs)} epoch(s) [{per}]; {np_} placed, "
                f"{self.unplaced} unplaced, {self.inconsistent} inconsistent")


# ---- the arms ------------------------------------------------------------------------------------------------------
class ArmRun:
    """One arm's drafter (ref.Drafter over a weights.Arm) and its per-epoch context K / V rows."""

    def __init__(self, name, lfmt, hfmt, a, cache):
        self.name, self.lfmt, self.hfmt = name, lfmt, hfmt
        self.src = WS.Arm(name, lfmt, hfmt, a.drafter, a.target_hf, a.container, cache)
        t0 = time.time()
        self.D = R.Drafter(a.drafter, None, torch.device("cpu"), src=self.src)
        log(f"arm {name}: Linears {lfmt}, head {hfmt} ({len(self.src.ids)} rows) loaded in {time.time() - t0:.0f} s")
        self.kv = {}

    def context(self, eid, base, taps_u16):
        """K / V of positions base .. base + n - 1 (taps u16 [n][width]) for epoch eid"""
        kvs = None
        for a in range(0, taps_u16.shape[0], 1024):
            t = R.u16(taps_u16[a:a + 1024], self.D.dev)
            _fc, cn = self.D.context(t)
            part = self.D.ctx_kv(cn, torch.arange(base + a, base + a + t.shape[0]))
            kvs = part if kvs is None else [(torch.cat([k0, k1]), torch.cat([v0, v1]))
                                            for (k0, v0), (k1, v1) in zip(kvs, part)]
        self.kv[eid] = (base, kvs)

    def draft(self, eid, p, carry):
        """(chain [B-1], log-probs over the draft vocabulary [B-1][rows]) of the draft anchored at p"""
        base, kvs = self.kv[eid]
        B = self.D.block
        lo = max(0, p - (self.D.W - 1) - (B - 1))
        if lo < base:
            raise SystemExit(f"anchor {p}: context from {lo} below the computed rows ({base})")
        kv = [(k[lo - base:p - base], v[lo - base:p - base]) for k, v in kvs]
        _inter, o, lp = self.D.forward(None, torch.arange(lo, p), p, carry, kv=kv)
        return self.D.select(o, lp, carry), lp[1:]


def accept(chain, S, p, B):
    """(accepted, censored): the chain at anchor p against the known tokens S"""
    a = 0
    while a < B - 1:
        t = S.get(p + 1 + a)
        if t is None:
            return a, True
        if chain[a] != t:
            return a, False
        a += 1
    return a, False


def by_position(acc, B):
    """P(slot j accepted | slot j verified alive) for j = 0 .. B-2, and tokens / round"""
    R_ = len(acc)
    alive = [sum(1 for a in acc if a >= j) for j in range(B - 1)]
    ok = [sum(1 for a in acc if a >= j + 1) for j in range(B - 1)]
    return dict(rounds=R_, tokens_per_round=(sum(a + 1 for a in acc) / R_) if R_ else None,
                acc_by_pos=[round(o / al, 4) if al else None for o, al in zip(ok, alive)],
                p_kept_ge=[round(o / R_, 4) if R_ else None for o in ok])


# ---- the byte model ------------------------------------------------------------------------------------------------
def linear_bytes(fmt, out, inp, arm: WS.Arm, hf):
    if fmt == "bf16":
        return 2 * out * inp
    if fmt in ("nvfp4", "nvfp4q"):
        return out * inp // 2 + WS.PR._rup(out, 128) * (inp // 16) + WS.PR.NVFP4_TAIL
    if fmt.startswith("exl3:"):
        ex = arm.cache.exl3(fmt[5:])
        name = hf[:-len(".weight")]
        tr = ex.s.entry(name + ".trellis")["shape"]
        return 2 * tr[0] * tr[1] * tr[2] + 2 * (16 * tr[0] + 16 * tr[1])
    raise SystemExit(fmt)


def cost(arm: WS.Arm, cfg):
    """bytes per draft pass (the block forward's Linears + the head + the selector projection) and per absorb (fc and
    each layer's k / v over the round's rows: the weights once a round)"""
    H, L, NQ, NKV, HD, FF = (cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"],
                             cfg["num_key_value_heads"], cfg["head_dim"], cfg["intermediate_size"])
    dc = cfg["dflash_config"]
    kp = 2 * dc["conv_kernel_size"] * H // dc["conv_group_size"]
    shapes = {"self_attn.q_proj": (NQ * HD, H), "self_attn.k_proj": (NKV * HD, H), "self_attn.v_proj": (NKV * HD, H),
              "self_attn.o_proj": (H, NQ * HD), "mlp.gate_proj": (FF, H), "mlp.up_proj": (FF, H),
              "mlp.down_proj": (H, FF), "attention_conv.kernel_projection": (kp, H),
              "mlp_conv.kernel_projection": (kp, H)}
    draft = absorb = 0
    for l in range(L):
        for n, (o, i) in shapes.items():
            hf = f"layers.{l}.{n}.weight"
            b = linear_bytes(arm.lfmt, o, i, arm, hf)
            draft += b
            if n in ("self_attn.k_proj", "self_attn.v_proj"):
                absorb += b
    absorb += linear_bytes(arm.lfmt, H, len(dc["target_layer_ids"]) * H, arm, "fc.weight")
    head = linear_bytes(arm.hfmt, len(arm.ids), H, arm, "lm_head.weight")
    sel = 2 * dc["selector_rank"] * H
    draft_total = draft + head + sel
    return dict(draft_linears=draft, head=head, selector=sel, draft_pass=draft_total, absorb=absorb,
                round=draft_total + absorb, floor_ms_round=round(1e3 * (draft_total + absorb) / GB10_BW, 3))


# ---- commands ------------------------------------------------------------------------------------------------------
def make_arms(a, cache):
    specs = a.arm or DEFAULT_ARMS
    return [ArmRun(*WS.parse_arm(s), a, cache) for s in specs]


def rel(x, y):
    return float((x - y).norm() / y.norm().clamp_min(1e-30))


def compare_drafts(ref_arm, arm, eid, anchors):
    """per anchor: KL(ref || arm) per slot, top-1 agreement, chain equality"""
    kl, t1, same = [], [], []
    for p, carry in anchors:
        c0, l0 = ref_arm.draft(eid, p, carry)
        c1, l1 = arm.draft(eid, p, carry)
        kl.append((l0.exp() * (l0 - l1)).sum(-1))
        t1.append((l0.argmax(-1) == l1.argmax(-1)).float())
        same.append(float(c0 == c1))
    kl, t1 = torch.stack(kl), torch.stack(t1)
    return dict(kl_by_slot=[round(float(x), 5) for x in kl.mean(0)], kl=round(float(kl.mean()), 5),
                top1_by_slot=[round(float(x), 3) for x in t1.mean(0)], chain_equal=round(float(np.mean(same)), 3))


def cmd_validate(a):
    cache = WS.Cache()
    c = cache.shards(a.container)
    d = cache.shards(a.drafter)
    log("1. the container's native drafter tensors vs the BF16 checkpoint's bytes")
    nat = bad = 0
    for name in sorted(c.at):
        if not name.startswith(WS.DRAFTER) or c.decl(name).get("layout") != "bf16":
            continue
        nat += 1
        if c.raw(name) != d.raw(name[len(WS.DRAFTER):]):
            bad += 1
            print(f"  DIFFER {name}")
    print(f"  {nat - bad}/{nat} byte-identical")
    arms = make_arms(a, cache)
    base = arms[0]
    log("2. nvfp4q (producers.nvfp4_from_bf16 on the BF16 source) vs the container's nvfp4 bytes")
    same = tot = 0
    for name in sorted(c.at):
        if not name.startswith(WS.DRAFTER) or c.decl(name).get("layout") != "nvfp4":
            continue
        tot += 1
        eq = cache.nvfp4q_entry(name[len(WS.DRAFTER):], base.src)[2] == c.raw(name)
        same += int(eq)
        if not eq:
            print(f"  DIFFER {name}")
    print(f"  {same}/{tot} byte-identical (the head's rows: the target's BF16 lm_head at the draft vocabulary's ids)")
    log("3. every arm's Linears vs BF16, per tensor (rel Frobenius error, max |d|, cos)")
    names = [n for n in sorted(c.at) if n.startswith(WS.DRAFTER) and c.decl(n).get("layout") == "nvfp4"]
    stats = {}
    for arm in arms:
        rows = []
        for name in names:
            hf = name[len(WS.DRAFTER):]
            Wr = cache.linear("bf16", hf, arm.src)[0]
            fmt = arm.hfmt if name == WS.HEAD else arm.lfmt
            Wq, s2 = cache.linear(fmt, hf, arm.src)
            Wq = Wq * s2 if s2 is not None else Wq
            e = Wq - Wr
            rows.append((hf, rel(Wq, Wr), float(e.abs().max()), float(torch.nn.functional.cosine_similarity(
                Wq.reshape(1, -1), Wr.reshape(1, -1)))))
        r = np.array([x[1] for x in rows])
        stats[arm.name] = dict(per_tensor={x[0]: dict(rel=round(x[1], 5), max_abs=x[2], cos=round(x[3], 6))
                                           for x in rows}, rel_mean=float(r.mean()), rel_max=float(r.max()))
        worst = max(rows, key=lambda x: x[1])
        print(f"  {arm.name:10s} rel mean {r.mean():.4f} max {r.max():.4f} ({worst[0]}); head "
              f"{dict((x[0], x[1]) for x in rows)['lm_head.weight']:.4f}")
    log("4. drafts vs the reference arm on a small context")
    out = dict(natives=[nat - bad, nat], nvfp4q_vs_container=[same, tot], weights=stats, drafts={})
    g = torch.Generator().manual_seed(1234)
    n = 96
    taps = (torch.randn(n, base.D.fc[0].shape[1], generator=g) * 2.0).to(BF16)
    u = taps.view(torch.int16).numpy().view(np.uint16)
    V = base.D.emb.shape[0]
    anchors = [(p, int(torch.randint(0, V, (1,), generator=g))) for p in (24, 48, 72, 95)]
    for arm in arms:
        arm.context("synthetic", 0, u)
    for arm in arms[1:]:
        r = compare_drafts(base, arm, "synthetic", anchors)
        out["drafts"].setdefault("synthetic", {})[arm.name] = r
        print(f"  synthetic taps ({n} rows, {len(anchors)} anchors) {arm.name:10s} vs {base.name}: KL {r['kl']:.4f} "
              f"(by slot {r['kl_by_slot']}), top-1 {r['top1_by_slot']}, chain equal {r['chain_equal']}")
    if a.dump:
        dm = Dump(a.dump)
        e = max((e for e in dm.epochs if e["passes"]), key=lambda e: len(e["passes"]))
        ps = e["passes"][:a.rounds or None]
        lo = max(0, min(p["P"] for p in ps) - (base.D.W - 1) - (dm.B - 1))
        hi = max(p["P"] for p in ps)
        t, have = dm.rows(e, lo, hi)
        if not have.all():
            raise SystemExit(f"the dump lacks {int((~have).sum())} context rows in {lo}..{hi}")
        for arm in arms:
            arm.context(e["id"], lo, t)
        anchors = [(p["P"], p["carry"]) for p in ps]
        for arm in arms[1:]:
            r = compare_drafts(base, arm, e["id"], anchors)
            out["drafts"].setdefault("dump", {})[arm.name] = r
            print(f"  dump taps ({len(anchors)} anchors) {arm.name:10s} vs {base.name}: KL {r['kl']:.4f} "
                  f"(by slot {r['kl_by_slot']}), top-1 {r['top1_by_slot']}, chain equal {r['chain_equal']}")
        for arm in arms:
            eq = [float(arm.draft(e["id"], p["P"], p["carry"])[0] == p["chain"]) for p in ps]
            print(f"  {arm.name:10s} chain == the engine's dumped chain: {np.mean(eq):.3f} of {len(eq)} passes")
            out["drafts"].setdefault("engine_chain_equal", {})[arm.name] = float(np.mean(eq))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        log(f"wrote {a.out}")


def taps_fixture(spec, B):
    """NAME:TAPSDIR:IDS:PROMPT_LEN -- an L292 tap fixture (tools/dflash2/replay.py's inputs: per tap layer the HC
    streams dumped by the engine, the prompt + greedy reply ids) as a grade source: no engine passes, served rounds
    over the whole reply"""
    name, tdir, ids, plen = spec.split(":")
    S = np.fromfile(ids, dtype=np.int32).tolist()
    P = int(plen)
    feats = []

    for il in L292_TAPS:
        rows = np.full((len(S), 4, L292_H), np.nan, dtype=np.float32)
        for f in os.listdir(tdir):
            m = re.match(rf"t_qwen_h-{il}_pos(\d+)(?:_c(\d+))?\.bin$", f)
            if not m or int(m[2] or 1) != 1:
                continue
            x = np.fromfile(os.path.join(tdir, f), dtype=np.float32).reshape(-1, 4, L292_H)
            last = int(m[1])
            rows[last - x.shape[0] + 1:last + 1] = x
        if np.isnan(rows).any():
            raise SystemExit(f"{tdir}: layer {il} positions missing")
        # the 4-stream mean rounded to bf16, as replay.py's load_taps (SGLang's .mean on a bf16 tensor)
        feats.append(torch.from_numpy(rows).mean(dim=1).to(BF16))
    taps = torch.cat(feats, dim=-1).view(torch.int16).numpy().view(np.uint16)

    def rows_fn(lo, hi):
        return taps[lo:hi], np.ones(hi - lo, dtype=bool)

    return dict(id=f"fx-{name}", label=name, bank=None, passes=[], rows=rows_fn,
                S={x: S[x] for x in range(P, len(S))}, spans=[(P, len(S) - 1)])


def cmd_grade(a):
    sources, B, dm = [], None, None
    if a.dump:
        dm = Dump(a.dump)
        log("dump:", dm.summary())
        B = dm.B
        for e in dm.epochs:
            if e["passes"]:
                e["label"] = f"bank{e['bank']}.e{e['id']}"
                e["rows"] = (lambda e_: (lambda lo, hi: dm.rows(e_, lo, hi)))(e)
                e["spans"] = [s for s in dm.spans(e) if s[1] > s[0]]
                sources.append(e)
    B = B or a.block
    for spec in a.l292 or []:
        sources.append(taps_fixture(spec, B))
    if not sources:
        raise SystemExit("grade: --dump DIR and / or --l292 NAME:TAPSDIR:IDS:PROMPT_LEN")
    cache = WS.Cache()
    arms = make_arms(a, cache)
    cfg = json.load(open(os.path.join(a.drafter, "config.json")))
    if cfg["dflash_config"]["block_size"] != B:
        log(f"note: grading at block {B}; the checkpoint's block_size is {cfg['dflash_config']['block_size']}")
    Wn = arms[0].D.W
    eng = {}
    res = {arm.name: {} for arm in arms}
    for e in sources:
        ps = e["passes"][:a.rounds] if a.rounds else e["passes"]
        spans = e["spans"]
        starts = [p["P"] for p in ps] + [s[0] for s in spans]
        hi = max([p["P"] for p in ps] + [s[1] for s in spans])
        lo = max(0, min(starts) - (Wn - 1) - (B - 1))
        t0 = time.time()
        taps, have = e["rows"](lo, hi)
        miss = np.flatnonzero(~have)
        log(f"source {e['label']}: {len(ps)} engine passes, known spans {spans[:4]}{' ...' if len(spans) > 4 else ''}"
            f", context rows {lo}..{hi - 1} ({len(miss)} missing), read in {time.time() - t0:.0f} s")
        if len(miss):
            log("  rows missing (the dump was not cold, or a request resumed a prefix): an anchor whose window "
                "lacks one is skipped")
        for arm in arms:
            t0 = time.time()
            arm.context(e["id"], lo, taps)
            log(f"  {arm.name}: context K/V of {hi - lo} rows in {time.time() - t0:.0f} s")

        def window_ok(p):
            lo_p = max(0, p - (Wn - 1) - (B - 1))
            return not ((miss >= lo_p - lo) & (miss < p - lo)).any()

        S = e["S"]
        if ps:   # the engine's own rounds as dumped (its stop rule and pricing included)
            eng[e["label"]] = [p["absorbed"] - 1 for p in ps if "absorbed" in p]
        for arm in arms:
            res[arm.name][e["label"]] = dict(served=[], anchors=[], censored_served=0, censored_anchor=0, kl=[],
                                             top1=[], chain_ref=[], chain_engine=[], top1_engine=[])
        t0 = time.time()
        for p in ps:   # every arm at the engine's anchors
            if not window_ok(p["P"]):
                continue
            outs = {arm.name: arm.draft(e["id"], p["P"], p["carry"]) for arm in arms}
            c0, l0 = outs[arms[0].name]
            for arm in arms:
                c, lp = outs[arm.name]
                r = res[arm.name][e["label"]]
                acc, cen = accept(c, S, p["P"], B)
                if cen:
                    r["censored_anchor"] += 1
                else:
                    r["anchors"].append(acc)
                r["kl"].append([float(x) for x in (l0.exp() * (l0 - lp)).sum(-1)])
                r["top1"].append([int(x) for x in (l0.argmax(-1) == lp.argmax(-1))])
                r["chain_ref"].append(int(c == c0))
                r["chain_engine"].append(int(c == p["chain"]))
                r["top1_engine"].append([int(x == y) for x, y in zip(arm.src.ids[lp.argmax(-1)].tolist(), p["top1"])])
        if ps:
            log(f"  anchors: {len(ps)} x {len(arms)} arms in {time.time() - t0:.0f} s")
        t0 = time.time()
        for arm in arms:   # each arm's own greedy rounds over the known spans
            r = res[arm.name][e["label"]]
            for s0, s1 in spans:
                p = s0
                while p < s1:
                    if not window_ok(p):
                        p += 1
                        continue
                    c, lp = arm.draft(e["id"], p, S[p])
                    acc, cen = accept(c, S, p, B)
                    if cen:                    # the draft ran past the span's known tokens
                        r["censored_served"] += 1
                        break
                    r["served"].append(acc)
                    p += acc + 1
        log(f"  served rounds: {len(arms)} arms in {time.time() - t0:.0f} s")
        for arm in arms:
            arm.kv.pop(e["id"], None)
    report(a, arms, res, eng, B, cfg, dm)


def report(a, arms, res, eng, B, cfg, dm):
    labels = list(next(iter(res.values())).keys())
    allv = lambda arm, k: [x for lab in labels for x in res[arm][lab][k]]   # noqa: E731
    f3 = lambda x: f"{x:.3f}" if x is not None else "-"                     # noqa: E731
    out = dict(dump=a.dump, l292=a.l292, dump_summary=dm.summary() if dm else None, block=B, reference=arms[0].name,
               sources=labels, engine_as_dumped={lab: by_position(v, B) for lab, v in eng.items()}, arms={})
    if eng:
        ev = [x for v in eng.values() for x in v]
        out["engine_as_dumped"]["all"] = by_position(ev, B)
        print(f"\nengine as dumped (its stop rule and pricing): tok/rnd {f3(by_position(ev, B)['tokens_per_round'])}"
              f" over {len(ev)} rounds; acc by pos {by_position(ev, B)['acc_by_pos']}")
    print(f"\n{'arm':10s} {'served':>7s} {'rounds':>6s} {'anchor':>7s} {'KL':>8s} {'top-1':>6s} {'=ref':>6s} "
          f"{'=eng':>6s} {'MB/pass':>8s} {'MB/rnd':>7s} {'floor ms':>8s}   per source served tok/rnd")
    for arm in arms:
        n = arm.name
        sv, an = by_position(allv(n, "served"), B), by_position(allv(n, "anchors"), B)
        kl, t1 = np.array(allv(n, "kl") or [[0.0] * (B - 1)]), np.array(allv(n, "top1") or [[0] * (B - 1)])
        te, cr, ce = allv(n, "top1_engine"), allv(n, "chain_ref"), allv(n, "chain_engine")
        co = cost(arm.src, cfg)
        per = {lab: by_position(res[n][lab]["served"], B)["tokens_per_round"] for lab in labels}
        o = dict(linears=arm.lfmt, head=arm.hfmt, served=sv, anchors=an, per_source_served=per,
                 per_source_anchors={lab: by_position(res[n][lab]["anchors"], B) for lab in labels},
                 censored_served=sum(res[n][lab]["censored_served"] for lab in labels),
                 censored_anchor=sum(res[n][lab]["censored_anchor"] for lab in labels),
                 kl_by_slot=[round(float(x), 5) for x in kl.mean(0)], kl=round(float(kl.mean()), 5),
                 top1_vs_ref_by_slot=[round(float(x), 4) for x in t1.mean(0)],
                 top1_vs_engine_by_slot=[round(float(x), 4) for x in np.array(te).mean(0)] if te else None,
                 chain_equal_ref=round(float(np.mean(cr)), 4) if cr else None,
                 chain_equal_engine=round(float(np.mean(ce)), 4) if ce else None, cost=co,
                 acc={lab: dict(served=res[n][lab]["served"], anchors=res[n][lab]["anchors"]) for lab in labels})
        out["arms"][n] = o
        print(f"{n:10s} {f3(sv['tokens_per_round']):>7s} {sv['rounds']:>6d} {f3(an['tokens_per_round']):>7s} "
              f"{o['kl']:>8.4f} {float(t1.mean()):>6.3f} {f3(o['chain_equal_ref']):>6s} "
              f"{f3(o['chain_equal_engine']):>6s} {co['draft_pass'] / 1e6:>8.1f} {co['round'] / 1e6:>7.1f} "
              f"{co['floor_ms_round']:>8.2f}   " + " ".join(f"{lab} {f3(v)}" for lab, v in per.items()))
        print(f"{'':10s} served acc by pos {sv['acc_by_pos']}; anchors {an['acc_by_pos']}; KL by slot "
              f"{o['kl_by_slot']}")
    print("\nserved = the arm's own greedy rounds on the known target tokens (full block, no stop rule); anchor = its "
          "draft at the engine's anchors; KL / top-1 / =ref against the first arm per draft slot (draft vocabulary); "
          "=eng: chain identical to the engine's dumped chain; MB/pass = the draft pass's weight bytes, MB/rnd + the "
          f"absorb's (fc, k, v); floor ms = MB/rnd at {GB10_BW / 1e9:.0f} GB/s")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        log(f"wrote {a.out}")


def cmd_cost(a):
    cache = WS.Cache()
    cfg = json.load(open(os.path.join(a.drafter, "config.json")))
    print(f"{'arm':12s} {'linears':>10s} {'head':>8s} {'draft MB':>9s} {'absorb MB':>9s} {'round MB':>9s} "
          f"{'floor ms @' + str(int(GB10_BW / 1e9)) + 'GB/s':>16s}")
    for s in a.arm or DEFAULT_ARMS:
        n, lf, hf = WS.parse_arm(s)
        arm = WS.Arm(n, lf, hf, a.drafter, a.target_hf, a.container, cache)
        c = cost(arm, cfg)
        print(f"{n:12s} {c['draft_linears'] / 1e6:>10.1f} {c['head'] / 1e6:>8.1f} {c['draft_pass'] / 1e6:>9.1f} "
              f"{c['absorb'] / 1e6:>9.1f} {c['round'] / 1e6:>9.1f} {c['floor_ms_round']:>16.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["validate", "grade", "cost"])
    ap.add_argument("--drafter", required=True, help="the BF16 drafter checkpoint (config.json + model.safetensors)")
    ap.add_argument("--target-hf", required=True, help="the target's HF checkpoint (BF16 lm_head + input embedding)")
    ap.add_argument("--container", required=True, help="the pulsar container served (nvfp4 entries, head ids)")
    ap.add_argument("--arm", action="append", help="NAME=linears:FMT,head:FMT (repeatable; first = reference)")
    ap.add_argument("--dump", help="a PULSAR_BLOCK_DUMP directory")
    ap.add_argument("--l292", action="append",
                    help="NAME:TAPSDIR:IDS:PROMPT_LEN, an L292 tap fixture (tools/dflash2/replay.py's inputs; repeatable)")
    ap.add_argument("--block", type=int, default=8, help="the block size for --l292 sources alone (a dump has its own)")
    ap.add_argument("--rounds", type=int, default=0, help="passes per epoch to grade (0 = all)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    {"validate": cmd_validate, "grade": cmd_grade, "cost": cmd_cost}[a.cmd](a)


if __name__ == "__main__":
    main()
