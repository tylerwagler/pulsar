"""L286 cost model: predicted dKL(allocation vs U-e4-d5) = sum over (block, family) of a MEASURED response curve
evaluated at the block's local-error change.

local error  L251's full-KL record (`research/l251/full-kl/results.json` record.layers): the U-e4-d5 tensors' EXL3
             Hessian proxy (experts gate_up / down K4 mean over 512 experts; dense K5 per tensor) and the proxy at every
             other K the recipes needed.  proxy(K) / proxy(K_U) is universal (cv <= 3% across layers and tensors;
             `ratios()` fits it from that record), so local_error(L, K) = proxy_U(L) x RATIO[K].  MXFP8 dense tensors
             have no proxy in the record; gains.py's REF pass measures every quantized dense tensor's held-out local
             error (||X dW^T||^2 / ||X W^T||^2 on the BF16 input) and the dense family uses those.
curve        gains.py moves one family of a block of layers along its own quantization error by s: the block's proxy
             changes by x = s^2 - 1 units of its U proxy (s = 0.5, 2, 4 -> x = -0.75, 3, 15: K5-, K3-, K2-like).  The
             measured paired dKL at those x, plus (0, 0), is the block's response curve; it is evaluated
             piecewise-linearly (the end segments' slopes continue outside the measured range).
An allocation's terms are added over blocks and families: additivity ACROSS blocks is the assumption the known
recipes test (`analyze.py`); within a block the curve is measured, interactions included.
"""
import random, statistics

PFX = "model.language_model.layers."
EXPERT = {"gu": "mlp.experts.gate_up_proj", "dn": "mlp.experts.down_proj"}
REC_PART = {"gu": "gate_up", "dn": "down"}


def ratios(record):
    """proxy(K) / proxy(K_U) from the record: experts vs K4, dense EXL3 vs K5 (mean over every layer that has both)."""
    ex, de = {}, {}
    for r in record.values():
        for p in ("gate_up", "down"):
            b = r["experts"].get(f"{p}|K4")
            for K in (2, 3, 5):
                o = r["experts"].get(f"{p}|K{K}")
                if b and o:
                    ex.setdefault(K, []).append(o["proxy_err_mean"] / b["proxy_err_mean"])
        for k, v in r["dense"].items():
            nm, f = k.split("|")
            if f == "EXL3_K5":
                for K in (2, 3, 4):
                    o = r["dense"].get(f"{nm}|EXL3_K{K}")
                    if o:
                        de.setdefault(K, []).append(o["proxy_err"] / v["proxy_err"])
    rx = {K: statistics.mean(v) for K, v in ex.items()}
    rx[4] = 1.0
    rx[6] = rx[5] ** 2                         # NOT measured: one more bit at K5's per-bit factor (flagged)
    rd = {K: statistics.mean(v) for K, v in de.items()}
    rd[5] = 1.0
    return {"expert_vs_K4": rx, "dense_vs_K5": rd,
            "cv": {"expert": {K: statistics.pstdev(v) / statistics.mean(v) for K, v in ex.items()},
                   "dense": {K: statistics.pstdev(v) / statistics.mean(v) for K, v in de.items()}},
            "n": {"expert": {K: len(v) for K, v in ex.items()}, "dense": {K: len(v) for K, v in de.items()}}}


def expert_proxy(record, L, f):
    return record[str(L)]["experts"][f"{REC_PART[f]}|K4"]["proxy_err_mean"]


def dense_proxy(dense_local, L):
    return sum(v for k, v in dense_local.items() if k.startswith(f"{PFX}{L}."))


def curve_points(branches, x0, x1, f):
    """[(x, dkl, lo, hi)] for one (block, family), the origin included."""
    pts = [(0.0, 0.0, 0.0, 0.0)]
    for b in branches:
        if b["layers"] == [x0, x1] and b["family"] == f:
            s = b["s"]
            pts.append((s * s - 1, b["dkl"], b["ci"][0], b["ci"][1]))
    return sorted(pts)


def curve_eval(pts, x, which=1):
    """Piecewise-linear dKL at proxy change x (which: 1 point, 2 CI low, 3 CI high); end slopes continue."""
    if len(pts) < 2:
        raise SystemExit("a curve needs a measured point besides the origin")
    i = 0
    while i < len(pts) - 2 and x > pts[i + 1][0]:
        i += 1
    a, b = pts[i], pts[i + 1]
    return a[which] + (x - a[0]) / (b[0] - a[0]) * (b[which] - a[which])


def expert_x(record, R, x0, x1, f, Ks):
    """The block's proxy change (units of its U proxy) when layer L takes K = Ks[L]."""
    base = sum(expert_proxy(record, L, f) for L in range(x0, x1 + 1))
    return sum(expert_proxy(record, L, f) * (R["expert_vs_K4"][Ks[L]] - 1) for L in range(x0, x1 + 1)) / base


# ------------------------------------------------------------------ bytes (the allocator's resident accounting)
def exl3_bytes(k_in, n_out, K):
    return (k_in // 16) * (n_out // 16) * 16 * K * 2 + (k_in + n_out) * 2


def expert_layer_bytes(f, K):
    return 512 * (exl3_bytes(2560, 1280, K) if f == "gu" else exl3_bytes(640, 2560, K))


# ------------------------------------------------------------------ statistics
def bootstrap_ratio(num_seq, den_seq, n=10000, seed=0):
    """Paired bootstrap over sequences of sum(num) / sum(den): (point, lo, hi)."""
    rnd, m = random.Random(seed), len(num_seq)
    rs = []
    for _ in range(n):
        idx = [rnd.randrange(m) for _ in range(m)]
        rs.append(sum(num_seq[i] for i in idx) / sum(den_seq[i] for i in idx))
    rs.sort()
    return sum(num_seq) / sum(den_seq), rs[int(0.025 * n)], rs[int(0.975 * n)]


def bootstrap_mean_diff(diff_seq_sums, tok_counts, n=10000, seed=0):
    """Paired per-token mean difference, bootstrap over sequences: (point, lo, hi)."""
    rnd, m = random.Random(seed), len(diff_seq_sums)
    rs = []
    for _ in range(n):
        idx = [rnd.randrange(m) for _ in range(m)]
        rs.append(sum(diff_seq_sums[i] for i in idx) / sum(tok_counts[i] for i in idx))
    rs.sort()
    return sum(diff_seq_sums) / sum(tok_counts), rs[int(0.025 * n)], rs[int(0.975 * n)]


def spearman(a, b):
    def rank(x):
        r = [0] * len(x)
        for i, j in enumerate(sorted(range(len(x)), key=lambda i: x[i])):
            r[j] = i
        return r
    ra, rb, n = rank(a), rank(b), len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))
