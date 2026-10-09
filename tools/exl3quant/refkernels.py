"""The DeepSeek-V4.1 reference's `kernel` module (inference/kernel.py), in plain torch.

The reference's model.py does `from kernel import ...`; v41_stream.py installs this
module under that name before importing it, so the reference forward runs unmodified
on any device torch runs on -- no tilelang, whose fp8 GEMM returns NaN for M > 1 on
sm_100 (memory: v41-reference-sm100-broken).

Exactness.  Every GEMM operand the reference builds is an e4m3 or e2m1 value times a
power-of-two (e8m0) scale, which bf16 holds exactly.  fp8_gemm / fp4_gemm therefore
dequantize both operands to bf16 and run a bf16 matmul: fp32 accumulation and a bf16
result, the kernels' own contract, up to summation order (the driver turns off cuBLAS's
reduced-precision bf16 split-K reduction).  sparse_attn keeps fp32 scores as its kernel
does, so it multiplies in fp32 (TF32 on CUDA: bf16 inputs are exact there too).
Quantizers reproduce the kernels' scale rules bit for bit (amax floors, the fp32
`amax * (1/448)` product before the power-of-two ceiling, round-to-nearest-even casts
after the clamp).

Only the entry points and modes model.py calls exist here.
"""

import torch
import torch.nn.functional as F

FP8_MAX = 448.0
FP4_MAX = 6.0
# e2m1 code -> value, low nibble first (inference/convert.py FP4_TABLE)
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)


def _pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(v)) for positive normal fp32 v -- kernel.py fast_round_scale's bit trick:
    the exponent, plus one unless the mantissa is zero."""
    m, e = torch.frexp(v)  # v = m * 2^e, m in [0.5, 1)
    return torch.ldexp(torch.ones_like(v), torch.where(m == 0.5, e - 1, e))


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 already clamped to [-6, 6] onto the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6},
    ties to the even code (0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4)."""
    a = x.abs()
    r = torch.where(a <= 0.25, 0.0,
        torch.where(a < 0.75, 0.5,
        torch.where(a <= 1.25, 1.0,
        torch.where(a < 1.75, 1.5,
        torch.where(a <= 2.5, 2.0,
        torch.where(a < 3.5, 3.0,
        torch.where(a <= 5.0, 4.0, 6.0)))))))
    return torch.copysign(r, x)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: str | None = None,
              scale_dtype: torch.dtype = torch.float32, inplace: bool = False):
    """Block-wise e4m3 over the last dim.  Returns (e4m3, scales), or with inplace=True writes
    the dequantized values back into x (the reference's fused quant+dequant)."""
    n = x.size(-1)
    assert n % block_size == 0
    xb = x.float().unflatten(-1, (-1, block_size))
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil(amax * (1.0 / FP8_MAX)) if scale_fmt is not None else amax * (1.0 / FP8_MAX)
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), s.squeeze(-1).to(scale_dtype)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False,
                  scale_dtype: torch.dtype = torch.float8_e8m0fnu):
    """e2m1 over the last dim with an e8m0 scale per block (index keys / queries) or an e4m3
    scale per block (compressed KV), dequantized back into x.  model.py only calls the
    in-place form."""
    assert inplace, "model.py only uses fp4_act_quant in place"
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    n = x.size(-1)
    assert n % block_size == 0
    xb = x.float().unflatten(-1, (-1, block_size))
    amax = xb.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        # training's compressed KV: an all-zero group keeps a nonzero scale
        s = (amax.clamp_min(6 * 2.0**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = _pow2_ceil(amax.clamp_min(6 * 2.0**-126) * (1.0 / FP4_MAX))
    q = _round_e2m1((xb / s).clamp(-FP4_MAX, FP4_MAX))
    x.copy_((q * s).flatten(-2).to(x.dtype))
    return x


def _dequant_rows(q: torch.Tensor, s: torch.Tensor, block: int) -> torch.Tensor:
    """[..., K] codes times one scale per `block` along K -> fp32."""
    return (q.float().unflatten(-1, (-1, block)) * s.float().unsqueeze(-1)).flatten(-2)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
             scale_dtype: torch.dtype = torch.float32, block_size: int = 128) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K]^T; A e4m3 with a scale per (row, K block), B e4m3 with a scale
    per (N block, K block)."""
    n = b.size(0)
    bs = b_s.float().repeat_interleave(block_size, dim=0)[:n]
    a16 = _dequant_rows(a, a_s, block_size).bfloat16()
    return F.linear(a16, _dequant_rows(b, bs, block_size).bfloat16())


def fp4_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
             scale_dtype: torch.dtype = torch.float32, act_block_size: int = 128) -> torch.Tensor:
    """C[M,N] = A_e4m3[M,K] @ B_e2m1[N,K]^T; B packed two codes per byte along K (low nibble
    first) with an e8m0 scale per 32 along K."""
    a16 = _dequant_rows(a, a_s, act_block_size).bfloat16()
    return F.linear(a16, dequant_fp4(b, b_s).bfloat16())


def dequant_fp4(b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """An e2m1 weight [N, K/2] (two codes per byte along K, low nibble first) with an e8m0
    scale per 32 along K -> fp32 [N, K], exact."""
    codes = b.view(torch.uint8)
    codes = torch.stack([codes & 0x0F, codes >> 4], dim=-1).flatten(-2)
    return _dequant_rows(FP4_TABLE.to(b.device)[codes.long()], b_s, 32)


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                softmax_scale: float, query_chunk_bytes: int = 1 << 30) -> torch.Tensor:
    """Attention over gathered KV positions with a learned sink.  q [b, m, h, d] bf16, kv
    [b, n, d] bf16, topk_idxs [b, m, k] int32 with -1 for an empty slot.  The kernel's finite
    running-max floor (-1e30) makes an all-empty row come out zero; its bf16 cast of the
    probabilities before the PV product is kept (taken against the row's final max rather than
    the online one -- a rounding-order difference)."""
    b, m, h, d = q.shape
    k = topk_idxs.size(-1)
    out = torch.empty_like(q)
    step = max(1, query_chunk_bytes // max(1, b * k * d * 4))
    batch = torch.arange(b, device=q.device)[:, None, None]
    sink = attn_sink.float()[None, None, :]
    for m0 in range(0, m, step):
        m1 = min(m, m0 + step)
        idx = topk_idxs[:, m0:m1].long()
        valid = idx >= 0
        g = kv[batch, idx.clamp_min(0)].float()  # [b, mc, k, d]
        s = torch.matmul(q[:, m0:m1].float(), g.transpose(-1, -2)) * softmax_scale  # [b, mc, h, k]
        s = s.masked_fill(~valid[:, :, None, :], float("-inf"))
        mx = s.amax(-1).clamp_min(-1e30)  # [b, mc, h]
        p = torch.exp(s - mx[..., None])
        denom = p.sum(-1) + torch.exp(sink - mx)
        o = torch.matmul(p.to(torch.bfloat16).float(), g) / denom[..., None]
        out[:, m0:m1] = o.to(out.dtype)
    return out


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                      hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    """Split the hyper-connection mix projection into pre / post / comb, comb made doubly
    stochastic by Sinkhorn.  mixes [b, s, (2 + hc) * hc] fp32."""
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
