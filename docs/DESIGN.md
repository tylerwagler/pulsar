# pulsar — design overview

CUDA inference engine for DeepSeek-V4-Flash, targeting a single NVIDIA GB10
(sm_120/sm_121, 48 SMs, unified memory). This page is the entry point for the
generated API documentation; it describes the shape of the system and where to
look for what.

## Layout

| directory | what lives there |
|---|---|
| `src/engine/` | session lifecycle, prefill/decode graph encoding, KV bookkeeping, tokenizer |
| `src/cuda/` | kernels: attention, MoE, matmul/GEMV, compressor, indexer, drafter |
| `src/server/` | HTTP server, scheduler, request/slot machinery, OpenAI/Anthropic-shaped APIs |
| `src/cli/` | `pulsar-cli` (chat), `pulsar-bench` (throughput), `pulsar-eval` (quality) |
| `src/agent/` | agentic tool loop built on the server |
| `src/lib/` | shared helpers (KV store, help text) |
| `src/vendor/` | third-party: `rax` radix tree, `linenoise` (upstream copyright; not documented here) |
| `tests/` | gates and probes — see `docs/QA_BEFORE_RELEASES.md` |

## The decode path

A request moves through three stages:

1. **Sync / prefill** — `pulsar_session_sync()` brings the session's KV up to the
   prompt. If the live checkpoint is already a prefix, only the suffix is
   evaluated; otherwise the KV is rebuilt. Long prompts go through chunked
   prefill (`gpu_graph_prefill_chunked_range`).
2. **Decode** — tokens are produced one round at a time. Production decode is
   *batched*: `pulsar_session_decode_mixed()` runs one shared forward over rows
   belonging to several concurrent sequences.
3. **Speculation** — a small drafter (DSpark) proposes `k` tokens per round; the
   target model verifies them in the same batched forward. Accepted tokens are
   emitted; the rest are discarded and the KV rewound.

### Banks

Concurrent sequences live in **banks**: per-sequence KV slabs inside one session
object. The scheduler installs one bank at a time for classic work
(`server::bank_switch`) and batches rows from several banks for a decode round.

A session with no pool allocated behaves as **bank 0** — `gpu_graph_bank_pool_count()`
reports 1 and the bank accessors fall back to the classic tensors — so
single-sequence and multi-sequence are the same code path with a different bank
count, not two implementations.

### KV compression

Layers compress at a per-layer *ratio*: every `ratio` positions emit one
compressed row. Ratio-4 layers additionally maintain an indexer cache. The
**frontier** (`ms_n_comp[bank][layer]`) is the count of compressed rows a bank
has emitted, and the engine's central invariant is that it is *position-true*:

```
ms_n_comp[bank][layer] == (last_position_of_bank + 1) / ratio
```

`gpu_graph_multiseq_step_end()` asserts this after every batched step and fails
loud, because a violation is the silent-KV-corruption class rather than a
performance problem.

### Rewind

When emission stops mid-batch (a stop sequence, a token cap, a client
disconnect), tokens already committed to the KV but never emitted are "ghosts"
and must be undone. `pulsar_session::rewind(pos)` trims the checkpoint and
clamps every layer's frontier back to `pos / ratio`.

A best-effort second half restores the *values* of re-emitted compressed rows
from a projection ring, when that ring covers the rewound span. The ring is
deposited only for committed, non-multiseq chunks, so this restoration does not
apply to every path; the counter clamp does.

## Numeric formats

The checkpoint is BF16 residual, E4M3 dynamic activations, FP4 experts — the
engine reproduces that rather than promoting to f32. KV rows are NVFP4
(E2M1 + E4M3 block scales) at 384 B/row, unified across the raw ring, the
compressed pool, the drafter and chunked prefill. Weights are MXFP8/MXFP4
depending on tier; the MoE runs a grouped CUTLASS path for uniform-MXFP4 layers
and a per-expert tiled path otherwise.

## Correctness discipline

The engine is gated by a battery of GPU-resident tests (`make gates`) covering
bit-exactness against a recorded reference, frontier position-truth, rewind,
bank fork/evict/restore, mixed-batch M-independence, and speculative sampling.
`docs/QA_BEFORE_RELEASES.md` is the ledger of what each gate pins.

Two properties are load-bearing and worth stating plainly:

* **Bit-exactness is the currency.** Most changes are expected to leave decode
  output byte-identical; a moved hash is a claim that needs evidence, not a
  detail.
* **A gate that cannot fail is worse than no gate.** New gates are
  mutation-tested — broken deliberately to confirm they go red — before being
  trusted.

## Generated documentation

`doxygen docs/Doxyfile` → `docs/api/html/index.html`. Undocumented declarations
are reported in `docs/api/doxygen-warnings.log`; that log is the progress meter
for the in-flight documentation pass.
