# pulsar — design overview

CUDA inference engine for a fixed set of model families — DeepSeek V4
(Flash 0731 / Vision-Exp, V4.1), Qwen3.8-Flash-Next, Inkling, Nemotron-3.5
Super, and the Nemotron-3.5 encoder (Embed / Rerank) and Parse models —
targeting a single NVIDIA GB10 (sm_120/sm_121, 48 SMs, unified memory) or a
tensor-parallel group of them. This page is the entry point for the generated
API documentation; it describes the shape of the system and where to look for
what.

## Layout

| directory | what lives there |
|---|---|
| `src/engine/` | the family registry (`family.cpp`, `family.h`), the DeepSeek and Qwen families, session lifecycle, banks and grid checkpoints, the prefill loop and row-step driver, the speculation lane, KV bookkeeping and payloads, tokenizers, images |
| `src/families/` | family modules added as plugins: `inkling/`, `nemotron_h/`, `nemotron_enc/`, `nemotron_parse/` |
| `src/cuda/` | kernels: attention (MLA, GQA, QSA, GDN, Mamba-2), MoE (MXFP4, IQ2, EXL3), matmul/GEMV, compressor, indexer, drafters, sampler, vision, TP |
| `src/tp/` | the tensor-parallel transport (TCP / RDMA), gate scheduler and verify |
| `src/server/` | HTTP server, scheduler, request/slot machinery, OpenAI/Anthropic/Responses APIs, structured output, encoder/parser endpoints, the stall watchdog |
| `src/cli/` | `pulsar` (chat; built from `pulsar_cli.cpp`), `pulsar-bench` (throughput), `pulsar-eval` (quality) |
| `src/agent/` | agentic tool loop built on the engine |
| `src/lib/` | shared helpers: disk KV segment store, help text, tokenizer and chat frontends, the grammar boundary, JSON, unit types |
| `src/vendor/` | third-party: `linenoise`, `xgrammar` (upstream copyright; not documented here) |
| `tests/` | gates and probes — see `docs/QA_BEFORE_RELEASES.md` |

## Families

A model family is a `pulsar_family` (`src/engine/family.h`): its loader,
tokenizer and chat format, layer plan, session state, drafter, and a
capability mask (`PULSAR_FAMILY_CAP_*`). The registry in `family.cpp` maps the
container's `general.architecture` to exactly one family; everything else in
the core is family-neutral and an operation a family does not implement
refuses by name (`pulsar_family_require`). DeepSeek keeps its own forward; the
other generating families run their forward through the core row-step driver
(`step_driver.h`), one block op per layer kind.

## The decode path

A request moves through three stages:

1. **Sync / prefill** — `pulsar_session_sync()` brings the session's KV up to the
   prompt. If the live state already covers a prefix, only the suffix is
   evaluated; otherwise the deepest valid state is restored (a grid
   checkpoint in the bank, another bank's published checkpoint, or a disk
   segment chain) and the rest prefilled. Long prompts go through the
   family-neutral chunked prefill loop (`prefill_loop.cpp`), which yields
   between chunks so the server can interleave other work.
2. **Decode** — tokens are produced one round at a time. Production decode is
   *batched*: `pulsar_session_decode_multiseq()` / `_decode_mixed()` /
   `_decode_fused()` run one shared forward over rows belonging to several
   concurrent sequences (a fused step also carries queued prompts' chunks).
3. **Speculation** — a drafter (DSpark for DeepSeek, DFlash2 for Qwen,
   LiLiCorr for Nemotron-H, or a chained MTP head) proposes tokens per round;
   the target model verifies them in the same batched forward. Accepted tokens
   are emitted; the rest are discarded and the KV rewound. Acceptance is the
   plain sampler's own draw, so speculation never changes the sampled
   distribution.

### Banks

Concurrent sequences live in **banks**: per-sequence KV slabs inside one session
object. A decode round batches rows from several banks at once; the scheduler
installs a single bank (`server::bank_switch`) only for the operations that are
still one-bank-at-a-time — finishing a request, writing a checkpoint. There is
no per-slot decode lane: since L118 the batched quanta run at every n >= 1, so
a solo session is a batch of one.

A session with no pool allocated behaves as **bank 0** — `gpu_graph_bank_pool_count()`
reports 1 and the bank accessors fall back to the classic tensors — so
single-sequence and multi-sequence are the same code path with a different bank
count, not two implementations.

### Checkpoints

Each bank keeps a few **grid checkpoints** (`checkpoint.cpp`, L264): the state
at a grid point that its append-only pools do not already hold. A resume
restores the deepest checkpoint at or below the prefill frontier and prefills
from there, which is the cold prefill's own computation. On disk the same
states persist as **segment chains** (`src/lib/pulsar_segstore`,
`src/lib/pulsar_kvchain`), mirrored on every rank of a TP group.

### KV compression (DeepSeek V4)

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
and must be undone. A rewind trims the checkpoint and clamps every layer's
frontier back to `pos / ratio`.

A best-effort second half restores the *values* of re-emitted compressed rows
from a projection ring, when that ring covers the rewound span. The ring is
deposited only for committed, non-multiseq chunks, so this restoration does not
apply to every path; the counter clamp does.

## Numeric formats

Each container declares every tensor's storage layout by name (`native`,
`mxfp8_lt`, `cutlass_mxfp4`, `iq2_xxs_mmq_k`, the EXL3 `exl3m_k*` rates,
`nvfp4`, `fp8_e4m3_soa_k`), and the engine converts nothing at load. For
DeepSeek V4 the checkpoint is BF16 residual, E4M3 dynamic activations, FP4
experts — the engine reproduces that rather than promoting to f32. Its KV rows
are NVFP4 (E2M1 + E4M3 block scales) at 384 B/row on the 0731 profile, unified
across the raw ring, the compressed pool, the drafter and chunked prefill; V4.1
stores an E4M3 window row and an E2M1 compressed row (`pulsar_kv_row_style`).
Weights are MXFP8 / MXFP4 / IQ2 / EXL3 depending on tier and family.

## Correctness discipline

The engine is gated by a battery of GPU-resident tests (`make gates`) covering
bit-exactness against a recorded reference, fidelity against the B300
reference, frontier position-truth, rewind, bank fork/evict/restore,
mixed-batch neutrality, resume == cold, and speculative sampling, across every
hosted family. `docs/QA_BEFORE_RELEASES.md` is the ledger of what each gate
pins; `docs/ENGINEERING-RULES.md` is the rule book the gates enforce.

Two properties are load-bearing and worth stating plainly:

* **Bit-exactness is the currency.** Most changes are expected to leave decode
  output byte-identical; a moved hash is a claim that needs evidence, not a
  detail.
* **A gate that cannot fail is worse than no gate.** New gates are
  mutation-tested — broken deliberately to confirm they go red — before being
  trusted.

## Generated documentation

`doxygen docs/Doxyfile` (from the engine root) → `docs/api/html/index.html`,
with call, caller and collaboration graphs (needs graphviz `dot`).

Coverage is complete: every declaration in scope carries documentation, and
`docs/api/doxygen-warnings.log` is the check that it stays that way — treat a
new warning as a regression, not as a backlog item. Vendored files are
deliberately OUT of scope; the Doxyfile's exclude list names them (they are
byte-identical to their upstream pins and must stay that way for a clean
re-vendor).
