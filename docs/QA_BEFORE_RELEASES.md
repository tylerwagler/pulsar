# QA Before Releases

This is the release gate for Pulsar.  Run it before tagging or pushing a
release build.  The goal is not to prove every code path exhaustively; it is to
exercise the paths that have historically regressed: CUDA graph inference,
disk KV cache, server APIs, and the agent TUI/tool state
machine.

This fork is CUDA-only and targets the DGX Spark (GB10, ~128 GB unified
memory); all release testing happens on that host.  Do not run multiple huge
model processes at the same time.  Record the commit, GGUF file, context size,
and any non-default flags for every manual run.

## 1. Repository And Build Sanity

- Start from a clean tree except intentional release notes:
  `git status --short`.
- Build the release binary:
  `make clean && make cuda-spark` (this builds **only `pulsar-server`** — the
  shipped release).  Build the development tools by name with the same arch:
  `make pulsar pulsar-agent pulsar-bench pulsar-eval CUDA_ARCH=sm_120f`.
- Run whitespace checks before committing:
  `git diff --check`.
- Confirm `./pulsar --help`, `./pulsar-server --help`, and `./pulsar-agent --help` render
  cleanly, with readable section colors and no broken wrapping.

## 2. Core Regression Tests

- Run the default suite:
  `make test` (also the first entry of `make gates`, as `unit-test-gate`,
  since 2026-09-03 -- a battery run covers it).
- Run the CUDA smoke regression:
  `make cuda-regression`.
- Run the vector checks explicitly after any tokenizer, template, KV, kernel,
  quantization, or prompt-rendering change:
  `./pulsar_test --logprob-vectors`
  (the local golden-vector leg was retired 2026-09-02, L156; the reference
  gate `make cuda-reference-gate` with `PULSAR_REF_DIR` set is the fidelity
  oracle).
- Run server tests when HTTP, SSE, prompt rendering, cache policy, or tool-call
  replay changed:
  `./pulsar_test --server`.
- Build and run the agent unit tests: `make pulsar_agent_test CUDA_ARCH=sm_120f
  && ./pulsar_agent_test` (they are not part of `make test`).
- Run `./pulsar-eval --self-test-extractors`.

### CUDA gates ledger

**Run them with one command:**

```
make gates FRONTIER_MODEL=/srv/models/<artifact>.gguf
```

It runs every gate below, continues past failures so one break does not hide
the rest, prints a PASS/FAIL summary, and exits non-zero if any failed.

> ⚠ **Do not hand-run these individually and tick them off.** That is what this
> checklist used to say, and on 2026-08-14 a sweep found the prefill gate
> rotted (its baseline ref predated type-43, so it could not run *at all*), two
> attention gates uncompilable since an API parameter was removed that morning,
> and a real production bug — fp16 attention breaking mixed-batch prefill —
> that had shipped six days earlier.  **None of it was visible to the product
> build.**  A gate nobody invokes is a gate that silently stops compiling.
>
> Two rules that follow:
> * After removing or changing any engine API parameter, run `make gates`.
>   Gates are separate targets; the product build will not tell you.
> * Re-baseline `PREFILL_BASELINE_REF` whenever a numerics change **ships**, and
>   only then.  Anchoring it before a deliberate precision change makes the gate
>   fail forever, and a gate that always fails is one you learn to ignore.

The individual targets, all needing the GB10 and (except the first two) the
model.  Record pass/fail against the release commit:

- `make cuda-regression` — modelless kernel smokes.
- `make cuda-attn-gates` — fp16 attention kernel oracle (dense, compressed,
  indexed, decode-batch, one-row), banked cross-session KV-leak isolation
  (modelless).
- `make cuda-prefill-gate` — full-vocab frontier byte-compare against the
  `PREFILL_BASELINE_REF` blob.  **The baseline ref must postdate type-43**
  (the aligned-MMQ pre-store): a pre-type-43 baseline build cannot load the
  shipped artifact and the gate cannot certify the release.  Rebuild the blob
  with `make cuda-prefill-gate-baseline` after bumping the ref.
- `make cuda-evict-restore-gate` and `make cuda-fork-gate` — bank evict/restore
  bit-identity and fork==cold oracle (covers warm-fork routing).  Since L115
  the fork validators compare BYTES, not token ids, so this gate also covers
  forking a conversation onto its own history across token-boundary seams.
- `make cuda-rewind-gate` — compressor frontier position-truth across ghost
  rewinds (L120): counters equal pos/ratio at every mod-4 residue and the
  ratio-128 boundary, plus the value leg (a ghost rewind must not contaminate
  re-emitted comp rows) and the ratio-128 undo walk (L124).
- `make cuda-mseq-rewind-gate` — a rewind must clamp BOTH frontier
  representations. L120 clamped the scalars; the served path validates the next
  multiseq step against the PER-BANK slots (ms_n_comp[bank]), which rewind did
  not touch, so a rewound bank decoded again with no bank switch-away between
  was rejected with production's own "frontier not position-true". Reproduces
  that deterministically and asserts both copies agree.
- `make cuda-seam-gate` — token-boundary seams keep the live KV (L115).
  Discovers a seam pair from the model's own vocabulary, then pins the byte
  walk, the sync rescue, the shorter-echo shape, and a fork across seams.
  Needs `PULSAR_MSEQ_BANKS>=2` for its fork leg (the target sets it).
- `warm_fork_3way` / `warm_partial_fork_3way` (`make warm-fork-3way
  warm-partial-fork-3way`) — server-level warm-fork determinism.
- `make cuda-frontier-gate`, `make cuda-multiseq-gate`,
  `make cuda-multiseq-gate-nodspark` — multiseq isolation and throughput.
- `make cuda-bank-spec-gate`, `make cuda-accounting-gate`,
  `make cuda-algo-stability-gate` — bank/spec interaction, admission
  accounting exactness, cuBLASLt M-independence.
- `make cuda-mixed-prefill-gate`, `make cuda-mixed-neutrality-gate` — mixed
  decode+prefill batching neutrality.
- `make cuda-spec-sampling-gate` — speculative-sampling chi-square exactness.

## 3. Flash Inference Path

Use the normal Flash GGUF that 128 GB users run.

- One-shot CLI:
  `./pulsar -m ds4flash.gguf --ctx 32768 --nothink -p "Explain C pointers in one paragraph."`
- Thinking and max-thinking prompts:
  run one short coding prompt with default thinking and one with max thinking.
- Long-context recall:
  run the long name/number or archive recall test used for catching attention
  and MoE routing drift.
- Logprob sanity:
  `./pulsar --nothink --temp 0 --dump-logprobs /tmp/ds4-logprobs.json --logprobs-top-k 20 -p "..."`
  and inspect that the continuation is sane.
- Speed sanity:
  run `pulsar-bench` with `speed-bench/promessi_sposi.txt` and compare prefill,
  generation speed, and KV bytes with the last known good numbers for the same
  machine.
- Run a longer prompt that exercises routed experts past a few thousand tokens.

## 4. Disk KV Cache

Disk KV cache bugs are high impact for server users.

- Start the server with:
  `./pulsar-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192`.
- Run the same request twice and verify the second request hits cache.
- Fill the cache enough to trigger eviction; verify the newly-written entry is
  not evicted and useful anchors are retained.
- Test rejection of incompatible checkpoints when model, quantization, context,
  or raw/compressed KV layout changes.
- Test stripped agent sessions: `/strip <id>` then `/switch <id>` should rebuild
  by prefill and render sane history.

## 5. Server APIs

The server must keep compatibility across OpenAI, Responses, and Anthropic
clients.

- `GET /v1/models/deepseek-v4-flash` and `GET /v1/models/deepseek-v4-pro`
  should both serve whichever GGUF is loaded.
- Test OpenAI chat completion, OpenAI Responses, and Anthropic messages.
- Test SSE streaming with thinking enabled and disabled.
- Test keepalive during long prefill and confirm clients do not time out.
- Test `--trace` and confirm rendered prompts, cache decisions, generated text,
  and tool-parser events are useful without leaking unrelated state.
- `curl /metrics` and confirm the Prometheus page renders: spec-decode
  acceptance, token totals, request latency, per-slot generation phase, pool
  churn and admission pressure counters all present and moving after a request.
- Reasoning effort: send Anthropic requests at `low`, `high`, and `max` effort
  and confirm the reported `prompt_tokens` reflects the three distinct effort
  prefixes (and that `xhigh` maps to `max`).
- Anthropic server tools: send a Claude-Code-style request advertising
  `web_search_20250305` next to an ordinary tool and confirm the server-tool
  entry is dropped at parse (tool log line), the ordinary tool is rendered,
  and the model never emits a dead search call. The server executes no
  Anthropic server tools; the router in front of it owns web search.

## 6. pulsar-agent

The agent is the most stateful component.  Test it manually, not only by build.

- Startup banner, status bar, help, `/save`, `/list`, `/switch`,
  `/history`, `/compact`, `/new`, `/del`, and `/strip`.
- Ctrl+C during generation, during prefill, during a web fetch, and during a
  long tool call.  After `Stopped by user`, typing a new prompt must work.
- Queue messages while the model is busy.  Queued messages must not skip tool
  execution; after tool results, the queued user text must be provided.
- Read/search/edit/write tools:
  create a temp project, ask for edits, verify old/new and `[upto]` anchored
  edits fail safely on ambiguous matches and do not require retyping whole files.
- Real coding edit loop:
  delete `/tmp/mymandel`, ask pulsar-agent to create a small C ASCII Mandelbrot
  program there, build and run it, then in a second user turn ask for a small
  modification that should naturally use the edit tool, such as changing the
  ASCII character ramp or output dimensions.  Verify the agent edits the
  existing file instead of rewriting the whole project, and that the final
  program still builds and runs.
- Bash tools:
  test short output, large output truncation, non-zero exit output, long-running
  jobs, `bash_status`, and `bash_stop`.
- TUI:
  test multiline prompt editing, history navigation, queued prompt display,
  status bar fill to terminal width, syntax highlighting in Markdown/code blocks,
  and SSH/remote terminal flicker.

## 7. Download Script And Model Files

- Test `download_model.sh` in a temporary directory so local weights are not
  overwritten.
- Test one Flash target enough to verify URL, resume, Hugging Face CLI/curl
  behavior, file naming, and symlink policy.
- Verify legacy removed targets fail clearly.
- Verify README model names match the script and Hugging Face repository.
- Run the artifact-type gate on every shipped gguf:
  `python3 gguf-tools/audit_artifact_types.py MODEL.gguf`
  It fails if any tensor ships in a PLAIN type that has a pre-formatted twin
  (16 -> 42/43, 38 -> 41, 39 -> 40). Those are pure byte permutations, so the
  plain form is never *wrong* -- the engine just converts at first use and
  keeps a second device copy beside the mmap, silently. The drafter shipped
  0.429 GiB of type-38 double-store this way from its first build until
  2026-08-12, because it is built from a separate pinned type table
  (`gguf-tools/dspark_type_flags.txt`) that was never revisited when the main
  model moved to MXFP8_LT. Use `--census` for the full type breakdown.

## 8. Performance

- Run `pulsar-bench` on the release machine and compare with tracked CSV baselines.
- Confirm context buffer size, raw KV rows, compressed KV rows, and mmap behavior
  match expectations for 32k, 100k, and any release-advertised context size.

### Client-side serving gates

These talk HTTP to an already-running `pulsar-server`, because they measure what
a client experiences rather than what the engine reports about itself.  Start the
server with `--no-kv-disk` for all three: a warm disk checkpoint skips prefill
outright, so TTFT and wall-clock t/s from a warm run cannot be compared against a
cold one.  Each writes a machine-readable JSON record — keep it with the release
notes, since a number with no run record is not a baseline.

- `make decode-floor-gate` — five workloads (Python, Rust, TypeScript, CUDA, Go)
  at 512 tokens, concurrency 1.  **Scored on the worst workload, not the mean**:
  averaging one bad prompt shape against four healthy ones produces a number that
  still looks fine while the regression ships.  Fails (exit 1) when the worst
  workload drops more than `--tolerance` (default 10%) below the recorded floor.

  The floor is ours, captured with `make decode-floor-baseline` on a known-good
  build — not a competitor's published figure, which was measured on a different
  harness, depth and sampler and is therefore not a floor.  Re-record it
  deliberately when a change is *expected* to move decode, and say so in the
  release notes; never re-record it to make a red gate go green.

- `make sse-decode-bench` — decode rate measured between SSE delta arrivals,
  reported **alongside** the wall-clock rate rather than replacing it.  When the
  two disagree the gap is the finding: a slot starved behind another job's
  prefill shows a healthy inter-delta rate and a poor wall-clock one.

- `make context-coherence-probe` — plants three facts at the beginning, middle
  and end of a long context and asks for them back in free prose, with no grammar
  mask or constrained decoding (under a mask a wrong answer still comes out
  structurally perfect, so the probe cannot fail in the way that matters).  A
  checksum question requiring all three values at once guards against partial
  latching.  Fails (exit 1) if any depth misses a fact or the sum.

## 9. Release Sign-off

Do not sign off until:

- `make test` and `make cuda-regression` passed on the GB10 host.
- The CUDA gates ledger (section 2) passed against the release commit and the
  shipped artifact, with a post-type-43 prefill-gate baseline.
- The Flash inference path (CLI, thinking modes, long-context recall) was
  exercised.
- Disk KV cache was exercised.
- Server API streaming was exercised.
- Agent interruption and tool loops were exercised manually.
- Speed is within expected variance for the same hardware and model.
- `make decode-floor-gate` passed against the recorded floor, and its JSON run
  record is filed with the release notes.
- Any skipped item is written down with the reason.
