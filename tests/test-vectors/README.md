# Test vectors

This directory holds the fixtures the test runners and gates read.  The
official DeepSeek V4 Flash API vectors are described first; the rest are
listed at the end.

## Official API vectors

These vectors were captured from the official DeepSeek V4 Flash API using
`deepseek-v4-flash`, greedy decoding, thinking disabled, and
`top_logprobs=20`. The hosted API does not expose full logits, so these files
store the best logprob slice the API provides.

Files:

- `prompts/*.txt`: exact user prompts.
- `official/*.official.json`: official API continuations and top-logprobs.
- `official.vec`: compact C-test fixture generated from the official JSON.

Regenerate official vectors:

```sh
DEEPSEEK_API_KEY=... ./tests/test-vectors/fetch_official_vectors.py
```

Running the fetcher without `--only` also regenerates `official.vec`.

The C runner consumes `official.vec` directly:

```sh
./pulsar_test --logprob-vectors
```

(The `local-golden.vec` fixture and `--local-golden-vectors` were deleted
2026-09-02, L156: a May-2026 Metal capture of a different quantization that
had been red since this fork's first format change; fidelity to the source is
graded by `cuda-reference-gate` against the B300 capture instead.)

The `--logprob-vectors` runner opens its own engine with
`PULSAR_CUDA_PREFILL_CHUNK=2048` pinned for this strict official-vector check
(`PULSAR_TEST_VECTOR_FILE` points it at another `.vec`).  It is an
informational DeepSeek drift dashboard, not a gate: the served model is
quantized, so mismatches against the full-precision API are expected and
reported without failing the suite.  It skips an expert-pruned (REAP) model,
and skips the `long_memory_archive` case by design (see `pulsar_test.cpp`).

`official.vec` is intentionally trivial to parse from C: each case points to a
prompt file and each expected token is hex-encoded by bytes. The official JSON
files remain in the tree so the compact fixture can be audited against the raw
API response.

To inspect a local top-logprob dump manually:

```sh
./pulsar --nothink -sys "" --temp 0 -n 4 --ctx 16384 \
  --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
  --dump-logprobs /tmp/long_code_audit.ds4.json \
  --logprobs-top-k 20
```

## Other fixtures

Each of these belongs to the gate or test that reads it; the gate's recipe in
the top-level `Makefile` or its source under `tests/` is the authority for
what it holds and how it is regenerated.

- `prefill_bitexact_baseline-<ref>.bin`, `prefill_decode_baseline-<ref>.bin`,
  `prefill_cutlass_mxfp4_baseline-<ref>.bin`: the committed blobs of
  `make cuda-prefill-gate`, `make cuda-prefill-decode-gate` and
  `make cuda-prefill-gate-cutlass-mxfp4` (the Makefile's `*_BASELINE_REF`
  variables pick the ref; the `.md` beside a blob records its provenance).
- `kl-budget-*.txt`: the KL-direction budgets of the reference gates
  (`make cuda-reference-gate`; the `kl-budget-qwen-*` pair is Qwen's).
- `decode_floor_baseline.json`: `make decode-floor-gate`'s recorded floor.
- `exl3/`: the EXL3 codebook and trellis vectors of `make exl3-dequant-gate`.
- `opencode-nemotron-turn3.json`: a captured client turn for `tests/opencode_turns_live.py`.
- `vision-*-goldens.*`, `image-rope-goldens.bin`: the vision gates' goldens.
- `render-bytes-<ref>.txt`: `pulsar_test`'s chat-render byte goldens.
- `engram-l*.fix`, `qwen-ngram-small.vec`: the Engram and Qwen n-gram table gates.
- `qwen38/`, `nemotron_h/`, `nemotron_enc/`, `inkling/`: each family's
  tokenizer / chat-render gate fixtures.
