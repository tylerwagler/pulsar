# Pulsar

Pulsar is a native CUDA inference server for a small, fixed set of open-weight
model families. The product is one binary, **`pulsar-server`**, which loads one
model container and serves it over OpenAI-, Anthropic- and Responses-compatible
HTTP APIs, with tool calling, structured output, image input (where the model
has a vision tower), speculative decoding, multi-session batching and an on-disk
KV cache built for long agent sessions.

It is narrow on purpose. Each family is implemented end to end in this tree:
weight layouts, kernels, tokenizer, chat template, tool-call parser and KV state.
There is no generic model loader, and nothing links against another runtime. The
production target is one **NVIDIA DGX Spark (GB10, `sm_121`, ~128 GB unified
memory)**. Two-node tensor-parallel serving over a GB10 pair is supported for the
families that implement it.

Pulsar started as the CUDA-only fork of [antirez/ds4](https://github.com/antirez/ds4),
a DeepSeek V4 Flash engine. It has since grown a family plug-in layer and several
more models; see [Acknowledgements](#acknowledgements).

Status: beta. Releases are tagged on `dev`; what changed in this one is in
[docs/RELEASE_NOTES_0.5.0.md](docs/RELEASE_NOTES_0.5.0.md).

## Model families

A model is a **pulsar safetensors container** (see [Models](#models)). The
container's `general.architecture` key picks the family; anything else is refused
at load.

**GB10 (DGX Spark, `sm_121`) is the focus.** It is the production machine and
the only one the release battery runs on. **The only model with good test
coverage is Qwen3.8-Flash-Next.** It is the production model, and most of the
battery's serving legs run on it. Treat every other row below as less proven, to
the degree its coverage column says.

| Model (family) | Served id | Weight formats | Runs on | Features | Test coverage |
| --- | --- | --- | --- | --- | --- |
| **Qwen3.8-Flash-Next** (`qwen4_exp`) | `qwen3.8-flash-next` | EXL3 experts and head, MXFP8 / BF16 dense, NVFP4 drafter; PLE n-gram row file beside the container | GB10 | DFlash2 block drafter (default) or MTP; 262k context, 1M with `--rope-yarn 4`; banks + disk KV; images; TP=2 | **Good.** Production; most of the battery's live serving legs (scheduling, preemption, prefix reuse, resume, fan-out, structured output); its session, speculation, chunk-neutrality and KL-vs-BF16 reference gates; chat-template and vision-tower gates. Declared gaps (`tests/contracts.tsv`): TP on a real pair, whole-session payloads, mixed-prefill neutrality |
| **DeepSeek V4 Flash** Vision-Exp / 0731 (`deepseek4`) | `deepseek-v4-flash` | MXFP8 dense, MXFP4 / IQ2_XXS / EXL3 routed experts, BF16 head | GB10 (needs `sm_12x`) | DSpark drafter merged in the container; 1M context; banks + disk KV; images (Vision-Exp); TP=2 | **Partial.** The Vision-Exp container is the battery runner's primary model (kernels, bit-exact prefill and decode baselines, the reference gates, sessions, speculation) and runs four live legs. Not exercised in production in this release |
| **DeepSeek V4.1 Flash** (`deepseek4`) | `deepseek-v4-pro` (see note) | as above, EXL3 routed experts; Engram table on disk | GB10 | DSpark when the container carries it; 1M context; banks + disk KV; no images (its image path refuses); TP=2 | **Thin.** Renderer goldens only (`render-gate`); no model gate in the battery |
| **Nemotron 3.5 Super** (`nemotron_h`) | `nemotron-3.5-super` | EXL3 K4, NVFP4 W4A16 LiLiCorr drafter | GB10 | LiLiCorr or MTP drafter; banks + disk KV; no images; no TP | **Partial.** Its LiLiCorr container runs three runner legs (block speculation, scratch poison, bank adoption) and two live legs; KV-admission legs on a mini fixture; chat-template gate |
| **Inkling** (`inkling`) | `inkling-small` | EXL3 K3 | GB10 | single session; no drafter, banks, images or TP | **Minimal.** Tokenizer and chat-template gate only |
| **Nemotron 3.5 Embed-8B / Rerank-8B** (`nemotron_enc`) | `nemotron-embed` / `nemotron-rerank` | EXL3 4.0 | Turing `sm_75` (in service on an RTX 5000); GB10 builds | `/v1/embeddings`, `/v1/rerank`; several in one process | **Manual.** `tests/nemotron_enc_gate` run by hand; not in the battery |
| **Nemotron 3.5 Parse** (`nemotron_parse`) | `nemotron-parse` | f16 | Turing `sm_75` (in service on an RTX 5000); GB10 builds | `/v1/parse`, one page image a call | **Manual.** `tests/nemotron_parse_gate` against the HF fp32 reference, run by hand; not in the battery |

"Runs on" lists what has been run. Other GPUs are not tested. What they could
run follows from the code: the runtime floor is `sm_75`, and anything below it is
refused. Containers are refused at load by storage layout:

- `mxfp8_lt` needs `sm_100` or newer.
- MXFP4, IQ2_XXS and NVFP4 need the `sm_12x` family.
- `fp8_e4m3_soa_k` needs `sm_89` or newer.

Individual lanes refuse below their own floor: bf16 rows, EXL3 MoE and QSA
attention need `sm_80`, Tessera needs `sm_89`, and Qwen's GDN recurrence needs
`sm_90`. The decoder families therefore need GB10-class hardware. The gates build
only `sm_120f` and `sm_75`. Any other `CUDA_ARCH` is untested, whether or not it
builds.

The family capability flags (`src/engine/family.h`, `PULSAR_FAMILY_CAP_*`) are the
authority for features: a feature a family does not declare is refused by name,
never approximated.

Notes:

- **Qwen3.8-Flash-Next** is served from EXL3 weights. Its per-layer n-gram (PLE)
  table is a separate row file that the container names. Pulsar streams it from
  disk while the model runs, so keep the container on local NVMe. The trained
  context is 262,144 positions; `--rope-yarn 4` serves 1M (1,048,576).
- **DeepSeek V4.1** containers currently report the served id `deepseek-v4-pro`.
  This is a leftover of the old Flash/Pro naming (`ds4_served_model_id` in
  `src/engine/family_deepseek.cpp`). DeepSeek V4 **Pro** itself is no longer a
  loadable shape.
- The encoder and Parse families serve their own endpoints (`/v1/embeddings`,
  `/v1/rerank`, `/v1/parse`). They do not chat. One process can host several of
  them (`-m` repeated).

## Hardware

- **GB10 / DGX Spark**: build with `make cuda-spark` (`CUDA_ARCH=sm_120f`).
- **Turing** (encoder and Parse families): build with
  `make pulsar-server CUDA_ARCH=sm_75`. Other GPUs: see the table above.
- A binary started on a GPU it was not compiled for fails at startup with
  rebuild instructions. It does not crash later inside a kernel.
- Weights are fully resident. A model that does not fit is rejected at load;
  there is no partial or streamed weight path. The Qwen PLE rows are table data,
  not weights.

## Build

Requirements: Linux, the CUDA toolkit (the tree is built with 13.x), a C++23
compiler, `libpng` and `libjpeg` (the image decoder), and CUTLASS, which is an
external header-only dependency pinned in [`cutlass.pin`](cutlass.pin). It is
deliberately not a submodule.

```sh
make cutlass            # once: clone the pinned CUTLASS into ./cutlass
                        # (or pass CUTLASS_DIR=/path/to/cutlass to every make)
make cuda-spark         # GB10: rebuilds pulsar-server at CUDA_ARCH=sm_120f
make pulsar-server CUDA_ARCH=native   # another local GPU
make help               # the other targets
```

Plain `make` prints the targets; it does not pick one. The binaries land in the
repository root. Only `pulsar-server` is the release. `pulsar` (CLI and
diagnostics), `pulsar-bench`, `pulsar-eval` and `pulsar-agent` are development
tools: build them by name.

Tests and release gates are described in
[docs/QA_BEFORE_RELEASES.md](docs/QA_BEFORE_RELEASES.md) and
[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md). In short:

- `make test`: the main test runner. It needs a model in `PULSAR_TEST_MODEL`.
- `make cuda-regression`: modelless kernel smokes.
- `make gates-dev`: the fast tier.
- `make gates`: the full release battery.

## Models

Pulsar reads one format: a **safetensors container** in which every tensor is
stored in the layout its kernel reads, declared by name in the shard metadata
(`pulsar.kv`, `pulsar.tensors`, `pulsar.experts`; reader in
`src/engine/safetensors.cpp`). The engine converts nothing at load, and an
unknown or retired layout name is refused. GGUF is not read or written anywhere.
The old GGUF tooling is archived at the tag `archive/gguf-tooling-2026-09-24`.

Containers are built from Hugging Face checkpoints, or from EXL3 packs for the
EXL3 builds, by [`tools/container/build.py`](tools/container/README.md). How to
run it, the recipe format and how a build is checked against a served artifact
are in [docs/ARTIFACT_BUILD.md](docs/ARTIFACT_BUILD.md).

<!-- VERIFY-LOAD (0.5.0-rc1): confirm this container loads on this release before tagging; then drop this comment. -->
A published DeepSeek V4 Flash container (the Vision-Exp repack with MXFP4 routed
experts and the DSpark drafter merged in) is on Hugging Face as
[`ElytronAI/DeepSeek-v4-Flash`](https://huggingface.co/ElytronAI/DeepSeek-v4-Flash):

```sh
hf download ElytronAI/DeepSeek-v4-Flash --local-dir /path/to/DeepSeek-v4-Flash
```

## Run

```sh
./pulsar-server -m /path/to/container --ctx 262144
# Qwen3.8-Flash-Next at its 1M context:
./pulsar-server -m /path/to/qwen-container --rope-yarn 4
```

`-m` takes a shard directory or a single file. Without `-m` the server looks for
`./model` or `./model.safetensors`, then the same names under `$PULSAR_MODEL_DIR`.
The server listens on `0.0.0.0:8000` by default and has **no authentication**: run
it on a network you control, or bind it with `--host 127.0.0.1` and put your own
authenticating proxy in front.

Flags a user normally sets:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m, --model PATH` | `./model`, then `$PULSAR_MODEL_DIR/model` | The container. Repeat it to host several encoder / Parse models in one process. |
| `-c, --ctx N` | 1048576 | Context per session. A context past the model's trained positions is refused. |
| `--rope-yarn F` | off | YaRN scaling over a native rope (Qwen: `4` = 1M positions). Refused under TP and by families without a rope. |
| `--host H` / `--port N` | `0.0.0.0` / `8000` | Bind address. |
| `--kv-disk-dir DIR` | `$XDG_CACHE_HOME/pulsar/kv-<model>` (else `~/.cache/pulsar/...`) | Disk KV cache directory. `""` disables it. |
| `--no-kv-disk` | | Disable the disk KV cache. |
| `--kv-disk-space-mb N` | no limit | Disk KV budget. |
| `--no-dspark` | | Run without the container's drafter (any family). |
| `--drafter NAME` | the container's | `mtp`, `dflash2` (Qwen) or `lilicorr` (Nemotron). |
| `--spec-depth N`, `--spec-tau F\|off` | the drafter's | Starting draft depth and draft stop threshold. |
| `--trace FILE` | | Write rendered prompts, cache decisions, output and tool-call events. |

The complete list (tensor parallelism, steering, request capture), every
`PULSAR_*` environment variable, and the server's behaviour in detail are in
**[docs/SERVER.md](docs/SERVER.md)**. `./pulsar-server --help [topic]` prints the
built-in help.

## API

| Endpoint | |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions: streaming, tools, `tool_choice`, `response_format`, `logprobs`, images |
| `POST /v1/responses` | OpenAI Responses (Codex CLI): streaming, function and custom tools, tool search, `text.format`, images |
| `POST /v1/messages` | Anthropic Messages (Claude Code): streaming, thinking, tools, `output_config`, images |
| `POST /v1/messages/count_tokens` | Anthropic token counting, rendered exactly as `/v1/messages` would |
| `POST /v1/completions` | OpenAI Completions: the prompt is continued raw, with no chat template |
| `GET /v1/models`, `GET /v1/models/{id}` | The loaded model only |
| `GET /health` | Readiness and status: 200 `ok`; 503 `draining` or `stalled` |
| `GET /healthz`, `GET /ping` | Liveness: always 200 while the process runs |
| `GET /version`, `GET /` | Build and model identity; endpoint list |
| `GET /metrics`, `GET /metrics/stream` | Prometheus text (vLLM-compatible names plus `pulsar:*`); an SSE change feed |
| `POST /v1/embeddings`, `POST /v1/rerank`, `POST /v1/parse` | Encoder and Parse families only |

Request fields, thinking and effort controls, tool-call handling, structured
output and image rules are in [docs/SERVER.md#requests](docs/SERVER.md#requests).
Ready-made settings for opencode, Pi, Codex CLI and Claude Code are in
[docs/SERVER.md#client-setup](docs/SERVER.md#client-setup).

## How serving behaves

- **Banks.** A server holds a pool of warm session states ("banks"), sized at
  startup against a measured memory budget (up to 16). Concurrent decoders share
  one batched step, and queued prompts ride the decode rounds. A request is
  routed to the bank holding the longest matching prefix. A prompt that does not
  fit can preempt a running decoder, which resumes later exactly where it was.
  Requests that arrive together on a new shared prefix wait for its first prefill
  instead of each prefilling it.
- **Disk KV cache.** On by default. Conversations persist as content-addressed
  segment chains, cut on a fixed token grid, so copies and branches of one
  conversation share segments. A returning conversation restores the deepest
  stored state that matches it, in slices between decode rounds, and only when
  reading it beats recomputing it.
- **Memory floor and stall watchdog.** The server derives one free-memory floor
  at startup and checks it before every bank, prefill chunk and disk write. When
  memory runs short it spills idle banks, then preempts, and only then refuses
  the request with a retryable 503. A watchdog thread reports a worker that has
  stopped making progress on `/health` (503 `stalled`) and dumps its state to the
  log once.
- **Speculative decoding** is exact. A draft is accepted only if it is the token
  the plain sampler would draw at that position, at every temperature. Greedy
  continuations can still differ from plain decode by float rounding on near
  ties, as with any batched engine.

The details, defaults and the metrics that show each mechanism are in
[docs/SERVER.md](docs/SERVER.md).

## Documentation

- [docs/SERVER.md](docs/SERVER.md): the server reference (flags, environment,
  endpoints, request fields, scheduling, disk KV, memory, metrics, TP, clients).
- [docs/RELEASE_NOTES_0.5.0.md](docs/RELEASE_NOTES_0.5.0.md): what changed since 0.4.2.
- [docs/DESIGN.md](docs/DESIGN.md): the shape of the engine. Also the Doxygen
  main page: `doxygen docs/Doxyfile` from the repository root.
- [docs/ENGINEERING-RULES.md](docs/ENGINEERING-RULES.md),
  [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md),
  [docs/QA_BEFORE_RELEASES.md](docs/QA_BEFORE_RELEASES.md): how changes are made
  and gated.
- [AGENTS.md](AGENTS.md): source layout and conventions (written for coding
  agents; useful for people too).
- [docs/ARTIFACT_BUILD.md](docs/ARTIFACT_BUILD.md) and
  [tools/container/README.md](tools/container/README.md): building model
  containers.
- [docs/tensor-parallel-port.md](docs/tensor-parallel-port.md) and
  [docs/tensor-parallel-split.md](docs/tensor-parallel-split.md): the
  tensor-parallel design records.
- [dir-steering/README.md](dir-steering/README.md): directional steering vectors.
- [speed-bench/README.md](speed-bench/README.md) and
  [tools/serve-bench/](tools/serve-bench/): speed measurement.
- [tests/test-vectors/README.md](tests/test-vectors/README.md): reference vectors
  used by the gates.
- [docs/history/](docs/history/): dated reviews, bring-up notes and finished
  plans, kept for the record and not maintained.

## Acknowledgements

Pulsar exists because of the work below. Code that is vendored or ported keeps
its upstream licence and notice.

- **[antirez/ds4](https://github.com/antirez/ds4)** (Salvatore Sanfilippo and
  contributors): the DeepSeek V4 engine this tree was forked from. Its MIT
  copyright notice is kept in [`LICENSE`](LICENSE).
- **[llama.cpp / GGML](https://github.com/ggml-org/llama.cpp)** (Georgi Gerganov
  and contributors): Pulsar does not link GGML, but its quantization layouts and
  tables, host-side quant and dot logic, and the MMQ kernels that the 2-bit
  routed-expert path descends from were retained or adapted under the MIT
  licence. The GGML authors' notice is kept in [`LICENSE`](LICENSE); see
  [src/cuda/mmq/VENDOR.md](src/cuda/mmq/VENDOR.md).
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** (NVIDIA, BSD-3-Clause): the
  MXFP4 block-scaled grouped GEMM, and the memory-efficient attention kernel from
  its examples (originally from xFormers), vendored verbatim in `src/cuda/fmha/`
  ([VENDOR.md](src/cuda/fmha/VENDOR.md)).
- **[xgrammar](https://github.com/mlc-ai/xgrammar)** (MLC, Apache-2.0): the
  grammar engine behind structured output, vendored verbatim in
  `src/vendor/xgrammar/` with its `LICENSE` and `NOTICE` (pin in `UPSTREAM`).
- **[exllamav3](https://github.com/turboderp-org/exllamav3)** (turboderp): the
  EXL3 trellis format. Pulsar's EXL3 decode is graded byte-exact against it, and
  the Qwen3.8-Flash-Next containers are built from turboderp's EXL3 packs.
- **[Tessera](https://github.com/RobTand/tessera)** (Robert Tand): the fused
  routed-window kernel, vendored in `src/cuda/mmq/` with its licence
  ([VENDOR-TESSERA.md](src/cuda/mmq/VENDOR-TESSERA.md)), and
  **[PrismaQuant](https://github.com/RobTand/prismaquant)**, the measured-KL
  mixed-precision allocation used for the DeepSeek builds.
- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** and its **b12x**
  kernels: the model for the SM12x NVFP4 decode and the scale-plane layout
  (`src/cuda/pulsar_cuda_nvfp4.cu`).
- **[linenoise](https://github.com/antirez/linenoise)** (BSD-2-Clause): line
  editing in the development CLI (`src/vendor/linenoise.*`).
- The model authors: **DeepSeek** (V4 Flash, V4.1), **Qwen** (Qwen3.8-Flash-Next),
  **NVIDIA** (Nemotron 3.5 Super, Embed, Rerank and Parse, and the LiLiCorr
  drafter), the **Inkling** authors, and **incoai** (the DFlash2 drafter).
- **[REAP](https://arxiv.org/abs/2510.13999)**: the router-weighted expert pruning
  used by earlier DeepSeek builds.
- **NVIDIA Flare**: the value-per-time idea behind the fused-lane round
  selection (L312).

## Licence

MIT; see [`LICENSE`](LICENSE). Vendored components carry their own licences
beside them: CUTLASS and its examples (BSD-3-Clause), xgrammar (Apache-2.0, with
dlpack under Apache-2.0 and picojson under BSD-2-Clause), Tessera
(`src/cuda/mmq/LICENSE-TESSERA`) and linenoise (BSD-2-Clause).
