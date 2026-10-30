# Imatrix collection

This directory contains the calibration dataset and the tools to collect
activation importance matrices (llama.cpp's legacy `.dat` format) with the
`pulsar` development CLI (`make pulsar`).  Two families implement collection
(`PULSAR_FAMILY_CAP_IMATRIX`): DeepSeek V4 (Flash / Pro) and Qwen4-exp
(Qwen3.8-Flash-Next).

The current imatrix target is the routed MoE path.  Flash has 43 layers and
256 routed experts per layer.  Pro has 61 layers and 384 routed experts per
layer.  Both variants expose three routed expert tensors per layer:

- `blk.N.ffn_gate_exps.weight`
- `blk.N.ffn_up_exps.weight`
- `blk.N.ffn_down_exps.weight`

For gate/up tensors, the collector records the squared FFN-normalized input
activation.  For down tensors, it records the squared routed SwiGLU row after
route weighting.  The result tells the quantizer which input columns are used
more heavily by the actual inference graph.

## 1. Build The Calibration Dataset

The tracked dataset is in `tools/imatrix/dataset/`.  Treat the tracked files as
the dataset: the generator below still names the upstream ds4 source files
(`ds4.c`, `ds4_server.c`, `ds4_eval.c`, `metal/*.metal`, ...) that this tree no
longer has and skips missing ones silently, so re-running it does not reproduce
the tracked prompts.  The generator is:

```sh
python3 tools/imatrix/dataset/build_ds4_imatrix_dataset.py
```

The important output is:

```text
tools/imatrix/dataset/rendered_prompts.txt
```

It contains DeepSeek-rendered chat prompts, separated by visible
`DS4_IMATRIX_PROMPT` markers.  The prompts include:

- C/CUDA source-review prompts from the upstream ds4 tree at generation time.
- Long-context snippets.
- Agent/tool-call prompts using DeepSeek's DSML syntax.
- Language/prose rewriting, summarization, extraction, and translation prompts.
- `pulsar-eval` GPQA Diamond, SuperGPQA, and AIME2025 benchmark prompts.
- Both thinking and non-thinking assistant prefixes.

The current tracked dataset has 4690 rendered prompts (manifest version 4) and
roughly 2.92M tokens by the coarse bytes/4 estimate.  Check
`tools/imatrix/dataset/manifest.json` for the exact generated-file
summary.

## 2. Collect The Imatrix

Use the `pulsar` CLI to collect routed MoE activation statistics.  The
collector uses the loaded checkpoint's metadata, so the same command shape works for
Flash and Pro.

Flash example:

```sh
./pulsar \
  -m /path/to/<container-dir> \
  --imatrix-dataset tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out DeepSeek-V4-Flash-routed-moe.dat \
  --ctx 32768
```

Pro example with a smaller calibration budget:

```sh
./pulsar \
  -m /path/to/<container-dir> \
  --imatrix-dataset tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out DeepSeek-V4-Pro-routed-moe-small.dat \
  --imatrix-max-prompts 16 \
  --imatrix-max-tokens 32768 \
  --ctx 32768
```

Useful smoke-test limits:

```sh
./pulsar \
  -m /path/to/<container-dir> \
  --imatrix-dataset tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out /tmp/ds4-test.imatrix.dat \
  --imatrix-max-prompts 1 \
  --imatrix-max-tokens 4096
```

The collector is CUDA-only because it hooks the layer-major CUDA prefill graph.
It does not change inference math; it reads the already materialized MoE inputs
and accumulates `sum(x[column]^2)` per routed expert.

### Qwen4-exp

The same command collects on a Qwen4-exp checkpoint (`PULSAR_FAMILY_CAP_IMATRIX`; the dataset walk, the caps
and the `.dat` frame are shared, `pulsar_family_imatrix` is the per-family half).  Each prompt is prefilled
whole through a session with an observer on the trunk's linears, so the entries are the exact bf16 rows the
weights read: every dense linear (`step_linear`: GDN / QSA projections, PLE), the shared expert's gate / up /
down, and per routed expert the gate_up input and the down input (the fold's SwiGLU rows, route weight folded
in).  Entry names are the checkpoint's tensor names.  The dataset must be rendered for Qwen (its chat template,
the same `===== DS4_IMATRIX_PROMPT` markers); the tracked `rendered_prompts*.txt` are DeepSeek-rendered.
Collection runs on one GPU (a tensor-parallel engine refuses).  `read_dat.py FILE` parses and validates any
`.dat` this tool writes:

```sh
python3 tools/imatrix/read_dat.py out.dat --quiet --expect-substr experts
```

The output format is llama.cpp's legacy binary `.dat` imatrix format.  DeepSeek packs
per-expert vectors into one entry per routed expert tensor:

```text
entry length = n_expert * n_columns
```

The quantizer slices the right expert's segment when quantizing each expert.
For Pro, that means each routed tensor entry contains 384 vectors instead of
Flash's 256.

## 3. Using the imatrix

Nothing in this tree consumes the `.dat` today.  The GGUF quantizer that did
(`deepseek4-quantize`) was archived 2026-09-24 (tag
`archive/gguf-tooling-2026-09-24`, L247), and the direct container builder
(`tools/container`) has no imatrix-weighted producer: its routed-expert formats
come from the source checkpoint (`cutlass_mxfp4` from DeepSeek's FP4) or from an
exllamav3 EXL3 pack.  `act_cache_to_imatrix_dat.py` converts a PrismaQuant
activation cache into the same `.dat` layout, and `read_dat.py` parses and
validates any `.dat` either tool writes.

## Compatibility

The `.dat` file is llama.cpp's legacy imatrix format.  DeepSeek entries are
named by the engine's binding names (`blk.N.ffn_{gate,up,down}_exps.weight`)
and pack every expert's vector into one entry, so a quantizer must slice
`n_columns` values per expert; a generic imatrix loader will read valid data
but will not know how to apply the packed routed-expert vectors.  Qwen entries
are named by the checkpoint's tensor names.
