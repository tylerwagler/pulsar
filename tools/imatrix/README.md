# DS4 Imatrix Pipeline

This directory contains the calibration dataset and instructions used to build
activation importance matrices for DeepSeek V4 Flash and Pro
quantization (the direct container builder, tools/container, L247).

The current imatrix target is the routed MoE path.  Flash has 43 layers and
256 routed experts per layer.  Pro has 61 layers and 384 routed experts per
layer.  Both variants expose three routed expert tensors per layer:

- `blk.N.ffn_gate_exps.weight`
- `blk.N.ffn_up_exps.weight`
- `blk.N.ffn_down_exps.weight`

For gate/up tensors, the collector records the squared FFN-normalized input
activation.  For down tensors, it records the squared routed SwiGLU row after
route weighting.  The result tells the quantizer which input columns are used
more heavily by the actual DS4 inference graph.

## 1. Build The Calibration Dataset

The tracked dataset is in `tools/imatrix/dataset/`.  Regenerate it from
the repository root with:

```sh
python3 tools/imatrix/dataset/build_ds4_imatrix_dataset.py
```

The important output is:

```text
tools/imatrix/dataset/rendered_prompts.txt
```

It contains DS4-rendered chat prompts, separated by visible
`DS4_IMATRIX_PROMPT` markers.  The prompts include:

- C/CUDA source-review prompts from this repository.
- Long-context snippets.
- Agent/tool-call prompts using DS4's DSML syntax.
- Language/prose rewriting, summarization, extraction, and translation prompts.
- `ds4-eval` GPQA Diamond, SuperGPQA, and AIME2025 benchmark prompts.
- Both thinking and non-thinking assistant prefixes.

The current tracked dataset has 4682 rendered prompts and roughly 2.91M tokens
by the coarse bytes/4 estimate.  Check
`tools/imatrix/dataset/manifest.json` for the exact generated-file
summary.

## 2. Collect The Imatrix

Use the DS4 runtime itself to collect routed MoE activation statistics.  The
collector uses the loaded checkpoint's metadata, so the same command shape works for
Flash and Pro.

Flash example:

```sh
./pulsar \
  -m /srv/models/<checkpoint-dir> \
  --imatrix-dataset tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out ../deepseek-v4-quants/imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat \
  --ctx 32768
```

Pro example with a smaller calibration budget:

```sh
./pulsar \
  -m /srv/models/<checkpoint-dir> \
  --imatrix-dataset tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out ../deepseek-v4-quants/imatrix/DeepSeek-V4-Pro-Instruct-routed-moe-ds4-small.dat \
  --imatrix-max-prompts 16 \
  --imatrix-max-tokens 32768 \
  --ctx 32768
```

Useful smoke-test limits:

```sh
./pulsar \
  -m /srv/models/<checkpoint-dir> \
  --imatrix-dataset tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out /tmp/ds4-test.imatrix.dat \
  --imatrix-max-prompts 1 \
  --imatrix-max-tokens 4096
```

The collector is CUDA-only because it hooks the layer-major CUDA prefill graph.
It does not change inference math; it reads the already materialized MoE inputs
and accumulates `sum(x[column]^2)` per routed expert.

The output format is llama.cpp's legacy binary `.dat` imatrix format.  DS4 packs
per-expert vectors into one entry per routed expert tensor:

```text
entry length = n_expert * n_columns
```

The quantizer slices the right expert's segment when quantizing each expert.
For Pro, that means each routed tensor entry contains 384 vectors instead of
Flash's 256.

## 3. Using the imatrix

The GGUF quantizer this section used to describe was archived 2026-09-24
(tag `archive/gguf-tooling-2026-09-24`, L247).  The `.dat` file is consumed by the
direct container builder's quantising producers (tools/container).

## 4. Evaluate

Useful local tools:

```text
tools/quality-testing/
```

`tools/quality-testing/` can score local GGUFs against official DeepSeek
API continuations by target-token negative log likelihood.

The Q4 imatrix file uploaded to Hugging Face was tested on 100 official
DeepSeek V4 Flash continuations:

```text
old Q4 avg NLL:         0.177357819
Q4 imatrix avg NLL:     0.173895148
relative NLL change:   -1.95%
case wins:              54 imatrix / 46 old
first-token matches:    83 imatrix / 81 old
avg greedy LCP:         12.21 imatrix / 11.94 old
```

## Compatibility

The `.dat` file is intentionally in llama.cpp's legacy imatrix format, so the
data is not conceptually tied to DS4.  In practice, it is immediately useful
only with a quantizer that understands DS4's tensor names and packed per-expert
entries.  The current `deepseek4-quantize` tooling does that.

Other quantizers can use the same imatrix if they implement the same
name mapping and per-expert slicing convention.  Without that DS4-specific
mapping, a generic imatrix loader will see valid data but will not know how to
apply the packed routed-expert vectors correctly.
