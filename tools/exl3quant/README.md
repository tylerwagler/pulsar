# tools/exl3quant -- DeepSeek-V4.1-Flash routed experts to EXL3, our own quantization (L285 E5 / L269 W3)

V4.1's routed experts (40 layers x 384, plus the DSpark drafter's 3 x 128) re-quantized from the FP4 source to
EXL3 (exllamav3's trellis format) at a per-layer rate K sized for the two-Spark pair, from Hessians collected on
the **unquantized** model over a coding-heavy calibration set.  Dense / attention / shared-expert tensors stay as
the builder's default recipe says (MXFP8 from the FP8 source); Engram rows stay row files on disk; the tower
passes through.  The output is what `tools/container/build.py --exl3` (L279's pulsar.recipe.v1 builder) reads.

| file | what |
|---|---|
| `calib.py` | the calibration rows: coding-heavy, rendered by V4.1's own `encoding/encoding.py` (below) |
| `deepseek_v41.py` | `DeepseekV41Config`, written as an exllamav3 architecture file: config, layer schedule (CSA sources, Engram, DSpark 128/3), the quantization plan, the census the checkpoint must pass |
| `v41_stream.py` | the driver: layer-streamed reference forward, Hessians, `quantize_exl3_batch`, the drafter, per-stage time + peak memory |
| `refkernels.py` | the reference's tilelang `kernel` module in plain torch (exact operands, summation order only) |
| `kmap.py` | the K map -> per-rank bytes (`budget`), the pulsar.recipe.v1 (`recipe`), the `--exl3` directory (`assemble`) |
| `grade_layer.py` | a quantized block vs the FP4 source on held-out activations (weight, functional, MoE output) |
| `test_refkernels.py`, `test_quantize.py` | the kernel port (CPU) and the quantizer path on two real experts (GPU) |

## How it works

- **The forward is DeepSeek's.** The snapshot's own `inference/model.py` runs unmodified, one `Block` resident
  at a time, on `refkernels.py`.  Every GEMM operand the reference builds (e4m3 / e2m1 times a power-of-two
  scale) is exact in bf16, so the port changes summation order only, and it runs on GB10.
- **Layer streaming.** Every calibration row passes layer L before layer L+1 loads; the hyper-connection stream
  and what V4.1 hands down the stack (compressed KV, index keys, top-k, candidate blocks), plus the attention
  inputs of the drafter's target layers 37-39, live in host memory between layers.  Engram layers (1, 14) read
  their 94 GiB table once, sequentially, keeping the rows the calibration set looks up.
- **Hessians: exllamav3's block-sparse recipe.** One gate/up Hessian per layer over every token's FFN input; a
  down Hessian per expert with every expert run on every token (`calibration_all_experts`), both from the fp8
  activation the GEMM sees.  Because the stream is the SOURCE model's, one Hessian set serves every K and
  per-layer shards of different K mix freely; `--from-hessians` re-quantizes a layer at another K without the
  forward.
- **Quantization: exllamav3's `quantize_exl3_batch`**, mul1 codebook (the only one pulsar reads), output scales
  always, sigma_reg 0.025, seed = the block's layer id; gate/up batched over the shared Hessian, down 16 at a time.
- **The drafter** is calibrated as it is served: at 256 sampled positions p of every row, the block [token p+1,
  4 noise tokens] runs through mtp.0-2 with main_x = main_norm(main_proj(attention inputs of 37-39 at p)) and
  each stage's window KV holding main_x's KV for p-127..p.
- **Held-out rows** (calib.py marks 4) are forwarded but kept out of every Hessian; at the quantized blocks their
  FFN inputs and routes are saved for `grade_layer.py`.

## Calibration set (`calib.py`; v1 = 64 + 4 held-out rows x 4096 tokens = 262,144 calibration tokens)

The served workload is Claude Code-style agentic coding: long sessions whose tokens are mostly tool results (file
contents, command output) around short tool calls and code.  So, by rows (~91% code / agentic):

| source | rows | content |
|---|---:|---|
| agentic-swe | 20 | SWE-agent trajectories over real GitHub repos (file views, searches, edits, test runs) |
| repo-reads | 16 | synthetic coding-agent sessions over local source trees (this engine, exllamav3, llama.cpp, vLLM): Grep -> Read -> Read -> Edit, tool results = the real files `cat -n` numbered |
| code-opencode | 12 | code generation / explanation (OpenCodeInstruct) |
| agentic-tools | 10 | function calling with reasoning (hermes reasoning-tool-use, thinking mode) and without (hermes-fc) |
| prose | 6 | ultrachat dialogue + wikitext |

Corpus: `/mnt/models/reap-corpus` (the REAP case corpus; its `heldout/` is not used).  A long conversation gives at
most 2 of its row windows, picked at random (not always its opening system prompt); short ones are packed.  4096
tokens per row (agentic contexts are long; V4.1's compressed attention only engages past the 128-token window).
262k tokens is 51x the 5120-dim gate/up Hessian and every expert's down Hessian sees every token.  exllamav3's
default mix (250 x 2048 web/wiki/random rows, what the public packs used) is not used.

## Run (sparky; every GPU command under `flock ~/gpu.lock`)

```sh
V41=/mnt/models/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/2cba9e42aa026125f3ed06c6d98c1db82f7ca027
python tools/exl3quant/calib.py --ref $V41 --corpus /mnt/models/reap-corpus --repo engine=$PWD/src \
    --repo exllamav3=$EXL3/exllamav3 --repo llama.cpp=.../llama.cpp/src --repo vllm=.../vllm/vllm --out calib.safetensors

python tools/exl3quant/test_refkernels.py --ref $V41
python tools/exl3quant/test_quantize.py --ref $V41 --exllamav3 $EXL3 --scratch $SCRATCH

# one layer end to end (the estimate): forward 0-19, Hessians + K2/K3/K4 at 20
python tools/exl3quant/v41_stream.py --ref $V41 --exllamav3 $EXL3 --calib calib.safetensors --out $OUT \
    --quant-layers 20 --layers 21 --k 2,3,4 --save-every 0
python tools/exl3quant/grade_layer.py --ref $V41 --exllamav3 $EXL3 --run $OUT --block layers.20

# the full run (Tyler's go): every layer at its K map rate, then the drafter
python tools/exl3quant/v41_stream.py --ref $V41 --exllamav3 $EXL3 --calib calib.safetensors --out $OUT \
    --kmap kmap.json --container $L279/tools/container --drafter
```

A killed run restarts at the layer after the last checkpointed stream (same command).  `--until HH:MM` starts no
layer the previous layer's time says would end after then, `--min-free-gb` none with too little disk.

## Into a container (L279's builder)

```sh
kmap.py budget   --kmap kmap.json --ref $V41 --container $L279/tools/container --dump plan-dump.json
kmap.py recipe   --kmap kmap.json --ref $V41 --container $L279/tools/container --default default-recipe.json --out recipe.json
kmap.py assemble --kmap kmap.json --ref $V41 --container $L279/tools/container --run $OUT --out $OUT/exl3-mixed
build.py emit --hf $V41 --exl3 $OUT/exl3-mixed --recipe recipe.json --out $CONTAINER --all
```

`default-recipe.json` is `build.py recipe --hf $V41`; `plan-dump.json` is `build.py plan --hf $V41 --recipe
default-recipe.json --dump plan-dump.json` (the non-expert tensors' served bytes).

## Output (under --out)

| path | contents |
|---|---|
| `exl3-kK/model-layerNN.safetensors`, `model-mtpNN` | the block's experts at K: `<block>.N.ffn.experts.E.wP.{trellis,suh,svh,mul1}` |
| `exl3-kK/proxy-*.json` | exllamav3's proxy error per tensor (the K-map ranking signal) |
| `hessians/*.safetensors` | `gate_up` [5120, 5120] and `down` [E, packed upper triangle of 2304 x 2304] fp32 sums; `count` in the metadata (4.2 GB per trunk layer) |
| `holdout/*.safetensors` | held-out rows' FFN inputs, route indices and weights at each quantized block |
| `state/after.pt` | the stream after the last checkpointed layer |
| `log.jsonl` | one record per stage: seconds, peak system memory (MemTotal - MemAvailable), peak CUDA allocation |
