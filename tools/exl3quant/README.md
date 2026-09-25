# tools/exl3quant -- V4.1-Flash routed experts to EXL3, calibrated on the source model

`v41_stream.py` quantizes all 40 x 384 x 3 routed-expert projections of DeepSeek-V4.1-Flash
to EXL3 (exllamav3's trellis format) at every requested rate K in one pass, from Hessians
collected on the **unquantized** model. The output shards are what `tools/container/build.py
--exl3` already takes (the same tensor names as the MiaAI checkpoint), one file per layer per K,
so a mixed-K artifact is a choice of shard per layer.

## How it works

- **The forward is DeepSeek's.** The snapshot's own `inference/model.py` runs unmodified, one
  `Block` resident at a time; `refkernels.py` replaces its tilelang `kernel` module with plain
  torch. Every GEMM operand the reference builds (e4m3 / e2m1 times a power-of-two scale) is
  exact in bf16, so the port changes summation order only -- and it runs on any GPU (the
  tilelang fp8 GEMM returns NaN on sm_100).
- **Layer streaming.** Every calibration row passes layer L before layer L+1 loads; the
  hyper-connection stream and what V4.1 hands down the stack (compressed KV, index keys, top-k,
  candidate blocks) live in host memory between layers. Engram rows are read from the shard by
  offset; the 2 x 94 GiB tables are never loaded.
- **Hessians: exllamav3's block-sparse recipe.** One gate/up Hessian per layer over every token's
  FFN input; per expert, a down Hessian with every expert run on every token (the converter's
  `calibration_all_experts`). Both from the fp8 activation the GEMM sees.
- **Quantization: exllamav3's `quantize_exl3_batch`** at the converter's defaults (mul1 codebook,
  output scales always, sigma_reg 0.025, seed = layer), batched as the converter batches a
  block-sparse MLP. The source-model stream makes one Hessian set serve every K.
- **Calibration rows:** exllamav3's default mix (the MiaAI V4.1 quant used 250 x 2048), plus any
  `--extra-corpus` jsonl packed into more rows.

## Requirements

- One CUDA GPU with ~40 GB free for the forward (a layer's experts are 6.8 GB e2m1, its down
  Hessians 8.2 GB fp32); more GPUs speed up quantization (`--quant-devices`).
- Host RAM ~ 90 KB per calibration token for the stream (250 x 2048 rows: ~45 GB) plus page
  cache for the Engram tables if you want them warm.
- Disk: the HF snapshot (443 GB, all 48 shards -- 47/48 hold Engram), `hessians/` 4.1 GB per
  layer, `state/` one stream copy, outputs ~3.2 GB per layer at K=2 and ~4.8 GB at K=3.
- Python: torch >= 2.9 (float4_e2m1fn_x2 / float8_e8m0fnu), transformers, safetensors, numpy,
  sympy, pillow; an exllamav3 checkout with its extension built (quantizer and calibration data).

## Run

```sh
# checks first: kernel port (CPU is fine), then the quantizer path on the GPU
python tools/exl3quant/test_refkernels.py --ref $V41
python tools/exl3quant/test_quantize.py --ref $V41 --exllamav3 $EXL3 --device cuda:0

# Engram reads ~24 random rows per token from shards 47/48 (2 x 94 GiB): read them once
# sequentially so the run's row gathers hit page cache, not the disk
cat $V41/model-00047-of-00048.safetensors $V41/model-00048-of-00048.safetensors > /dev/null

# the whole forward on a few rows, no quantization: the perplexity at the end is the gate
python tools/exl3quant/v41_stream.py --ref $V41 --exllamav3 $EXL3 --out $OUT-ppl \
    --rows 16 --k none --device cuda:0

python tools/exl3quant/v41_stream.py --ref $V41 --exllamav3 $EXL3 --out $OUT \
    --k 2,3 --device cuda:0 --quant-devices 0,1,2,3 \
    --extra-corpus /path/calib-diverse-ds4-v1.jsonl
```

Measured on GB10 (sparky, 7 rows x 128 tokens, shards over NFS): layer 0 in 16 s, the Engram
layer 1 in 290 s (random row reads over NFS), the first compressed-KV/indexer source layer 2
in 75 s; Hessians finite and symmetric. Checkpoint tensors are cloned into host memory before
they go to the GPU (a page-faulting mmap -> CUDA copy runs at ~5 MB/s on GB10).

A killed run restarts at the layer after the last one in `$OUT/state/after.safetensors`
(same command). `log.jsonl` has one record per layer stage; after layer 39 the run reports
perplexity over the calibration text rows -- the end-to-end check that the streamed forward is
the model (a broken layer shows up as a perplexity in the hundreds).

## Output

| path | contents |
|---|---|
| `exl3-kK/model-layerNN.safetensors` | layer NN's experts at rate K: `layers.NN.ffn.experts.E.wN.{trellis,suh,svh,mul1}` |
| `exl3-kK/proxy-layerNN.json` | exllamav3's proxy error per tensor -- the ordering signal for which layers earn the higher K |
| `hessians/layerNN.safetensors` | the raw sums: `gate_up` [5120, 5120], `down` [384, packed upper triangle of 2304 x 2304]; `count` in the metadata |
| `calib_rows.safetensors` | the token rows and which of them are text |

Bring every per-K directory home before the box is destroyed; then a mixed artifact is a
directory of symlinks to the chosen `model-layerNN.safetensors` per layer, fed to
`tools/container/build.py emit --exl3 <dir>`.
