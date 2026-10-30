# Recipes and format maps

Since L279 the builder (`../build.py --recipe JSON`) takes its formats from a recipe (`pulsar.recipe.v1`,
`../recipe.py`): one row per emitted tensor, by role (+ block, layers) or by HF-name pattern.  DeepSeek's default
recipe is generated from `policy.py` (`build.py recipe` prints it); every other family names one of these files.

| File | What it builds |
|---|---|
| `qwen38fn-u-e4-d5-dflash2.json` | Qwen3.8-Flash-Next U-e4-d5 with z-lab's DFlash2 block drafter (needs `--drafter dflash2 DIR`; the drafter's Linears and draft-vocabulary head `nvfp4` from BF16, the rest BF16; L299) |
| `qwen38fn-u-e4-d5.json` | Qwen3.8-Flash-Next U-e4-d5 without a block drafter |
| `qwen38fn-turboderp-4.05.json`, `qwen38fn-turboderp-6.05.json` | Qwen3.8-Flash-Next on turboderp's EXL3 4.05 / 6.05 bpw packs (L266) |
| `nemotron_h-k4-lilicorr.json` | Nemotron 3.5 Super VL on its K4 EXL3 pack, MTP omitted and NVIDIA's LiLiCorr block drafter written in its place (needs `--drafter lilicorr DIR`; its NVFP4 Linears `nvfp4`, the rest native; L298) |
| `nemotron_h-exl3-k4.json` | Nemotron 3.5 Super VL on its K4 EXL3 pack (with its MTP module) |
| `nemotron_h-bf16.json` | the BF16 reference container of any Nemotron-H Omni checkpoint (role rows; the vision tower consumed by name; L296) |
| `nemotron_enc-embed-exl3-k4.json`, `nemotron_enc-rerank-exl3-k4.json` | Nemotron-3.5 Embed-8B / Rerank-8B on a uniform 4.0 bpw EXL3 pack of the text trunk (L321) |
| `nemotron_parse-f16.json` | Nemotron-3.5 Parse, every Linear f16 from the F32 checkpoint (L321) |
| `inkling-bf16.json` | the BF16 reference container of any Inkling checkpoint (role rows) |
| `inkling-mini-k3.json` | the 3-layer Inkling mini on its K3 EXL3 pack (L294) |
| `v41-tp1-fixture.json` | a DeepSeek V4.1 layer-subset fixture (`build.py --layers`) for one-GPU bring-up (L269) |

Two files are GGUF-era records that nothing reads any more: `v5mx4-format-map.json` (the PrismaQuant allocator's
measured per-layer expert-format map for the 0731 / Vision-Exp builds) and `ds4_engine.json` (its engine
descriptor).  Both are keyed by GGUF-era tensor names; the `--format-map` path that read them left with L279, and
there is no IQ2 producer for its 91 IQ2 rows (see docs/ARTIFACT_BUILD.md).  The scripts that produced them were
archived 2026-09-24 (tag `archive/gguf-tooling-2026-09-24`).
