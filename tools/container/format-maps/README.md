# Per-layer format maps

`v5mx4-format-map.json` is the prisma allocator's measured per-layer expert-format map for the
0731 / Vision-Exp builds and `ds4_engine.json` its engine descriptor.  Both are keyed by the
GGUF-era tensor names.  Since L279 the builder reads neither: `--format-map` (and the re-key through
`names.py`) is gone -- formats are recipe rows (`pulsar.recipe.v1`, `recipe.py`).  Whether the 91 IQ2 rows get a
producer or the map retires is L279 step 7.  `qwen38fn-*.json` are the Qwen recipes (pulsar.recipe.v1, name rows).
The scripts that produced them were archived 2026-09-24 (tag `archive/gguf-tooling-2026-09-24`).
