# Per-layer format maps

`v5mx4-format-map.json` is the prisma allocator's measured per-layer expert-format map for the
0731 / Vision-Exp builds and `ds4_engine.json` its engine descriptor.  Both are keyed by the
GGUF-era tensor names; the direct builder (L247) re-keys them by HF name through `names.py`.
The scripts that produced them were archived 2026-09-24 (tag `archive/gguf-tooling-2026-09-24`).
