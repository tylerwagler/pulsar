# tools/reap

`audit_reap_router.py MODEL_DIR...` -- the battery's `cuda-reap-router-audit`: a REAP-compacted
checkpoint whose router was left in source expert order produces fluent text from the wrong
experts, so every artifact is checked before it is loaded.  Today's checkpoints do not declare
`reap.enabled` and pass with a note; one that does is refused until a layout-aware router reader
exists.  The REAP-25 survivor map, its recovery script and the GGUF-era transplant tooling were
archived 2026-09-24 (tag `archive/gguf-tooling-2026-09-24`, L247).
