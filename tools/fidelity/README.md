# Fidelity metrology (Plan 90 Track A)

Quantifies divergence-from-source for every pulsar configuration (and for
competitor stacks) by teacher-forcing one fixed suite of documents on every
rig and comparing per-token distributions. Kills the measure-against-ourselves
bias: the reference is the full-fat DSv4-Flash-0731 on the work rig.

## Pieces

- `build_suite.py` — assembles a content-addressed suite (short calib docs +
  one ~64k-token long doc for the drift-vs-depth curve + a DSML transcript).
  Suites are immutable; a change is a new suite version.
- `capture_vllm_prompt_logprobs.py` — run against a vLLM server (the work
  rig); captures per-prompt-token top-K logprobs via the `prompt_logprobs`
  sampling extension into the common JSONL format. Stdlib only, resumable.
- pulsar side: the engine's own `--kl-file X --kl-ref-dump Y [--kl-stride N]`
  teacher-forces the same document and dumps full-vocab logits (DS4KLRF1) plus
  a `.tokens.json` sidecar.
  - pulsar-vs-pulsar (exact, full-vocab KL): `--kl-file X --kl-score Y`.
  - cross-rig: `pulsar_kldump_to_jsonl.py` converts the dump to the common
    JSONL (top-K), then `fidelity_compare.py`.
- `fidelity_compare.py` — truncated KL (union-of-top-K + tail), top-1
  agreement, forced-logprob delta; bucketed by depth; markdown tables.

## Canonical suite

`/mnt/pve1-models/fidelity/suite-v1/` (NAS). sha256 of every doc is in its
`manifest.json`. Reference captures live next to it: `suite-v1/ref-<model>/`.

## Work-rig capture (one command)

    python3 capture_vllm_prompt_logprobs.py \
        --endpoint http://<rig>:8000 \
        --suite /path/to/suite-v1 --topk 20

Notes: needs vLLM `--max-logprobs >= topk` (default 20 works; 64 is better if
the rig can restart with `--max-logprobs 64`). The 64k-token long doc prefills
in one request — a few minutes on the rig. Output drops into
`suite-v1/ref-<model>/`; copy to the NAS path above if run elsewhere.

**Name the output dir after the WEIGHTS, not the served id.** vLLM reports
whatever display name the rig was launched with (ours answers "DeepSeek v4
Flash"), which does not identify the checkpoint. Pass `--out ref-<weights>`
and keep `model_root` in the header — A1 versions the reference by weights
identity, and a reference you cannot attribute to a checkpoint is not a
reference.

### Captured 2026-08-05 — `ref-DeepSeek-V4-Flash-0731/`

14 docs, 107,388 scored positions, top-20, `MANIFEST.json` carries sha256 per
file. Verified: 100% position coverage (rows = tokens−1; position 0 has no
prediction), zero ordering errors, and **the forced token was inside top-20 at
every position** — so the cap costs tail resolution but loses no forced-logprob
data. `long-mixed` came in at 83,643 tokens, which reaches past the 64k depth
point. NOT yet measured: the reference's own run-to-run noise floor under batch
contention (see "Reproducibility control" below).

## Pulsar capture (maintenance window — CLI loads the model, so the server
must be down; follow the model-load discipline)

Use the script; it preflights the things that have bitten us before (a live
`pulsar-server`, low MemAvailable, a plain-`make` binary):

    tools/fidelity/capture_pulsar_side.sh /path/to/ds4flash.gguf

## Reproducibility control (do this on an IDLE rig, ~1 min)

vLLM batches concurrent requests, and batch composition changes reduction
order — our own engine's 9.3k greedy spec-fork investigation traced a
reproducible divergence to exactly this (M=2 co-batched vs M=1 differing ~1
ULP, enough to flip a near-tie argmax). So a reference captured while someone
is using the rig is not guaranteed bit-reproducible.

Re-capture ONE short doc into a scratch dir and compare against the stored
reference:

    python3 capture_vllm_prompt_logprobs.py --endpoint http://<rig>:8000 \
        --suite <one-doc-suite> --topk 20 --out /tmp/ctl
    python3 fidelity_compare.py suite-v1/ref-<weights>/short-00.jsonl \
        /tmp/ctl/short-00.jsonl

Byte-identical (KL 0, top-1 100%) means the apparatus is inert and every ΔKL
is trustworthy to the floor. Anything else IS the noise floor and must be
quoted alongside every A3 row — a lever whose ΔKL sits under it has not been
measured, it has been guessed.

## The ledger table

    python3 fidelity_compare.py --dir suite-v1/ref-<model> caps/jsonl

Report the "all" row's KL median/p95 + per-depth buckets per candidate. The
FIRST candidate measured must be current production pulsar — that row is the
standing divergence budget every speed lever is priced against (Plan 90 A2).

KL columns print in scientific notation deliberately: a calibration sweep on
the real reference puts a 0.1% logit-scale perturbation at KL median 1.0e-07,
and the earlier `%.4f` formatting rendered that as `0.0000`. Since A2's budget
rule is a RATIO against the baseline row, a baseline that prints as zero makes
the gate unevaluable — the precision is load-bearing, not cosmetic.

Default depth buckets are `2048,9216,38912,65536`. The 65536 bound matters:
without it everything past 38912 collapses into one bin, and depth is where
the competitor gap grows (2.4x@2k → 3.3x@64k), so the deepest bucket is the
most diagnostic one in the table.

Sanity checks worth re-running after any change here (all verified 2026-08-05
against the real reference): ref-vs-itself must give KL 0 / top-1 100% / 0
skipped; ref-vs-a-different-doc must collapse to a near-zero align rate and
mass skips, so a mis-paired comparison cannot quietly produce a credible row.
