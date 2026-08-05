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

## Pulsar capture (maintenance window — CLI loads the model, so the server
must be down; follow the model-load discipline)

    for f in suite-v1/*.txt; do
      ./pulsar -m ds4flash.gguf --cuda --ctx 262144 --kl-stride 4 \
          --kl-file "$f" --kl-ref-dump "caps/$(basename $f .txt).klrf"
      python3 tools/fidelity/pulsar_kldump_to_jsonl.py \
          "caps/$(basename $f .txt).klrf" --name "$(basename $f .txt)" \
          --tag pulsar-$(git rev-parse --short HEAD) \
          --out "caps/jsonl/$(basename $f .txt).jsonl"
    done

## The ledger table

    python3 fidelity_compare.py --dir suite-v1/ref-<model> caps/jsonl

Report the "all" row's KL median/p95 + per-depth buckets per candidate. The
FIRST candidate measured must be current production pulsar — that row is the
standing divergence budget every speed lever is priced against (Plan 90 A2).
