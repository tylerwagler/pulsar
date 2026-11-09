# Official-Continuation Quality Testing

This directory contains the 100 prompts and scripts used to compare local
DeepSeek V4 checkpoint variants (pulsar containers) against official DeepSeek V4
API continuations.

The metric is target-token negative log likelihood: collect a deterministic
official continuation, then ask each local checkpoint how much probability it assigns
to that exact continuation token by token.  This avoids judging quality from one
sampled answer.

## 1. Collect Official Continuations

```sh
export DEEPSEEK_API_KEY=...
python3 tools/quality-testing/collect_official.py \
  --prompts tools/quality-testing/prompts.jsonl \
  --out tools/quality-testing/data/flash \
  --count 100 \
  --max-tokens 24
```

Use one output directory per official model.  The default model is Flash, so
`data/flash` is the recommended path for Flash continuations.  For PRO:

```sh
python3 tools/quality-testing/collect_official.py \
  --model deepseek-v4-pro \
  --prompts tools/quality-testing/prompts.jsonl \
  --out tools/quality-testing/data/pro \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 20
```

The script writes:

- `data/<model>/prompts/case_*.txt`
- `data/<model>/continuations/case_*.txt`
- `data/<model>/responses/case_*.json`
- `data/<model>/manifest.tsv`

The prompt list is tracked in `prompts.jsonl`; the official responses are not
tracked because they are derived from an external API.

## 2. Build The Local Scorer

```sh
make -C tools/quality-testing
```

The scorer links against the engine's objects (its Makefile builds
`pulsar-bench` at the repo root first, which compiles them) and runs on CUDA.
`CUDA_ARCH=...` passes an explicit `-arch`.

## 3. Score checkpoint variants

```sh
tools/quality-testing/score_official \
  /path/to/OLD-container \
  tools/quality-testing/data/flash/manifest.tsv \
  /tmp/old.tsv \
  4096

tools/quality-testing/score_official \
  /path/to/NEW-container \
  tools/quality-testing/data/flash/manifest.tsv \
  /tmp/new.tsv \
  4096
```

The last argument is the context size (default 4096).  Use `data/pro/manifest.tsv` for PRO checkpoints.  The scorer and comparator do not
care which model produced the manifest; the manifest path selects the
continuation set.

## 4. Compare

```sh
python3 tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```

Output fields:

- `avg_nll`: average negative log likelihood; lower is better.
- `delta_new_minus_old`: negative means the new checkpoint fits the official
  continuation better.
- `case_wins_new_old_ties`: per-prompt NLL wins.
- `first_token_matches`: how often the local greedy first token matches the
  official first token.
- `avg_greedy_lcp`: average greedy longest common prefix against the official
  continuation.
