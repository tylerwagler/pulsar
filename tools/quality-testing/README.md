# Official-Continuation Quality Testing

This directory contains the 100 prompts and scripts used to compare local
checkpoint variants (pulsar containers) against a hosted model's official API
continuations.  The collector and the tracked `data/flash` set are DeepSeek V4's;
the scorer runs any generative family (it renders with the loaded family's own
chat renderer and tokenizer), and reads antirez/ds4's Qwen sets in place (section 5).

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
make -C tools/quality-testing        # = make tools/quality-testing/score_official at the repo root
```

The scorer links the engine's objects (`CORE_OBJS`, the rule is in the repo
root's Makefile) and runs on CUDA.  `--dry-run` needs no GPU.

## 3. Score checkpoint variants

```sh
tools/quality-testing/score_official /path/to/OLD-container \
  --out /tmp/old.tsv tools/quality-testing/data/flash

tools/quality-testing/score_official /path/to/NEW-container \
  --out /tmp/new.tsv tools/quality-testing/data/flash
```

```
score_official MODEL [--out OUT.tsv] [--ctx N] [--limit N] [--dry-run] [--dump-ids] SUITE...
```

- `SUITE` is a suite directory (its `manifest.tsv` is read and every row's
  `prompts/`, `continuations/`, `responses/` file is opened inside that
  directory, wherever the manifest says it was collected), or a manifest file
  (its paths are opened as written, relative to the working directory).
  Several suites share one model load; each gets its own summary.
- `--limit N` scores the first N cases of each suite (a smoke).
- `--ctx N` is the session's context; the default fits the longest case (at
  least 4096).
- `--dry-run` loads the container inspect-only (no GPU), renders and tokenizes
  every case and prints the token counts and the API alignment.  DeepSeek's
  vocabulary is not loaded by an inspect-only open, so a DeepSeek dry run is
  refused by name; Qwen's is.
- `--dump-ids` prints each case's prompt and continuation ids and the prompt's
  text on stderr.
- `--out` writes one row per case (`compare_scores.py` reads it).

Each prompt is rendered as one user message with thinking off and no system
message, by the family's renderer.  Every case is a cold prefill of its prompt
(the session is invalidated first), then the continuation's tokens are fed one
at a time; nothing is sampled.

Per suite the scorer prints:

- `cases`, `tokens`: cases and continuation tokens scored.
- `mean_nll`: the sum of `-log p(token)` over every continuation token, divided
  by the token count.  Lower is better.
- `top_token_agreement`: over the API-aligned cases, how often the local argmax
  is the API's most likely token at that position (the `top_logprobs` entry
  with the highest logprob; tied maxima agree if any is the local argmax).
  `first_listed_rule` takes the first listed alternative instead, which is how
  ds4 scored before its commit d6b516a.
- `mean_p_on_api_top1`: the mean probability the local model gives the API's
  most likely token.  `greedy_in_api_topk`: how often the local argmax is one
  of the API's listed alternatives (top 5 in the Qwen sets).
- `api_target_mae`: mean `|local logprob - API logprob|` of the emitted token.
- `greedy_is_continuation`, `first_match`, `mean_greedy_lcp`: the local argmax
  against the continuation itself (no API logprobs needed).

A case is API-aligned when its response has as many logprob positions as the
local tokenizer makes of the continuation and each position's bytes are the
local token's bytes.  A case that is not aligned still counts in `mean_nll` and
is left out of every API figure; the scorer names it.

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

## 5. Qwen3.8-Flash-Next: ds4's official sets

The Qwen continuations are not in this repository.  They are antirez/ds4's
(added in its commit `63b5dca`; read here at `b3f5018`, 2026-11-03), under
`gguf-tools/quality-testing/data/` of a ds4 checkout
(`~/Projects/AI/upstream/ds4` on the dev VM):

| Directory | Cases | Continuation tokens | Prompt tokens |
| --- | ---: | ---: | --- |
| `qwen38-flash-alibaba-100` | 100 | 5,696 | 20 to 34 each |
| `qwen38-flash-alibaba-long` | 12 | 766 | 1,971 to 23,985 each |

Alibaba's hosted `qwen/qwen3.8-flash` through OpenRouter, temperature 0, 64
tokens, top-5 logprobs, thinking off, an explicitly empty system message (which
the template renders as no system turn).  Copy the two directories to the GPU
host (5 MB) and pass them as suites:

```sh
tools/quality-testing/score_official $QWEN_GATE_MODEL --out /tmp/qwen-official.tsv \
  /path/to/qwen38-flash-alibaba-100 /path/to/qwen38-flash-alibaba-long
```

`--dry-run` on the published K4 container gives ds4's counts exactly: 5,696 and
766 continuation tokens, every prompt's token count equal to the API's
`usage.prompt_tokens`, 98 of 100 short cases aligned (cases 058 and 069 carry
damaged emoji bytes in the API's logprob metadata; 5,568 aligned tokens) and 12
of 12 long ones.  The container's `tokenizer.json` and `chat_template.jinja`
have the sha256 ds4's `collection.json` records.

ds4's own figures on a DGX Spark (its GGUF Q2 / Q4, 2026-09-15, first-listed
rule): short mean NLL 0.353 / 0.290, top-token agreement 88.90 % / 92.08 %;
long 0.154 / 0.125, 93.99 % / 97.26 %.  The hosted endpoint's precision is not
disclosed and it is "based on" Flash Next, so agreement with it is an external
check, not a proof of the weights.
